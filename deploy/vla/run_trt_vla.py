#!/usr/bin/env python3
"""
End-to-end VLA inference with TensorRT.

Pipeline:
  4 history images -> TRT Vision Encoder -> visual_features
  text prompt -> tokenizer -> input_ids -> text embeddings (PyTorch embed_tokens)
  inject visual_features at image token positions
  append 24 learned query embeddings
  TRT/PyTorch LLM (single forward) -> hidden_states
  TRT Action Head -> 24 trajectory points (x, y)
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import tensorrt as trt
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from deploy.vla.utils import load_checkpoint
from vla.lightning import VLAPlModule
from vla.utils.common import build_transform

DEFAULT_CHECKPOINT = os.path.join(_PROJECT_ROOT, "outputs", "nuscenes_internvl2_5_real_v2", "checkpoints", "last.ckpt")
DEFAULT_BASE_MODEL = "/root/embodvla/models/OpenGVLab/InternVL2_5-1B"
DEFAULT_VISION_ENGINE = os.path.join(_PROJECT_ROOT, "deploy", "outputs", "internvl2_vision_dynamic.trt")
DEFAULT_LLM_ENGINE = os.path.join(_PROJECT_ROOT, "deploy", "outputs", "vla_llm_hidden.trt")
DEFAULT_ACTION_ENGINE = os.path.join(_PROJECT_ROOT, "deploy", "outputs", "vla_action_head.trt")


def _get_model_path_from_checkpoint(checkpoint_path: str, fallback: str = DEFAULT_BASE_MODEL) -> str:
    """Read the saved model_path from the Lightning checkpoint hyper-parameters."""
    try:
        ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
        hp = ckpt.get("hyper_parameters", {})
        model_path = hp.get("model_path")
        if model_path and os.path.exists(model_path):
            print(f"[Load] Detected backbone model_path from checkpoint: {model_path}")
            return model_path
    except Exception as e:
        print(f"[Warn] Could not read model_path from checkpoint: {e}")
    print(f"[Load] Using fallback base model: {fallback}")
    return fallback

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


class TRTRunner:
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]

    def infer(self, feed_dict: dict) -> dict:
        device_inputs = {}
        outputs = {}
        for name, arr in feed_dict.items():
            arr = np.ascontiguousarray(arr)
            self.context.set_input_shape(name, arr.shape)
            device_inputs[name] = torch.from_numpy(arr).cuda()

        for name in self.input_names:
            if name not in feed_dict:
                shape = tuple(self.context.get_tensor_shape(name))
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                out = np.empty(shape, dtype=dtype)
                outputs[name] = torch.from_numpy(out).cuda()

        for name, tensor in device_inputs.items():
            self.context.set_tensor_address(name, tensor.data_ptr())
        for name, tensor in outputs.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()

        return {name: tensor.cpu().numpy() for name, tensor in outputs.items()}


class PyTorchLLMRunner:
    def __init__(self, llm: torch.nn.Module):
        self.llm = llm
        self.llm.eval()

    def infer(self, inputs_embeds: np.ndarray, attention_mask: np.ndarray, position_ids: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = self.llm(
                inputs_embeds=torch.from_numpy(inputs_embeds).to(self.llm.device),
                attention_mask=torch.from_numpy(attention_mask).to(self.llm.device),
                position_ids=torch.from_numpy(position_ids).to(self.llm.device),
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            return out.hidden_states[-1].cpu().numpy()


class VLATRTInference:
    """Loads all VLA deployment components once and provides an `infer` method."""

    def __init__(
        self,
        checkpoint_path: str,
        base_model_path: str,
        vision_engine_path: str,
        llm_engine_path: str,
        action_engine_path: str,
        llm_backend: str = "trt",
        dtype: torch.dtype = torch.float32,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.num_queries = 24

        print(f"[Load] Loading VLA components from checkpoint {checkpoint_path}...")
        pl_module = VLAPlModule.load_from_checkpoint(
            checkpoint_path,
            model_path=base_model_path,
            num_queries=self.num_queries,
            freeze_vision=True,
            freeze_llm=False,
            freeze_mlp1=True,
            action_head_dims=(512, 2),
            lr=1e-4,
            weight_decay=0.01,
            warmup_steps=100,
            max_steps=5000,
            dtype=dtype,
        )
        pl_module.eval().to(self.device)
        for p in pl_module.parameters():
            p.requires_grad = False

        model = pl_module.model
        self.tokenizer = model.tokenizer
        self.embed_tokens = model.internvl.get_input_embeddings()
        self.query_tokens = model.query_tokens
        self.llm = model.internvl.language_model
        self.num_image_token = model.internvl.num_image_token
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

        print("[Load] Loading TensorRT vision engine...")
        self.vision_runner = TRTRunner(vision_engine_path)
        print("[Load] Loading TensorRT action head engine...")
        self.action_runner = TRTRunner(action_engine_path)

        if llm_backend == "trt" and os.path.exists(llm_engine_path):
            print("[Load] Loading TensorRT LLM hidden engine...")
            self.llm_runner = TRTRunner(llm_engine_path)
            self.llm_backend = "trt"
        else:
            if llm_backend == "trt":
                print(f"[Warn] LLM TRT engine not found at {llm_engine_path}, falling back to PyTorch LLM.")
            print("[Load] Using PyTorch LLM for hidden states...")
            self.llm_runner = PyTorchLLMRunner(self.llm)
            self.llm_backend = "pytorch"

    def preprocess_images(self, image_paths: list):
        transform = build_transform(448)
        images = [Image.open(p).convert("RGB") for p in image_paths]
        tensors = [transform(img) for img in images]
        pixel_values = torch.stack(tensors, dim=0).unsqueeze(0).to(torch.float32).to(self.device)
        return pixel_values

    def build_prompt(self, num_images: int, prompt: str):
        image_token_str = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token + IMG_END_TOKEN
        image_tokens = "\n".join([image_token_str] * num_images)
        text = f"{image_tokens}\nUser: {prompt}\nAssistant:"
        model_inputs = self.tokenizer(text, return_tensors="pt", padding=False, truncation=True, max_length=2048)
        return model_inputs["input_ids"], model_inputs["attention_mask"]

    def infer(self, image_paths: list, prompt: str = "Predict the future trajectory of the ego vehicle."):
        pixel_values = self.preprocess_images(image_paths)
        B, N_imgs, C, H, W = pixel_values.shape
        pixel_values_flat = pixel_values.view(B * N_imgs, C, H, W)

        # Vision TRT.
        t0 = time.time()
        visual_features = self.vision_runner.infer({"pixel_values": pixel_values_flat.cpu().numpy()})
        t_vision = time.time() - t0
        visual_features = visual_features[list(visual_features.keys())[0]]

        # Prompt embeddings.
        input_ids, attention_mask = self.build_prompt(N_imgs, prompt)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids = position_ids.clamp(min=0).to(self.device)

        with torch.no_grad():
            inputs_embeds = self.embed_tokens(input_ids).clone()

        # Inject visual features.
        selected = (input_ids == self.img_context_token_id)
        assert selected.sum() == visual_features.shape[0] * visual_features.shape[1]
        inputs_embeds[selected] = torch.from_numpy(visual_features.reshape(-1, visual_features.shape[-1])).to(self.device).to(inputs_embeds.dtype)

        # Append query tokens.
        query_ids = torch.arange(self.num_queries, device=self.device).unsqueeze(0)
        query_embeds = self.query_tokens(query_ids).to(inputs_embeds.dtype)
        full_inputs_embeds = torch.cat([inputs_embeds, query_embeds], dim=1)
        full_attention_mask = torch.cat([attention_mask, torch.ones(B, self.num_queries, device=self.device)], dim=1)
        full_position_ids = full_attention_mask.cumsum(dim=-1) - 1
        full_position_ids = full_position_ids.clamp(min=0).to(self.device)

        # LLM forward.
        t0 = time.time()
        if self.llm_backend == "trt":
            hidden_states = self.llm_runner.infer({
                "inputs_embeds": full_inputs_embeds.cpu().numpy(),
                "attention_mask": full_attention_mask.cpu().numpy(),
                "position_ids": full_position_ids.cpu().numpy(),
            })
            hidden_states = hidden_states[list(hidden_states.keys())[0]]
        else:
            hidden_states = self.llm_runner.infer(
                full_inputs_embeds.cpu().numpy(),
                full_attention_mask.cpu().numpy(),
                full_position_ids.cpu().numpy(),
            )
        t_llm = time.time() - t0

        # Action head TRT.
        query_hidden = hidden_states[:, -self.num_queries:, :].astype(np.float32)
        t0 = time.time()
        pred_traj = self.action_runner.infer({"query_hidden": query_hidden})
        t_action = time.time() - t0
        pred_traj = pred_traj[list(pred_traj.keys())[0]]

        return {
            "pred_traj": pred_traj[0],  # (24, 2)
            "time": {
                "vision": t_vision,
                "llm": t_llm,
                "action": t_action,
                "total": t_vision + t_llm + t_action,
            },
        }


def run_trt_vla(
    checkpoint_path: str,
    base_model_path: str,
    vision_engine_path: str,
    llm_engine_path: str,
    action_engine_path: str,
    image_paths: list,
    prompt: str,
    llm_backend: str = "trt",
):
    """Convenience function for one-shot inference (loads components each call)."""
    runner = VLATRTInference(
        checkpoint_path, base_model_path,
        vision_engine_path, llm_engine_path, action_engine_path,
        llm_backend=llm_backend,
    )
    result = runner.infer(image_paths, prompt)
    t = result["time"]
    print(f"[Timing] Vision: {t['vision']:.3f}s | LLM: {t['llm']:.3f}s | Action: {t['action']:.3f}s | Total: {t['total']:.3f}s")
    return result["pred_traj"]


def main():
    parser = argparse.ArgumentParser(description="Run VLA end-to-end with TensorRT")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base_model_path", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--vision_engine", type=str, default=DEFAULT_VISION_ENGINE)
    parser.add_argument("--llm_engine", type=str, default=DEFAULT_LLM_ENGINE)
    parser.add_argument("--action_engine", type=str, default=DEFAULT_ACTION_ENGINE)
    parser.add_argument("--image_paths", type=str, nargs="+", required=True)
    parser.add_argument("--prompt", type=str, default="Predict the future trajectory of the ego vehicle.")
    parser.add_argument("--llm_backend", type=str, default="pytorch", choices=["trt", "pytorch"])
    args = parser.parse_args()

    for p in [args.vision_engine, args.action_engine]:
        if not os.path.exists(p):
            print(f"[Error] Engine not found: {p}")
            sys.exit(1)

    # Resolve backbone model_path from checkpoint to support InternVL2.5 / InternVL3 seamlessly.
    resolved_base_model_path = _get_model_path_from_checkpoint(args.checkpoint, args.base_model_path)

    traj = run_trt_vla(
        args.checkpoint, resolved_base_model_path,
        args.vision_engine, args.llm_engine, args.action_engine,
        args.image_paths, args.prompt, args.llm_backend,
    )

    print("\n" + "=" * 60)
    print("TRT VLA PREDICTED TRAJECTORY")
    print("=" * 60)
    print(traj)
    print("=" * 60)


if __name__ == "__main__":
    main()
