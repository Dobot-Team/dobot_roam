#!/usr/bin/env python3
"""
End-to-end Qwen2-VL VLA inference with TensorRT.

Pipeline:
  4 history images -> Qwen2VLProcessor -> pixel_values + image_grid_thw + input_ids
  TRT Vision Encoder (qwen2vl_2b_vision.trt) -> image_embeds
  embed_tokens + masked_scatter at image_token_id -> inputs_embeds
  append 24 learned query embeddings
  TRT/PyTorch LLM (single forward) with MRoPE -> hidden_states
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
from transformers import Qwen2VLProcessor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from deploy.vla.utils import load_checkpoint
from vla.lightning import VLAPlModule

DEFAULT_CHECKPOINT = os.path.join(_PROJECT_ROOT, "outputs", "nuscenes_qwen2vl_real_v2", "checkpoints", "last.ckpt")
DEFAULT_BASE_MODEL = os.path.join(_PROJECT_ROOT, "models", "Qwen", "Qwen2-VL-2B-Instruct")
DEFAULT_VISION_ENGINE = os.path.join(_PROJECT_ROOT, "deploy", "outputs", "qwen2vl_2b_vision.trt")
DEFAULT_LLM_ENGINE = os.path.join(_PROJECT_ROOT, "deploy", "outputs", "qwen2vl_vla_llm_hidden.trt")
DEFAULT_ACTION_ENGINE = os.path.join(_PROJECT_ROOT, "deploy", "outputs", "qwen2vl_vla_action_head.trt")


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


class TRTRunner:
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
        self.output_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                             if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]

    def infer(self, feed_dict: dict) -> dict:
        device_inputs = {}
        outputs = {}
        for name, arr in feed_dict.items():
            arr = np.ascontiguousarray(arr)
            self.context.set_input_shape(name, arr.shape)
            device_inputs[name] = torch.from_numpy(arr).cuda()

        for name in self.output_names:
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


class Qwen2VLVLATRTInference:
    """Loads all Qwen2-VL VLA deployment components once and provides an `infer` method."""

    def __init__(
        self,
        checkpoint_path: str,
        base_model_path: str,
        vision_engine_path: str,
        llm_engine_path: str,
        action_engine_path: str,
        llm_backend: str = "pytorch",
        dtype: torch.dtype = torch.float32,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.num_queries = 24

        print(f"[Load] Loading Qwen2-VL VLA components from checkpoint {checkpoint_path}...")
        pl_module = VLAPlModule.load_from_checkpoint(
            checkpoint_path,
            model_path=base_model_path,
            model_type="qwen2vl",
            num_queries=self.num_queries,
            freeze_vision=True,
            freeze_llm=True,
            freeze_action_head=False,
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
        self.processor = model.tokenizer
        self.tokenizer = self.processor.tokenizer
        self.embed_tokens = model.qwen_model.model.embed_tokens
        self.query_tokens = model.query_tokens
        self.llm = model.qwen_model.model
        self.qwen2vl_model = model.qwen_model.model
        self.llm_hidden_size = model.llm_hidden_size
        self.image_token_id = model.image_token_id

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
        images = [Image.open(p).convert("RGB") for p in image_paths]
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": "Predict the future trajectory of the ego vehicle."})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors="pt", padding=True)
        return inputs

    def _get_mrope_position_ids(self, input_ids, image_grid_thw, mm_token_type_ids, attention_mask):
        """Compute MRoPE position_ids including query tokens as text."""
        B = input_ids.shape[0]
        query_ids = torch.arange(self.num_queries, device=input_ids.device).unsqueeze(0).expand(B, -1)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        full_input_ids = torch.cat([input_ids, torch.full_like(query_ids, pad_token_id)], dim=1)
        full_attention_mask = torch.cat([attention_mask, torch.ones(B, self.num_queries, device=input_ids.device, dtype=attention_mask.dtype)], dim=1)
        query_mm = torch.zeros(B, self.num_queries, device=input_ids.device, dtype=mm_token_type_ids.dtype)
        full_mm_token_type_ids = torch.cat([mm_token_type_ids, query_mm], dim=1)

        helper = getattr(self.qwen2vl_model, "get_rope_index", None)
        if helper is not None:
            try:
                out = helper(
                    full_input_ids,
                    mm_token_type_ids=full_mm_token_type_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=None,
                    attention_mask=full_attention_mask,
                )
                if isinstance(out, tuple):
                    return out[0]
                return out
            except Exception as e:
                print(f"[Warn] get_rope_index failed ({e}); falling back to plain arange.")

        seq_len = full_input_ids.shape[1]
        return torch.arange(seq_len, dtype=torch.long, device=input_ids.device).view(1, 1, -1).expand(3, 1, -1)

    def infer(self, image_paths: list):
        inputs = self.preprocess_images(image_paths)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        pixel_values = inputs["pixel_values"].to(torch.float32).to(self.device)
        image_grid_thw = inputs["image_grid_thw"].to(torch.int64).to(self.device)
        mm_token_type_ids = inputs.get("mm_token_type_ids", torch.zeros_like(input_ids, dtype=torch.int64)).to(self.device)
        B = input_ids.shape[0]

        print(f"[Preprocess] input_ids shape: {tuple(input_ids.shape)}, pixel_values shape: {tuple(pixel_values.shape)}")

        # Vision TRT: the engine is fixed for a single sample image, so run it
        # once per history image and concatenate the outputs.
        t0 = time.time()
        visual_features_list = []
        start = 0
        num_images = image_grid_thw.shape[0]
        for i in range(num_images):
            n_patches = int(image_grid_thw[i].prod().item())
            # temporal_patch_size * patch_size^2 patches per grid cell; each patch is 1176 dims.
            # The processor already flattens patches, so n_patches = grid_thw.prod() for a single image.
            chunk = pixel_values[start:start + n_patches]
            feat = self.vision_runner.infer({"pixel_values": chunk.cpu().numpy()})
            feat = feat[list(feat.keys())[0]]
            visual_features_list.append(feat)
            start += n_patches
        visual_features = np.concatenate(visual_features_list, axis=0)
        t_vision = time.time() - t0
        print(f"[Vision] TRT output shape: {visual_features.shape}, time: {t_vision:.3f}s")

        # Prompt embeddings.
        with torch.no_grad():
            inputs_embeds = self.embed_tokens(input_ids).clone()
            n_image_tokens = (input_ids == self.image_token_id).sum().item()
            n_image_features = visual_features.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(f"Image token mismatch: tokens={n_image_tokens}, features={n_image_features}")
            image_mask = (input_ids == self.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            image_embeds_tensor = torch.from_numpy(visual_features).to(self.device).to(inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds_tensor)

        # Append query tokens.
        query_ids = torch.arange(self.num_queries, device=self.device).unsqueeze(0).expand(B, -1)
        query_embeds = self.query_tokens(query_ids).to(inputs_embeds.dtype)
        full_inputs_embeds = torch.cat([inputs_embeds, query_embeds], dim=1)
        full_attention_mask = torch.cat([attention_mask, torch.ones(B, self.num_queries, device=self.device, dtype=attention_mask.dtype)], dim=1)

        # MRoPE position_ids.
        position_ids = self._get_mrope_position_ids(input_ids, image_grid_thw, mm_token_type_ids, attention_mask)

        # LLM forward.
        t0 = time.time()
        if self.llm_backend == "trt":
            hidden_states = self.llm_runner.infer({
                "inputs_embeds": full_inputs_embeds.cpu().numpy(),
                "attention_mask": full_attention_mask.cpu().numpy(),
                "position_ids": position_ids.cpu().numpy(),
            })
            hidden_states = hidden_states[list(hidden_states.keys())[0]]
        else:
            hidden_states = self.llm_runner.infer(
                full_inputs_embeds.cpu().numpy(),
                full_attention_mask.cpu().numpy(),
                position_ids.cpu().numpy(),
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
    llm_backend: str = "pytorch",
):
    """Convenience function for one-shot inference (loads components each call)."""
    runner = Qwen2VLVLATRTInference(
        checkpoint_path, base_model_path,
        vision_engine_path, llm_engine_path, action_engine_path,
        llm_backend=llm_backend,
    )
    result = runner.infer(image_paths)
    t = result["time"]
    print(f"[Timing] Vision: {t['vision']:.3f}s | LLM: {t['llm']:.3f}s | Action: {t['action']:.3f}s | Total: {t['total']:.3f}s")
    return result["pred_traj"]


def main():
    parser = argparse.ArgumentParser(description="Run Qwen2-VL VLA end-to-end with TensorRT")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base_model_path", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--vision_engine", type=str, default=DEFAULT_VISION_ENGINE)
    parser.add_argument("--llm_engine", type=str, default=DEFAULT_LLM_ENGINE)
    parser.add_argument("--action_engine", type=str, default=DEFAULT_ACTION_ENGINE)
    parser.add_argument("--image_paths", type=str, nargs="+", required=True)
    parser.add_argument("--llm_backend", type=str, default="pytorch", choices=["trt", "pytorch"])
    args = parser.parse_args()

    for p in [args.vision_engine, args.action_engine]:
        if not os.path.exists(p):
            print(f"[Error] Engine not found: {p}")
            sys.exit(1)

    resolved_base_model_path = _get_model_path_from_checkpoint(args.checkpoint, args.base_model_path)

    traj = run_trt_vla(
        args.checkpoint, resolved_base_model_path,
        args.vision_engine, args.llm_engine, args.action_engine,
        args.image_paths, args.llm_backend,
    )

    print("\n" + "=" * 60)
    print("TRT QWEN2-VL VLA PREDICTED TRAJECTORY")
    print("=" * 60)
    print(traj)
    print("=" * 60)


if __name__ == "__main__":
    main()
