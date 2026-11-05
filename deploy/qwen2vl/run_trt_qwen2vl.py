#!/usr/bin/env python3
"""
End-to-end Qwen2-VL inference using TensorRT.

Pipeline:
  Image -> processor -> pixel_values + image_grid_thw + input_ids
  TRT Vision Encoder -> image_embeds
  embed_tokens + masked_scatter at image_token_id -> inputs_embeds
  TRT LLM (inputs_embeds mode) with 3-D mrope position_ids -> greedy decode -> response
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
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Qwen", "Qwen2-VL-2B-Instruct")
DEFAULT_VISION_ENGINE = os.path.join(PROJECT_ROOT, "deploy", "outputs", "qwen2vl_2b_vision.trt")
DEFAULT_LLM_ENGINE = os.path.join(PROJECT_ROOT, "deploy", "outputs", "qwen2vl_2b_llm_embed.trt")
DEFAULT_IMAGE = os.path.join(PROJECT_ROOT, "assets", "test_image.jpg")
MAX_SEQ_LEN = 2048  # must match the TRT LLM engine's max_seq


class TRTVisionRunner:
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
        self.output_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                             if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]

    def infer(self, pixel_values: np.ndarray, image_grid_thw: np.ndarray = None) -> np.ndarray:
        self.context.set_input_shape("pixel_values", pixel_values.shape)

        out_name = self.output_names[0]
        out_shape = tuple(s if s >= 0 else pixel_values.shape[0] for s in self.context.get_tensor_shape(out_name))
        out = np.empty(out_shape, dtype=np.float32)

        d_pv = torch.from_numpy(pixel_values.astype(np.float32)).cuda()
        d_out = torch.from_numpy(out).cuda()

        self.context.set_tensor_address("pixel_values", d_pv.data_ptr())
        self.context.set_tensor_address(out_name, d_out.data_ptr())
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return d_out.cpu().numpy()


class TRTLLMEmbedRunner:
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def infer(self, inputs_embeds: np.ndarray, attention_mask: np.ndarray, position_ids: np.ndarray) -> np.ndarray:
        seq_len = inputs_embeds.shape[1]
        self.context.set_input_shape("inputs_embeds", inputs_embeds.shape)
        self.context.set_input_shape("attention_mask", attention_mask.shape)
        self.context.set_input_shape("position_ids", position_ids.shape)

        out_name = "logits"
        out_shape = tuple(s if s >= 0 else seq_len for s in self.context.get_tensor_shape(out_name))
        out = np.empty(out_shape, dtype=np.float32)

        d_embeds = torch.from_numpy(inputs_embeds.astype(np.float32)).cuda()
        d_mask = torch.from_numpy(attention_mask.astype(np.int64)).cuda()
        d_pos = torch.from_numpy(position_ids.astype(np.int64)).cuda()
        d_out = torch.from_numpy(out).cuda()

        self.context.set_tensor_address("inputs_embeds", d_embeds.data_ptr())
        self.context.set_tensor_address("attention_mask", d_mask.data_ptr())
        self.context.set_tensor_address("position_ids", d_pos.data_ptr())
        self.context.set_tensor_address(out_name, d_out.data_ptr())
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return d_out.cpu().numpy()


def get_mrope_position_ids(model, input_ids: torch.Tensor, image_grid_thw: torch.Tensor,
                            mm_token_type_ids: torch.Tensor,
                            attention_mask: torch.Tensor = None) -> torch.Tensor:
    """Compute Qwen2-VL MRoPE 3-D position_ids (3, batch, seq_len).

    Tries the model's official helper first; falls back to a simple 1-D arange
    repeated 3 times if the helper is unavailable.
    """
    # Newer transformers exposes get_rope_index on model.model.
    helper = getattr(getattr(model, "model", None), "get_rope_index", None)
    if helper is None:
        helper = getattr(model, "get_rope_index", None)
    if helper is not None:
        try:
            out = helper(
                input_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=None,
                attention_mask=attention_mask,
            )
            # The helper returns either position_ids or (position_ids, mrope_section).
            if isinstance(out, tuple):
                return out[0]
            return out
        except Exception as e:
            print(f"[Warn] model.get_rope_index failed ({e}); falling back to plain arange position_ids.")

    seq_len = input_ids.shape[1]
    device = input_ids.device
    return torch.arange(seq_len, dtype=torch.long, device=device).view(1, 1, -1).expand(3, 1, -1)


def load_model_components(model_path: str, device: torch.device):
    print(f"[Load] Loading processor and embed_tokens from {model_path}...")
    processor = Qwen2VLProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map=None,
        local_files_only=True,
    )
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    eos_token_id = processor.tokenizer.eos_token_id
    pad_token_id = processor.tokenizer.pad_token_id or eos_token_id
    image_token_id = model.config.image_token_id

    return {
        "processor": processor,
        "model": model,
        "embed_tokens": model.model.language_model.embed_tokens,
        "device": device,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "image_token_id": image_token_id,
    }


def run_trt_inference(model_path: str, vision_engine_path: str, llm_engine_path: str,
                      image_path: str, prompt: str, max_new_tokens: int = 128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    comps = load_model_components(model_path, device)
    processor = comps["processor"]
    model = comps["model"]
    embed_tokens = comps["embed_tokens"]
    eos_token_id = comps["eos_token_id"]
    image_token_id = comps["image_token_id"]

    print("[Load] Loading TensorRT vision engine...")
    vision_runner = TRTVisionRunner(vision_engine_path)
    print("[Load] Loading TensorRT LLM engine...")
    llm_runner = TRTLLMEmbedRunner(llm_engine_path)

    print(f"[Image] Loading and preprocessing {image_path}...")
    img = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True)

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    pixel_values = inputs["pixel_values"].to(torch.float32).to(device)
    image_grid_thw = inputs["image_grid_thw"].to(torch.int64).to(device)
    # mm_token_type_ids is required for correct M-RoPE; processor returns it alongside input_ids.
    mm_token_type_ids = inputs.get("mm_token_type_ids", torch.zeros_like(input_ids, dtype=torch.int64)).to(device)

    print(f"[Preprocess] input_ids shape: {tuple(input_ids.shape)}, pixel_values shape: {tuple(pixel_values.shape)}")

    # Vision encoder TRT
    t0 = time.time()
    image_embeds = vision_runner.infer(pixel_values.cpu().numpy())
    t_vision = time.time() - t0
    print(f"[Vision] TRT output shape: {image_embeds.shape}, time: {t_vision:.3f}s")

    # Build inputs_embeds and merge image features
    with torch.no_grad():
        inputs_embeds = embed_tokens(input_ids)
        n_image_tokens = (input_ids == image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(f"Image token mismatch: tokens={n_image_tokens}, features={n_image_features}")
        image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        image_embeds_tensor = torch.from_numpy(image_embeds).to(device).to(inputs_embeds.dtype)
        merged_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds_tensor)

    # Greedy generation
    print(f"[Inference] Generating up to {max_new_tokens} tokens...")
    prompt_len = input_ids.shape[1]
    generated_ids = input_ids[0].tolist()
    t_start = time.time()
    first_token_time = None

    for step in range(max_new_tokens):
        seq_len = merged_embeds.shape[1]

        # Truncate before the TRT LLM call to avoid exceeding the engine's max_seq.
        if seq_len > MAX_SEQ_LEN:
            keep = MAX_SEQ_LEN
            merged_embeds = merged_embeds[:, -keep:, :]
            input_ids = input_ids[:, -keep:]
            mm_token_type_ids = mm_token_type_ids[:, -keep:]
            seq_len = keep

        cur_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
        # MRoPE 3-D position_ids; image_grid_thw is unchanged during generation.
        cur_pos = get_mrope_position_ids(
            model, input_ids, image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids, attention_mask=cur_mask,
        )

        logits = llm_runner.infer(
            merged_embeds.cpu().numpy(),
            cur_mask.cpu().numpy(),
            cur_pos.cpu().numpy(),
        )
        next_token_id = int(logits[0, -1, :].argmax())
        generated_ids.append(next_token_id)

        if first_token_time is None:
            first_token_time = time.time() - t_start

        if next_token_id == eos_token_id:
            break

        with torch.no_grad():
            next_embed = embed_tokens(torch.tensor([[next_token_id]], device=device))
        merged_embeds = torch.cat([merged_embeds, next_embed], dim=1)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=device)], dim=1)
        # New generated tokens are text tokens.
        mm_token_type_ids = torch.cat([
            mm_token_type_ids,
            torch.zeros((1, 1), dtype=mm_token_type_ids.dtype, device=device),
        ], dim=1)

    total_time = time.time() - t_start
    n_gen = len(generated_ids) - prompt_len

    response_ids = generated_ids[prompt_len:]
    response = processor.tokenizer.decode(response_ids, skip_special_tokens=True)

    print(f"\n[Timing] First token: {first_token_time:.3f}s" if first_token_time else "")
    print(f"[Timing] Total: {total_time:.3f}s, Generated tokens: {n_gen}, Throughput: {n_gen/total_time:.2f} tokens/s")

    return response


def main():
    parser = argparse.ArgumentParser(description="Run Qwen2-VL end-to-end with TensorRT")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vision_engine", type=str, default=DEFAULT_VISION_ENGINE)
    parser.add_argument("--llm_engine", type=str, default=DEFAULT_LLM_ENGINE)
    parser.add_argument("--image_path", type=str, default=DEFAULT_IMAGE,
                        help="Input image; must match the resolution used for vision engine export")
    parser.add_argument("--prompt", type=str, default="描述这张图片")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    if not os.path.exists(args.vision_engine):
        print(f"[Error] Vision engine not found: {args.vision_engine}")
        sys.exit(1)
    if not os.path.exists(args.llm_engine):
        print(f"[Error] LLM engine not found: {args.llm_engine}")
        sys.exit(1)

    response = run_trt_inference(
        args.model_path, args.vision_engine, args.llm_engine,
        args.image_path, args.prompt, args.max_new_tokens
    )

    print("\n" + "=" * 60)
    print("TRT MODEL RESPONSE")
    print("=" * 60)
    print(response)
    print("=" * 60)


if __name__ == "__main__":
    main()
