#!/usr/bin/env python3
"""
End-to-end SmolVLM2 inference using TensorRT.

Pipeline:
  Image -> processor -> pixel_values + pixel_attention_mask + input_ids
  TRT Vision Encoder (vision_model + connector) -> image_hidden_states
  embed_tokens + inputs_merger -> inputs_embeds
  TRT LLM (inputs_embeds mode) -> greedy decode -> response
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
from transformers import AutoModelForImageTextToText, AutoProcessor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_MODEL_PATH = "/home/jovyan/work/vla_smolvlm2_256m/models/HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
DEFAULT_VISION_ENGINE = os.path.join(os.path.dirname(__file__), "..", "outputs", "smolvlm2_256m_vision.trt")
DEFAULT_LLM_ENGINE = os.path.join(os.path.dirname(__file__), "..", "outputs", "smolvlm2_256m_llm_embed.trt")
DEFAULT_IMAGE = os.path.join(os.path.dirname(__file__), "..", "..", "20260610151154.jpg")


class TRTVisionRunner:
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def infer(self, pixel_values: np.ndarray) -> np.ndarray:
        self.context.set_input_shape("pixel_values", pixel_values.shape)

        out_name = "image_hidden_states"
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


def load_model_components(model_path: str, device: torch.device):
    print(f"[Load] Loading processor and embed_tokens from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
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
    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")

    return {
        "processor": processor,
        "model": model,
        "embed_tokens": model.model.text_model.get_input_embeddings(),
        "inputs_merger": model.model.inputs_merger,
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
    inputs_merger = comps["inputs_merger"]
    eos_token_id = comps["eos_token_id"]
    image_token_id = comps["image_token_id"]

    print("[Load] Loading TensorRT vision engine...")
    vision_runner = TRTVisionRunner(vision_engine_path)
    print("[Load] Loading TensorRT LLM engine...")
    llm_runner = TRTLLMEmbedRunner(llm_engine_path)

    print(f"[Image] Loading and preprocessing {image_path}...")
    img = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    pixel_values = inputs["pixel_values"].to(torch.float32).to(device)

    print(f"[Preprocess] input_ids shape: {tuple(input_ids.shape)}, pixel_values shape: {tuple(pixel_values.shape)}")

    # Vision encoder TRT
    t0 = time.time()
    image_hidden_states = vision_runner.infer(pixel_values.cpu().numpy())
    t_vision = time.time() - t0
    print(f"[Vision] TRT output shape: {image_hidden_states.shape}, time: {t_vision:.3f}s")

    # Build inputs_embeds and merge image features
    with torch.no_grad():
        inputs_embeds = embed_tokens(input_ids)
        # image_hidden_states: (1, 9, 64, hidden_size) -> squeeze batch -> (9, 64, hidden_size)
        image_hidden_states_tensor = torch.from_numpy(image_hidden_states).to(device).squeeze(0)
        merged_embeds = inputs_merger(input_ids, inputs_embeds, image_hidden_states_tensor)

    # Greedy generation
    print(f"[Inference] Generating up to {max_new_tokens} tokens...")
    generated_ids = input_ids[0].tolist()
    t_start = time.time()
    first_token_time = None

    for step in range(max_new_tokens):
        # Keep entire context; engine max_seq is 1024
        cur_embeds = merged_embeds
        seq_len = cur_embeds.shape[1]
        cur_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
        cur_pos = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)

        logits = llm_runner.infer(
            cur_embeds.cpu().numpy(),
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
        if merged_embeds.shape[1] > 1024:
            merged_embeds = merged_embeds[:, -1024:, :]

    total_time = time.time() - t_start
    n_gen = len(generated_ids) - input_ids.shape[1]

    prompt_len = input_ids.shape[1]
    response_ids = generated_ids[prompt_len:]
    response = processor.tokenizer.decode(response_ids, skip_special_tokens=True)

    print(f"\n[Timing] First token: {first_token_time:.3f}s" if first_token_time else "")
    print(f"[Timing] Total: {total_time:.3f}s, Generated tokens: {n_gen}, Throughput: {n_gen/total_time:.2f} tokens/s")

    return response


def main():
    parser = argparse.ArgumentParser(description="Run SmolVLM2 end-to-end with TensorRT")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vision_engine", type=str, default=DEFAULT_VISION_ENGINE)
    parser.add_argument("--llm_engine", type=str, default=DEFAULT_LLM_ENGINE)
    parser.add_argument("--image_path", type=str, default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.")
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
