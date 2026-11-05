#!/usr/bin/env python3
"""
End-to-end InternVL2.5-1B inference using TensorRT.

Pipeline:
  Image -> TRT Vision Encoder -> visual_features
  Text prompt -> tokenizer -> input_ids -> text embeddings (PyTorch embed_tokens)
  Inject visual_features at image token positions
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
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_MODEL_PATH = "/home/jovyan/work/vla_internvl2/models/OpenGVLab/InternVL2_5-1B"
DEFAULT_VISION_ENGINE = os.path.join(os.path.dirname(__file__), "..", "outputs", "internvl2_vision.trt")
DEFAULT_LLM_ENGINE = os.path.join(os.path.dirname(__file__), "..", "outputs", "internvl2_llm_embed.trt")
DEFAULT_IMAGE = os.path.join(os.path.dirname(__file__), "..", "..", "20260610151154.jpg")

# InternVL constants from modeling_internvl_chat.py
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


def build_transform(input_size: int = 448):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class TRTVisionRunner:
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))
        self.dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))

    def infer(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(self.dtype)
        out = np.empty(self.output_shape, dtype=self.dtype)
        d_input = torch.from_numpy(x).cuda()
        d_output = torch.from_numpy(out).cuda()
        self.context.set_tensor_address(self.input_name, d_input.data_ptr())
        self.context.set_tensor_address(self.output_name, d_output.data_ptr())
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return d_output.cpu().numpy().astype(np.float32)


class TRTLLMEmbedRunner:
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]

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
    print(f"[Load] Loading tokenizer and embed_tokens from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map=None,
        low_cpu_mem_usage=True,
        local_files_only=True,
        use_flash_attn=False,
    )
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    # Read config values
    config = model.config
    image_size = config.force_image_size or config.vision_config.image_size
    num_image_token = model.num_image_token
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    eos_token_id = config.llm_config.eos_token_id

    return {
        "tokenizer": tokenizer,
        "embed_tokens": model.language_model.get_input_embeddings(),
        "device": device,
        "image_size": image_size,
        "num_image_token": num_image_token,
        "img_context_token_id": img_context_token_id,
        "eos_token_id": eos_token_id,
        "sep": config.llm_config.sep if hasattr(config.llm_config, "sep") else "<|im_end|>",
    }


def build_query(question: str, system_message: str) -> str:
    # InternVL2.5 template (MPT style)
    query = f"<|im_start|>system\n{system_message}<|im_end|>\n"
    query += f"<|im_start|>user\n<image>\n{question}<|im_end|>\n"
    query += f"<|im_start|>assistant\n"
    return query


def run_trt_inference(model_path: str, vision_engine_path: str, llm_engine_path: str,
                      image_path: str, prompt: str, max_new_tokens: int = 128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    comps = load_model_components(model_path, device)
    tokenizer = comps["tokenizer"]
    embed_tokens = comps["embed_tokens"]
    image_size = comps["image_size"]
    num_image_token = comps["num_image_token"]
    img_context_token_id = comps["img_context_token_id"]
    eos_token_id = comps["eos_token_id"]

    # Load TRT engines
    print("[Load] Loading TensorRT vision engine...")
    vision_runner = TRTVisionRunner(vision_engine_path)
    print("[Load] Loading TensorRT LLM engine...")
    llm_runner = TRTLLMEmbedRunner(llm_engine_path)

    # Preprocess image
    print(f"[Image] Loading and preprocessing {image_path}...")
    img = Image.open(image_path).convert("RGB")
    transform = build_transform(image_size)
    pixel_values = transform(img).unsqueeze(0).to(torch.float32).to(device)

    # Vision encoder TRT
    t0 = time.time()
    visual_features = vision_runner.infer(pixel_values.cpu().numpy())
    t_vision = time.time() - t0
    print(f"[Vision] TRT output shape: {visual_features.shape}, time: {t_vision:.3f}s")

    # Build prompt and tokenize
    system_message = "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。"
    query = build_query(prompt, system_message)
    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token + IMG_END_TOKEN
    query = query.replace("<image>", image_tokens, 1)

    model_inputs = tokenizer(query, return_tensors="pt")
    input_ids = model_inputs["input_ids"].to(device)
    attention_mask = model_inputs["attention_mask"].to(device)
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.long, device=device).unsqueeze(0)

    # Get text embeddings and inject visual features
    with torch.no_grad():
        inputs_embeds = embed_tokens(input_ids).clone()
    selected = (input_ids == img_context_token_id)
    assert selected.sum() == visual_features.shape[1], (
        f"Image token count mismatch: {selected.sum()} vs {visual_features.shape[1]}"
    )
    inputs_embeds[selected] = torch.from_numpy(visual_features[0]).to(device).to(inputs_embeds.dtype)

    # Greedy generation
    print(f"[Inference] Generating up to {max_new_tokens} tokens...")
    generated_ids = input_ids[0].tolist()
    t_start = time.time()
    first_token_time = None

    for step in range(max_new_tokens):
        cur_embeds = inputs_embeds[:, -512:, :]  # truncate to max engine seq len if needed
        cur_mask = torch.ones((1, cur_embeds.shape[1]), dtype=torch.long, device=device)
        cur_pos = torch.arange(cur_embeds.shape[1], dtype=torch.long, device=device).unsqueeze(0)

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

        # Append embedding of next token
        with torch.no_grad():
            next_embed = embed_tokens(torch.tensor([[next_token_id]], device=device))
        inputs_embeds = torch.cat([inputs_embeds, next_embed], dim=1)

    total_time = time.time() - t_start
    n_gen = len(generated_ids) - input_ids.shape[1]

    # Decode only the newly generated tokens (after the prompt)
    prompt_len = input_ids.shape[1]
    response_ids = generated_ids[prompt_len:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    response = response.split("<|im_end|>")[0].strip()

    print(f"\n[Timing] First token: {first_token_time:.3f}s" if first_token_time else "")
    print(f"[Timing] Total: {total_time:.3f}s, Generated tokens: {n_gen}, Throughput: {n_gen/total_time:.2f} tokens/s")

    return response


def main():
    parser = argparse.ArgumentParser(description="Run InternVL2.5 end-to-end with TensorRT")
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
