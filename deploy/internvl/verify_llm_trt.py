#!/usr/bin/env python3
"""
Verify TensorRT LLM engine against PyTorch and ONNX.
"""
import argparse
import os
import sys
import warnings

import numpy as np
import onnxruntime as ort
import tensorrt as trt
import torch
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_MODEL_PATH = "/home/jovyan/work/vla_internvl2/models/OpenGVLab/InternVL2_5-1B"
DEFAULT_ONNX = os.path.join(os.path.dirname(__file__), "..", "outputs", "internvl2_llm.onnx")
DEFAULT_ENGINE = os.path.join(os.path.dirname(__file__), "..", "outputs", "internvl2_llm.trt")


def load_llm_tokenizer(model_path: str):
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map=None,
        low_cpu_mem_usage=True,
        local_files_only=True,
        use_flash_attn=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,
    )
    llm = model.language_model
    llm.eval().cpu()
    return llm, tokenizer


def compare_logits(ref_logits: np.ndarray, trt_logits: np.ndarray, tag: str):
    diff = np.abs(ref_logits - trt_logits)
    max_err = float(diff.max())
    mean_err = float(diff.mean())
    rel_err = max_err / (np.abs(ref_logits).max() + 1e-12)
    ref_argmax = ref_logits.argmax(-1)
    trt_argmax = trt_logits.argmax(-1)
    token_match = float((ref_argmax == trt_argmax).mean())

    print(f"[{tag}] max_abs_error={max_err:.6e} mean_abs_error={mean_err:.6e} rel_error={rel_err:.6e} token_match={token_match:.4f}")

    ok = True
    # TRT kernel fusion introduces larger numeric differences than eager CPU; token-level consistency is the main goal.
    # We keep max/mean as diagnostic checks but do not fail on them because logits can have large absolute magnitudes.
    if rel_err >= 5e-2:
        print(f"[{tag}] FAIL: relative_error >= 5e-2")
        ok = False
    if token_match < 1.0:
        print(f"[{tag}] FAIL: token_match < 100%")
        ok = False
    return ok


class TRTLLMRunner:
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        print(f"[TRT] Engine loaded. IO tensors: {self.input_names}")

    def infer(self, input_ids: np.ndarray, attention_mask: np.ndarray, position_ids: np.ndarray) -> np.ndarray:
        seq_len = input_ids.shape[1]
        # Set input shapes for dynamic axes
        for name, arr in [("input_ids", input_ids), ("attention_mask", attention_mask), ("position_ids", position_ids)]:
            self.context.set_input_shape(name, arr.shape)

        out_name = "logits"
        out_shape = self.context.get_tensor_shape(out_name)
        # Output shape may contain dynamic dim; replace with actual if needed
        out_shape = tuple(seq_len if s < 0 else s for s in out_shape)
        out = np.empty(out_shape, dtype=np.float32)

        d_input_ids = torch.from_numpy(input_ids.astype(np.int64)).cuda()
        d_attention_mask = torch.from_numpy(attention_mask.astype(np.int64)).cuda()
        d_position_ids = torch.from_numpy(position_ids.astype(np.int64)).cuda()
        d_output = torch.from_numpy(out).cuda()

        self.context.set_tensor_address("input_ids", d_input_ids.data_ptr())
        self.context.set_tensor_address("attention_mask", d_attention_mask.data_ptr())
        self.context.set_tensor_address("position_ids", d_position_ids.data_ptr())
        self.context.set_tensor_address(out_name, d_output.data_ptr())
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()

        return d_output.cpu().numpy()


def greedy_generate_pt(llm, input_ids: torch.Tensor, max_new_tokens: int):
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        attention_mask = torch.ones_like(generated)
        position_ids = torch.arange(generated.shape[1], dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            logits = llm(generated, attention_mask=attention_mask, position_ids=position_ids, use_cache=False).logits
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
    return generated[0].tolist()


def greedy_generate_ort(session, input_ids: np.ndarray, max_new_tokens: int):
    generated = input_ids.copy()
    for _ in range(max_new_tokens):
        attention_mask = np.ones_like(generated)
        position_ids = np.arange(generated.shape[1], dtype=np.int64).reshape(1, -1)
        logits = session.run(None, {
            "input_ids": generated,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        })[0]
        next_token = logits[:, -1, :].argmax(axis=-1, keepdims=True)
        generated = np.concatenate([generated, next_token], axis=-1)
    return generated[0].tolist()


def greedy_generate_trt(runner, input_ids: np.ndarray, max_new_tokens: int):
    generated = input_ids.copy()
    for _ in range(max_new_tokens):
        attention_mask = np.ones_like(generated)
        position_ids = np.arange(generated.shape[1], dtype=np.int64).reshape(1, -1)
        logits = runner.infer(generated, attention_mask, position_ids)
        next_token = logits[:, -1, :].argmax(axis=-1, keepdims=True)
        generated = np.concatenate([generated, next_token], axis=-1)
    return generated[0].tolist()


def verify(model_path: str, onnx_path: str, engine_path: str, n_prompts: int = 5, gen_len: int = 10):
    if not os.path.exists(engine_path):
        print(f"[Error] Engine not found: {engine_path}")
        return 1

    print("[Load] Loading PyTorch LLM and tokenizer...")
    llm, tokenizer = load_llm_tokenizer(model_path)
    vocab_size = llm.config.vocab_size

    print("[Load] Loading ONNX Runtime CPU session...")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    print("[Load] Loading TensorRT engine...")
    runner = TRTLLMRunner(engine_path)

    all_ok = True

    print(f"\n[Random] Running {n_prompts} random prompts...")
    for i in range(n_prompts):
        seq_len = np.random.randint(5, 32)
        input_ids = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)
        attention_mask = torch.ones((1, seq_len), dtype=torch.long)
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            pt_logits = llm(input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False).logits.numpy()
        trt_logits = runner.infer(input_ids.numpy(), attention_mask.numpy(), position_ids.numpy())

        ok = compare_logits(pt_logits, trt_logits, f"TRT-vs-PT-{i+1}")
        if not ok:
            all_ok = False
            break

    if all_ok:
        print(f"\n[Generation] Greedy generating {gen_len} tokens from a real prompt...")
        prompt_text = "The image shows"
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
        pt_tokens = greedy_generate_pt(llm, input_ids, gen_len)
        ort_tokens = greedy_generate_ort(session, input_ids.numpy(), gen_len)
        trt_tokens = greedy_generate_trt(runner, input_ids.numpy(), gen_len)

        print(f"[Generation] PyTorch tokens: {pt_tokens}")
        print(f"[Generation] ONNX tokens:    {ort_tokens}")
        print(f"[Generation] TRT tokens:     {trt_tokens}")
        if pt_tokens != trt_tokens or ort_tokens != trt_tokens:
            print("[Generation] FAIL: token sequences differ")
            all_ok = False
        else:
            print("[Generation] PASS: token sequences match")

    if all_ok:
        print("\n[Result] LLM TensorRT verification PASSED.")
        return 0
    else:
        print("\n[Result] LLM TensorRT verification FAILED.")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Verify InternVL2.5 LLM TensorRT engine")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--onnx_path", type=str, default=DEFAULT_ONNX)
    parser.add_argument("--engine_path", type=str, default=DEFAULT_ENGINE)
    parser.add_argument("--n_prompts", type=int, default=5)
    parser.add_argument("--gen_len", type=int, default=10)
    args = parser.parse_args()

    sys.exit(verify(args.model_path, args.onnx_path, args.engine_path, args.n_prompts, args.gen_len))


if __name__ == "__main__":
    main()
