"""GPTQ 4-bit quantization pipeline using AutoGPTQ."""
from __future__ import annotations

import argparse


def quantize_gptq(model_id: str, output_dir: str, bits: int = 4, group_size: int = 128):
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    from transformers import AutoTokenizer

    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=False,
    )

    print(f"Loading model for quantization: {model_id}")
    model = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config)

    calibration_data = [
        "NVIDIA DCGM monitors GPU health and performance in data center environments.",
        "vLLM uses PagedAttention to efficiently manage KV cache memory.",
        "Triton Inference Server supports HTTP and gRPC protocols for model serving.",
        "TRT-LLM compiles LLMs to TensorRT engines for maximum throughput on NVIDIA GPUs.",
        "GPTQ quantization reduces model weights to 4-bit integers with minimal accuracy loss.",
    ]
    examples = [{"input_ids": tokenizer.encode(t, return_tensors="pt")} for t in calibration_data]

    print(f"Quantizing to {bits}-bit GPTQ...")
    model.quantize(examples)

    print(f"Saving quantized model to {output_dir}")
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args()
    quantize_gptq(args.model, args.output, args.bits, args.group_size)
