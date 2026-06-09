"""AWQ 4-bit quantization pipeline using AutoAWQ."""
from __future__ import annotations

import argparse


def quantize_awq(model_id: str, output_dir: str, bits: int = 4, group_size: int = 128, zero_point: bool = True):  # noqa: E501
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    print(f"Loading model: {model_id}")
    model = AutoAWQForCausalLM.from_pretrained(model_id, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    quant_config = {
        "zero_point": zero_point,
        "q_group_size": group_size,
        "w_bit": bits,
        "version": "GEMM",
    }

    print(f"Quantizing to {bits}-bit AWQ...")
    model.quantize(tokenizer, quant_config=quant_config)

    print(f"Saving to {output_dir}")
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
    quantize_awq(args.model, args.output, args.bits, args.group_size)
