"""Run a TRT-LLM compiled engine and return generation output."""
from __future__ import annotations
import argparse
import os


def run_trtllm(engine_dir: str, prompt: str, max_tokens: int = 512) -> str:
    from tensorrt_llm.runtime import ModelRunner
    import torch

    runner = ModelRunner.from_dir(engine_dir=engine_dir, rank=0)
    input_ids = runner.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output_ids = runner.generate(
            batch_input_ids=[input_ids[0]],
            max_new_tokens=max_tokens,
        )
    return runner.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()
    output = run_trtllm(args.engine_dir, args.prompt, args.max_tokens)
    print(output)
