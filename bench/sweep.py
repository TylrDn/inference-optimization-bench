"""Full benchmark sweep across config combinations."""
from __future__ import annotations
import argparse
import subprocess
import sys
import yaml


def run_single(model: str, backend: str, quant: str, batch_size: int, seq_len: int, num_requests: int, output_dir: str):
    cmd = [
        sys.executable, "-m", "bench.run_bench",
        "--model", model,
        "--backend", backend,
        "--quant", quant,
        "--batch-size", str(batch_size),
        "--seq-len", str(seq_len),
        "--num-requests", str(num_requests),
        "--output-dir", output_dir,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED: {model}/{backend}/{quant} bs={batch_size}")
        print(result.stderr)
    else:
        print(f"  OK: {model}/{backend}/{quant} bs={batch_size}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quick.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    total = (len(cfg["models"]) * len(cfg["backends"]) * len(cfg["quantization"]) *
             len(cfg["batch_sizes"]) * len(cfg["sequence_lengths"]))
    print(f"Total combinations: {total}")

    for model in cfg["models"]:
        for backend in cfg["backends"]:
            for quant in cfg["quantization"]:
                for bs in cfg["batch_sizes"]:
                    for sl in cfg["sequence_lengths"]:
                        run_single(
                            model=model["name"],
                            backend=backend,
                            quant=quant,
                            batch_size=bs,
                            seq_len=sl,
                            num_requests=cfg.get("num_requests", 50),
                            output_dir=cfg.get("output_dir", "./results"),
                        )


if __name__ == "__main__":
    main()
