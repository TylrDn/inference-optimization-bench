#!/usr/bin/env bash
# TRT-LLM engine build script for LLaMA 3
# Requires: TensorRT-LLM installed (nvcr.io/nvidia/tensorrt-llm or pip install tensorrt-llm)

set -euo pipefail

MODEL_DIR=${MODEL_DIR:-"./models/llama3-8b"}
ENGINE_DIR=${ENGINE_DIR:-"./trtllm/engines/llama3-8b"}
TP_SIZE=${TP_SIZE:-1}      # Tensor parallel size
PP_SIZE=${PP_SIZE:-1}      # Pipeline parallel size
DTYPE=${DTYPE:-"float16"}

echo "Converting HF model to TRT-LLM checkpoint..."
python /opt/tensorrt_llm/examples/llama/convert_checkpoint.py \
  --model_dir "$MODEL_DIR" \
  --output_dir "${ENGINE_DIR}_ckpt" \
  --dtype "$DTYPE" \
  --tp_size "$TP_SIZE" \
  --pp_size "$PP_SIZE"

echo "Building TRT-LLM engine..."
trtllm-build \
  --checkpoint_dir "${ENGINE_DIR}_ckpt" \
  --output_dir "$ENGINE_DIR" \
  --gpt_attention_plugin "$DTYPE" \
  --gemm_plugin "$DTYPE" \
  --max_batch_size 32 \
  --max_input_len 2048 \
  --max_output_len 2048

echo "Engine built: $ENGINE_DIR"
