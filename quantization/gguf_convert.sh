#!/usr/bin/env bash
# Convert a HuggingFace model to GGUF format using llama.cpp
# Requires: llama.cpp cloned to LLAMACPP_DIR

set -euo pipefail

MODEL_DIR=${MODEL_DIR:-"./models/llama3-8b"}
OUTPUT_DIR=${OUTPUT_DIR:-"./models/llama3-8b-gguf"}
QUANT_TYPE=${QUANT_TYPE:-"Q4_K_M"}  # Q4_K_M, Q8_0, Q5_K_M, F16
LLAMACPP_DIR=${LLAMACPP_DIR:-"./llama.cpp"}

echo "Converting $MODEL_DIR to GGUF..."
python "$LLAMACPP_DIR/convert_hf_to_gguf.py" \
  "$MODEL_DIR" \
  --outfile "$OUTPUT_DIR/model_f16.gguf" \
  --outtype f16

echo "Quantizing to $QUANT_TYPE..."
"$LLAMACPP_DIR/llama-quantize" \
  "$OUTPUT_DIR/model_f16.gguf" \
  "$OUTPUT_DIR/model_${QUANT_TYPE}.gguf" \
  "$QUANT_TYPE"

echo "GGUF conversion complete: $OUTPUT_DIR/model_${QUANT_TYPE}.gguf"
