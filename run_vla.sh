#!/bin/bash
# Convenience wrapper for VLA training / evaluation.
# Usage:
#   bash run_vla.sh train [config_overrides...]
#   bash run_vla.sh eval  <checkpoint_path> [config_overrides...]
# Examples:
#   bash run_vla.sh train training.max_steps=5000
#   bash run_vla.sh eval outputs/vla/checkpoints/last.ckpt

set -e

PYTHON="${PYTHON:-python3}"
CONFIG="${CONFIG:-configs/nuscenes_mini_internvl2_5_vla.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/nuscenes_mini_internvl2_5_vla}"

CMD=$1
shift

case "$CMD" in
  train)
    $PYTHON scripts/train.py \
      --config "$CONFIG" \
      --output_dir "$OUTPUT_DIR" \
      "$@"
    ;;
  eval)
    CHECKPOINT=$1
    shift
    if [ -z "$CHECKPOINT" ]; then
      echo "Usage: bash run_vla.sh eval <checkpoint_path> [overrides...]"
      exit 1
    fi
    $PYTHON scripts/eval.py \
      --config "$CONFIG" \
      --checkpoint "$CHECKPOINT" \
      "$@"
    ;;
  *)
    echo "Unknown command: $CMD"
    echo "Usage: bash run_vla.sh {train|eval} ..."
    exit 1
    ;;
esac
