#!/usr/bin/env bash
set -euo pipefail

# Run inside dbn_wcr. This launches this repository's Lightning entry point.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv-omni/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
CONFIG="${CONFIG:-configs/omni_v2/qwen35_08B_omni_v2_base_v2v3data.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen35_08B_omni_v2_base_v2v3data}"
NCCL_LIBRARY="${NCCL_LIBRARY:-/opt/conda/envs/dbn_nccl/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2}"
[[ -z "${LOCAL_RANK:-}" ]] || { echo 'Run this script directly, not inside torchrun.' >&2; exit 2; }
[[ "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'GPU_IDS must be comma-separated GPU indices.' >&2; exit 2; }
IFS=',' read -r -a devices <<< "$GPU_IDS"
declare -A seen=()
for device in "${devices[@]}"; do
    [[ -z "${seen[$device]:-}" ]] || { echo "Duplicate GPU: $device" >&2; exit 2; }
    seen[$device]=1
done
[[ -f "$CONFIG" && -x "$PYTHON_BIN" && -r "$NCCL_LIBRARY" ]] || {
    echo 'Check CONFIG, PYTHON_BIN and NCCL_LIBRARY paths.' >&2; exit 2;
}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export LD_PRELOAD="${NCCL_LIBRARY}${LD_PRELOAD:+:$LD_PRELOAD}"
strategy=ddp_find_unused_parameters_true
[[ ${#devices[@]} -gt 1 ]] || strategy=auto
# IPv6 loopback avoids this container's slow IPv4 reverse-DNS lookup.
command=("$PYTHON_BIN" -m torch.distributed.run --nnodes=1
         --nproc_per_node="${#devices[@]}" --max_restarts=0
         --rdzv-backend=c10d --rdzv-endpoint='[::1]:0' --local-addr=::1
         scripts/train.py --config "$CONFIG" --output_dir "$OUTPUT_DIR"
         "$@" "trainer.devices=${#devices[@]}" trainer.num_nodes=1 "trainer.strategy=$strategy")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
else
    exec "${command[@]}"
fi
