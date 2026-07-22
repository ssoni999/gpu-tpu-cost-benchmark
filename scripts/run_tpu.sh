#!/usr/bin/env bash
# Run on a TPU v5e VM. Provisioning uses v5litepod-1 (single chip for Qwen3-4B).
# Benchmark parameters come from benchmark_config.yaml, not tpu-recipes defaults.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUN_ID:-run_01}"
RESULT_DIR="${ROOT}/results/tpu"
CONTAINER_NAME="${USER}-vllm"
mkdir -p "${RESULT_DIR}"

if [[ -f "${ROOT}/configs/tpu.env" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/configs/tpu.env"
fi

eval "$(python3 "${ROOT}/scripts/config.py" --shell)"

: "${PROJECT_ID:?Set PROJECT_ID in configs/tpu.env}"
: "${ZONE:?Set ZONE in configs/tpu.env}"
: "${TPU_NAME:?Set TPU_NAME in configs/tpu.env}"
TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v5litepod-1}"
TPU_RUNTIME_VERSION="${TPU_RUNTIME_VERSION:-v2-alpha-tpuv5-lite}"

print_provision_commands() {
  cat <<EOF
# --- Provision TPU v5e (queued resource, recommended) ---
gcloud alpha compute tpus queued-resources create ${QR_ID:-${USER}-v5e-qr} \\
  --node-id ${TPU_NAME} \\
  --project ${PROJECT_ID} \\
  --zone ${ZONE} \\
  --accelerator-type ${TPU_ACCELERATOR_TYPE} \\
  --runtime-version ${TPU_RUNTIME_VERSION}

gcloud alpha compute tpus queued-resources list --project ${PROJECT_ID} --zone ${ZONE}

# --- SSH ---
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --project ${PROJECT_ID} --zone=${ZONE}
EOF
}

print_serve_commands() {
  cat <<EOF
# --- On TPU VM: start vLLM TPU container (interactive) ---
export DOCKER_URI=${TPU_DOCKER_IMAGE}
sudo docker run -it --rm --name ${CONTAINER_NAME} --privileged --net=host \\
  -v /dev/shm:/dev/shm --shm-size 20gb \\
  --entrypoint /bin/bash \${DOCKER_URI}

# Inside container:
export HF_HOME=/dev/shm
export HF_TOKEN=\${HF_TOKEN}

vllm serve ${BENCH_MODEL} \\
  --seed ${BENCH_SEED} \\
  --disable-log-requests \\
  --gpu-memory-utilization 0.98 \\
  --max-num-batched-tokens ${TPU_MAX_BATCHED_TOKENS} \\
  --max-num-seqs ${TPU_MAX_NUM_SEQS} \\
  --tensor-parallel-size ${TPU_TP_SIZE} \\
  --max-model-len ${TPU_MAX_MODEL_LEN}
EOF
}

write_environment_json() {
  python3 - <<PY
import json, os
from pathlib import Path
count = 1
if os.environ.get("TPU_ACCELERATOR_TYPE", "").endswith("-4"):
    count = 4
elif os.environ.get("TPU_ACCELERATOR_TYPE", "").endswith("-8"):
    count = 8
elif os.environ.get("TPU_ACCELERATOR_TYPE", "") == "v5litepod-1":
    count = 1
env = {
    "platform": "tpu",
    "generation": "v5e",
    "project_id": os.environ.get("PROJECT_ID"),
    "zone": os.environ.get("ZONE"),
    "tpu_name": os.environ.get("TPU_NAME"),
    "accelerator": os.environ.get("TPU_ACCELERATOR_TYPE", "v5litepod-1"),
    "accelerator_count": count,
    "runtime_version": os.environ.get("TPU_RUNTIME_VERSION"),
    "framework": "vllm",
    "docker_image": os.environ.get("TPU_DOCKER_IMAGE"),
    "accelerator_hourly_usd": float(os.environ.get("TPU_ACCELERATOR_HOURLY_USD", "0") or 0),
    "vm_hourly_usd": float(os.environ.get("TPU_VM_HOURLY_USD", "0") or 0),
}
env["hourly_cost_usd"] = env["accelerator_hourly_usd"] * env["accelerator_count"] + env["vm_hourly_usd"]
Path("${RESULT_DIR}/${RUN_ID}_environment.json").write_text(json.dumps(env, indent=2) + "\\n")
PY
}

run_benchmark_in_container() {
  local out_txt="${RESULT_DIR}/${RUN_ID}.txt"
  echo "Running vLLM bench inside ${CONTAINER_NAME} -> ${out_txt}"

  sudo docker exec -i "${CONTAINER_NAME}" bash -lc "
    cd /workspace/vllm 2>/dev/null || cd /vllm-workspace 2>/dev/null || true
    vllm bench serve \\
      --host http://127.0.0.1:8000 \\
      --backend vllm \\
      --model '${BENCH_MODEL}' \\
      --dataset-name random \\
      --num-prompts ${BENCH_NUM_PROMPTS} \\
      --random-input-len ${BENCH_INPUT_TOKENS} \\
      --random-output-len ${BENCH_OUTPUT_TOKENS} \\
      --seed ${BENCH_SEED} \\
      --request-rate ${BENCH_REQUEST_RATE} \\
      --ignore-eos
  " | tee "${out_txt}"
}

case "${1:-help}" in
  provision-cmd)
    print_provision_commands
    ;;
  serve-cmd)
    print_serve_commands
    ;;
  env)
    write_environment_json
    echo "Wrote ${RESULT_DIR}/${RUN_ID}_environment.json"
    ;;
  bench)
    write_environment_json
    run_benchmark_in_container
    echo ""
    echo "Next:"
    echo "  PYTHONPATH=scripts python3 scripts/normalize_results.py --platform tpu \\"
    echo "    --raw results/tpu/${RUN_ID}.txt \\"
    echo "    --environment results/tpu/${RUN_ID}_environment.json"
    ;;
  bench-cmd)
    cat <<EOF
vllm bench serve \\
  --host http://127.0.0.1:8000 \\
  --backend vllm \\
  --model ${BENCH_MODEL} \\
  --dataset-name random \\
  --num-prompts ${BENCH_NUM_PROMPTS} \\
  --random-input-len ${BENCH_INPUT_TOKENS} \\
  --random-output-len ${BENCH_OUTPUT_TOKENS} \\
  --seed ${BENCH_SEED} \\
  --request-rate ${BENCH_REQUEST_RATE} \\
  --ignore-eos
EOF
    ;;
  *)
    echo "Usage: $0 [provision-cmd|serve-cmd|bench-cmd|env|bench]" >&2
    echo "  provision-cmd  Print gcloud v5e queued-resource commands"
    echo "  serve-cmd      Print vLLM TPU serve commands (Qwen3-4B, v5e)"
    echo "  bench-cmd      Print vllm bench serve command from benchmark_config.yaml"
    echo "  bench          Run benchmark inside running container ${CONTAINER_NAME}"
    exit 1
    ;;
esac
