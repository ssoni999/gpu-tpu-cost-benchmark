#!/usr/bin/env bash
# Run on a G4 GCE VM after cloning this repo and copying configs/gpu.env.
# Uses benchmark_config.yaml for all vLLM bench parameters (not recipe defaults).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUN_ID:-run_01}"
RESULT_DIR="${ROOT}/results/gpu"
mkdir -p "${RESULT_DIR}"

if [[ -f "${ROOT}/configs/gpu.env" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/configs/gpu.env"
fi

eval "$(python3 "${ROOT}/scripts/config.py" --shell)"

write_environment_json() {
  python3 - <<PY
import json, os
from pathlib import Path
env = {
    "platform": "gpu",
    "project_id": os.environ.get("PROJECT_ID"),
    "zone": os.environ.get("ZONE"),
    "vm_name": os.environ.get("VM_NAME"),
    "accelerator": "G4",
    "accelerator_count": 1,
    "framework": "vllm",
    "docker_image": os.environ.get("GPU_DOCKER_IMAGE"),
    "accelerator_hourly_usd": float(os.environ.get("GPU_ACCELERATOR_HOURLY_USD", "0") or 0),
    "vm_hourly_usd": float(os.environ.get("GPU_VM_HOURLY_USD", "0") or 0),
}
hourly = env["accelerator_hourly_usd"] * env["accelerator_count"] + env["vm_hourly_usd"]
env["hourly_cost_usd"] = hourly
Path("${RESULT_DIR}/${RUN_ID}_environment.json").write_text(json.dumps(env, indent=2) + "\\n")
PY
}

print_serve_command() {
  cat <<EOF
# Start vLLM server (background example — run in tmux/screen):
sudo docker run -d --name vllm-qwen3-4b \\
  --runtime nvidia --gpus all \\
  -v ~/.cache/huggingface:/root/.cache/huggingface \\
  --env "HUGGING_FACE_HUB_TOKEN=\${HF_TOKEN}" \\
  -p 8000:8000 --ipc=host \\
  ${GPU_DOCKER_IMAGE} \\
  --model ${BENCH_MODEL} \\
  --max-model-len ${GPU_MAX_MODEL_LEN} \\
  --max-num-batched-tokens ${GPU_MAX_BATCHED_TOKENS} \\
  --max-num-seqs ${GPU_MAX_NUM_SEQS} \\
  --tensor-parallel-size ${GPU_TP_SIZE} \\
  --gpu-memory-utilization 0.95
EOF
}

run_benchmark() {
  local out_txt="${RESULT_DIR}/${RUN_ID}.txt"
  echo "Running vLLM bench (contract from benchmark_config.yaml) -> ${out_txt}"

  sudo docker run --rm \
    --runtime nvidia --gpus all \
    --network host \
    --entrypoint vllm \
    "${GPU_DOCKER_IMAGE}" bench serve \
    --host http://127.0.0.1:8000 \
    --backend vllm \
    --model "${BENCH_MODEL}" \
    --dataset-name random \
    --num-prompts "${BENCH_NUM_PROMPTS}" \
    --random-input-len "${BENCH_INPUT_TOKENS}" \
    --random-output-len "${BENCH_OUTPUT_TOKENS}" \
    --seed "${BENCH_SEED}" \
    --request-rate "${BENCH_REQUEST_RATE}" \
    --ignore-eos \
    | tee "${out_txt}"
}

case "${1:-bench}" in
  env)
    write_environment_json
    echo "Wrote ${RESULT_DIR}/${RUN_ID}_environment.json"
    ;;
  serve-cmd)
    print_serve_command
    ;;
  bench)
    write_environment_json
    run_benchmark
    echo ""
    echo "Next (on laptop or VM with repo):"
    echo "  PYTHONPATH=scripts python3 scripts/normalize_results.py --platform gpu \\"
    echo "    --raw results/gpu/${RUN_ID}.txt \\"
    echo "    --environment results/gpu/${RUN_ID}_environment.json"
    ;;
  *)
    echo "Usage: $0 [env|serve-cmd|bench]" >&2
    exit 1
    ;;
esac
