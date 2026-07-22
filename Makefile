.PHONY: help install bench-cmd normalize cost compare provision-gpu provision-tpu

ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

help:
	@echo "GPU vs TPU cost benchmark (Qwen/Qwen3-4B, vLLM, shared benchmark_config.yaml)"
	@echo ""
	@echo "Local (Mac/Cursor):"
	@echo "  make install          Install Python deps"
	@echo "  make bench-cmd        Show vllm bench command from config"
	@echo "  make normalize GPU_RAW=... TPU_RAW=..."
	@echo "  make compare          Build comparison.json"
	@echo ""
	@echo "On GCE (GPU):  bash scripts/run_gpu.sh bench"
	@echo "On TPU VM:     bash scripts/run_tpu.sh bench"

install:
	pip install -r requirements.txt

PYTHONPATH=scripts

bench-cmd:
	PYTHONPATH=scripts python3 scripts/config.py --bench-cmd

provision-tpu:
	bash scripts/run_tpu.sh provision-cmd

serve-gpu-cmd:
	bash scripts/run_gpu.sh serve-cmd

serve-tpu-cmd:
	bash scripts/run_tpu.sh serve-cmd

normalize:
	@test -n "$(GPU_RAW)" || (echo "GPU_RAW=results/gpu/run_01.txt" && exit 1)
	PYTHONPATH=scripts python3 scripts/normalize_results.py --platform gpu \
		--raw $(GPU_RAW) \
		--environment $(or $(GPU_ENV),results/gpu/run_01_environment.json)
	@test -n "$(TPU_RAW)" || (echo "TPU_RAW=results/tpu/run_01.txt" && exit 1)
	PYTHONPATH=scripts python3 scripts/normalize_results.py --platform tpu \
		--raw $(TPU_RAW) \
		--environment $(or $(TPU_ENV),results/tpu/run_01_environment.json)

cost-gpu:
	PYTHONPATH=scripts python3 scripts/calculate_cost.py --input results/normalized/gpu.json

cost-tpu:
	PYTHONPATH=scripts python3 scripts/calculate_cost.py --input results/normalized/tpu.json

compare:
	PYTHONPATH=scripts python3 scripts/compare.py --output comparison.json
