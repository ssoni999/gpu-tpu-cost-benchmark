.PHONY: help install trace replay normalize-replay normalize cost compare bench-cmd

ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PYTHONPATH=scripts
TRACE ?= workload/prompts.jsonl
MODEL ?= $(shell PYTHONPATH=scripts python3 -c "from config import load_config; print(load_config()['model'])")
WARMUP ?= $(shell PYTHONPATH=scripts python3 -c "from config import load_config; print(load_config()['benchmark']['warmup_requests'])")

help:
	@echo "GPU vs TPU cost benchmark — Option A: trace + replay"
	@echo ""
	@echo "  make install              Python dependencies"
	@echo "  make trace                New JSONL workload (random seed unless SEED= set)"
	@echo "  make trace SEED=100       Reproducible workload"
	@echo "  make replay TARGET=... PLATFORM=tpu   Replay trace against vLLM"
	@echo "  make normalize-replay PLATFORM=tpu    Normalize replay JSON"
	@echo "  make compare              GPU vs TPU comparison.json"
	@echo ""
	@echo "See docs/OPTION_A_WORKFLOW.md for the full GKE step list."

install:
	pip install -r requirements.txt

trace:
	PYTHONPATH=scripts python3 scripts/generate_trace.py -o $(TRACE) \
		$(if $(SEED),--seed $(SEED),)

replay:
	@test -n "$(TARGET)" || (echo "TARGET is required, e.g. TARGET=http://127.0.0.1:8000" && exit 1)
	@test -n "$(PLATFORM)" || (echo "PLATFORM is required: gpu or tpu" && exit 1)
	PYTHONPATH=scripts python3 scripts/replay.py \
		--target $(TARGET) \
		--model $(MODEL) \
		--trace $(TRACE) \
		--output results/$(PLATFORM)/run_01_replay.json \
		--warmup $(WARMUP) \
		$(if $(SPEED),--speed $(SPEED),) \
		$(if $(CONCURRENCY),--concurrency $(CONCURRENCY),)

normalize-replay:
	@test -n "$(PLATFORM)" || (echo "PLATFORM is required: gpu or tpu" && exit 1)
	PYTHONPATH=scripts python3 scripts/normalize_results.py --platform $(PLATFORM) \
		--replay-json results/$(PLATFORM)/run_01_replay.json \
		--environment results/$(PLATFORM)/run_01_environment.json

normalize:
	@test -n "$(GPU_RAW)" || (echo "GPU_RAW=results/gpu/run_01.txt" && exit 1)
	PYTHONPATH=scripts python3 scripts/normalize_results.py --platform gpu \
		--raw $(GPU_RAW) \
		--environment $(or $(GPU_ENV),results/gpu/run_01_environment.json)
	@test -n "$(TPU_RAW)" || (echo "TPU_RAW=results/tpu/run_01.txt" && exit 1)
	PYTHONPATH=scripts python3 scripts/normalize_results.py --platform tpu \
		--raw $(TPU_RAW) \
		--environment $(or $(TPU_ENV),results/tpu/run_01_environment.json)

bench-cmd:
	PYTHONPATH=scripts python3 scripts/config.py --bench-cmd

cost-gpu:
	PYTHONPATH=scripts python3 scripts/calculate_cost.py --input results/normalized/gpu.json

cost-tpu:
	PYTHONPATH=scripts python3 scripts/calculate_cost.py --input results/normalized/tpu.json

compare:
	PYTHONPATH=scripts python3 scripts/compare.py --output comparison.json

serve-gpu-cmd:
	bash scripts/run_gpu.sh serve-cmd

serve-tpu-cmd:
	bash scripts/run_tpu.sh serve-cmd
