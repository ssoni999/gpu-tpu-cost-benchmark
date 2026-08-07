.PHONY: help install trace trace-all replay replay-all replay-live dashboard ui manifests migration-diff upload-results normalize-replay normalize cost compare bench-cmd

ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PYTHONPATH=scripts
TIER ?= medium
TRACE ?= workload/$(TIER)/prompts.jsonl
REPLAY_OUT ?= results/$(PLATFORM)/$(TIER)/run_01_replay.json
MODEL ?= $(shell PYTHONPATH=scripts python3 -c "from config import load_config; print(load_config()['model'])")
WARMUP ?= $(shell PYTHONPATH=scripts python3 -c "from config import load_config; print(load_config()['benchmark']['warmup_requests'])")
DASHBOARD_PORT ?= 8765
UI_PORT ?= 8787
RESULTS_SOURCE ?= auto

help:
	@echo "GPU vs TPU cost benchmark — Option A: trace + replay"
	@echo ""
	@echo "  make install              Python dependencies"
	@echo "  make trace TIER=medium       JSONL workload for tier (small|medium|high)"
	@echo "  make trace-all               Generate traces for all tiers"
	@echo "  make replay-all TARGET=...   Replay all tiers on tpu + gpu (needs port-forward)"
	@echo "  make replay TARGET=... PLATFORM=tpu TIER=medium"
	@echo "  make ui                                 Results UI on port $(UI_PORT) (GCS or local)"
	@echo "  make upload-results                     Push local results/*.json to GCS"
	@echo "  make manifests                          Render tiny-model*.yaml from benchmark_config.yaml"
	@echo "  make migration-diff                     Print GPU vs TPU parameter diff"
	@echo "  make dashboard                        Legacy live dashboard on $(DASHBOARD_PORT)"
	@echo "  make normalize-replay PLATFORM=tpu    Normalize replay JSON"
	@echo "  make compare              GPU vs TPU comparison.json"
	@echo ""
	@echo "See docs/OPTION_A_WORKFLOW.md, docs/MIGRATION.md, docs/GCS.md, and docs/UI.md"

manifests:
	PYTHONPATH=scripts python3 scripts/render_manifest.py

migration-diff:
	PYTHONPATH=scripts python3 -c "from config import migration_diff, load_config; import json; print(json.dumps(migration_diff(load_config()), indent=2))"

install:
	pip install -r requirements.txt

trace:
	PYTHONPATH=scripts python3 scripts/generate_trace.py --tier $(TIER) -o $(TRACE) \
		$(if $(SEED),--seed $(SEED),)

trace-all:
	@for t in small medium high; do \
		echo "=== trace $$t ==="; \
		$(MAKE) trace TIER=$$t TRACE=workload/$$t/prompts.jsonl || exit 1; \
	done

replay-all:
	@test -n "$(TARGET)" || (echo "TARGET is required, e.g. TARGET=http://127.0.0.1:8000" && exit 1)
	@for p in tpu gpu; do \
		for t in small medium high; do \
			echo "=== replay $$p/$$t ==="; \
			$(MAKE) replay TARGET=$(TARGET) PLATFORM=$$p TIER=$$t MODEL='$(MODEL)' || exit 1; \
		done; \
	done

replay:
	@test -n "$(TARGET)" || (echo "TARGET is required, e.g. TARGET=http://127.0.0.1:8000" && exit 1)
	@test -n "$(PLATFORM)" || (echo "PLATFORM is required: gpu or tpu" && exit 1)
	@set -a; [ -f configs/gcs.env ] && . configs/gcs.env; set +a; \
	PYTHONPATH=scripts python3 scripts/replay.py \
		--target $(TARGET) \
		--model '$(MODEL)' \
		--platform $(PLATFORM) \
		--tier $(TIER) \
		--trace $(TRACE) \
		--output $(REPLAY_OUT) \
		--warmup $(WARMUP) \
		$(if $(SPEED),--speed $(SPEED),) \
		$(if $(CONCURRENCY),--concurrency $(CONCURRENCY),) \
		$(if $(PARAMS_B),--params-b $(PARAMS_B),) \
		$(if $(PEAK_TFLOPS),--peak-tflops $(PEAK_TFLOPS),) \
		$(if $(DASHBOARD_URL),--dashboard-url $(DASHBOARD_URL),)

ui:
	@set -a; [ -f configs/gcs.env ] && . configs/gcs.env; set +a; \
	PYTHONPATH=scripts python3 scripts/results_server.py --port $(UI_PORT) \
		--source $(RESULTS_SOURCE) \
		$(if $(GCS_BUCKET),--gcs-bucket $(GCS_BUCKET),) \
		$(if $(GCS_PROJECT),--gcs-project $(GCS_PROJECT),)

upload-results:
	@set -a; [ -f configs/gcs.env ] && . configs/gcs.env; set +a; \
	PYTHONPATH=scripts python3 scripts/upload_results.py \
		$(if $(GCS_BUCKET),--bucket $(GCS_BUCKET),) \
		$(if $(GCS_PROJECT),--project $(GCS_PROJECT),) \
		$(if $(PLATFORM),--platform $(PLATFORM),)

dashboard:
	PYTHONPATH=scripts python3 scripts/dashboard_server.py --port $(DASHBOARD_PORT)

replay-live:
	@test -n "$(TARGET)" || (echo "TARGET is required" && exit 1)
	@test -n "$(PLATFORM)" || (echo "PLATFORM is required: gpu or tpu" && exit 1)
	$(MAKE) replay TARGET=$(TARGET) PLATFORM=$(PLATFORM) MODEL='$(MODEL)' \
		DASHBOARD_URL=http://127.0.0.1:$(DASHBOARD_PORT) \
		$(if $(SPEED),SPEED=$(SPEED),) \
		$(if $(CONCURRENCY),CONCURRENCY=$(CONCURRENCY),) \
		$(if $(PARAMS_B),PARAMS_B=$(PARAMS_B),)

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
