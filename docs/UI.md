# Benchmark results UI

View **real** metrics from `make replay` on port **8787** (does not use 8765).

## Quick start

**Terminal 1** — results UI (leave running):
```bash
make ui
```
Cloud Shell → **Web Preview → port 8787**

**Terminal 2** — port-forward vLLM (if not already):
```bash
kubectl port-forward svc/tiny-model-service 8000:8000
```

**Terminal 3** — run replay:
```bash
make replay \
  TARGET=http://127.0.0.1:8000 \
  PLATFORM=tpu \
  MODEL='Qwen/Qwen2.5-0.5B-Instruct' \
  PARAMS_B=0.49
```

The UI reads **`results/tpu/run_01_replay.json`** automatically (refreshes every 3s).
When you run GPU replay too, it shows a TPU vs GPU comparison bar.

## Data sources

| File | When |
|------|------|
| `gs://gpu-tpu-benchmark-storage/latest/tpu/run_01_replay.json` | Latest TPU metrics (default for UI) |
| `gs://gpu-tpu-benchmark-storage/latest/gpu/run_01_replay.json` | Latest GPU metrics |
| `results/tpu/run_01_replay.json` | Local fallback when GCS unavailable |
| `results/tpu/live.json` | During replay (in-progress) |

See [GCS.md](GCS.md) for bucket setup and IAM.

## View on Mac

```bash
gcloud cloud-shell ssh --authorize-session \
  --ssh-flag="-L 8787:localhost:8787"
```
Open http://localhost:8787

## Port 8765

Port 8765 may be used by **gpu-tpu-sim-poc** (mock matmul UI). This results UI uses **8787** only — no conflict.

## Migration parameters

The UI loads GPU vs TPU serving/infra differences from `configs/benchmark_config.yaml` via `/api/migration`.

## Cost & energy metrics

The UI computes cost metrics from replay results + pricing in `benchmark_config.yaml` (or `results/{platform}/run_01_environment.json` if present):

- Run cost, cost per output token, cost per wall second
- Performance per dollar and **performance per watt** (tok/s ÷ TDP)
- Green **↑** on the TPU panel when TPU beats GPU on that metric

Set real rates in `configs/gpu.env` / `configs/tpu.env` and write `run_01_environment.json` via normalize-replay for accurate dollars.

See [MIGRATION.md](MIGRATION.md) for deploy checklists. After editing config:

```bash
make manifests
kubectl apply -f tiny-model.yaml   # or tiny-model-gpu.yaml
```
