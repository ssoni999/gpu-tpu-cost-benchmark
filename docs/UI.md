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
| `results/tpu/run_01_replay.json` | After TPU `make replay` |
| `results/gpu/run_01_replay.json` | After GPU `make replay` |
| `results/tpu/live.json` | During replay (shows "running") |

## View on Mac

```bash
gcloud cloud-shell ssh --authorize-session \
  --ssh-flag="-L 8787:localhost:8787"
```
Open http://localhost:8787

## Port 8765

Port 8765 may be used by **gpu-tpu-sim-poc** (mock matmul UI). This results UI uses **8787** only — no conflict.
