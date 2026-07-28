# Live benchmark dashboard

Displays the **same metrics as `make replay` terminal output**, loaded from `results/{platform}/run_01_replay.json`.

## Quick start (Cloud Shell)

**Terminal 1** — vLLM port-forward:
```bash
kubectl port-forward svc/tiny-model-service 8000:8000
```

**Terminal 2** — dashboard (leave running):
```bash
cd ~/gpu-tpu-cost-benchmark
git pull origin main   # get latest dashboard files
pip install -r requirements.txt
make dashboard
```

Open **Web Preview → port 8765**. Select **TPU** or **GPU** in the dropdown.

**Terminal 3** — run replay (dashboard auto-connects):
```bash
make replay \
  TARGET=http://127.0.0.1:8000 \
  PLATFORM=tpu \
  MODEL='Qwen/Qwen2.5-0.5B-Instruct' \
  PARAMS_B=0.49
```

When replay finishes, the UI shows the full **WORKLOAD REPLAY SUMMARY** block (TPS, TTFT, MFU, KV cache, etc.) — identical to the terminal.

During the run, progress and rolling partial metrics update live. On completion, the dashboard loads `results/tpu/run_01_replay.json` automatically (polls every 2s).

---

## What the UI shows

Mirrors `scripts/replay.py` `print_summary()`:

- Infrastructure, model, wall time, ok/err counts
- Throughput (RPS, TPS), token counts
- Hardware: MFU, achieved TFLOPS
- Client latency: TTFT p50/p95/p99, latency p95, ITL
- Server: KV cache, prefix hit rate, queued/running

---

## View on your Mac

```bash
gcloud cloud-shell ssh --authorize-session \
  --ssh-flag="-L 8765:localhost:8765"
```

Open http://localhost:8765

---

## Disable dashboard push

```bash
make replay NO_DASHBOARD=1 TARGET=... PLATFORM=tpu ...
```

The UI still loads results from `run_01_replay.json` if the dashboard server is running.

---

## Manual server

```bash
PYTHONPATH=scripts python3 scripts/dashboard_server.py --port 8765
```

API:
- `GET /api/replay/tpu` — full replay JSON
- `GET /api/state?platform=tpu` — merged live + replay state
- `GET /api/stream` — SSE updates
