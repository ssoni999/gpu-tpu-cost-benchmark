# Live benchmark dashboard

View TPU/GPU replay metrics in a browser **while the workload runs** in Cloud Shell.

## Architecture

```
Cloud Shell
  Terminal 1: kubectl port-forward svc/tiny-model-service 8000:8000
  Terminal 2: make dashboard          → http://0.0.0.0:8765  (SSE UI)
  Terminal 3: make replay-live ...    → pushes metrics to dashboard + live.json

Your Mac (optional)
  SSH tunnel → http://localhost:8765  (same UI in local browser)
```

Replay writes:
- `results/tpu/live.json` — snapshot updated after each request (for SCP/sync)
- POST events to `http://127.0.0.1:8765/api/event` — drives the live UI

---

## Option A — View in Cloud Shell (simplest)

**Terminal 1** — vLLM port-forward:
```bash
kubectl port-forward svc/tiny-model-service 8000:8000
```

**Terminal 2** — start dashboard:
```bash
cd ~/gpu-tpu-cost-benchmark
pip install -r requirements.txt
make dashboard
```

**Terminal 3** — run replay with live feed:
```bash
make replay-live \
  TARGET=http://127.0.0.1:8000 \
  PLATFORM=tpu \
  MODEL='Qwen/Qwen2.5-0.5B-Instruct' \
  PARAMS_B=0.49
```

**Open the UI:** Cloud Shell → **Web Preview** → **Change port** → `8765`

You should see progress bar, TPS, TTFT, MFU, KV cache, and recent requests updating live.

---

## Option B — View on your Mac (“here” in Cursor)

Cloud Shell runs the dashboard; your Mac tunnels port 8765.

**On your Mac** (new terminal, not Cloud Shell):
```bash
gcloud cloud-shell ssh --authorize-session \
  --ssh-flag="-L 8765:localhost:8765"
```

Keep that SSH session open. Then on Cloud Shell run `make dashboard` and `make replay-live` as above.

**On your Mac browser:** open [http://localhost:8765](http://localhost:8765)

---

## Option C — Poll JSON file (no SSE)

If you skip the dashboard server, replay still writes `results/tpu/live.json` when `PLATFORM=tpu`:

```bash
make replay TARGET=http://127.0.0.1:8000 PLATFORM=tpu MODEL='Qwen/Qwen2.5-0.5B-Instruct'
```

Sync to Mac periodically:
```bash
gcloud cloud-shell scp cloudshell:~/gpu-tpu-cost-benchmark/results/tpu/live.json ./live.json
```

Open `dashboard/index.html` locally only works with a local server + CORS workaround — prefer Option A or B.

---

## Manual CLI

```bash
# Terminal 2
PYTHONPATH=scripts python3 scripts/dashboard_server.py --port 8765

# Terminal 3
PYTHONPATH=scripts python3 scripts/replay.py \
  --target http://127.0.0.1:8000 \
  --platform tpu \
  --model 'Qwen/Qwen2.5-0.5B-Instruct' \
  --dashboard-url http://127.0.0.1:8765 \
  --params-b 0.49
```

---

## API endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /` | Dashboard HTML |
| `GET /api/state` | Current metrics JSON |
| `GET /api/stream` | Server-Sent Events (live updates) |
| `POST /api/event` | Replay pushes updates (internal) |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Dashboard empty | Start `make dashboard` **before** `make replay-live` |
| UI not reachable on Mac | SSH tunnel must stay open; check port 8765 |
| Metrics stuck at 0 | vLLM port-forward dead; verify `curl localhost:8000/v1/models` |
| Server metrics all 0 | vLLM `/metrics` not exposed — check pod logs |
