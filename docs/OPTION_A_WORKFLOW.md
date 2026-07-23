# Option A: Trace + Replay workflow

Real GPU vs TPU numbers using the **same JSONL workload** on both platforms.

## Overview

```
make trace          → workload/prompts.jsonl  (new requests each run unless SEED= set)
       ↓
Deploy vLLM on GKE TPU (your YAML) — then GPU when ready
       ↓
make replay         → results/{tpu|gpu}/run_01_replay.json  (measured)
       ↓
make normalize-replay → results/normalized/{tpu|gpu}.json
       ↓
make compare        → comparison.json
```

---

## Step 0 — Cloud Shell setup

```bash
git clone https://github.com/ssoni999/gpu-tpu-cost-benchmark.git
cd gpu-tpu-cost-benchmark
pip install -r requirements.txt

cp configs/tpu.env.example configs/tpu.env
# Edit: PROJECT_ID, ZONE, HF_TOKEN, TPU hourly rate
```

---

## Step 1 — Generate a trace (no GPU/TPU needed)

```bash
make trace
# Prints: Trace seed: 1234567890

# Same workload again later:
make trace SEED=1234567890
```

Outputs:
- `workload/prompts.jsonl` — 100 requests, ~1024 input / 128 output tokens each
- `workload/trace_meta.json` — seed + params (commit this for reproducibility)

**Rule:** Use **one trace file** for both GPU and TPU in a single comparison. Regenerate only when you want a new evaluation session.

---

## Step 2 — Deploy vLLM on GKE TPU (your existing YAML)

You already have a Deployment YAML — use it if the pod reaches Ready.

```bash
gcloud container clusters get-credentials YOUR_CLUSTER --zone YOUR_ZONE
kubectl apply -f your-vllm-tpu.yaml
kubectl get pods -w
kubectl logs -l app=vllm-tpu --tail=30
```

Confirm the model is **Qwen/Qwen3-4B** (or update `benchmark_config.yaml` to match your YAML).

Smoke test:

```bash
kubectl port-forward svc/vllm-service 8000:8000 &
curl http://127.0.0.1:8000/v1/models
```

---

## Step 3 — Replay trace against TPU (real numbers)

Keep port-forward running in one terminal. In another:

```bash
make replay TARGET=http://127.0.0.1:8000 PLATFORM=tpu
```

Creates `results/tpu/run_01_replay.json`.

Write environment metadata (edit for your actual GKE TPU type):

```bash
cat > results/tpu/run_01_environment.json << 'EOF'
{
  "platform": "tpu",
  "orchestrator": "gke",
  "generation": "v6e",
  "accelerator": "tpu-v6e-slice",
  "accelerator_count": 1,
  "accelerator_hourly_usd": 1.20,
  "vm_hourly_usd": 0.0,
  "hourly_cost_usd": 1.20
}
EOF
```

Normalize:

```bash
make normalize-replay PLATFORM=tpu
```

---

## Step 4 — Repeat on GPU (same trace file)

Deploy vLLM on your GPU GKE pool (or GCE). **Do not run `make trace` again.**

```bash
kubectl port-forward svc/YOUR_GPU_VLLM_SERVICE 8001:8000 &
make replay TARGET=http://127.0.0.1:8001 PLATFORM=gpu
```

Create `results/gpu/run_01_environment.json` with GPU pricing, then:

```bash
make normalize-replay PLATFORM=gpu
```

---

## Step 5 — Compare

```bash
make compare
cat comparison.json
```

---

## Step 6 — Teardown (stop billing)

Scale down node pools or delete benchmark Deployments when done.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Cannot reach .../models` | Port-forward not running or wrong service name |
| All requests error | HF token, model name mismatch, pod not Ready |
| Slow replay | Increase `SPEED=20` (time compression) |
| Different GPU vs TPU workload | Reused wrong trace — use same `workload/prompts.jsonl` |

---

## Optional: vllm bench path (legacy)

You can still use `bash scripts/run_tpu.sh bench` + `make normalize` with `.txt` bench output.
Option A (trace + replay) is preferred for customer-like prompts and shared workload fairness.
