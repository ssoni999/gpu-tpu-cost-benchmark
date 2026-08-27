# GPU ↔ TPU migration guide

Same **workload**, different **serving + infra**. The trace (`workload/prompts.jsonl`) stays identical on both platforms; only deployment parameters change.

Source of truth: `configs/benchmark_config.yaml`

## Three layers

| Layer | Shared? | File / command |
|-------|---------|----------------|
| Workload | Yes — same trace | `make trace` → `workload/prompts.jsonl` |
| Serving | No — per platform | `gpu.serving` / `tpu.serving` in config |
| Infra | No — per platform | `gpu.infra` / `tpu.infra`, K8s manifests |

## Quick commands

```bash
# See side-by-side parameter diff
make migration-diff

# Regenerate K8s YAML from config (after editing benchmark_config.yaml)
make manifests
kubectl apply -f tiny-model.yaml        # TPU
kubectl apply -f tiny-model-gpu.yaml    # GPU
```

## Switching GPU → TPU

1. **Keep the same trace** — do not regenerate unless you intentionally want a new workload.
2. **Deploy TPU vLLM** (not the GPU manifest):

```bash
make manifests
kubectl apply -f tiny-model.yaml
kubectl rollout status deployment/tiny-model-server --timeout=600s
kubectl port-forward svc/tiny-model-service 8000:8000
```

3. **Replay with `PLATFORM=tpu`**:

```bash
make replay TARGET=http://127.0.0.1:8000 PLATFORM=tpu \
  MODEL='Qwen/Qwen2.5-0.5B-Instruct' PARAMS_B=0.49
```

4. Results include `platform_config` in `results/tpu/run_01_replay.json` (serving args, docker image, node selectors).

## Switching TPU → GPU

1. **Same trace file** as the TPU run.
2. **Deploy GPU vLLM**:

```bash
make manifests
kubectl apply -f tiny-model-gpu.yaml
kubectl rollout status deployment/tiny-model-gpu-server --timeout=600s
kubectl port-forward svc/tiny-model-gpu-service 8001:8000
```

3. **Replay on a different local port** (so both port-forwards can coexist):

```bash
make replay TARGET=http://127.0.0.1:8001 PLATFORM=gpu \
  MODEL='Qwen/Qwen2.5-0.5B-Instruct' PARAMS_B=0.49
```

4. Compare:

```bash
make normalize-replay PLATFORM=tpu
make normalize-replay PLATFORM=gpu
make compare
```

## What changes per platform

| Parameter | GPU | TPU |
|-----------|-----|-----|
| Docker image | `vllm/vllm-openai:latest` | `vllm/vllm-tpu:latest` |
| Node selector | `nvidia-h100-80gb` | `tpu-v6e-slice` + topology `1x1` |
| Resource | `nvidia.com/gpu: 1` | `google.com/tpu: 1` |
| Runtime version | — | `v2-alpha-tpuv6e` |
| K8s service | `tiny-model-gpu-service` | `tiny-model-service` |
| Memory util | `0.95` | `0.98` |

Shared serving (must match for fair comparison):

| Parameter | Value |
|-----------|-------|
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| `max_model_len` | 4096 |
| `max_num_batched_tokens` | 4096 |
| `max_num_seqs` | 128 |
| `tensor_parallel_size` | 1 |

## VM path (non-GKE)

For bare-metal / VM deployments, shell scripts read the same config:

```bash
bash scripts/run_gpu.sh serve-cmd   # prints docker run with GPU_* exports
bash scripts/run_tpu.sh serve-cmd   # prints vLLM TPU serve with TPU_* exports
```

Exports come from `eval $(python3 scripts/config.py --shell)`.

## UI

The results UI (`make ui`, port 8787) shows:

- Per-platform **serving config** from the last replay JSON
- **Migration diff** table at `/api/migration` (GPU vs TPU parameters from config)

## Editing platform params

1. Edit `configs/benchmark_config.yaml` under `gpu:` or `tpu:`.
2. Run `make manifests` to refresh K8s YAML.
3. Redeploy and re-run replay.

Do **not** put platform-specific request shapes in the trace — that breaks apples-to-apples comparison.

## Common migration mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Wrong docker image on TPU | Pod crash / no TPU detected | Use `vllm/vllm-tpu:latest` |
| `max-model-len=2048` with 1024-word trace | 100 errors, context length | Set `max_model_len: 4096`, `make manifests`, redeploy |
| Different model name in replay vs server | `model does not exist` | Match `MODEL=` to server's `--model` |
| Regenerated trace between GPU and TPU runs | Invalid comparison | Reuse same `prompts.jsonl` + seed |
