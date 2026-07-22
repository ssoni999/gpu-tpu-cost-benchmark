# GPU vs TPU Cost Benchmark

Measured **Qwen/Qwen3-4B** inference economics on **G4 (vLLM)** vs **TPU v5e (vLLM TPU)**. Every headline number comes from a real `vllm bench serve` run — not placeholders.

Recipe repos (`gpu-recipes`, `tpu-recipes`) stay **adjacent and read-only** for infra patterns. This repo owns the **benchmark contract**, parsing, and comparison.

## Layout

```
configs/benchmark_config.yaml   # Single source of truth (model, prompts, tokens, seed)
configs/gpu.env.example         # Copy → gpu.env (pricing + GCP IDs)
configs/tpu.env.example         # Copy → tpu.env (v5e + pricing)
scripts/run_gpu.sh              # Run on G4 GCE VM
scripts/run_tpu.sh              # Run on TPU v5e VM
scripts/normalize_results.py    # Raw bench text → normalized JSON
scripts/calculate_cost.py       # Normalized → cost metrics
scripts/compare.py              # GPU + TPU → comparison.json
results/gpu/                    # Raw .txt + environment.json
results/tpu/
results/normalized/             # gpu.json, tpu.json
comparison.json                 # Headline comparison (after both sides complete)
```

## Frozen benchmark contract

From `configs/benchmark_config.yaml` (identical on both platforms):

| Parameter | Value |
|-----------|-------|
| Model | `Qwen/Qwen3-4B` |
| Prompts | 100 |
| Input tokens | 1024 |
| Output tokens | 128 |
| Seed | 100 |
| Request rate | `inf` |

Recipe defaults (e.g. TPU guide’s 1000 prompts / 1800 input) are **not** used.

## TPU v5e note

Official `tpu-recipes` Qwen3 guide targets **v6e**. This project pins **v5e**:

- Accelerator: `v5litepod-1` (1 chip for Qwen3-4B)
- Runtime: `v2-alpha-tpuv5-lite`
- Image: `vllm/vllm-tpu:latest`

See [vLLM TPU setup](https://docs.vllm.ai/projects/tpu/en/latest/getting_started/tpu_setup/) for v5e provisioning.

## Quick start

### 1. Local setup (Cursor / Mac)

```bash
pip install -r requirements.txt
make bench-cmd    # verify vllm bench flags from config
```

### 2. GPU path (G4 GCE)

```bash
cp configs/gpu.env.example configs/gpu.env   # edit PROJECT_ID, ZONE, rates
bash scripts/run_gpu.sh serve-cmd            # print docker serve command
bash scripts/run_gpu.sh bench                # after server is up on :8000
```

### 3. TPU path (v5e)

```bash
cp configs/tpu.env.example configs/tpu.env    # edit PROJECT_ID, ZONE, rates
bash scripts/run_tpu.sh provision-cmd        # gcloud queued-resource (v5litepod-1)
bash scripts/run_tpu.sh serve-cmd            # vLLM TPU serve in container
bash scripts/run_tpu.sh bench                # inside running container
```

### 4. Normalize and compare (Mac)

```bash
make normalize GPU_RAW=results/gpu/run_01.txt TPU_RAW=results/tpu/run_01.txt
make compare
cat comparison.json
```

## Headline metrics

- Cost per 1M **output** tokens
- Cost per 1,000 successful requests
- Output tokens per second
- p95 TTFT
- Performance per dollar (tok/s / $)
- Projected annual cost (configurable volume in `calculate_cost.py`)

## Related repos (adjacent, not submodules)

- `../gpu-recipes` — G4 vLLM serving pattern
- `../tpu-recipes` — Qwen3 vLLM guide (adapted here for v5e)
- `../MeasurementLayer` — source for cost math ideas (not a runtime dependency)
- `../production-stack` — archived; not used in this path

## Honest limits

- Verify **G4** and **TPU v5e** quota before provisioning.
- Set real hourly rates in `gpu.env` / `tpu.env` before quoting dollars.
- v1 uses `vllm bench serve` random dataset with fixed seed — not customer trace replay.
- Run `measured_runs: 3` manually until automation is added.
