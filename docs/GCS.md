# GCS results storage

Benchmark metrics are uploaded to **`gs://gpu-tpu-benchmark-storage`** (project `gpu-tpu-benchmark-results`, region `us-central1`).

The UI reads the **latest** results from the bucket by default.

## Bucket layout

```
gs://gpu-tpu-benchmark-storage/
  latest/
    manifest.json                 # pointers + upload timestamps
    tpu/run_01_replay.json        # latest TPU metrics
    gpu/run_01_replay.json        # latest GPU metrics
    tpu/live.json                 # in-progress run (optional)
    comparison.json               # after make compare
  runs/
    tpu/2026-08-04T173000Z_run_01_replay.json   # archived history
    gpu/...
```

## One-time setup (Cloud Shell)

```bash
# 1. Enable API (if not already)
gcloud services enable storage.googleapis.com --project=gpu-tpu-benchmark-results

# 2. Grant your account write access to the bucket
gcloud storage buckets describe gs://gpu-tpu-benchmark-storage --project=gpu-tpu-benchmark-results

# 3. Application Default Credentials (Cloud Shell usually has this already)
gcloud auth application-default login --project=gpu-tpu-benchmark-results

# 4. Configure repo
cp configs/gcs.env.example configs/gcs.env
pip install -r requirements.txt
```

### IAM

The identity running `make replay` / `make ui` needs:

- `storage.objects.create`
- `storage.objects.get`
- `storage.objects.list` (optional)

Example role: **Storage Object Admin** on the bucket, or **Storage Object User** for read-only UI.

## Upload (automatic)

Every successful `make replay` uploads to GCS when `GCS_UPLOAD=1` (default in `gcs.env.example`):

```bash
source configs/gcs.env
make replay TARGET=http://127.0.0.1:8000 PLATFORM=tpu MODEL='Qwen/Qwen2.5-0.5B-Instruct'
# → Wrote results/tpu/run_01_replay.json
# → Uploaded to gs://gpu-tpu-benchmark-storage/latest/tpu/run_01_replay.json
```

Manual upload of existing local files:

```bash
make upload-results              # all platforms + comparison
make upload-results PLATFORM=tpu # one platform
```

`make compare` also uploads `latest/comparison.json`.

## UI from GCS

```bash
source configs/gcs.env
make ui
# Web Preview → port 8787
```

The UI uses `--source auto` (default): reads GCS when the bucket is configured, falls back to local `results/` if GCS is empty or unreachable.

Force GCS-only:

```bash
make ui RESULTS_SOURCE=gcs
```

Force local-only:

```bash
make ui RESULTS_SOURCE=local
```

Check data source:

```bash
curl -s http://127.0.0.1:8787/api/status | python3 -m json.tool
```

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `GCS_RESULTS_BUCKET` | `gpu-tpu-benchmark-storage` | Bucket name |
| `GCS_RESULTS_PROJECT` | `gpu-tpu-benchmark-results` | GCP project for API client |
| `GCS_RESULTS_PREFIX` | `latest` | Prefix for “current” results |
| `GCS_UPLOAD` | `1` | Upload after replay (`0` to disable) |
| `GCS_UPLOAD_REQUIRED` | `0` | Fail replay if upload fails |

Values are also in `configs/benchmark_config.yaml` under `storage.gcs`.

## Cross-machine workflow

1. **Cloud Shell** runs replay → uploads to GCS automatically.
2. **Mac / another machine** runs `make ui` with the same `gcs.env` → UI shows latest Cloud Shell results.
3. No need to copy `results/` manually.
