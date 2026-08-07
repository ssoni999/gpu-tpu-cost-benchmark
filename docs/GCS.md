# GCS results storage

Benchmark metrics are uploaded to **`gs://gpu-tpu-benchmark-storage`** (project `gpu-tpu-benchmark-results`, region `us-central1`).

The UI reads the **latest** results from the bucket by default.

## Bucket layout

```
gs://gpu-tpu-benchmark-storage/
  latest/
    manifest.json                 # pointers + upload timestamps per tier
    small/tpu/run_01_replay.json  # tier-specific (Simple)
    medium/tpu/run_01_replay.json # tier-specific (Medium)
    high/tpu/run_01_replay.json   # tier-specific (Complex)
    tpu/run_01_replay.json        # legacy medium layout (fallback)
    gpu/...
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

# 3. Cloud Shell: credentials come from the VM automatically — do NOT run
#    gcloud auth application-default login (it crashes with scope errors).
#    Verify access instead:
make check-gcs

# 3b. Local laptop only (not Cloud Shell):
# gcloud auth application-default login --project=gpu-tpu-benchmark-results \
#   --scopes=https://www.googleapis.com/auth/cloud-platform

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
cp configs/gcs.env.example configs/gcs.env   # if not done yet
source configs/gcs.env
make ui
# Web Preview → port 8787
```

Default is **`RESULTS_SOURCE=gcs`** — UI reads directly from the bucket.

If `manifest.json` is missing (common for older uploads), rebuild it once:

```bash
make rebuild-gcs-manifest
```

Or pull all replays locally **and** rebuild manifest:

```bash
make pull-gcs
make ui RESULTS_SOURCE=local   # optional: serve from downloaded results/
```

Debug what GCS objects the UI can see:

```bash
curl -s http://127.0.0.1:8787/api/gcs/inspect | python3 -m json.tool
```

The UI uses `--source gcs` by default. To merge with local files:

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
