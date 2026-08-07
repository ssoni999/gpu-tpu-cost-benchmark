#!/usr/bin/env python3
"""Pull benchmark replays from GCS into local results/ and optionally rebuild manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from gcs_results import (
    gcs_settings,
    pull_all_replays,
    rebuild_manifest_from_bucket,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Pull replay JSON from GCS bucket")
    parser.add_argument("--bucket", help="Override GCS_RESULTS_BUCKET")
    parser.add_argument("--project", help="Override GCS_RESULTS_PROJECT")
    parser.add_argument("--prefix", help="Override GCS_RESULTS_PREFIX")
    parser.add_argument(
        "--rebuild-manifest",
        action="store_true",
        help="Write latest/manifest.json from objects found in the bucket",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only rebuild manifest.json (do not download to results/)",
    )
    args = parser.parse_args()

    settings = gcs_settings(bucket=args.bucket, project=args.project, prefix=args.prefix)
    print(f"Bucket: gs://{settings.bucket}/{settings.prefix}/")
    print(f"Project: {settings.project or '(default)'}")

    if args.rebuild_manifest or args.manifest_only:
        manifest = rebuild_manifest_from_bucket(settings=settings)
        print(f"Wrote gs://{settings.bucket}/{settings.prefix}/manifest.json")
        for platform, block in (manifest.get("platforms") or {}).items():
            tiers = (block.get("tiers") or {}).keys()
            print(f"  {platform}: {', '.join(sorted(tiers)) or 'none'}")

    if args.manifest_only:
        return 0

    paths = pull_all_replays(ROOT, settings=settings)
    if not paths:
        print("No replays downloaded. Check bucket paths and IAM.")
        print("Expected: latest/{small,medium,high}/{tpu,gpu}/run_01_replay.json")
        return 1

    for path in paths:
        print(f"  {path}")
    print(f"Pulled {len(paths)} replay file(s) into results/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
