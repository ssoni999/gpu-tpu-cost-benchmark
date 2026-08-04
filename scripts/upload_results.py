#!/usr/bin/env python3
"""Upload local benchmark results to GCS."""

from __future__ import annotations

import argparse
from pathlib import Path

from gcs_results import PLATFORMS, gcs_settings, upload_all_local, upload_replay_file

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload benchmark results to GCS")
    parser.add_argument("--platform", choices=[*PLATFORMS, "all"], default="all")
    parser.add_argument("--bucket", help="Override GCS_RESULTS_BUCKET")
    parser.add_argument("--project", help="Override GCS_RESULTS_PROJECT")
    args = parser.parse_args()

    settings = gcs_settings(bucket=args.bucket, project=args.project)
    print(f"Bucket: gs://{settings.bucket}/ ({settings.project or 'default project'})")

    if args.platform == "all":
        uris = upload_all_local(ROOT, settings=settings)
    else:
        path = ROOT / "results" / args.platform / "run_01_replay.json"
        if not path.exists():
            print(f"Missing {path}")
            return 1
        uri = upload_replay_file(args.platform, path, settings=settings)
        uris = [uri] if uri else []

    if not uris:
        print("Nothing uploaded.")
        return 1

    for uri in uris:
        print(f"  {uri}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
