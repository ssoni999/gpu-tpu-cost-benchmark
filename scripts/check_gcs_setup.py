#!/usr/bin/env python3
"""Verify GCS credentials and bucket access for the results UI."""

from __future__ import annotations

import json
import os
import sys

from gcs_results import check_gcs_access, gcs_settings


def main() -> int:
    settings = gcs_settings()
    result = check_gcs_access(settings)

    print(f"Bucket:  gs://{result['bucket']}/{result['prefix']}/")
    print(f"Project: {result['project']}")
    if result.get("adc_project"):
        print(f"ADC:     {result['adc_project']}")

    if result.get("ok"):
        print("GCS access: OK")
        return 0

    print(f"GCS access: FAILED")
    print(f"Error: {result.get('error')}")

    if os.environ.get("CLOUD_SHELL"):
        print()
        print("Cloud Shell already provides credentials via the VM.")
        print("Do NOT run 'gcloud auth application-default login' here (it often crashes).")
        print("If bucket access is denied, ask the bucket admin to grant your account:")
        print(f"  Storage Object Viewer on gs://{settings.bucket}")
        print(f"  (project: {settings.project})")
    else:
        print()
        print("Fix (local machine):")
        print(
            "  gcloud auth application-default login "
            f"--project={settings.project} "
            "--scopes=https://www.googleapis.com/auth/cloud-platform"
        )

    if os.environ.get("JSON"):
        print(json.dumps(result, indent=2))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
