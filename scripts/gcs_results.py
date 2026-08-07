"""Upload and read benchmark results from Google Cloud Storage."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config import load_config, workload_tier_ids

PLATFORMS = ("tpu", "gpu")
DEFAULT_BUCKET = "gpu-tpu-benchmark-storage"
DEFAULT_PROJECT = "gpu-tpu-benchmark-results"


@dataclass
class GcsSettings:
    bucket: str
    project: str | None
    prefix: str
    archive_prefix: str

    @property
    def enabled(self) -> bool:
        return bool(self.bucket)


def gcs_settings(
    bucket: str | None = None,
    project: str | None = None,
    prefix: str | None = None,
) -> GcsSettings:
    cfg = load_config()
    storage = cfg.get("storage", {}).get("gcs", {})
    return GcsSettings(
        bucket=bucket or os.environ.get("GCS_RESULTS_BUCKET") or storage.get("bucket") or DEFAULT_BUCKET,
        project=project or os.environ.get("GCS_RESULTS_PROJECT") or storage.get("project") or DEFAULT_PROJECT,
        prefix=prefix or os.environ.get("GCS_RESULTS_PREFIX") or storage.get("prefix", "latest"),
        archive_prefix=storage.get("archive_prefix", "runs"),
    )


def _client(settings: GcsSettings):
    from google.cloud import storage

    if settings.project:
        return storage.Client(project=settings.project)
    return storage.Client()


def _bucket(settings: GcsSettings):
    return _client(settings).bucket(settings.bucket)


def upload_bytes(
    data: bytes,
    object_name: str,
    content_type: str = "application/json",
    settings: GcsSettings | None = None,
) -> str:
    settings = settings or gcs_settings()
    blob = _bucket(settings).blob(object_name)
    blob.upload_from_string(data, content_type=content_type)
    return f"gs://{settings.bucket}/{object_name}"


def upload_json(
    payload: dict[str, Any],
    object_name: str,
    settings: GcsSettings | None = None,
) -> str:
    body = json.dumps(payload, indent=2) + "\n"
    return upload_bytes(body.encode("utf-8"), object_name, settings=settings)


def download_json(object_name: str, settings: GcsSettings | None = None) -> dict[str, Any] | None:
    settings = settings or gcs_settings()
    blob = _bucket(settings).blob(object_name)
    if not blob.exists():
        return None
    return json.loads(blob.download_as_text(encoding="utf-8"))


def latest_replay_object(
    platform: str,
    settings: GcsSettings | None = None,
    tier: str = "medium",
) -> str:
    settings = settings or gcs_settings()
    return f"{settings.prefix}/{tier}/{platform}/run_01_replay.json"


def legacy_latest_replay_object(platform: str, settings: GcsSettings | None = None) -> str:
    settings = settings or gcs_settings()
    return f"{settings.prefix}/{platform}/run_01_replay.json"


def latest_live_object(
    platform: str,
    settings: GcsSettings | None = None,
    tier: str = "medium",
) -> str:
    settings = settings or gcs_settings()
    return f"{settings.prefix}/{tier}/{platform}/live.json"


def legacy_latest_live_object(platform: str, settings: GcsSettings | None = None) -> str:
    settings = settings or gcs_settings()
    return f"{settings.prefix}/{platform}/live.json"


def latest_comparison_object(settings: GcsSettings | None = None) -> str:
    settings = settings or gcs_settings()
    return f"{settings.prefix}/comparison.json"


def manifest_object(settings: GcsSettings | None = None) -> str:
    settings = settings or gcs_settings()
    return f"{settings.prefix}/manifest.json"


def gs_uri_to_object(uri: str, bucket: str) -> str | None:
    """Convert gs://bucket/key to object key."""
    prefix = f"gs://{bucket}/"
    if uri.startswith(prefix):
        return uri[len(prefix):]
    return None


def manifest_tier_entry(
    manifest: dict[str, Any],
    platform: str,
    tier: str,
) -> dict[str, Any]:
    platforms = manifest.get("platforms") or {}
    block = platforms.get(platform) or {}
    tiers = block.get("tiers") or {}
    return tiers.get(tier) or {}


def manifest_replay_candidates(
    manifest: dict[str, Any],
    platform: str,
    tier: str,
    settings: GcsSettings,
) -> list[str]:
    """Object paths to try for a tier replay (manifest URI first, then canonical paths)."""
    objects: list[str] = []
    entry = manifest_tier_entry(manifest, platform, tier)
    uri = entry.get("latest_uri")
    if uri:
        obj = gs_uri_to_object(str(uri), settings.bucket)
        if obj:
            objects.append(obj)
    objects.append(latest_replay_object(platform, settings, tier=tier))
    if tier == "medium":
        objects.append(legacy_latest_replay_object(platform, settings))
    # De-dupe while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for obj in objects:
        if obj not in seen:
            seen.add(obj)
            out.append(obj)
    return out


def read_manifest(settings: GcsSettings | None = None) -> dict[str, Any]:
    settings = settings or gcs_settings()
    return download_json(manifest_object(settings), settings=settings) or {}


def _update_manifest(
    platform: str,
    latest_uri: str,
    summary: dict[str, Any],
    settings: GcsSettings,
    *,
    tier: str = "medium",
) -> None:
    manifest = read_manifest(settings)
    manifest["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest.setdefault("platforms", {})
    manifest["platforms"].setdefault(platform, {})
    manifest["platforms"][platform].setdefault("tiers", {})
    manifest["platforms"][platform]["tiers"][tier] = {
        "latest_uri": latest_uri,
        "uploaded_at": manifest["updated_at"],
        "successful_requests": summary.get("successful_requests"),
        "failed_requests": summary.get("failed_requests"),
        "output_tokens_per_second": summary.get("output_tokens_per_second"),
        "model": summary.get("model"),
        "workload_tier": tier,
    }
    # Backward-compatible flat fields (latest upload wins)
    manifest["platforms"][platform].update(manifest["platforms"][platform]["tiers"][tier])
    upload_json(manifest, manifest_object(settings), settings=settings)


def upload_replay_summary(
    platform: str,
    summary: dict[str, Any],
    *,
    settings: GcsSettings | None = None,
    archive: bool = True,
    tier: str = "medium",
) -> str | None:
    """Upload replay JSON to GCS latest/{tier}/ + optional timestamped archive."""
    if platform not in PLATFORMS:
        raise ValueError(f"platform must be one of {PLATFORMS}")

    settings = settings or gcs_settings()
    if not settings.enabled:
        return None

    summary = dict(summary)
    summary.setdefault("workload_tier", tier)
    bc = summary.get("benchmark_contract")
    if isinstance(bc, dict):
        bc = dict(bc)
        bc.setdefault("workload_tier", tier)
        summary["benchmark_contract"] = bc
    latest_obj = latest_replay_object(platform, settings, tier=tier)
    latest_uri = upload_json(summary, latest_obj, settings=settings)

    if archive:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
        archive_obj = f"{settings.archive_prefix}/{tier}/{platform}/{ts}_run_01_replay.json"
        upload_json(summary, archive_obj, settings=settings)

    _update_manifest(platform, latest_uri, summary, settings, tier=tier)
    return latest_uri


def upload_replay_file(
    platform: str,
    path: Path,
    *,
    settings: GcsSettings | None = None,
    tier: str | None = None,
) -> str | None:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if tier is None:
        tier = path.parent.name if path.parent.name in workload_tier_ids() else "medium"
    return upload_replay_summary(platform, summary, settings=settings, tier=tier)


def upload_live_snapshot(
    platform: str,
    payload: dict[str, Any],
    *,
    settings: GcsSettings | None = None,
) -> str | None:
    settings = settings or gcs_settings()
    if not settings.enabled:
        return None
    obj = latest_live_object(platform, settings)
    return upload_json(payload, obj, settings=settings)


def upload_comparison(
    comparison: dict[str, Any],
    *,
    settings: GcsSettings | None = None,
) -> str | None:
    settings = settings or gcs_settings()
    if not settings.enabled:
        return None
    obj = latest_comparison_object(settings)
    uri = upload_json(comparison, obj, settings=settings)
    manifest = read_manifest(settings)
    manifest["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest["comparison_uri"] = uri
    upload_json(manifest, manifest_object(settings), settings=settings)
    return uri


def upload_all_local(
    root: Path,
    *,
    settings: GcsSettings | None = None,
) -> list[str]:
    settings = settings or gcs_settings()
    uris: list[str] = []
    for platform in PLATFORMS:
        for tier in workload_tier_ids():
            replay_path = root / "results" / platform / tier / "run_01_replay.json"
            if replay_path.exists():
                summary = json.loads(replay_path.read_text(encoding="utf-8"))
                uri = upload_replay_summary(
                    platform, summary, settings=settings, tier=tier,
                )
                if uri:
                    uris.append(uri)
        legacy = root / "results" / platform / "run_01_replay.json"
        medium_path = root / "results" / platform / "medium" / "run_01_replay.json"
        if legacy.exists() and not medium_path.exists():
            summary = json.loads(legacy.read_text(encoding="utf-8"))
            uri = upload_replay_summary(
                platform, summary, settings=settings, tier="medium",
            )
            if uri:
                uris.append(uri)
        for tier in workload_tier_ids():
            live_path = root / "results" / platform / tier / "live.json"
            if live_path.exists():
                payload = json.loads(live_path.read_text(encoding="utf-8"))
                obj = f"{settings.prefix}/{tier}/{platform}/live.json"
                uri = upload_json(payload, obj, settings=settings)
                if uri:
                    uris.append(uri)
        legacy_live = root / "results" / platform / "live.json"
        if legacy_live.exists():
            payload = json.loads(legacy_live.read_text(encoding="utf-8"))
            uri = upload_live_snapshot(platform, payload, settings=settings)
            if uri:
                uris.append(uri)
    comparison_path = root / "comparison.json"
    if comparison_path.exists():
        payload = json.loads(comparison_path.read_text(encoding="utf-8"))
        uri = upload_comparison(payload, settings=settings)
        if uri:
            uris.append(uri)
    return uris


class GcsResultsStore:
    """Read-through cache for UI/API."""

    def __init__(
        self,
        settings: GcsSettings | None = None,
        cache_ttl_s: float = 3.0,
    ) -> None:
        self.settings = settings or gcs_settings()
        self.cache_ttl_s = cache_ttl_s
        self._cache: dict[str, tuple[float, Any]] = {}

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    def _cached(self, key: str, loader) -> Any:
        now = time.monotonic()
        hit = self._cache.get(key)
        if hit and now - hit[0] < self.cache_ttl_s:
            return hit[1]
        value = loader()
        self._cache[key] = (now, value)
        return value

    def read_replay(self, platform: str, tier: str = "medium") -> dict[str, Any] | None:
        if not self.enabled:
            return None

        def _load() -> dict[str, Any] | None:
            manifest = read_manifest(self.settings)
            for obj in manifest_replay_candidates(manifest, platform, tier, self.settings):
                data = download_json(obj, settings=self.settings)
                if data is not None:
                    return data
            return None

        return self._cached(f"replay:{platform}:{tier}", _load)

    def tier_upload_epoch(self, platform: str, tier: str) -> float:
        """Unix timestamp from manifest uploaded_at, or 0 if unknown."""
        if not self.enabled:
            return 0.0

        def _load() -> float:
            manifest = read_manifest(self.settings)
            entry = manifest_tier_entry(manifest, platform, tier)
            uploaded = entry.get("uploaded_at")
            if not uploaded:
                return 0.0
            try:
                dt = datetime.fromisoformat(str(uploaded).replace("Z", "+00:00"))
                return dt.timestamp()
            except ValueError:
                return 0.0

        return float(self._cached(f"tier_ts:{platform}:{tier}", _load) or 0.0)

    def read_live(self, platform: str, tier: str = "medium") -> dict[str, Any] | None:
        if not self.enabled:
            return None
        obj = latest_live_object(platform, self.settings, tier=tier)

        def _load() -> dict[str, Any] | None:
            data = download_json(obj, settings=self.settings)
            if data is not None:
                return data
            if tier == "medium":
                return download_json(
                    legacy_latest_live_object(platform, self.settings),
                    settings=self.settings,
                )
            return None

        return self._cached(f"live:{platform}:{tier}", _load)

    def read_comparison(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        obj = latest_comparison_object(self.settings)

        def _load() -> dict[str, Any] | None:
            return download_json(obj, settings=self.settings)

        return self._cached(f"comparison", _load)

    def read_manifest(self) -> dict[str, Any]:
        if not self.enabled:
            return {}

        def _load() -> dict[str, Any]:
            return read_manifest(self.settings)

        return self._cached("manifest", _load) or {}


def try_upload_replay(
    platform: str,
    summary: dict[str, Any],
    *,
    tier: str = "medium",
) -> str | None:
    """Best-effort upload; prints warning on failure."""
    settings = gcs_settings()
    if not settings.enabled:
        return None
    if os.environ.get("GCS_UPLOAD", "1") not in ("1", "true", "yes"):
        return None
    try:
        uri = upload_replay_summary(platform, summary, settings=settings, tier=tier)
        print(f"Uploaded to {uri}")
        return uri
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: GCS upload failed: {exc}")
        if os.environ.get("GCS_UPLOAD_REQUIRED", "0") in ("1", "true", "yes"):
            raise
        return None
