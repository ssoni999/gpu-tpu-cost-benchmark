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


def blob_updated_epoch(object_name: str, settings: GcsSettings | None = None) -> float:
    """Unix timestamp from GCS object updated time, or 0."""
    settings = settings or gcs_settings()
    try:
        blob = _bucket(settings).blob(object_name)
        if not blob.exists():
            return 0.0
        blob.reload()
        if blob.updated:
            return blob.updated.timestamp()
    except Exception:  # noqa: BLE001
        return 0.0
    return 0.0


def object_implies_tier(object_name: str, tier: str, settings: GcsSettings) -> bool:
    """True when object path is tier-specific (latest/small/tpu/...), not legacy flat layout."""
    prefix = f"{settings.prefix}/{tier}/"
    return object_name.startswith(prefix)


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


def rebuild_manifest_from_bucket(settings: GcsSettings | None = None) -> dict[str, Any]:
    """Scan known GCS paths and write latest/manifest.json (fixes missing manifest)."""
    from config import replay_matches_tier

    settings = settings or gcs_settings()
    manifest: dict[str, Any] = {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "platforms": {},
    }
    for platform in PLATFORMS:
        manifest["platforms"].setdefault(platform, {"tiers": {}})
        for tier in workload_tier_ids():
            for obj in manifest_replay_candidates({}, platform, tier, settings):
                summary = download_json(obj, settings=settings)
                if summary is None:
                    continue
                trusted = object_implies_tier(obj, tier, settings)
                if not trusted and not replay_matches_tier(summary, tier):
                    continue
                uri = f"gs://{settings.bucket}/{obj}"
                ts = blob_updated_epoch(obj, settings)
                uploaded_at = (
                    datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    if ts
                    else manifest["updated_at"]
                )
                manifest["platforms"][platform]["tiers"][tier] = {
                    "latest_uri": uri,
                    "uploaded_at": uploaded_at,
                    "successful_requests": summary.get("successful_requests"),
                    "failed_requests": summary.get("failed_requests"),
                    "output_tokens_per_second": summary.get("output_tokens_per_second"),
                    "model": summary.get("model"),
                    "workload_tier": tier,
                }
                break
    upload_json(manifest, manifest_object(settings), settings=settings)
    return manifest


def pull_all_replays(root: Path, settings: GcsSettings | None = None) -> list[str]:
    """Download all tier replays from GCS into results/{platform}/{tier}/."""
    from config import replay_matches_tier, replay_result_path

    settings = settings or gcs_settings()
    written: list[str] = []
    for platform in PLATFORMS:
        for tier in workload_tier_ids():
            for obj in manifest_replay_candidates({}, platform, tier, settings):
                summary = download_json(obj, settings=settings)
                if summary is None:
                    continue
                trusted = object_implies_tier(obj, tier, settings)
                if not trusted and not replay_matches_tier(summary, tier):
                    continue
                out = replay_result_path(platform, tier, root=root)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
                written.append(str(out.relative_to(root)))
                break
    return written


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

    def read_replay_object(
        self, platform: str, tier: str = "medium",
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Return (replay JSON, object key) from first matching GCS object."""
        if not self.enabled:
            return None, None

        def _load() -> tuple[dict[str, Any] | None, str | None]:
            manifest = read_manifest(self.settings)
            for obj in manifest_replay_candidates(manifest, platform, tier, self.settings):
                data = download_json(obj, settings=self.settings)
                if data is not None:
                    return data, obj
            return None, None

        return self._cached(f"replay_obj:{platform}:{tier}", _load)

    def read_replay(self, platform: str, tier: str = "medium") -> dict[str, Any] | None:
        replay, _obj = self.read_replay_object(platform, tier)
        return replay

    def tier_upload_epoch(self, platform: str, tier: str) -> float:
        """Unix timestamp from manifest or GCS blob updated time."""
        if not self.enabled:
            return 0.0

        def _load() -> float:
            manifest = read_manifest(self.settings)
            entry = manifest_tier_entry(manifest, platform, tier)
            uploaded = entry.get("uploaded_at")
            if uploaded:
                try:
                    dt = datetime.fromisoformat(str(uploaded).replace("Z", "+00:00"))
                    return dt.timestamp()
                except ValueError:
                    pass
            best = 0.0
            for obj in manifest_replay_candidates(manifest, platform, tier, self.settings):
                best = max(best, blob_updated_epoch(obj, self.settings))
            return best

        return float(self._cached(f"tier_ts:{platform}:{tier}", _load) or 0.0)

    def inspect_tier(self, platform: str, tier: str) -> dict[str, Any]:
        """Diagnostic: which GCS objects exist and whether they'd be loaded."""
        from config import replay_matches_tier, workload_profile, load_config

        cfg = load_config()
        expected = int(workload_profile(cfg, tier).get("num_prompts") or 0)
        manifest = read_manifest(self.settings)
        objects: list[dict[str, Any]] = []
        chosen: str | None = None
        for obj in manifest_replay_candidates(manifest, platform, tier, self.settings):
            row: dict[str, Any] = {"object": obj, "exists": False}
            try:
                blob = _bucket(self.settings).blob(obj)
                row["exists"] = blob.exists()
                if row["exists"]:
                    blob.reload()
                    row["updated"] = blob.updated.isoformat() if blob.updated else None
                    data = download_json(obj, settings=self.settings)
                    if data:
                        actual = int(data.get("successful_requests") or 0) + int(
                            data.get("failed_requests") or 0
                        )
                        trusted = object_implies_tier(obj, tier, self.settings)
                        matches = trusted or replay_matches_tier(data, tier, cfg)
                        row["requests"] = actual
                        row["expected_requests"] = expected
                        row["trusted_path"] = trusted
                        row["matches_tier"] = matches
                        row["output_tokens_per_second"] = data.get("output_tokens_per_second")
                        if matches and chosen is None:
                            chosen = obj
            except Exception as exc:  # noqa: BLE001
                row["error"] = str(exc)
            objects.append(row)
        return {
            "platform": platform,
            "tier": tier,
            "chosen_object": chosen,
            "candidates": objects,
        }

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


def check_gcs_access(settings: GcsSettings | None = None) -> dict[str, Any]:
    """Verify Application Default Credentials and bucket read access."""
    settings = settings or gcs_settings()
    out: dict[str, Any] = {
        "ok": False,
        "bucket": settings.bucket,
        "project": settings.project,
        "prefix": settings.prefix,
        "adc_project": None,
        "error": None,
        "fix": "",
    }
    if os.environ.get("CLOUD_SHELL"):
        out["fix"] = (
            f"Grant Storage Object Viewer on gs://{settings.bucket} to your Cloud Shell "
            f"user/service account (project {settings.project or DEFAULT_PROJECT}). "
            "Do not run gcloud auth application-default login on Cloud Shell."
        )
    else:
        out["fix"] = (
            "gcloud auth application-default login "
            f"--project={settings.project or DEFAULT_PROJECT} "
            "--scopes=https://www.googleapis.com/auth/cloud-platform"
        )
    try:
        import google.auth

        credentials, adc_project = google.auth.default()
        out["adc_project"] = adc_project
        if credentials is None:
            out["error"] = "No Application Default Credentials found"
            return out
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
        return out

    try:
        bucket = _bucket(settings)
        # Probe read access using a tier path that should exist after replays.
        probe = bucket.blob(latest_replay_object("tpu", settings, tier="medium"))
        probe.exists()
        out["ok"] = True
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
    return out


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
