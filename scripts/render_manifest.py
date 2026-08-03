#!/usr/bin/env python3
"""Render GKE Deployment/Service YAML from configs/benchmark_config.yaml."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from config import ROOT, load_config, platform_block, serving_vllm_args

GENERATED_HEADER = (
    "# Generated from configs/benchmark_config.yaml — do not edit by hand.\n"
    "# Regenerate: make manifests\n"
)


def _deployment(platform: str, cfg: dict[str, Any]) -> dict[str, Any]:
    block = platform_block(cfg, platform)
    k8s = block["k8s"]
    infra = block["infra"]
    app = k8s["app_label"]
    args = serving_vllm_args(cfg, platform)

    container: dict[str, Any] = {
        "name": k8s["container_name"],
        "image": block["docker_image"],
        "command": ["python3", "-m", "vllm.entrypoints.openai.api_server"],
        "args": args,
        "ports": [{"containerPort": k8s.get("port", 8000)}],
        "resources": infra.get("resources", {}),
    }
    if platform == "tpu":
        container["imagePullPolicy"] = "IfNotPresent"

    pod_spec: dict[str, Any] = {
        "nodeSelector": infra.get("node_selector", {}),
        "tolerations": infra.get("tolerations", []),
        "containers": [container],
    }

    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": k8s["deployment_name"]},
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": {"app": app}},
            "template": {
                "metadata": {"labels": {"app": app}},
                "spec": pod_spec,
            },
        },
    }


def _service(platform: str, cfg: dict[str, Any]) -> dict[str, Any]:
    block = platform_block(cfg, platform)
    k8s = block["k8s"]
    app = k8s["app_label"]
    port = k8s.get("port", 8000)
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": k8s["service_name"]},
        "spec": {
            "type": "ClusterIP",
            "selector": {"app": app},
            "ports": [{"port": port, "targetPort": port}],
        },
    }


def render_platform(platform: str, cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg or load_config()
    docs = [_deployment(platform, cfg), _service(platform, cfg)]
    chunks = [yaml.safe_dump(doc, sort_keys=False, default_flow_style=False).rstrip() for doc in docs]
    return GENERATED_HEADER + "\n---\n".join(chunks) + "\n"


def write_manifests(cfg: dict[str, Any] | None = None) -> list[Path]:
    cfg = cfg or load_config()
    written: list[Path] = []
    for platform in platforms:
        block = platform_block(cfg, platform)
        out = ROOT / block["k8s"]["output_file"]
        out.write_text(render_platform(platform, cfg), encoding="utf-8")
        written.append(out)
        print(f"Wrote {out.relative_to(ROOT)}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Render K8s manifests from benchmark_config.yaml")
    parser.add_argument("--platform", choices=["gpu", "tpu", "all"], default="all")
    parser.add_argument("--stdout", action="store_true", help="Print to stdout instead of writing files")
    args = parser.parse_args()

    cfg = load_config()
    platforms: tuple[str, ...] = ("tpu", "gpu") if args.platform == "all" else (args.platform,)

    if args.stdout:
        for i, platform in enumerate(platforms):
            if i:
                print("---")
            print(render_platform(platform, cfg), end="")
        return

    for platform in platforms:
        block = platform_block(cfg, platform)
        out = ROOT / block["k8s"]["output_file"]
        out.write_text(render_platform(platform, cfg), encoding="utf-8")
        print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
