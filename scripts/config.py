"""Load benchmark_config.yaml for scripts and shell wrappers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "benchmark_config.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def platform_block(cfg: dict[str, Any], platform: str) -> dict[str, Any]:
    if platform not in ("gpu", "tpu"):
        raise ValueError(f"platform must be gpu or tpu, got {platform!r}")
    return cfg[platform]


def serving_vllm_args(cfg: dict[str, Any], platform: str) -> list[str]:
    """OpenAI API server CLI flags for a platform (from benchmark_config.yaml)."""
    block = platform_block(cfg, platform)
    serving = block["serving"]
    model = cfg["model"]
    args = [
        f"--model={model}",
        f"--max-model-len={serving['max_model_len']}",
        f"--max-num-batched-tokens={serving['max_num_batched_tokens']}",
        f"--max-num-seqs={serving['max_num_seqs']}",
        f"--tensor-parallel-size={serving['tensor_parallel_size']}",
    ]
    if serving.get("gpu_memory_utilization") is not None:
        args.append(f"--gpu-memory-utilization={serving['gpu_memory_utilization']}")
    port = block.get("k8s", {}).get("port", 8000)
    args.append(f"--port={port}")
    for extra in serving.get("extra_args", []):
        token = str(extra).strip()
        if not token.startswith("--"):
            token = f"--{token}"
        args.append(token)
    return args


def platform_metadata(cfg: dict[str, Any], platform: str) -> dict[str, Any]:
    """Platform snapshot for replay JSON, UI, and migration docs."""
    block = platform_block(cfg, platform)
    return {
        "platform": platform,
        "accelerator": block.get("accelerator"),
        "accelerator_count": block.get("accelerator_count"),
        "generation": block.get("generation"),
        "runtime_version": block.get("runtime_version"),
        "framework": block.get("framework"),
        "docker_image": block.get("docker_image"),
        "recipe_ref": block.get("recipe_ref"),
        "k8s": block.get("k8s", {}),
        "infra": block.get("infra", {}),
        "serving": block.get("serving", {}),
        "vllm_args": serving_vllm_args(cfg, platform),
    }


def migration_diff(cfg: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Side-by-side GPU vs TPU parameters for migration checklists."""
    cfg = cfg or load_config()
    gpu = platform_metadata(cfg, "gpu")
    tpu = platform_metadata(cfg, "tpu")
    rows: list[dict[str, Any]] = []

    def add(category: str, key: str, gpu_val: Any, tpu_val: Any) -> None:
        rows.append({
            "category": category,
            "key": key,
            "gpu": gpu_val,
            "tpu": tpu_val,
            "same": gpu_val == tpu_val,
        })

    add("workload", "model", cfg["model"], cfg["model"])
    b = cfg["benchmark"]
    add("workload", "input_tokens", b["input_tokens"], b["input_tokens"])
    add("workload", "output_tokens", b["output_tokens"], b["output_tokens"])
    add("workload", "num_prompts", b["num_prompts"], b["num_prompts"])

    add("infra", "docker_image", gpu["docker_image"], tpu["docker_image"])
    add("infra", "accelerator", gpu["accelerator"], tpu["accelerator"])
    add("infra", "runtime_version", gpu.get("runtime_version"), tpu.get("runtime_version"))

    gs, ts = gpu["serving"], tpu["serving"]
    for key in (
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
        "tensor_parallel_size",
        "gpu_memory_utilization",
    ):
        add("serving", key, gs.get(key), ts.get(key))

    gsel = gpu.get("infra", {}).get("node_selector", {})
    tsel = tpu.get("infra", {}).get("node_selector", {})
    add("k8s", "node_selector", gsel, tsel)
    add("k8s", "deployment", gpu["k8s"].get("deployment_name"), tpu["k8s"].get("deployment_name"))
    add("k8s", "service", gpu["k8s"].get("service_name"), tpu["k8s"].get("service_name"))

    return rows


def bench_cli_args(config: dict[str, Any] | None = None) -> list[str]:
    """vllm bench serve flags derived from benchmark_config.yaml."""
    cfg = config or load_config()
    b = cfg["benchmark"]
    args = [
        "--backend",
        "vllm",
        "--model",
        cfg["model"],
        "--dataset-name",
        str(b["dataset_name"]),
        "--num-prompts",
        str(b["num_prompts"]),
        "--random-input-len",
        str(b["input_tokens"]),
        "--random-output-len",
        str(b["output_tokens"]),
        "--seed",
        str(b["seed"]),
        "--request-rate",
        str(b["request_rate"]),
    ]
    if b.get("ignore_eos"):
        args.append("--ignore-eos")
    return args


def bench_command_string(host: str = "http://localhost:8000") -> str:
    cfg = load_config()
    flags = " ".join(bench_cli_args(cfg))
    return f"vllm bench serve --host {host} {flags}"


def emit_shell_exports() -> None:
    """Print export statements for use in bash: eval $(python3 scripts/config.py --shell)."""
    cfg = load_config()
    b = cfg["benchmark"]
    g = cfg["gpu"]
    t = cfg["tpu"]
    gs = g["serving"]
    ts = t["serving"]

    exports = {
        "BENCH_MODEL": cfg["model"],
        "BENCH_NUM_PROMPTS": b["num_prompts"],
        "BENCH_INPUT_TOKENS": b["input_tokens"],
        "BENCH_OUTPUT_TOKENS": b["output_tokens"],
        "BENCH_SEED": b["seed"],
        "BENCH_REQUEST_RATE": b["request_rate"],
        "BENCH_MEASURED_RUNS": b["measured_runs"],
        "GPU_MAX_MODEL_LEN": gs["max_model_len"],
        "GPU_MAX_BATCHED_TOKENS": gs["max_num_batched_tokens"],
        "GPU_MAX_NUM_SEQS": gs["max_num_seqs"],
        "GPU_TP_SIZE": gs["tensor_parallel_size"],
        "GPU_DOCKER_IMAGE": g["docker_image"],
        "TPU_MAX_MODEL_LEN": ts["max_model_len"],
        "TPU_MAX_BATCHED_TOKENS": ts["max_num_batched_tokens"],
        "TPU_MAX_NUM_SEQS": ts["max_num_seqs"],
        "TPU_TP_SIZE": ts["tensor_parallel_size"],
        "TPU_DOCKER_IMAGE": t["docker_image"],
        "TPU_ACCELERATOR_TYPE": t["accelerator"],
        "TPU_RUNTIME_VERSION": t["runtime_version"],
    }
    for key, value in exports.items():
        print(f'export {key}="{value}"')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--shell", action="store_true", help="Emit bash exports")
    parser.add_argument("--json", action="store_true", help="Print full config as JSON")
    parser.add_argument("--bench-cmd", action="store_true", help="Print vllm bench serve command")
    args = parser.parse_args()

    if args.shell:
        emit_shell_exports()
    elif args.json:
        print(json.dumps(load_config(), indent=2))
    elif args.bench_cmd:
        print(bench_command_string())
    else:
        print(bench_command_string())
