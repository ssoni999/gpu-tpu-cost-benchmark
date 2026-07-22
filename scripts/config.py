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
