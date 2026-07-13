from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from round7.data import (
    LANGUAGES,
    assign_grouped_splits,
    atomic_write_jsonl,
    build_shared_vocabulary,
    load_and_validate_manifest,
    manifest_report,
    verify_audio_records,
)


IMPLEMENTED_STAGES = (
    "preflight",
    "prepare-sources",
    "build-splits",
    "seed-smoke",
    "train-seed",
    "align-pilot",
)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    parent = value.get("extends")
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        return deep_merge(load_yaml(parent_path.resolve()), value)
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def unresolved_fields(value: Any, prefix: str = "") -> list[str]:
    fields: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            fields.extend(unresolved_fields(child, name))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            fields.extend(unresolved_fields(child, f"{prefix}[{index}]"))
    elif value == "FILL_ME":
        fields.append(prefix)
    return fields


def validate_main_config(config: dict[str, Any]) -> None:
    languages = tuple(config.get("project", {}).get("languages", ()))
    if languages != LANGUAGES:
        raise ValueError(f"project.languages must be exactly {list(LANGUAGES)}")
    paths = config.get("paths", {})
    for key in ("source_manifest", "output_dir", "persistent_dir", "scratch_dir"):
        if not paths.get(key):
            raise ValueError(f"paths.{key} is required")


def stage_directory(config: dict[str, Any], stage: str) -> Path:
    return Path(config["paths"]["output_dir"]) / stage


def mark_complete(config: dict[str, Any], stage: str, details: dict[str, Any]) -> None:
    payload = {
        "stage": stage,
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "details": details,
    }
    atomic_write_json(stage_directory(config, stage) / "status.json", payload)


def preflight(config: dict[str, Any], cluster_config: dict[str, Any] | None) -> None:
    validate_main_config(config)
    unresolved = unresolved_fields(cluster_config) if cluster_config else []
    if unresolved:
        raise ValueError("cluster configuration has unresolved fields: " + ", ".join(unresolved))

    source_manifest = Path(config["paths"]["source_manifest"])

    scratch = Path(config["paths"]["scratch_dir"])
    output = Path(config["paths"]["output_dir"])
    scratch.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    minimum_disk = float(config["runtime"]["minimum_free_disk_gb"])
    free_disk = shutil.disk_usage(scratch).free / 1024**3
    if free_disk < minimum_disk:
        raise RuntimeError(f"scratch has {free_disk:.1f} GB free; {minimum_disk:.1f} GB required")

    torch_details: dict[str, Any]
    try:
        import torch

        cuda = torch.cuda.is_available()
        torch_details = {"version": torch.__version__, "cuda_available": cuda}
        if cuda:
            props = torch.cuda.get_device_properties(0)
            gpu_gb = props.total_memory / 1024**3
            torch_details.update({"gpu": props.name, "gpu_memory_gb": round(gpu_gb, 2)})
            minimum_gpu = float(config["runtime"]["minimum_gpu_memory_gb"])
            if gpu_gb < minimum_gpu:
                raise RuntimeError(f"GPU has {gpu_gb:.1f} GB; {minimum_gpu:.1f} GB required")
        elif config["runtime"].get("require_cuda", True):
            raise RuntimeError("CUDA is required but torch.cuda.is_available() is false")
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed") from exc

    details = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "free_scratch_gb": round(free_disk, 2),
        "source_manifest": str(source_manifest),
        "source_manifest_exists": source_manifest.is_file(),
        "torch": torch_details,
    }
    mark_complete(config, "preflight", details)
    print(json.dumps(details, indent=2))


def prepare_source_data(config: dict[str, Any]) -> None:
    from round7.sources import prepare_sources

    details = prepare_sources(config)
    mark_complete(config, "prepare-sources", details)
    print(json.dumps(details, indent=2))


def build_splits(config: dict[str, Any]) -> None:
    source = Path(config["paths"]["source_manifest"])
    records, rejected = load_and_validate_manifest(source)
    if config["data"].get("verify_audio_files", True):
        records, audio_rejected = verify_audio_records(
            records,
            audio_root=Path(config["paths"]["audio_root"]),
            minimum_duration=float(config["data"]["minimum_duration_seconds"]),
            maximum_duration=float(config["data"]["maximum_source_duration_seconds"]),
        )
        rejected.extend(audio_rejected)
    split_records = assign_grouped_splits(
        records,
        validation_fraction=float(config["data"]["validation_fraction"]),
        seed=int(config["project"]["seed"]),
    )

    directory = stage_directory(config, "build-splits")
    atomic_write_jsonl(directory / "manifest.jsonl", split_records)
    atomic_write_jsonl(directory / "rejected.jsonl", rejected)
    atomic_write_json(directory / "vocabulary.json", build_shared_vocabulary(split_records))
    report = manifest_report(split_records)
    report["rejected_rows"] = len(rejected)
    atomic_write_json(directory / "report.json", report)
    mark_complete(config, "build-splits", report)
    print(json.dumps(report, indent=2))


def seed_smoke(config: dict[str, Any]) -> None:
    from round7.training import seed_smoke_test

    split_directory = stage_directory(config, "build-splits")
    if not (split_directory / "status.json").is_file():
        raise RuntimeError("build-splits must complete before seed-smoke")
    directory = stage_directory(config, "seed-smoke")
    metrics = seed_smoke_test(config, split_directory, directory)
    atomic_write_json(directory / "metrics.json", metrics)
    mark_complete(config, "seed-smoke", metrics)
    print(json.dumps(metrics, indent=2))


def train_seed(config: dict[str, Any]) -> None:
    from round7.training import train_seed_model

    split_directory = stage_directory(config, "build-splits")
    smoke_directory = stage_directory(config, "seed-smoke")
    if not (smoke_directory / "status.json").is_file():
        raise RuntimeError("seed-smoke must complete before train-seed")
    directory = stage_directory(config, "train-seed")
    result = train_seed_model(config, split_directory, smoke_directory, directory)
    atomic_write_json(directory / "result.json", result)
    mark_complete(config, "train-seed", result)
    print(json.dumps(result, indent=2))


def align_pilot(config: dict[str, Any]) -> None:
    from round7.alignment import run_alignment_pilot

    seed_directory = stage_directory(config, "train-seed")
    if not (seed_directory / "status.json").is_file():
        raise RuntimeError("train-seed must complete before align-pilot")
    directory = stage_directory(config, "align-pilot")
    result = run_alignment_pilot(
        config, stage_directory(config, "build-splits"), seed_directory, directory
    )
    atomic_write_json(directory / "report.json", result)
    mark_complete(config, "align-pilot", result)
    print(json.dumps(result, indent=2))


def is_complete(config: dict[str, Any], stage: str) -> bool:
    if stage == "preflight":
        return False
    status = stage_directory(config, stage) / "status.json"
    if not status.is_file():
        return False
    try:
        complete = json.loads(status.read_text(encoding="utf-8")).get("status") == "complete"
        if not complete:
            return False
        if stage in {"prepare-sources", "build-splits", "align-pilot"}:
            from round7.sources import manifest_audio_complete

            manifests = {
                "prepare-sources": Path(config["paths"]["source_manifest"]),
                "build-splits": stage_directory(config, "build-splits") / "manifest.jsonl",
                "align-pilot": stage_directory(config, "align-pilot") / "accepted.jsonl",
            }
            return manifest_audio_complete(manifests[stage])
        return True
    except (OSError, json.JSONDecodeError):
        return False


def run_stage(
    config: dict[str, Any], stage: str, cluster_config: dict[str, Any] | None, force: bool
) -> None:
    if not force and is_complete(config, stage):
        print(f"{stage}: already complete; use --force to rerun")
        return
    if stage == "preflight":
        preflight(config, cluster_config)
    elif stage == "prepare-sources":
        prepare_source_data(config)
    elif stage == "build-splits":
        build_splits(config)
    elif stage == "seed-smoke":
        seed_smoke(config)
    elif stage == "train-seed":
        train_seed(config)
    elif stage == "align-pilot":
        align_pilot(config)
    else:
        raise ValueError(f"stage is not implemented: {stage}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 7 resumable pipeline")
    parser.add_argument("--config", type=Path, default=Path("round7/config.yaml"))
    parser.add_argument("--cluster-config", type=Path)
    parser.add_argument("--stage", choices=(*IMPLEMENTED_STAGES, "all"), default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    cluster_config = load_yaml(args.cluster_config) if args.cluster_config else None
    stages = IMPLEMENTED_STAGES if args.stage == "all" else (args.stage,)
    for stage in stages:
        run_stage(config, stage, cluster_config, args.force)


if __name__ == "__main__":
    main()
