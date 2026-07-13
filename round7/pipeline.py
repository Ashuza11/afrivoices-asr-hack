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
    "alignment-gate",
    "align-full",
    "train-final",
    "infer-test",
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


def alignment_gate(config: dict[str, Any]) -> None:
    from round7.gate import run_alignment_gate

    required = ("seed-smoke", "train-seed", "align-pilot")
    for stage in required:
        if not (stage_directory(config, stage) / "status.json").is_file():
            raise RuntimeError(f"{stage} must complete before alignment-gate")
    directory = stage_directory(config, "alignment-gate")
    result = run_alignment_gate(
        config,
        stage_directory(config, "build-splits"),
        stage_directory(config, "seed-smoke"),
        stage_directory(config, "train-seed"),
        stage_directory(config, "align-pilot"),
        directory,
    )
    atomic_write_json(directory / "gate.json", result)
    mark_complete(config, "alignment-gate", result)
    print(json.dumps(result, indent=2))


def _require_gate(config: dict[str, Any]) -> dict[str, Any]:
    path = stage_directory(config, "alignment-gate") / "gate.json"
    if not path.is_file():
        raise RuntimeError("alignment-gate must complete first")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not result.get("proceed"):
        raise RuntimeError("alignment gate rejected aligned data: " + "; ".join(result["reasons"]))
    return result


def align_full(config: dict[str, Any]) -> None:
    from round7.alignment import run_full_alignment

    gate = _require_gate(config)
    directory = stage_directory(config, "align-full")
    result = run_full_alignment(
        config,
        stage_directory(config, "build-splits"),
        stage_directory(config, "alignment-gate"),
        directory,
    )
    result["gate_macro_wer_improvement"] = gate["macro_wer_improvement"]
    atomic_write_json(directory / "report.json", result)
    mark_complete(config, "align-full", result)
    print(json.dumps(result, indent=2))


def train_final(config: dict[str, Any]) -> None:
    from round7.gate import select_lower_macro_candidate
    from round7.training import train_seed_model

    gate = _require_gate(config)
    aligned = stage_directory(config, "align-full") / "accepted.jsonl"
    if not aligned.is_file():
        raise RuntimeError("align-full must complete before train-final")
    directory = stage_directory(config, "train-final")
    result = train_seed_model(
        config,
        stage_directory(config, "build-splits"),
        stage_directory(config, "seed-smoke"),
        directory,
        additional_train_manifest=aligned,
        maximum_steps_override=int(config["final_training"]["maximum_steps"]),
        evaluation_steps_override=int(config["final_training"]["evaluation_steps"]),
        initial_checkpoint_override=Path(_require_gate(config)["aligned_checkpoint"]),
    )
    final_pointer = json.loads(
        (directory / "checkpoints" / "best.json").read_text(encoding="utf-8")
    )
    selected = select_lower_macro_candidate(
        {
            "name": "gated_pilot",
            "checkpoint": gate["aligned_checkpoint"],
            "metrics": {"macro_wer": gate["aligned_macro_wer"]},
        },
        {
            "name": "full_aligned",
            "checkpoint": final_pointer["checkpoint"],
            "metrics": final_pointer["metrics"],
        },
    )
    result["inference_selection"] = selected
    atomic_write_json(directory / "result.json", result)
    mark_complete(config, "train-final", result)
    print(json.dumps(result, indent=2))


def infer_test(config: dict[str, Any]) -> None:
    from round7.inference import run_test_inference

    if not (stage_directory(config, "train-final") / "status.json").is_file():
        raise RuntimeError("train-final must complete before infer-test")
    directory = stage_directory(config, "infer-test")
    result = run_test_inference(config, stage_directory(config, "train-final"), directory)
    atomic_write_json(directory / "report.json", result)
    mark_complete(config, "infer-test", result)
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
        if stage in {"prepare-sources", "build-splits", "align-pilot", "align-full"}:
            from round7.sources import manifest_audio_complete

            manifests = {
                "prepare-sources": Path(config["paths"]["source_manifest"]),
                "build-splits": stage_directory(config, "build-splits") / "manifest.jsonl",
                "align-pilot": stage_directory(config, "align-pilot") / "accepted.jsonl",
                "align-full": stage_directory(config, "align-full") / "accepted.jsonl",
            }
            return manifest_audio_complete(manifests[stage])
        if stage == "seed-smoke":
            return (stage_directory(config, stage) / "checkpoint" / "config.json").is_file()
        if stage in {"train-seed", "alignment-gate", "train-final"}:
            pointer = stage_directory(config, stage) / "checkpoints" / "best.json"
            if not pointer.is_file():
                return False
            checkpoint = Path(json.loads(pointer.read_text(encoding="utf-8"))["checkpoint"])
            return (checkpoint / "config.json").is_file() and (checkpoint / "model.safetensors").is_file()
        if stage == "infer-test":
            return (stage_directory(config, stage) / "submission_round7_xlsr_greedy.csv").is_file()
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
    elif stage == "alignment-gate":
        alignment_gate(config)
    elif stage == "align-full":
        align_full(config)
    elif stage == "train-final":
        train_final(config)
    elif stage == "infer-test":
        infer_test(config)
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
