from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from round7.data import LANGUAGES
from round7.training import read_jsonl, train_seed_model


def alignment_hours(rows: list[dict[str, Any]]) -> dict[str, float]:
    seconds: defaultdict[str, float] = defaultdict(float)
    for row in rows:
        seconds[row["language"]] += float(row["duration"])
    return {language: seconds[language] / 3600 for language in LANGUAGES}


def assert_no_alignment_leakage(
    split_rows: list[dict[str, Any]], aligned_rows: list[dict[str, Any]]
) -> None:
    validation_sources = {
        (row["language"], row["source_id"])
        for row in split_rows
        if row["split"] == "validation"
    }
    leaking = {
        (row["language"], row["source_id"])
        for row in aligned_rows
        if (row["language"], row["source_id"]) in validation_sources
    }
    if leaking:
        raise RuntimeError(f"aligned manifest leaks {len(leaking)} validation sources")


def evaluate_gate(
    clean: dict[str, float],
    aligned: dict[str, float],
    hours: dict[str, float],
    gate_config: dict[str, Any],
) -> dict[str, Any]:
    deltas = {
        language: float(clean[f"wer_{language}"]) - float(aligned[f"wer_{language}"])
        for language in LANGUAGES
    }
    macro_improvement = float(clean["macro_wer"]) - float(aligned["macro_wer"])
    improved_languages = sum(delta > 0 for delta in deltas.values())
    reasons = []
    minimum_hours = float(gate_config["minimum_accepted_hours_per_language"])
    required_languages = tuple(gate_config["required_alignment_languages"])
    insufficient = [
        language for language in required_languages if hours.get(language, 0.0) < minimum_hours
    ]
    if insufficient:
        reasons.append("insufficient_aligned_hours:" + ",".join(insufficient))
    if macro_improvement < float(gate_config["minimum_macro_wer_improvement"]):
        reasons.append("macro_wer_improvement_below_threshold")
    if improved_languages < int(gate_config["minimum_languages_improved"]):
        reasons.append("too_few_languages_improved")
    maximum_regression = float(gate_config["maximum_language_regression"])
    regressed = [language for language, delta in deltas.items() if delta < -maximum_regression]
    if regressed:
        reasons.append("excessive_language_regression:" + ",".join(regressed))
    return {
        "proceed": not reasons,
        "reasons": reasons,
        "clean_macro_wer": float(clean["macro_wer"]),
        "aligned_macro_wer": float(aligned["macro_wer"]),
        "macro_wer_improvement": macro_improvement,
        "languages_improved": improved_languages,
        "per_language_improvement": deltas,
        "accepted_hours": hours,
        "thresholds": dict(gate_config),
    }


def select_lower_macro_candidate(
    first: dict[str, Any], second: dict[str, Any]
) -> dict[str, Any]:
    required = {"name", "checkpoint", "metrics"}
    if not required.issubset(first) or not required.issubset(second):
        raise ValueError("checkpoint candidates require name, checkpoint, and metrics")
    return min((first, second), key=lambda item: float(item["metrics"]["macro_wer"]))


def run_alignment_gate(
    config: dict[str, Any],
    split_directory: Path,
    smoke_directory: Path,
    seed_directory: Path,
    alignment_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    accepted_path = alignment_directory / "accepted.jsonl"
    aligned_rows = read_jsonl(accepted_path)
    split_rows = read_jsonl(split_directory / "manifest.jsonl")
    if not aligned_rows:
        raise RuntimeError("alignment gate has no accepted pilot segments")
    assert_no_alignment_leakage(split_rows, aligned_rows)
    clean_pointer = json.loads(
        (seed_directory / "checkpoints" / "best.json").read_text(encoding="utf-8")
    )
    clean_metrics = clean_pointer["metrics"]
    candidate = train_seed_model(
        config,
        split_directory,
        smoke_directory,
        output_directory,
        additional_train_manifest=accepted_path,
        maximum_steps_override=int(config["gate"]["candidate_max_steps"]),
    )
    aligned_pointer = json.loads(
        (output_directory / "checkpoints" / "best.json").read_text(encoding="utf-8")
    )
    report = evaluate_gate(
        clean_metrics,
        aligned_pointer["metrics"],
        alignment_hours(aligned_rows),
        config["gate"],
    )
    report["clean_checkpoint"] = clean_pointer["checkpoint"]
    report["aligned_checkpoint"] = aligned_pointer["checkpoint"]
    report["candidate_training"] = candidate
    return report
