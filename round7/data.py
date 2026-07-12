from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator


LANGUAGES = ("swa", "som", "kik", "luo", "mas", "kln")
PUNCTUATION_RE = re.compile(r"[^\w\sÀ-ɏ̀-ͯḀ-ỿ'’ŋŊ]", flags=re.UNICODE)
FIELD_ALIASES = {
    "id": ("id", "utterance_id", "record_id"),
    "source_id": ("source_id", "recording_id", "original_id", "id"),
    "audio_path": ("audio_path", "path", "file", "filename"),
    "language": ("language", "lang", "lang3"),
    "dialect": ("dialect", "accent", "variety"),
    "subtype": ("subtype", "type", "scripted_type"),
    "speaker_id": ("speaker_id", "speaker", "client_id"),
    "duration": ("duration", "duration_seconds", "seconds"),
    "transcription": (
        "normalized_transcription",
        "transcription",
        "actualSentence",
        "transcript",
        "text",
        "sentence",
    ),
}


class ManifestError(ValueError):
    pass


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFC", text).lower()
    text = PUNCTUATION_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _first(record: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        value = record.get(name)
        if value is not None and str(value).strip():
            return value
    return None


def canonicalize_record(record: dict[str, Any], row_number: int) -> dict[str, Any]:
    values = {field: _first(record, aliases) for field, aliases in FIELD_ALIASES.items()}
    language = str(values["language"] or "").strip().lower()
    if language not in LANGUAGES:
        raise ManifestError(f"row {row_number}: unsupported language {language!r}")

    text = normalize_text(values["transcription"])
    if not text:
        raise ManifestError(f"row {row_number}: empty transcription")

    source_id = str(values["source_id"] or "").strip()
    if not source_id:
        raise ManifestError(f"row {row_number}: source_id is required")

    audio_path = str(values["audio_path"] or "").strip()
    if not audio_path:
        raise ManifestError(f"row {row_number}: audio_path is required")

    raw_duration = values["duration"]
    try:
        duration = float(raw_duration) if raw_duration is not None else None
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"row {row_number}: invalid duration {raw_duration!r}") from exc
    if duration is not None and duration <= 0:
        raise ManifestError(f"row {row_number}: duration must be positive")

    record_id = str(values["id"] or "").strip()
    if not record_id:
        digest = hashlib.sha1(
            f"{source_id}\0{audio_path}\0{text}".encode("utf-8")
        ).hexdigest()[:16]
        record_id = f"{language}-{digest}"

    return {
        "id": record_id,
        "source_id": source_id,
        "audio_path": audio_path,
        "language": language,
        "dialect": str(values["dialect"] or "unknown").strip().lower(),
        "subtype": str(values["subtype"] or "unknown").strip().lower(),
        "speaker_id": str(values["speaker_id"] or "").strip() or None,
        "duration": duration,
        "transcription": text,
        "split": None,
        "dataset": str(record.get("dataset") or "unknown"),
        "source_shard": str(record.get("source_shard") or "unknown"),
        "source_repo": str(record.get("source_repo") or "unknown"),
    }


def iter_source_records(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for row_number, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ManifestError(f"line {row_number}: invalid JSON: {exc}") from exc
        return
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            yield from csv.DictReader(handle)
        return
    raise ManifestError(f"manifest must be .jsonl or .csv: {path}")


def atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def verify_audio_records(
    records: Iterable[dict[str, Any]],
    audio_root: Path,
    minimum_duration: float,
    maximum_duration: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile is required for audio verification") from exc

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for record in records:
        candidate = Path(record["audio_path"])
        path = candidate if candidate.is_absolute() else audio_root / candidate
        try:
            if not path.is_file():
                raise ManifestError("audio file does not exist")
            info = sf.info(path)
            if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
                raise ManifestError("audio metadata is invalid")
            duration = info.frames / info.samplerate
            if not minimum_duration <= duration <= maximum_duration:
                raise ManifestError(
                    f"duration {duration:.3f}s outside {minimum_duration}-{maximum_duration}s"
                )
            row = dict(record)
            row["audio_path"] = str(path.resolve())
            row["duration"] = duration
            row["sample_rate"] = info.samplerate
            row["channels"] = info.channels
            accepted.append(row)
        except (OSError, RuntimeError, ManifestError) as exc:
            rejected.append({"id": record["id"], "reason": str(exc)})
    if not accepted:
        raise ManifestError("audio verification rejected every record")
    return accepted, rejected


def build_shared_vocabulary(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    characters = sorted(
        {
            character
            for record in records
            if record.get("split") == "train"
            for character in record["transcription"]
            if character != " "
        }
    )
    vocabulary = {"<pad>": 0, "<unk>": 1, "|": 2}
    for character in characters:
        if character not in vocabulary:
            vocabulary[character] = len(vocabulary)
    if len(vocabulary) <= 3:
        raise ManifestError("training split produced an empty character vocabulary")
    return vocabulary


def load_and_validate_manifest(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row_number, raw in enumerate(iter_source_records(path), 1):
        try:
            record = canonicalize_record(raw, row_number)
            if record["id"] in seen_ids:
                raise ManifestError(f"row {row_number}: duplicate id {record['id']!r}")
            seen_ids.add(record["id"])
            accepted.append(record)
        except ManifestError as exc:
            rejected.append({"row_number": row_number, "reason": str(exc)})
    if not accepted:
        raise ManifestError("manifest has no accepted records")
    return accepted, rejected


def _group_key(record: dict[str, Any]) -> str:
    # Speaker grouping is stronger than source grouping: all recordings from a
    # known speaker stay together. Source ID is the fallback when speaker is not
    # available. Language is included because source IDs need not be global.
    speaker = record.get("speaker_id")
    identity = f"speaker:{speaker}" if speaker else f"source:{record['source_id']}"
    return f"{record['language']}\0{identity}"


def assign_grouped_splits(
    records: list[dict[str, Any]], validation_fraction: float, seed: int
) -> list[dict[str, Any]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    groups_by_language: dict[str, set[str]] = {language: set() for language in LANGUAGES}
    for record in records:
        groups_by_language[record["language"]].add(_group_key(record))

    validation_groups: set[str] = set()
    for language, groups in groups_by_language.items():
        ordered = sorted(
            groups,
            key=lambda group: hashlib.sha256(f"{seed}\0{group}".encode()).digest(),
        )
        if len(ordered) < 2:
            raise ManifestError(f"{language}: at least two source groups are required")
        count = max(1, round(len(ordered) * validation_fraction))
        count = min(count, len(ordered) - 1)
        validation_groups.update(ordered[:count])

    result = []
    for original in records:
        record = dict(original)
        record["split"] = "validation" if _group_key(record) in validation_groups else "train"
        result.append(record)
    assert_no_group_leakage(result)
    return result


def assert_no_group_leakage(records: Iterable[dict[str, Any]]) -> None:
    assignments: dict[str, set[str]] = {}
    for record in records:
        assignments.setdefault(_group_key(record), set()).add(record["split"])
    leaking = [group for group, splits in assignments.items() if len(splits) > 1]
    if leaking:
        raise ManifestError(f"split leakage detected in {len(leaking)} source groups")


def manifest_report(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    counts = Counter((row["split"], row["language"], row["subtype"]) for row in rows)
    durations: Counter[tuple[str, str]] = Counter()
    unknown_duration = Counter()
    for row in rows:
        key = (row["split"], row["language"])
        if row["duration"] is None:
            unknown_duration[key] += 1
        else:
            durations[key] += row["duration"]
    return {
        "total_rows": len(rows),
        "rows": [
            {"split": key[0], "language": key[1], "subtype": key[2], "count": value}
            for key, value in sorted(counts.items())
        ],
        "hours": [
            {
                "split": split,
                "language": language,
                "hours": round(seconds / 3600, 3),
                "unknown_duration_rows": unknown_duration[(split, language)],
            }
            for (split, language), seconds in sorted(durations.items())
        ],
    }
