from __future__ import annotations

import gc
import hashlib
import io
import json
import math
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import soundfile as sf
from huggingface_hub import hf_hub_download, list_repo_files
from pydub import AudioSegment
from scipy.signal import resample_poly

from round7.data import LANGUAGES, atomic_write_jsonl, normalize_text


AUDIO_EXTENSIONS = {".webm", ".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"}
ANV_LANGUAGES = ("som", "kik", "luo", "mas", "kln")


def clean_optional(value: Any) -> Any | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str) and not value.strip():
        return None
    return value


def validate_source_policy(config: dict[str, Any]) -> None:
    policy = config["sources"]["policy"]
    if tuple(policy["digital_umuganda_languages"]) != ("swa",):
        raise ValueError("Digital Umuganda source policy must contain Swahili only")
    if tuple(policy["anv_languages"]) != ANV_LANGUAGES:
        raise ValueError(f"ANV source policy must be exactly {list(ANV_LANGUAGES)}")
    if policy.get("allow_digital_umuganda_somali") is not False:
        raise ValueError("Digital Umuganda Somali must remain disabled")
    if tuple(config["sources"]["anv"]["languages"]) != ANV_LANGUAGES:
        raise ValueError("sources.anv.languages violates the locked source policy")


def stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha1("\0".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}-{digest}"


def decode_audio_field(field: Any, target_sample_rate: int) -> np.ndarray:
    if isinstance(field, dict) and field.get("array") is not None:
        audio = np.asarray(field["array"], dtype=np.float32)
        sample_rate = int(field.get("sampling_rate") or target_sample_rate)
    else:
        raw = field.get("bytes") if isinstance(field, dict) else field
        if not isinstance(raw, (bytes, bytearray)) or not raw:
            raise ValueError(f"unsupported audio field: {type(field)}")
        try:
            audio, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        except Exception:
            segment = AudioSegment.from_file(io.BytesIO(raw))
            segment = segment.set_channels(1).set_frame_rate(target_sample_rate)
            scale = float(1 << (8 * segment.sample_width - 1))
            audio = np.asarray(segment.get_array_of_samples(), dtype=np.float32) / scale
            sample_rate = target_sample_rate
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != target_sample_rate:
        divisor = math.gcd(sample_rate, target_sample_rate)
        audio = resample_poly(
            audio,
            up=target_sample_rate // divisor,
            down=sample_rate // divisor,
        )
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("decoded audio is empty or non-finite")
    return np.asarray(audio, dtype=np.float32)


def write_flac(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".flac", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        sf.write(temporary, audio, sample_rate, format="FLAC", subtype="PCM_16")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def source_record(
    *,
    record_id: str,
    source_id: str,
    audio_path: Path,
    language: str,
    text: str,
    subtype: str,
    dataset: str,
    source_repo: str,
    source_shard: str,
    row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = row or {}
    audio_info = sf.info(audio_path)
    return {
        "id": record_id,
        "source_id": source_id,
        "audio_path": str(audio_path.resolve()),
        "language": language,
        "dialect": clean_optional(row.get("dialect"))
        or clean_optional(row.get("accent"))
        or clean_optional(row.get("variety"))
        or "unknown",
        "subtype": subtype,
        "speaker_id": clean_optional(row.get("speaker_id"))
        or clean_optional(row.get("speaker"))
        or clean_optional(row.get("client_id")),
        "duration": audio_info.frames / audio_info.samplerate,
        "transcription": normalize_text(text),
        "dataset": dataset,
        "source_repo": source_repo,
        "source_shard": source_shard,
    }


def _manifest_text(entry: dict[str, Any]) -> str:
    return str(
        entry.get("normalized_transcription")
        or entry.get("transcription")
        or entry.get("transcript")
        or entry.get("text")
        or ""
    ).strip()


def manifest_audio_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        return bool(rows) and all(Path(row["audio_path"]).is_file() for row in rows)
    except (OSError, KeyError, json.JSONDecodeError):
        return False


def prepare_swahili_shard(
    shard: int,
    source_config: dict[str, Any],
    audio_root: Path,
    manifest_root: Path,
    sample_rate: int,
    token: str | None,
) -> dict[str, Any]:
    repo_id = source_config["repo_id"]
    shard_name = f"swa-{shard:04d}"
    output_manifest = manifest_root / "swahili" / f"{shard_name}.jsonl"
    if manifest_audio_complete(output_manifest):
        return {"shard": shard_name, "status": "existing", "manifest": str(output_manifest)}
    manifest_path = hf_hub_download(
        repo_id, source_config["manifest_pattern"].format(shard=shard), repo_type="dataset", token=token
    )
    archive_path = hf_hub_download(
        repo_id, source_config["audio_pattern"].format(shard=shard), repo_type="dataset", token=token
    )
    wanted: dict[str, str] = {}
    with open(manifest_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            text = _manifest_text(entry)
            key = str(entry.get("key") or entry.get("audio_filepath") or "")
            if text and key:
                wanted[os.path.splitext(os.path.basename(key))[0]] = text
                wanted[key] = text

    records = []
    rejected = 0
    with tarfile.open(archive_path, "r:xz") as archive:
        for member in archive:
            extension = os.path.splitext(member.name)[1].lower()
            if member.isdir() or extension not in AUDIO_EXTENSIONS:
                continue
            base = os.path.splitext(os.path.basename(member.name))[0]
            text = wanted.get(base) or wanted.get(member.name)
            if not text:
                continue
            try:
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError("archive member has no data")
                audio = decode_audio_field(extracted.read(), sample_rate)
                record_id = stable_id("swa", shard_name, member.name)
                path = audio_root / "swa" / shard_name / f"{record_id}.flac"
                if not path.is_file():
                    write_flac(path, audio, sample_rate)
                records.append(
                    source_record(
                        record_id=record_id,
                        source_id=record_id,
                        audio_path=path,
                        language="swa",
                        text=text,
                        subtype="scripted",
                        dataset="digital_umuganda_swahili",
                        source_repo=repo_id,
                        source_shard=shard_name,
                    )
                )
            except Exception:
                rejected += 1
    if not records:
        raise RuntimeError(f"{shard_name}: no usable records")
    atomic_write_jsonl(output_manifest, records)
    return {"shard": shard_name, "status": "prepared", "rows": len(records), "rejected": rejected}


def list_anv_shards(source_config: dict[str, Any], token: str | None) -> list[tuple[str, str, str]]:
    repo_id = source_config["repo_id"]
    files = sorted(list_repo_files(repo_id, repo_type="dataset", token=token))
    selected: list[tuple[str, str, str]] = []
    for path in files:
        parts = path.split("/")
        if (
            path.endswith(".parquet")
            and len(parts) >= 4
            and parts[0] in source_config["languages"]
            and parts[1] == source_config["split"]
            and parts[2] in source_config["subtypes"]
        ):
            selected.append((parts[0], parts[2], path))
    maximum = source_config.get("maximum_shards_per_bucket")
    if maximum is not None:
        buckets: dict[tuple[str, str], list[str]] = {}
        for language, subtype, path in selected:
            buckets.setdefault((language, subtype), []).append(path)
        limited = []
        for (language, subtype), paths in sorted(buckets.items()):
            count = min(int(maximum), len(paths))
            if count == 1:
                indices = [0]
            else:
                indices = sorted({round(index * (len(paths) - 1) / (count - 1)) for index in range(count)})
            limited.extend((language, subtype, paths[index]) for index in indices)
        selected = sorted(limited)
    return selected


def prepare_anv_shard(
    language: str,
    subtype: str,
    shard_path: str,
    source_config: dict[str, Any],
    audio_root: Path,
    manifest_root: Path,
    sample_rate: int,
    token: str | None,
) -> dict[str, Any]:
    repo_id = source_config["repo_id"]
    shard_digest = hashlib.sha1(shard_path.encode()).hexdigest()[:12]
    shard_name = f"{language}-{subtype}-{shard_digest}"
    output_manifest = manifest_root / "anv" / language / subtype / f"{shard_name}.jsonl"
    if manifest_audio_complete(output_manifest):
        return {"shard": shard_name, "status": "existing", "manifest": str(output_manifest)}
    parquet_path = hf_hub_download(repo_id, shard_path, repo_type="dataset", token=token)
    frame = pd.read_parquet(parquet_path)
    text_column = next(
        (name for name in ("transcription", "actualSentence", "transcript", "text") if name in frame),
        None,
    )
    if text_column is None or "audio" not in frame:
        raise RuntimeError(f"{shard_path}: required audio/text columns not found")
    records = []
    rejected = 0
    for row_index, series in frame.iterrows():
        row = series.to_dict()
        text = str(clean_optional(row.get(text_column)) or "").strip()
        if not normalize_text(text):
            rejected += 1
            continue
        source_key = (
            clean_optional(row.get("id"))
            or clean_optional(row.get("path"))
            or clean_optional(row.get("audio_filepath"))
            or row_index
        )
        record_id = stable_id(language, shard_path, source_key)
        path = audio_root / language / subtype / shard_name / f"{record_id}.flac"
        try:
            if not path.is_file():
                write_flac(path, decode_audio_field(row["audio"], sample_rate), sample_rate)
            records.append(
                source_record(
                    record_id=record_id,
                    source_id=record_id,
                    audio_path=path,
                    language=language,
                    text=text,
                    subtype=subtype,
                    dataset="afrivoices_ke",
                    source_repo=repo_id,
                    source_shard=shard_path,
                    row=row,
                )
            )
        except Exception:
            rejected += 1
    del frame
    gc.collect()
    if not records:
        raise RuntimeError(f"{shard_path}: no usable records")
    atomic_write_jsonl(output_manifest, records)
    return {"shard": shard_name, "status": "prepared", "rows": len(records), "rejected": rejected}


def combine_manifests(manifest_root: Path, output_path: Path) -> dict[str, Any]:
    manifests = sorted(manifest_root.glob("**/*.jsonl"))
    records = []
    seen: set[str] = set()
    counts: dict[str, int] = {language: 0 for language in LANGUAGES}
    datasets: dict[str, int] = {}
    for path in manifests:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record["id"] in seen:
                    raise RuntimeError(f"duplicate source ID {record['id']} in {path}")
                seen.add(record["id"])
                counts[record["language"]] += 1
                datasets[record["dataset"]] = datasets.get(record["dataset"], 0) + 1
                records.append(record)
    missing = [language for language, count in counts.items() if count == 0]
    if missing:
        raise RuntimeError(f"source preparation has no rows for: {missing}")
    atomic_write_jsonl(output_path, records)
    return {"manifests": len(manifests), "rows": len(records), "languages": counts, "datasets": datasets}


def prepare_sources(config: dict[str, Any]) -> dict[str, Any]:
    validate_source_policy(config)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to decode Digital Umuganda archive audio")
    token = os.environ.get("HF_TOKEN")
    sample_rate = int(config["model"]["sample_rate"])
    scratch = Path(config["paths"]["scratch_dir"])
    audio_root = scratch / "source_audio"
    manifest_root = Path(config["paths"]["persistent_dir"]) / "source_shards"
    results = []
    swahili = config["sources"]["swahili"]
    for shard in range(int(swahili["shard_count"])):
        result = prepare_swahili_shard(shard, swahili, audio_root, manifest_root, sample_rate, token)
        results.append(result)
        print(json.dumps(result), flush=True)
    anv = config["sources"]["anv"]
    shards = list_anv_shards(anv, token)
    if not shards:
        raise RuntimeError("no ANV parquet shards found")
    for index, (language, subtype, shard_path) in enumerate(shards, 1):
        result = prepare_anv_shard(
            language, subtype, shard_path, anv, audio_root, manifest_root, sample_rate, token
        )
        results.append(result)
        print(f"ANV {index}/{len(shards)} {json.dumps(result)}", flush=True)
    summary = combine_manifests(manifest_root, Path(config["paths"]["source_manifest"]))
    summary["shards_processed"] = len(results)
    return summary
