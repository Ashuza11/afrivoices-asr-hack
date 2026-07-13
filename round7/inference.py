from __future__ import annotations

import glob
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import Wav2Vec2Processor

from round7.data import LANGUAGES, normalize_text
from round7.model import LanguageConditionedXLSRForCTC
from round7.sources import decode_audio_field
from round7.training import LANGUAGE_TO_ID


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".csv", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def validate_submission(frame: pd.DataFrame, expected: pd.DataFrame) -> None:
    required = ["id", "language", "transcription"]
    if list(frame.columns) != required:
        raise RuntimeError(f"submission columns must be exactly {required}")
    if frame["id"].astype(str).duplicated().any():
        raise RuntimeError("submission contains duplicate IDs")
    if expected["id"].astype(str).duplicated().any():
        raise RuntimeError("test data contains duplicate IDs")
    expected_pairs = set(zip(expected["id"].astype(str), expected["language"]))
    actual_pairs = set(zip(frame["id"].astype(str), frame["language"]))
    if actual_pairs != expected_pairs:
        raise RuntimeError(
            f"submission ID/language mismatch: missing={len(expected_pairs-actual_pairs)} "
            f"unexpected={len(actual_pairs-expected_pairs)}"
        )
    invalid_languages = sorted(set(frame["language"]) - set(LANGUAGES))
    if invalid_languages:
        raise RuntimeError(f"submission has unsupported languages: {invalid_languages}")
    if frame["transcription"].isna().any() or frame["transcription"].str.strip().eq("").any():
        raise RuntimeError("submission contains empty transcriptions")


def _cache_test_parquets(config: dict[str, Any]) -> list[Path]:
    cache = Path(config["paths"]["persistent_dir"]) / "test_parquets"
    existing = sorted(cache.glob("**/*.parquet"))
    if existing:
        return existing
    key = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")
    if not key:
        raise RuntimeError("KAGGLE_API_TOKEN is required to download test data")
    os.environ["KAGGLE_KEY"] = key
    import kagglehub

    downloaded_root = Path(kagglehub.dataset_download(config["inference"]["kaggle_dataset"]))
    downloaded = sorted(downloaded_root.glob("**/*.parquet"))
    if not downloaded:
        raise RuntimeError("Kaggle test dataset contains no parquet files")
    for source in downloaded:
        destination = cache / source.relative_to(downloaded_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return sorted(cache.glob("**/*.parquet"))


def _chunks(audio: np.ndarray, sample_rate: int, maximum_seconds: float) -> list[np.ndarray]:
    width = max(1, round(sample_rate * maximum_seconds))
    return [audio[start : start + width] for start in range(0, len(audio), width)]


@torch.no_grad()
def transcribe_rows(
    rows: list[Any],
    model: LanguageConditionedXLSRForCTC,
    processor: Wav2Vec2Processor,
    device: torch.device,
    sample_rate: int,
    batch_size: int,
    maximum_chunk_seconds: float,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    chunks: list[np.ndarray] = []
    owners: list[tuple[str, str]] = []
    failures = []
    for row in rows:
        record_id, language = str(row.id), str(row.language)
        try:
            audio = decode_audio_field(row.audio, sample_rate)
            for chunk in _chunks(audio, sample_rate, maximum_chunk_seconds):
                chunks.append(chunk)
                owners.append((record_id, language))
        except Exception as exc:
            failures.append({"id": record_id, "language": language, "reason": str(exc)})

    decoded: dict[tuple[str, str], list[str]] = {}
    for start in range(0, len(chunks), batch_size):
        audio_batch = chunks[start : start + batch_size]
        owner_batch = owners[start : start + batch_size]
        inputs = processor(
            audio_batch, sampling_rate=sample_rate, padding=True, return_tensors="pt"
        )
        language_ids = torch.tensor(
            [LANGUAGE_TO_ID[language] for _, language in owner_batch],
            dtype=torch.long,
            device=device,
        )
        output = model(
            input_values=inputs.input_values.to(device),
            attention_mask=inputs.attention_mask.to(device),
            language_ids=language_ids,
        )
        texts = processor.batch_decode(output.logits.argmax(dim=-1).cpu().numpy())
        for owner, text in zip(owner_batch, texts):
            decoded.setdefault(owner, []).append(normalize_text(text))
    results = [
        {"id": record_id, "language": language, "transcription": normalize_text(" ".join(parts))}
        for (record_id, language), parts in decoded.items()
    ]
    failures.extend(
        {"id": row["id"], "language": row["language"], "reason": "empty_decoding"}
        for row in results
        if not row["transcription"]
    )
    failed_ids = {row["id"] for row in failures}
    return [row for row in results if row["id"] not in failed_ids], failures


def run_test_inference(
    config: dict[str, Any], final_directory: Path, output_directory: Path
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("test inference requires CUDA")
    final_result = json.loads((final_directory / "result.json").read_text(encoding="utf-8"))
    checkpoint = Path(final_result["inference_selection"]["checkpoint"])
    processor = Wav2Vec2Processor.from_pretrained(checkpoint)
    model = LanguageConditionedXLSRForCTC.from_pretrained(checkpoint).to("cuda").eval()
    parquet_files = _cache_test_parquets(config)
    checkpoint_csv = output_directory / "inference_checkpoint.csv"
    if checkpoint_csv.is_file():
        existing = pd.read_csv(checkpoint_csv, dtype={"id": str}).to_dict("records")
    else:
        existing = []
    done_ids = {str(row["id"]) for row in existing}
    expected_frames = []
    failures = []
    started = time.time()
    for index, parquet in enumerate(parquet_files, 1):
        frame = pd.read_parquet(parquet)
        if not {"id", "language", "audio"}.issubset(frame.columns):
            raise RuntimeError(f"test parquet is missing required columns: {parquet}")
        frame["id"] = frame["id"].astype(str)
        expected_frames.append(frame[["id", "language"]])
        pending = frame[~frame["id"].isin(done_ids)]
        rows, row_failures = transcribe_rows(
            list(pending.itertuples(index=False)),
            model,
            processor,
            torch.device("cuda"),
            int(config["model"]["sample_rate"]),
            int(config["inference"]["batch_size"]),
            float(config["inference"]["maximum_chunk_seconds"]),
        )
        existing.extend(rows)
        failures.extend(row_failures)
        done_ids.update(row["id"] for row in rows)
        if index % int(config["inference"]["checkpoint_every_files"]) == 0:
            atomic_write_csv(checkpoint_csv, pd.DataFrame(existing))
        print(f"infer-test {index}/{len(parquet_files)} completed={len(existing)} failures={len(failures)}", flush=True)
    if failures:
        atomic_write_csv(output_directory / "failures.csv", pd.DataFrame(failures))
        raise RuntimeError(f"inference has {len(failures)} failed rows; see failures.csv")
    submission = pd.DataFrame(existing)[["id", "language", "transcription"]]
    expected = pd.concat(expected_frames, ignore_index=True)
    validate_submission(submission, expected)
    submission = submission.sort_values("id").reset_index(drop=True)
    final_path = output_directory / "submission_round7_xlsr_greedy.csv"
    atomic_write_csv(final_path, submission)
    return {
        "rows": len(submission),
        "checkpoint": str(checkpoint),
        "submission": str(final_path.resolve()),
        "elapsed_minutes": (time.time() - started) / 60,
        "languages": submission["language"].value_counts().sort_index().to_dict(),
    }
