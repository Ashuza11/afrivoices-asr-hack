from __future__ import annotations

import json
import math
import os
import tempfile
from copy import deepcopy
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import Wav2Vec2Processor

from round7.data import LANGUAGES, atomic_write_jsonl
from round7.model import LanguageConditionedXLSRForCTC
from round7.sources import write_flac
from round7.training import LANGUAGE_TO_ID, read_jsonl


class AlignmentError(RuntimeError):
    pass


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


def cached_alignment_result(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if all(Path(row["audio_path"]).is_file() for row in result.get("accepted", [])):
            return result
    except (OSError, KeyError, json.JSONDecodeError):
        pass
    return None


def forced_align_ctc(
    log_probabilities: torch.Tensor,
    targets: list[int],
    blank_id: int,
) -> list[dict[str, float]]:
    """Align target tokens with a standard blank-expanded CTC trellis."""
    if log_probabilities.ndim != 2:
        raise ValueError("log_probabilities must have shape [frames, vocabulary]")
    if not targets:
        raise AlignmentError("target sequence is empty")
    frames, vocabulary = log_probabilities.shape
    if min(targets) < 0 or max(targets) >= vocabulary:
        raise AlignmentError("target token is outside the emission vocabulary")

    symbols = [blank_id]
    for token in targets:
        symbols.extend((token, blank_id))
    states = len(symbols)
    if frames < len(targets):
        raise AlignmentError(f"only {frames} frames for {len(targets)} target tokens")

    emissions = log_probabilities.detach().to(dtype=torch.float32, device="cpu")
    symbol_tensor = torch.tensor(symbols, dtype=torch.long)
    negative_infinity = torch.tensor(float("-inf"))
    previous = torch.full((states,), negative_infinity)
    previous[0] = 0.0
    backpointers = torch.zeros((frames, states), dtype=torch.int8)

    skip_allowed = torch.zeros(states, dtype=torch.bool)
    for state in range(3, states, 2):
        skip_allowed[state] = symbols[state] != symbols[state - 2]

    for frame in range(frames):
        stay = previous
        advance = torch.cat((negative_infinity.reshape(1), previous[:-1]))
        skip = torch.cat((negative_infinity.repeat(2), previous[:-2]))
        skip = torch.where(skip_allowed, skip, negative_infinity)
        candidates = torch.stack((stay, advance, skip), dim=0)
        best_scores, choices = candidates.max(dim=0)
        previous = best_scores + emissions[frame, symbol_tensor]
        backpointers[frame] = choices.to(torch.int8)

    final_states = [states - 1, states - 2]
    state = max(final_states, key=lambda index: float(previous[index]))
    if not torch.isfinite(previous[state]):
        raise AlignmentError("no finite CTC alignment path")

    frames_by_target: list[list[int]] = [[] for _ in targets]
    scores_by_target: list[list[float]] = [[] for _ in targets]
    for frame in range(frames - 1, -1, -1):
        if state % 2 == 1:
            target_index = (state - 1) // 2
            frames_by_target[target_index].append(frame)
            scores_by_target[target_index].append(float(emissions[frame, targets[target_index]]))
        transition = int(backpointers[frame, state])
        state -= transition
        if state < 0:
            raise AlignmentError("CTC backtracking left the trellis")

    if state != 0 or any(not positions for positions in frames_by_target):
        raise AlignmentError("CTC path did not cover every target token")
    alignment = []
    for token, positions, scores in zip(targets, frames_by_target, scores_by_target):
        positions.reverse()
        scores.reverse()
        alignment.append(
            {
                "token_id": token,
                "start_frame": positions[0],
                "end_frame": positions[-1] + 1,
                "mean_posterior": float(np.exp(np.mean(scores))),
            }
        )
    return alignment


def load_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    audio, original_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if original_rate != sample_rate:
        divisor = math.gcd(original_rate, sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, original_rate // divisor)
    if not len(audio) or not np.isfinite(audio).all():
        raise AlignmentError("audio is empty or non-finite")
    return np.asarray(audio, dtype=np.float32)


@torch.no_grad()
def extract_chunked_emissions(
    model: LanguageConditionedXLSRForCTC,
    processor: Wav2Vec2Processor,
    audio: np.ndarray,
    language: str,
    sample_rate: int,
    chunk_seconds: float,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    chunk_samples = int(round(chunk_seconds * sample_rate))
    if chunk_samples <= 0:
        raise ValueError("chunk_seconds must be positive")
    emissions = []
    frame_times = []
    model.eval()
    for start in range(0, len(audio), chunk_samples):
        end = min(len(audio), start + chunk_samples)
        chunk = audio[start:end]
        inputs = processor(
            chunk,
            sampling_rate=sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        output = model(
            input_values=inputs.input_values.to(device),
            attention_mask=inputs.attention_mask.to(device),
            language_ids=torch.tensor([LANGUAGE_TO_ID[language]], device=device),
        )
        chunk_log_probs = output.logits[0].float().log_softmax(dim=-1).cpu()
        count = chunk_log_probs.shape[0]
        start_seconds = start / sample_rate
        duration_seconds = (end - start) / sample_rate
        times = start_seconds + (np.arange(count) + 0.5) * duration_seconds / count
        emissions.append(chunk_log_probs)
        frame_times.append(times)
    return torch.cat(emissions, dim=0), np.concatenate(frame_times)


def token_alignment_to_words(
    text: str,
    target_ids: list[int],
    token_alignment: list[dict[str, float]],
    frame_times: np.ndarray,
    delimiter_id: int,
) -> list[dict[str, Any]]:
    words = text.split()
    token_groups: list[list[int]] = [[]]
    for index, token_id in enumerate(target_ids):
        if token_id == delimiter_id:
            if token_groups[-1]:
                token_groups.append([])
        else:
            token_groups[-1].append(index)
    token_groups = [group for group in token_groups if group]
    if len(token_groups) != len(words):
        raise AlignmentError(
            f"tokenized word count {len(token_groups)} does not match text word count {len(words)}"
        )
    result = []
    for word, group in zip(words, token_groups):
        first = token_alignment[group[0]]
        last = token_alignment[group[-1]]
        start_frame = int(first["start_frame"])
        end_frame = min(int(last["end_frame"]) - 1, len(frame_times) - 1)
        result.append(
            {
                "word": word,
                "start": float(frame_times[start_frame]),
                "end": float(frame_times[end_frame]),
                "mean_token_posterior": float(
                    np.mean([token_alignment[index]["mean_posterior"] for index in group])
                ),
            }
        )
    return result


def group_words_into_segments(
    words: list[dict[str, Any]],
    audio_duration: float,
    minimum_seconds: float,
    maximum_seconds: float,
    padding_seconds: float,
) -> list[dict[str, Any]]:
    segments = []
    current: list[dict[str, Any]] = []
    for word in words:
        candidate_start = current[0]["start"] if current else word["start"]
        if current and word["end"] - candidate_start > maximum_seconds:
            segments.append(current)
            current = []
        current.append(word)
    if current:
        segments.append(current)

    output = []
    for group in segments:
        start = max(0.0, group[0]["start"] - padding_seconds)
        end = min(audio_duration, group[-1]["end"] + padding_seconds)
        if end - start < minimum_seconds or end - start > maximum_seconds:
            continue
        output.append(
            {
                "start": start,
                "end": end,
                "text": " ".join(word["word"] for word in group),
                "mean_token_posterior": float(
                    np.mean([word["mean_token_posterior"] for word in group])
                ),
            }
        )
    return output


def speech_occupancy(audio: np.ndarray, sample_rate: int) -> float:
    frame = max(1, round(sample_rate * 0.02))
    usable = len(audio) // frame * frame
    if usable == 0:
        return 0.0
    frames = audio[:usable].reshape(-1, frame)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    peak = float(np.max(rms))
    if peak <= 1e-8:
        return 0.0
    threshold = min(
        max(float(np.percentile(rms, 10)) * 3.0, peak * 0.02),
        peak * 0.5,
    )
    return float(np.mean(rms >= threshold))


def select_alignment_pilot(records: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    maximum_source = float(config["alignment"]["maximum_source_seconds"])
    minimum_source = float(config["data"]["seed_max_duration"])
    candidates = [
        row
        for row in records
        if row["split"] == "train"
        and row["subtype"] == "unscripted"
        and minimum_source < float(row["duration"]) <= maximum_source
    ]
    candidates.sort(key=lambda row: (row["language"], row["source_shard"], row["id"]))
    selected = []
    hours: defaultdict[str, float] = defaultdict(float)
    targets = config["alignment"]["pilot_hours"]
    for row in candidates:
        language = row["language"]
        if hours[language] >= float(targets[language]):
            continue
        selected.append(row)
        hours[language] += float(row["duration"]) / 3600
    required = tuple(config["gate"]["required_alignment_languages"])
    missing = [language for language in required if hours[language] == 0]
    if missing:
        raise RuntimeError(f"alignment pilot has no long unscripted audio for: {missing}")
    return selected


def run_alignment_pilot(
    config: dict[str, Any], split_directory: Path, seed_directory: Path, output_directory: Path
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("alignment pilot requires CUDA")
    best_pointer = seed_directory / "checkpoints" / "best.json"
    if not best_pointer.is_file():
        raise RuntimeError("best seed checkpoint was not found")
    checkpoint = Path(json.loads(best_pointer.read_text(encoding="utf-8"))["checkpoint"])
    processor = Wav2Vec2Processor.from_pretrained(checkpoint)
    model = LanguageConditionedXLSRForCTC.from_pretrained(checkpoint).to("cuda").eval()
    records = select_alignment_pilot(read_jsonl(split_directory / "manifest.jsonl"), config)
    sample_rate = int(config["model"]["sample_rate"])
    delimiter_id = int(processor.tokenizer.word_delimiter_token_id)
    blank_id = int(processor.tokenizer.pad_token_id)
    accepted = []
    rejected = []
    by_language: defaultdict[str, dict[str, float]] = defaultdict(
        lambda: {"sources": 0, "accepted_segments": 0, "accepted_hours": 0.0, "rejected": 0}
    )
    segment_root = Path(config["paths"]["scratch_dir"]) / "aligned_pilot"
    record_result_root = output_directory / "records"
    for index, record in enumerate(records, 1):
        language = record["language"]
        by_language[language]["sources"] += 1
        result_path = record_result_root / language / f"{record['id']}.json"
        cached = cached_alignment_result(result_path)
        if cached is not None:
            source_accepted = cached.get("accepted", [])
            source_rejected = cached.get("rejected", [])
            accepted.extend(source_accepted)
            rejected.extend(source_rejected)
            by_language[language]["accepted_segments"] += len(source_accepted)
            by_language[language]["accepted_hours"] += sum(
                float(row["duration"]) / 3600 for row in source_accepted
            )
            by_language[language]["rejected"] += len(source_rejected)
            continue
        source_accepted = []
        source_rejected = []
        try:
            audio = load_audio(record["audio_path"], sample_rate)
            target_ids = processor.tokenizer(
                record["transcription"], add_special_tokens=False
            ).input_ids
            if processor.tokenizer.unk_token_id in target_ids:
                raise AlignmentError("transcript contains an unknown vocabulary token")
            log_probs, frame_times = extract_chunked_emissions(
                model,
                processor,
                audio,
                language,
                sample_rate,
                float(config["alignment"]["emission_chunk_seconds"]),
                torch.device("cuda"),
            )
            token_alignment = forced_align_ctc(log_probs, target_ids, blank_id)
            words = token_alignment_to_words(
                record["transcription"], target_ids, token_alignment, frame_times, delimiter_id
            )
            segments = group_words_into_segments(
                words,
                len(audio) / sample_rate,
                float(config["alignment"]["minimum_segment_seconds"]),
                float(config["alignment"]["maximum_segment_seconds"]),
                float(config["alignment"]["boundary_padding_seconds"]),
            )
            for segment_index, segment in enumerate(segments):
                start_sample = round(segment["start"] * sample_rate)
                end_sample = round(segment["end"] * sample_rate)
                segment_audio = audio[start_sample:end_sample]
                occupancy = speech_occupancy(segment_audio, sample_rate)
                confidence = float(segment["mean_token_posterior"])
                reasons = []
                if occupancy < float(config["alignment"]["minimum_speech_occupancy"]):
                    reasons.append("low_speech_occupancy")
                if confidence < float(config["alignment"]["minimum_mean_token_posterior"]):
                    reasons.append("low_token_posterior")
                segment_id = f"{record['id']}-aligned-{segment_index:03d}"
                path = segment_root / language / f"{segment_id}.flac"
                result = {
                    "id": segment_id,
                    "source_id": record["source_id"],
                    "audio_path": str(path.resolve()),
                    "language": language,
                    "dialect": record["dialect"],
                    "subtype": "unscripted_aligned",
                    "speaker_id": record.get("speaker_id"),
                    "duration": len(segment_audio) / sample_rate,
                    "transcription": segment["text"],
                    "split": "train",
                    "dataset": record["dataset"],
                    "source_repo": record["source_repo"],
                    "source_shard": record["source_shard"],
                    "alignment_start": segment["start"],
                    "alignment_end": segment["end"],
                    "mean_token_posterior": confidence,
                    "speech_occupancy": occupancy,
                }
                if reasons:
                    result["reasons"] = reasons
                    rejected.append(result)
                    source_rejected.append(result)
                    by_language[language]["rejected"] += 1
                else:
                    if not path.is_file():
                        write_flac(path, segment_audio, sample_rate)
                    accepted.append(result)
                    source_accepted.append(result)
                    by_language[language]["accepted_segments"] += 1
                    by_language[language]["accepted_hours"] += result["duration"] / 3600
        except Exception as exc:
            failure = {"id": record["id"], "language": language, "reasons": [str(exc)]}
            rejected.append(failure)
            source_rejected.append(failure)
            by_language[language]["rejected"] += 1
        atomic_write_json(
            result_path,
            {"source_id": record["id"], "accepted": source_accepted, "rejected": source_rejected},
        )
        if index % 10 == 0:
            print(f"align-pilot {index}/{len(records)} accepted={len(accepted)} rejected={len(rejected)}", flush=True)
    if not accepted:
        raise RuntimeError("alignment pilot produced no accepted segments")
    output_directory.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(output_directory / "accepted.jsonl", accepted)
    atomic_write_jsonl(output_directory / "rejected.jsonl", rejected)
    return {
        "source_records": len(records),
        "accepted_segments": len(accepted),
        "rejected_records_or_segments": len(rejected),
        "languages": {language: dict(values) for language, values in sorted(by_language.items())},
        "provisional_thresholds": True,
    }


def run_full_alignment(
    config: dict[str, Any], split_directory: Path, seed_directory: Path, output_directory: Path
) -> dict[str, Any]:
    """Align every eligible long recording in the prepared, leakage-safe corpus."""
    full_config = deepcopy(config)
    full_config["alignment"]["maximum_source_seconds"] = float(
        config["full_alignment"]["maximum_source_seconds"]
    )
    full_config["alignment"]["pilot_hours"] = {language: float("inf") for language in LANGUAGES}
    result = run_alignment_pilot(full_config, split_directory, seed_directory, output_directory)
    result["scope"] = "all_eligible_records_in_prepared_manifest"
    result["provisional_thresholds"] = False
    return result
