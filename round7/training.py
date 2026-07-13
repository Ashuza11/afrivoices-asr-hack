from __future__ import annotations

import json
import math
import os
import random
import shutil
import tempfile
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
import torch
from jiwer import wer
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from scipy.signal import resample_poly
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

from round7.data import LANGUAGES
from round7.model import LanguageConditionedXLSRForCTC


LANGUAGE_TO_ID = {language: index for index, language in enumerate(LANGUAGES)}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_processor(vocabulary_path: Path, sample_rate: int) -> Wav2Vec2Processor:
    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocabulary_path),
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
    )
    extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=sample_rate,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    return Wav2Vec2Processor(feature_extractor=extractor, tokenizer=tokenizer)


class AudioManifestDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], sample_rate: int):
        self.records = records
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        audio, sample_rate = sf.read(record["audio_path"], dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sample_rate != self.sample_rate:
            divisor = math.gcd(sample_rate, self.sample_rate)
            audio = resample_poly(
                audio,
                up=self.sample_rate // divisor,
                down=sample_rate // divisor,
            )
        return {
            "audio": np.asarray(audio, dtype=np.float32),
            "text": record["transcription"],
            "language": record["language"],
            "id": record["id"],
        }


class CTCBatchCollator:
    def __init__(self, processor: Wav2Vec2Processor, sample_rate: int):
        self.processor = processor
        self.sample_rate = sample_rate

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        audio_batch = self.processor(
            [example["audio"] for example in examples],
            sampling_rate=self.sample_rate,
            padding=True,
            return_tensors="pt",
        )
        label_batch = self.processor.tokenizer(
            [example["text"] for example in examples],
            padding=True,
            return_tensors="pt",
        )
        labels = label_batch.input_ids.masked_fill(label_batch.attention_mask.ne(1), -100)
        return {
            "input_values": audio_batch.input_values,
            "attention_mask": audio_batch.attention_mask,
            "labels": labels,
            "language_ids": torch.tensor(
                [LANGUAGE_TO_ID[example["language"]] for example in examples], dtype=torch.long
            ),
            "languages": [example["language"] for example in examples],
            "ids": [example["id"] for example in examples],
        }


def select_seed_records(
    records: Iterable[dict[str, Any]],
    split: str,
    maximum_duration: float,
    maximum_per_language: int | None = None,
) -> list[dict[str, Any]]:
    selected = [
        record
        for record in records
        if record["split"] == split
        and record.get("duration") is not None
        and float(record["duration"]) <= maximum_duration
    ]
    selected.sort(key=lambda row: (row["language"], row["id"]))
    if maximum_per_language is None:
        return selected
    counts: defaultdict[str, int] = defaultdict(int)
    limited = []
    for record in selected:
        if counts[record["language"]] < maximum_per_language:
            limited.append(record)
            counts[record["language"]] += 1
    return limited


def decode_batch(processor: Wav2Vec2Processor, token_ids: torch.Tensor) -> list[str]:
    return [text.strip() for text in processor.batch_decode(token_ids.cpu().numpy())]


@torch.no_grad()
def evaluate_model(
    model: LanguageConditionedXLSRForCTC,
    loader: DataLoader,
    processor: Wav2Vec2Processor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    references: defaultdict[str, list[str]] = defaultdict(list)
    predictions: defaultdict[str, list[str]] = defaultdict(list)
    for batch in loader:
        output = model(
            input_values=batch["input_values"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            language_ids=batch["language_ids"].to(device),
        )
        texts = decode_batch(processor, output.logits.argmax(dim=-1))
        for language, prediction, label_ids in zip(batch["languages"], texts, batch["labels"]):
            reference_ids = label_ids[label_ids >= 0].unsqueeze(0)
            reference = decode_batch(processor, reference_ids)[0]
            references[language].append(reference)
            predictions[language].append(prediction)
    metrics = {
        f"wer_{language}": wer(references[language], predictions[language])
        for language in LANGUAGES
        if references[language]
    }
    if len(metrics) != len(LANGUAGES):
        missing = sorted(set(LANGUAGES) - {key.removeprefix("wer_") for key in metrics})
        raise RuntimeError(f"seed evaluation has no examples for: {missing}")
    metrics["macro_wer"] = sum(metrics.values()) / len(metrics)
    return metrics


def seed_smoke_test(config: dict[str, Any], split_directory: Path, output_directory: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("seed smoke test requires CUDA")
    device = torch.device("cuda")
    sample_rate = int(config["model"]["sample_rate"])
    processor = make_processor(split_directory / "vocabulary.json", sample_rate)
    records = read_jsonl(split_directory / "manifest.jsonl")
    train_records = select_seed_records(
        records, "train", float(config["data"]["seed_max_duration"])
    )
    eval_records = select_seed_records(
        records,
        "validation",
        float(config["data"]["seed_max_duration"]),
        int(config["training"]["smoke_eval_samples_per_language"]),
    )
    if not train_records or not eval_records:
        raise RuntimeError("seed train and validation records are required")

    collator = CTCBatchCollator(processor, sample_rate)
    batch_size = int(config["training"]["per_device_batch_size"])
    train_loader = DataLoader(
        AudioManifestDataset(train_records, sample_rate),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        AudioManifestDataset(eval_records, sample_rate),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )
    model = LanguageConditionedXLSRForCTC.from_xlsr_pretrained(
        config["model"]["id"], vocab_size=len(processor.tokenizer)
    ).to(device)
    if config["training"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
    model.freeze_feature_encoder()

    head_parameters = list(model.language_embedding.parameters()) + list(model.lm_head.parameters())
    head_ids = {id(parameter) for parameter in head_parameters}
    encoder_parameters = [parameter for parameter in model.parameters() if id(parameter) not in head_ids]
    optimizer = AdamW(
        [
            {"params": encoder_parameters, "lr": float(config["training"]["encoder_learning_rate"])},
            {"params": head_parameters, "lr": float(config["training"]["head_learning_rate"])},
        ]
    )
    steps = int(config["training"]["smoke_steps"])
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    use_bf16 = bool(config["training"]["bf16"] and torch.cuda.is_bf16_supported())
    iterator = iter(train_loader)
    losses = []
    optimizer.zero_grad(set_to_none=True)
    model.train()
    for step in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        context = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
        with context:
            output = model(
                input_values=batch["input_values"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                language_ids=batch["language_ids"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = output.loss / accumulation
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step + 1}")
        loss.backward()
        losses.append(float(loss.detach().cpu()) * accumulation)
        if (step + 1) % accumulation == 0 or step + 1 == steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["maximum_gradient_norm"]))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if (step + 1) % 10 == 0:
            print(f"seed-smoke step={step + 1}/{steps} loss={sum(losses[-10:]) / len(losses[-10:]):.4f}", flush=True)

    metrics = evaluate_model(model, eval_loader, processor, device)
    checkpoint = output_directory / "checkpoint"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint, safe_serialization=True)
    processor.save_pretrained(checkpoint)

    reloaded = LanguageConditionedXLSRForCTC.from_pretrained(checkpoint).to(device).eval()
    verification_batch = next(iter(eval_loader))
    with torch.no_grad():
        verification = reloaded(
            input_values=verification_batch["input_values"].to(device),
            attention_mask=verification_batch["attention_mask"].to(device),
            language_ids=verification_batch["language_ids"].to(device),
        )
    if not torch.isfinite(verification.logits).all():
        raise RuntimeError("reloaded checkpoint produced non-finite logits")
    return {
        "steps": steps,
        "train_records": len(train_records),
        "eval_records": len(eval_records),
        "bf16": use_bf16,
        "last_10_loss": sum(losses[-10:]) / len(losses[-10:]),
        **metrics,
    }


def language_balanced_sampler(
    records: list[dict[str, Any]], seed: int
) -> WeightedRandomSampler:
    counts: defaultdict[str, int] = defaultdict(int)
    hours: defaultdict[str, float] = defaultdict(float)
    for record in records:
        counts[record["language"]] += 1
        hours[record["language"]] += float(record["duration"]) / 3600
    missing = [language for language in LANGUAGES if counts[language] == 0]
    if missing:
        raise RuntimeError(f"seed training has no records for: {missing}")
    language_mass = {language: math.sqrt(hours[language]) for language in LANGUAGES}
    weights = [language_mass[row["language"]] / counts[row["language"]] for row in records]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(records), replacement=True, generator=generator)


def _atomic_json(path: Path, value: Any) -> None:
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


def _save_training_checkpoint(
    checkpoint_root: Path,
    step: int,
    model: LanguageConditionedXLSRForCTC,
    processor: Wav2Vec2Processor,
    optimizer: AdamW,
    scheduler: LambdaLR,
    state: dict[str, Any],
) -> Path:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    final = checkpoint_root / f"checkpoint-{step:07d}"
    temporary = checkpoint_root / f".checkpoint-{step:07d}.tmp"
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary, safe_serialization=True)
    processor.save_pretrained(temporary)
    torch.save(
        {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        temporary / "training.pt",
    )
    _atomic_json(temporary / "state.json", state)
    if final.exists():
        shutil.rmtree(final)
    os.replace(temporary, final)
    final = final.resolve()
    _atomic_json(checkpoint_root / "latest.json", {"checkpoint": str(final), "step": step})
    return final


def _latest_checkpoint(checkpoint_root: Path) -> Path | None:
    pointer = checkpoint_root / "latest.json"
    if not pointer.is_file():
        return None
    try:
        path = Path(json.loads(pointer.read_text(encoding="utf-8"))["checkpoint"])
        return path if (path / "state.json").is_file() else None
    except (OSError, KeyError, json.JSONDecodeError):
        return None


def _prune_checkpoints(checkpoint_root: Path, keep: set[Path]) -> None:
    resolved_keep = {path.resolve() for path in keep}
    for path in checkpoint_root.glob("checkpoint-*"):
        if path.is_dir() and path.resolve() not in resolved_keep:
            shutil.rmtree(path)


def _learning_rate_lambda(step: int, warmup_steps: int, maximum_steps: int) -> float:
    if step < warmup_steps:
        return max(step, 1) / max(warmup_steps, 1)
    return max(0.0, (maximum_steps - step) / max(maximum_steps - warmup_steps, 1))


def train_seed_model(
    config: dict[str, Any],
    split_directory: Path,
    smoke_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("seed training requires CUDA")
    seed = int(config["project"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda")
    sample_rate = int(config["model"]["sample_rate"])
    processor = Wav2Vec2Processor.from_pretrained(smoke_directory / "checkpoint")
    records = read_jsonl(split_directory / "manifest.jsonl")
    maximum_duration = float(config["data"]["seed_max_duration"])
    train_records = select_seed_records(records, "train", maximum_duration)
    eval_records = select_seed_records(records, "validation", maximum_duration)
    if not train_records or not eval_records:
        raise RuntimeError("seed train and validation records are required")

    collator = CTCBatchCollator(processor, sample_rate)
    workers = int(config["training"]["dataloader_workers"])
    batch_size = int(config["training"]["per_device_batch_size"])
    train_loader = DataLoader(
        AudioManifestDataset(train_records, sample_rate),
        batch_size=batch_size,
        sampler=language_balanced_sampler(train_records, seed),
        collate_fn=collator,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    eval_loader = DataLoader(
        AudioManifestDataset(eval_records, sample_rate),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )

    maximum_steps = int(config["training"]["seed_max_steps"])
    eval_steps = int(config["training"]["seed_eval_steps"])
    warmup_steps = max(1, round(maximum_steps * float(config["training"]["warmup_ratio"])))
    unfreeze_step = max(
        1, round(maximum_steps * float(config["training"]["feature_encoder_freeze_ratio"]))
    )
    checkpoint_root = output_directory / "checkpoints"
    resume = _latest_checkpoint(checkpoint_root)
    initial = resume or smoke_directory / "checkpoint"
    model = LanguageConditionedXLSRForCTC.from_pretrained(initial).to(device)
    if config["training"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    head_parameters = list(model.language_embedding.parameters()) + list(model.lm_head.parameters())
    head_ids = {id(parameter) for parameter in head_parameters}
    encoder_parameters = [parameter for parameter in model.parameters() if id(parameter) not in head_ids]
    optimizer = AdamW(
        [
            {"params": encoder_parameters, "lr": float(config["training"]["encoder_learning_rate"])},
            {"params": head_parameters, "lr": float(config["training"]["head_learning_rate"])},
        ]
    )
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda current: _learning_rate_lambda(current, warmup_steps, maximum_steps),
    )

    step = 0
    best_wer = math.inf
    best_checkpoint: str | None = None
    evaluations_without_improvement = 0
    if resume:
        state = json.loads((resume / "state.json").read_text(encoding="utf-8"))
        training_state = torch.load(resume / "training.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(training_state["optimizer"])
        scheduler.load_state_dict(training_state["scheduler"])
        step = int(state["step"])
        best_wer = float(state["best_wer"])
        best_checkpoint = state.get("best_checkpoint")
        evaluations_without_improvement = int(state["evaluations_without_improvement"])
        print(f"Resuming seed training from step {step}", flush=True)

    if step < unfreeze_step:
        model.freeze_feature_encoder()
    else:
        for parameter in model.wav2vec2.feature_extractor.parameters():
            parameter.requires_grad = True

    accumulation = int(config["training"]["gradient_accumulation_steps"])
    patience = int(config["training"]["early_stopping_patience"])
    use_bf16 = bool(config["training"]["bf16"] and torch.cuda.is_bf16_supported())
    losses: list[float] = []
    iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    stopped_early = False
    accumulated = 0
    while step < maximum_steps:
        if step == unfreeze_step:
            for parameter in model.wav2vec2.feature_extractor.parameters():
                parameter.requires_grad = True
            print(f"Unfroze feature encoder at step {step}", flush=True)
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        context = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
        with context:
            output = model(
                input_values=batch["input_values"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                language_ids=batch["language_ids"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = output.loss / accumulation
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite seed loss at step {step + 1}")
        loss.backward()
        losses.append(float(loss.detach().cpu()) * accumulation)
        accumulated += 1
        if accumulated == accumulation:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"]["maximum_gradient_norm"])
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
            step += 1
        else:
            continue
        if step % 25 == 0:
            print(
                f"train-seed step={step}/{maximum_steps} "
                f"loss={sum(losses[-25:]) / len(losses[-25:]):.4f}",
                flush=True,
            )
        if step % eval_steps != 0 and step != maximum_steps:
            continue

        metrics = evaluate_model(model, eval_loader, processor, device)
        improved = metrics["macro_wer"] < best_wer
        if improved:
            best_wer = metrics["macro_wer"]
            evaluations_without_improvement = 0
        else:
            evaluations_without_improvement += 1
        state = {
            "step": step,
            "best_wer": best_wer,
            "best_checkpoint": best_checkpoint,
            "evaluations_without_improvement": evaluations_without_improvement,
            "metrics": metrics,
        }
        checkpoint = _save_training_checkpoint(
            checkpoint_root, step, model, processor, optimizer, scheduler, state
        )
        if improved:
            best_checkpoint = str(checkpoint)
            state["best_checkpoint"] = best_checkpoint
            _atomic_json(checkpoint / "state.json", state)
            _atomic_json(checkpoint_root / "best.json", {"checkpoint": best_checkpoint, "metrics": metrics})
        _prune_checkpoints(
            checkpoint_root,
            {checkpoint, Path(best_checkpoint)} if best_checkpoint else {checkpoint},
        )
        print(json.dumps({"step": step, **metrics, "best": improved}), flush=True)
        model.train()
        if evaluations_without_improvement >= patience:
            stopped_early = True
            print(f"Early stopping at step {step}", flush=True)
            break

    if best_checkpoint is None:
        raise RuntimeError("seed training completed without a best checkpoint")
    return {
        "final_step": step,
        "stopped_early": stopped_early,
        "best_macro_wer": best_wer,
        "best_checkpoint": best_checkpoint,
        "train_records": len(train_records),
        "eval_records": len(eval_records),
    }
