from __future__ import annotations

import json
import math
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
import torch
from jiwer import wer
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
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
