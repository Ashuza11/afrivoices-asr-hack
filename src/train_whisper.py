"""
src/train_whisper.py — Whisper fine-tuning script (Track A)

Works both locally (smoke-test mode) and on Kaggle (full training).
Control behaviour via the CONFIG dict at the top.
"""
import os, sys, dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
import numpy as np
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate  # HF evaluate library (pip install evaluate)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — change these to switch between smoke-test and full Kaggle run
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # Model
    "model_name":    "openai/whisper-tiny",   # swap to "openai/whisper-small" on Kaggle
    "language":      "Swahili",               # or list for multilingual
    "task":          "transcribe",

    # Data
    "dataset_name":  "google/fleurs",         # swap to "MCAA1-MSU/anv_data_ke" when approved
    "dataset_config": "sw_ke",                # FLEURS Swahili config
    "max_train_samples": 20,                  # set None on Kaggle (use full dataset)
    "max_eval_samples":  5,

    # Training
    "output_dir":    "models/whisper-tiny-sw-smoke",
    "max_steps":     3,                       # set 2000+ on Kaggle
    "per_device_train_batch_size": 1,         # set 8 on Kaggle GPU
    "gradient_accumulation_steps": 1,         # set 4 on Kaggle (effective batch = 32)
    "learning_rate": 1e-5,
    "warmup_steps":  0,                       # set 200 on Kaggle
    "fp16":          False,                   # set True on Kaggle GPU
    "gradient_checkpointing": False,          # set True on Kaggle to save VRAM
    "eval_steps":    3,
    "save_steps":    3,
    "logging_steps": 1,
    "push_to_hub":   False,                   # set True on Kaggle with your HF token
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load feature extractor, tokenizer, processor
# ══════════════════════════════════════════════════════════════════════════════
# The feature extractor converts raw audio → log-mel spectrogram (what we
# learned in Lesson 2). The tokenizer converts text ↔ token IDs. The
# processor bundles both together for convenience.
print(f"\n[1] Loading processor for {CONFIG['model_name']} ...")
feature_extractor = WhisperFeatureExtractor.from_pretrained(CONFIG["model_name"])
tokenizer = WhisperTokenizer.from_pretrained(
    CONFIG["model_name"],
    language=CONFIG["language"],
    task=CONFIG["task"],
)
processor = WhisperProcessor.from_pretrained(
    CONFIG["model_name"],
    language=CONFIG["language"],
    task=CONFIG["task"],
)
print("   Done.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Load & prepare dataset
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[2] Loading dataset: {CONFIG['dataset_name']} ({CONFIG['dataset_config']}) ...")

# Stream → take only N examples → convert to in-memory Dataset.
# This avoids downloading the full split (which can be several GB).
def stream_take(split: str, n: int):
    """Stream a dataset split and materialise only the first n examples."""
    from datasets import Dataset
    ds_iter = load_dataset(
        CONFIG["dataset_name"], CONFIG["dataset_config"],
        split=split, streaming=True,
    )
    ds_iter = ds_iter.cast_column("audio", Audio(sampling_rate=16_000))
    examples = []
    for ex in ds_iter:
        examples.append(ex)
        if len(examples) >= n:
            break
    return Dataset.from_list(examples)

n_train = CONFIG["max_train_samples"] or 800
n_eval  = CONFIG["max_eval_samples"]  or 100
raw_train = stream_take("train",      n_train)
raw_eval  = stream_take("validation", n_eval)

print(f"   Train examples : {len(raw_train)}")
print(f"   Eval  examples : {len(raw_eval)}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Preprocessing function
# ══════════════════════════════════════════════════════════════════════════════
# This runs on every example and turns raw audio + text into tensors the
# model can consume. It is the bridge between "dataset" and "training loop."
def prepare_dataset(example):
    # Audio → log-mel spectrogram (feature extractor does this)
    audio = example["audio"]
    example["input_features"] = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="np",          # keep as numpy here; Trainer handles torch conversion
    ).input_features[0]

    # Text → token IDs (tokenizer does this)
    # -100 will be used as the ignore index so loss ignores padding
    transcript_key = "transcription" if "transcription" in example else "sentence"
    example["labels"] = tokenizer(example[transcript_key]).input_ids

    return example

print("\n[3] Preprocessing dataset (audio → features, text → token IDs) ...")
train_dataset = raw_train.map(
    prepare_dataset,
    remove_columns=raw_train.column_names,  # keep only input_features + labels
    num_proc=1,
)
eval_dataset = raw_eval.map(
    prepare_dataset,
    remove_columns=raw_eval.column_names,
    num_proc=1,
)
print("   Done.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — DataCollator
# ══════════════════════════════════════════════════════════════════════════════
# Pads audio features to the same length (30s max for Whisper) and pads
# labels to the longest transcript in the batch, using -100 as the ignore token.
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        # Pad audio features
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Pad token labels; replace padding token id with -100 so loss ignores them
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove the BOS token if present at start of labels
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
print("\n[4] DataCollator ready.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Metrics (WER)
# ══════════════════════════════════════════════════════════════════════════════
# This function is called by the Trainer at each eval step.
# It decodes the model's predicted token IDs back to text and computes WER.
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids   = pred.predictions
    label_ids  = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id  # restore padding

    # Decode predicted and reference token sequences back to text
    pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER — lower is better, 0.0 = perfect
    wer_score = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": round(wer_score, 4)}

print("[5] Metrics (WER) configured.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Load model
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[6] Loading model: {CONFIG['model_name']} ...")
model = WhisperForConditionalGeneration.from_pretrained(CONFIG["model_name"])

# Tell the model which language/task to use during generation
# These get baked in as "forced decoder IDs" — the first tokens the decoder
# always outputs, so it knows "I am transcribing Swahili" before hearing audio.
model.generation_config.language = CONFIG["language"].lower()
model.generation_config.task     = CONFIG["task"]
model.generation_config.forced_decoder_ids = None  # let Trainer handle this

n_params = sum(p.numel() for p in model.parameters())
print(f"   Parameters: {n_params:,}  ({n_params/1e6:.0f}M)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Training arguments
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7] Setting up training arguments ...")
training_args = Seq2SeqTrainingArguments(
    output_dir                  = CONFIG["output_dir"],
    max_steps                   = CONFIG["max_steps"],
    per_device_train_batch_size = CONFIG["per_device_train_batch_size"],
    gradient_accumulation_steps = CONFIG["gradient_accumulation_steps"],
    learning_rate               = CONFIG["learning_rate"],
    warmup_steps                = CONFIG["warmup_steps"],
    fp16                        = CONFIG["fp16"],
    gradient_checkpointing      = CONFIG["gradient_checkpointing"],
    predict_with_generate       = True,   # use model.generate() at eval time (greedy)
    generation_max_length       = 225,
    eval_strategy               = "steps",
    eval_steps                  = CONFIG["eval_steps"],
    save_strategy               = "steps",
    save_steps                  = CONFIG["save_steps"],
    logging_steps               = CONFIG["logging_steps"],
    report_to                   = ["none"],  # swap to "wandb" on Kaggle
    load_best_model_at_end      = True,
    metric_for_best_model       = "wer",
    greater_is_better           = False,  # lower WER = better
    push_to_hub                 = CONFIG["push_to_hub"],
)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Trainer + train
# ══════════════════════════════════════════════════════════════════════════════
trainer = Seq2SeqTrainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_dataset,
    eval_dataset    = eval_dataset,
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
    processing_class= processor.feature_extractor,
)

print("\n[8] Starting training ...")
print(f"    Steps       : {CONFIG['max_steps']}")
print(f"    Batch size  : {CONFIG['per_device_train_batch_size']}")
print(f"    Learning rate: {CONFIG['learning_rate']}")
print(f"    Device      : {'GPU' if torch.cuda.is_available() else 'CPU'}\n")

trainer.train()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — Final eval
# ══════════════════════════════════════════════════════════════════════════════
print("\n[9] Final evaluation ...")
metrics = trainer.evaluate()
print(f"\n    eval_wer  = {metrics.get('eval_wer', 'N/A')}")
print(f"    eval_loss = {metrics.get('eval_loss', 'N/A'):.4f}")

print("\n=== Training complete ===")
print(f"Checkpoint saved to: {CONFIG['output_dir']}")
print("On Kaggle: set push_to_hub=True to upload to HuggingFace Hub.")
