"""
Kaggle Notebook 2 — Multilingual data pipeline + Whisper fine-tuning (Track A)
================================================================================
"""

# ── CELL 1 — Install ────────────────────────────────────────────────────────
# !pip install -q -U transformers accelerate datasets evaluate jiwer soundfile librosa huggingface_hub


# ── CELL 2 — Auth + imports ─────────────────────────────────────────────────
import os, io, json, re, random
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import torch
import soundfile as sf
from datasets import Dataset, Audio, concatenate_datasets, DatasetDict
from huggingface_hub import HfApi, hf_hub_download, login
from transformers import (
    WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor,
    WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer,
)
import evaluate

from kaggle_secrets import UserSecretsClient
HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
login(token=HF_TOKEN)
print("Logged in to Hugging Face Hub.")

hf_api = HfApi()


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE A — MCAA1-MSU/anv_data_ke  (Kikuyu, Kalenjin, Luo, Maasai, Somali-Maxatire)
# Known schema: {lang}/{split}/{type}/audios/*.parquet + files/meta.csv + files/transcripts.csv
# ══════════════════════════════════════════════════════════════════════════════
ANV_REPO = "MCAA1-MSU/anv_data_ke"
ANV_LANGS = {"kik": "Kikuyu", "kln": "Kalenjin", "luo": "Luo", "mas": "Maasai", "som": "Somali"}

def load_anv_language(lang_code: str, split: str = "train", max_files: int = 2) -> Dataset:
    """
    Load a slice of one language from anv_data_ke.
    max_files caps how many parquet shards we pull per (scripted/unscripted) —
    keep this small while iterating, raise it for the real training run.
    """
    records = []
    for speech_type in ["scripted", "unscripted"]:
        prefix = f"{lang_code}/{split}/{speech_type}"
        try:
            all_files = hf_api.list_repo_files(ANV_REPO, repo_type="dataset", token=HF_TOKEN)
        except Exception as e:
            print(f"  Could not list repo files (likely gate not yet approved): {e}")
            return Dataset.from_list([])

        parquet_files = sorted(f for f in all_files if f.startswith(f"{prefix}/audios/") and f.endswith(".parquet"))[:max_files]
        transcripts_file = f"{prefix}/files/transcripts.csv"
        meta_file = f"{prefix}/files/meta.csv"

        if not parquet_files:
            continue

        # Download + read the text files first (small)
        try:
            transcripts_path = hf_hub_download(ANV_REPO, transcripts_file, repo_type="dataset", token=HF_TOKEN)
            transcripts_df = pd.read_csv(transcripts_path)
        except Exception as e:
            print(f"  Could not load {transcripts_file}: {e}")
            continue
        try:
            meta_path = hf_hub_download(ANV_REPO, meta_file, repo_type="dataset", token=HF_TOKEN)
            meta_df = pd.read_csv(meta_path)
        except Exception:
            meta_df = None

        print(f"  {prefix}: transcripts.csv columns = {list(transcripts_df.columns)}")
        if meta_df is not None:
            print(f"  {prefix}: meta.csv columns       = {list(meta_df.columns)}")

        for pq_file in parquet_files:
            pq_path = hf_hub_download(ANV_REPO, pq_file, repo_type="dataset", token=HF_TOKEN)
            audio_df = pd.read_parquet(pq_path)
            print(f"  {pq_file}: audio parquet columns = {list(audio_df.columns)}  ({len(audio_df)} rows)")

            merged = audio_df.merge(transcripts_df, on="mediaPathId", how="inner")
            text_col = "transcript" if speech_type == "unscripted" else "actualSentence"
            if text_col not in merged.columns:
                # fall back: try any column with 'sentence' or 'transcript' in the name
                candidates = [c for c in merged.columns if "sentence" in c.lower() or "transcript" in c.lower()]
                text_col = candidates[0] if candidates else None
                print(f"    [WARN] expected text col not found, using '{text_col}' instead")

            for _, row in merged.iterrows():
                audio_field = row.get("audio")
                if audio_field is None:
                    continue
                records.append({
                    "audio": audio_field,           # expect dict with 'bytes' or 'array'+'sampling_rate'
                    "text": row.get(text_col, ""),
                    "language": ANV_LANGS[lang_code],
                    "type": speech_type,
                })

    print(f"  -> {lang_code} ({split}): {len(records)} examples loaded")
    return Dataset.from_list(records) if records else Dataset.from_list([])


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE B — DigitalUmuganda/Afrivoice_Swahili
# Confirmed schema (dataset card inspected 2026-06-17):
#   Load via:   load_dataset(SWAHILI_REPO, name=<domain>, streaming=True)
#   Domains:    agriculture, education, financial, government, health
#   Text field: "transcription"  (raw, may contain [cs]...[cs] code-switch tags)
#   Split info: embedded in "dir_path" column, e.g. "agriculture_swahili_train"
#   Audio:      "audio" column (HF Audio object, decoded automatically)
# ══════════════════════════════════════════════════════════════════════════════
SWAHILI_REPO = "DigitalUmuganda/Afrivoice_Swahili"
SWAHILI_DOMAINS = ["agriculture", "education", "financial", "government", "health"]
_CS_TAG = re.compile(r'\[cs\](.*?)\[cs\]', re.IGNORECASE)

def clean_swahili_text(text: str) -> str:
    """Remove [cs]...[cs] code-switching markers but keep the word inside."""
    return _CS_TAG.sub(r'\1', text).strip()

def load_swahili_dataset(split_key: str = "train", max_per_domain: int = 500) -> Dataset:
    """
    Stream Swahili from 5 domains; filter by split_key via dir_path.
    split_key: "train" | "dev" | "dev_test"
    Raise max_per_domain for the full training run (current cap is ~2500 total).
    """
    from datasets import load_dataset as hf_load
    records = []
    for domain in SWAHILI_DOMAINS:
        print(f"  {domain} ...")
        try:
            ds_iter = hf_load(SWAHILI_REPO, name=domain, split="train",
                              streaming=True, token=HF_TOKEN)
        except Exception as e:
            print(f"    SKIP — could not load {domain}: {e}")
            continue

        count = 0
        for ex in ds_iter:
            if split_key not in ex.get("dir_path", "").lower():
                continue
            text = clean_swahili_text(ex.get("transcription", ""))
            if not text:
                continue
            audio = ex.get("audio")
            if audio is None:
                continue
            records.append({"audio": audio, "text": text, "language": "Swahili", "type": "unscripted"})
            count += 1
            if count >= max_per_domain:
                break
        print(f"    -> {count} examples")

    print(f"  Total Swahili: {len(records)}")
    return Dataset.from_list(records) if records else Dataset.from_list([])


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE C — DigitalUmuganda/Afrivoice  (Somali/*, Mogadishu dialect)
# Schema observed from Dataset Viewer (2026-06-17):
#   Format: WebDataset — tar.xz audio shards under Somali/audio_shards/
#   Viewer columns: "audio", "__key__", "__url__"
#   Dataset card fields: transcription, creator, speaker_id, locale, gender, age, year
#   Load via: load_dataset(SOMALI_REPO, streaming=True) — then filter by __url__ containing "Somali"
#   Text field: "transcription"  (confirmed from dataset card)
# ══════════════════════════════════════════════════════════════════════════════
SOMALI_REPO = "DigitalUmuganda/Afrivoice"

def load_somali_dataset(max_examples: int = 500) -> Dataset:
    """
    Stream Somali (Mogadishu) from the Afrivoice WebDataset repo.
    Filters to entries whose shard URL contains "Somali/".
    """
    from datasets import load_dataset as hf_load
    records = []
    try:
        ds_iter = hf_load(SOMALI_REPO, streaming=True, token=HF_TOKEN, split="train")
        for ex in ds_iter:
            url = ex.get("__url__", "")
            if "Somali" not in url:
                continue
            text = ex.get("transcription", "")
            if not text:
                continue
            audio = ex.get("audio")
            if audio is None:
                continue
            records.append({"audio": audio, "text": text, "language": "Somali", "type": "unscripted"})
            if len(records) >= max_examples:
                break
    except Exception as e:
        print(f"  [WARN] Somali load failed: {e}")
        print("  If streaming does not expose __url__, run the [INSPECT] cell below and paste results.")
    print(f"  Total Somali (Mogadishu): {len(records)}")
    return Dataset.from_list(records) if records else Dataset.from_list([])

# ── [INSPECT] Somali (run if load_somali_dataset returns 0 rows) ─────────────
def inspect_afrivoice_somali(n_preview: int = 3):
    from datasets import load_dataset as hf_load
    print("First 3 rows of Afrivoice (to identify real field names):")
    ds = hf_load(SOMALI_REPO, streaming=True, token=HF_TOKEN, split="train")
    for i, ex in enumerate(ds):
        if i >= n_preview:
            break
        print(f"  keys: {list(ex.keys())}")
        print(f"  __url__: {ex.get('__url__', 'N/A')}")
        print(f"  transcription sample: {str(ex.get('transcription', ex.get('text', 'KEY_MISSING')))[:80]}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# LOAD EVERYTHING (small caps for first iteration — raise once verified working)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Loading anv_data_ke languages ===")
anv_datasets = []
for code in ANV_LANGS:
    print(f"\n-- {code} --")
    ds = load_anv_language(code, split="train", max_files=2)
    if len(ds) > 0:
        anv_datasets.append(ds)

print("\n=== Loading Swahili (Afrivoice_Swahili, 5 domains) ===")
swahili_ds = load_swahili_dataset(split_key="train", max_per_domain=500)

print("\n=== Loading Somali Mogadishu (Afrivoice) ===")
somali_mog_ds = load_somali_dataset(max_examples=500)
if len(somali_mog_ds) == 0:
    print("  -> 0 rows. Run inspect_afrivoice_somali() to see real field names.")
    # inspect_afrivoice_somali()  # uncomment and run if the loader returned nothing

all_datasets = anv_datasets + [d for d in [swahili_ds, somali_mog_ds] if len(d) > 0]
print(f"\nTotal language datasets loaded: {len(all_datasets)}")
for d in all_datasets:
    if len(d) > 0:
        print(f"  {d[0]['language']}: {len(d)} examples")


# ══════════════════════════════════════════════════════════════════════════════
# TEMPERATURE-SAMPLED MULTILINGUAL MIXTURE  (Phase 2 plan, alpha=0.7)
# ══════════════════════════════════════════════════════════════════════════════
ALPHA = 0.7

def temperature_sample(datasets_list: List[Dataset], alpha: float = ALPHA) -> Dataset:
    sizes = np.array([len(d) for d in datasets_list], dtype=float)
    weights = sizes ** alpha
    weights = weights / weights.sum()
    target_total = int(sizes.sum())  # keep total dataset size roughly the same
    target_per_lang = (weights * target_total).astype(int)

    resampled = []
    for d, n in zip(datasets_list, target_per_lang):
        if len(d) == 0:
            continue
        idx = np.random.choice(len(d), size=min(n, len(d) * 3), replace=n > len(d))
        resampled.append(d.select(idx))
    return concatenate_datasets(resampled)

train_dataset_raw = temperature_sample(all_datasets)
print(f"\nFinal multilingual training set: {len(train_dataset_raw)} examples")
print(pd.Series([train_dataset_raw[i]["language"] for i in range(len(train_dataset_raw))]).value_counts())


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING — same pipeline as Lesson 4, now multilingual
# ══════════════════════════════════════════════════════════════════════════════
MODEL_NAME = "openai/whisper-small"

feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, task="transcribe")  # no fixed language — multilingual
processor = WhisperProcessor.from_pretrained(MODEL_NAME, task="transcribe")

LANG_TO_WHISPER_CODE = {
    "Swahili": "sw", "Kikuyu": "sw", "Kalenjin": "sw", "Luo": "sw", "Maasai": "sw", "Somali": "so",
    # NOTE: Whisper's tokenizer only knows ~98 languages — Kikuyu/Kalenjin/Luo/Maasai aren't
    # among them. We map them to the closest available token (Swahili) as a *starting point*;
    # the model still learns the correct mapping from audio -> target text during fine-tuning,
    # it just doesn't get a perfectly-matched language prior. This is a known limitation to
    # revisit (e.g. add new tokens, or drop forced language conditioning entirely).
}

def decode_audio_bytes(audio_field):
    """Decode parquet audio field to numpy array at 16 kHz.
    Primary: soundfile (WAV/FLAC). Fallback: pydub via ffmpeg (webm/mp3/opus/aac).
    """
    import numpy as np
    raw = audio_field["bytes"] if isinstance(audio_field, dict) else bytes(audio_field)
    try:
        arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != 16_000:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16_000)
        return arr
    except Exception:
        pass
    from pydub import AudioSegment
    audio = AudioSegment.from_file(io.BytesIO(raw)).set_frame_rate(16_000).set_channels(1)
    arr = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    return arr

def prepare_example(example):
    audio = example["audio"]
    if isinstance(audio, dict) and "bytes" in audio and audio.get("array") is None:
        arr = decode_audio_bytes(audio)
        sr = 16_000
    else:
        arr, sr = audio["array"], audio["sampling_rate"]

    example["input_features"] = feature_extractor(arr, sampling_rate=16_000).input_features[0]
    example["labels"] = tokenizer(example["text"]).input_ids
    return example

print("\nPreprocessing dataset (this can take a while on CPU) ...")
train_dataset = train_dataset_raw.map(prepare_example, remove_columns=train_dataset_raw.column_names, num_proc=2)


# ══════════════════════════════════════════════════════════════════════════════
# DataCollator + metrics  (identical to Lesson 4 / src/train_whisper.py)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids, label_ids = pred.predictions, pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": round(wer_metric.compute(predictions=pred_str, references=label_str), 4)}


# ══════════════════════════════════════════════════════════════════════════════
# Model + training (REAL settings — Kaggle GPU)
# ══════════════════════════════════════════════════════════════════════════════
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.generation_config.forced_decoder_ids = None

# Split off a small eval slice from the training mixture (proper dev set
# from anv_data_ke's dev_test split should replace this once available)
split = train_dataset.train_test_split(test_size=0.05, seed=42)
train_split, eval_split = split["train"], split["test"]

training_args = Seq2SeqTrainingArguments(
    output_dir="whisper-small-afrivoices",
    max_steps=3000,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,        # effective batch size 32
    learning_rate=1e-5,
    warmup_steps=200,
    fp16=True,
    gradient_checkpointing=True,
    predict_with_generate=True,
    generation_max_length=225,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    logging_steps=25,
    report_to=["none"],          # swap to "wandb" once you have an API key set up
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,           # set True + add hub_model_id to publish
)

trainer = Seq2SeqTrainer(
    model=model, args=training_args,
    train_dataset=train_split, eval_dataset=eval_split,
    data_collator=data_collator, compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

print("\nStarting training ...")
trainer.train()


# ══════════════════════════════════════════════════════════════════════════════
# Per-language eval breakdown
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Per-language WER on eval split ===")
# `train_dataset_raw` still has the "language" column (preprocessing dropped it via
# remove_columns). Filter the *raw* dataset per language, preprocess just that
# slice, and evaluate on it separately.
languages_present = sorted(set(train_dataset_raw["language"]))
for lang in languages_present:
    lang_raw = train_dataset_raw.filter(lambda x: x["language"] == lang)
    lang_eval = lang_raw.train_test_split(test_size=0.05, seed=42)["test"]
    if len(lang_eval) == 0:
        continue
    lang_eval = lang_eval.map(prepare_example, remove_columns=lang_eval.column_names, num_proc=1)
    metrics = trainer.evaluate(eval_dataset=lang_eval)
    print(f"  {lang:10s}  WER = {metrics['eval_wer']:.1%}  (n={len(lang_eval)})")


# ══════════════════════════════════════════════════════════════════════════════
# Save + (optionally) push to hub
# ══════════════════════════════════════════════════════════════════════════════
trainer.save_model("whisper-small-afrivoices-final")
processor.save_pretrained("whisper-small-afrivoices-final")
print("\nSaved to whisper-small-afrivoices-final/")
print("To publish: model.push_to_hub('your-username/whisper-small-afrivoices')")

print("\n=== Next step: run notebooks/kaggle_01_baseline_submission.py's Cell 5-7, ")
print("    but load THIS fine-tuned model instead of openai/whisper-small. ===")
