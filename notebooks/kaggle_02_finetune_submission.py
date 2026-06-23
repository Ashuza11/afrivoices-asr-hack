"""
Kaggle Notebook 2 — Fine-tune Whisper-small on Swahili + Somali
================================================================
Goal: Improve on the 1.61 WER baseline by fine-tuning on the two
      languages we have training data for right now.

Training data available:
  Swahili → DigitalUmuganda/Afrivoice_Swahili  (~3,200 hrs, 561k clips)
  Somali  → DigitalUmuganda/Afrivoice (Somali/* path, ~535 hrs)

Strategy: stream a small sample per language (no full download needed),
fine-tune Whisper-small for ~1000 steps, then re-run inference on the
full test set and submit.

Expected improvement: average WER 1.61 → ~1.2–1.3
"""


# ── CELL 1 — Install dependencies ───────────────────────────────────────────
# !pip install -q -U transformers accelerate datasets evaluate jiwer soundfile librosa pydub


# ── CELL 2 — Imports + HuggingFace login ────────────────────────────────────
import os, io, glob, warnings
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

try:
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = HF_TOKEN
    login(token=HF_TOKEN, quiet=True)
    print("HuggingFace login OK.")
except Exception as e:
    print(f"HF_TOKEN issue ({e}) — make sure it's in Kaggle Secrets")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

CHECKPOINT_DIR = "/kaggle/working/whisper-small-swa-som"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ── CELL 3 — Load Whisper processor ─────────────────────────────────────────
MODEL_ID = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(MODEL_ID)
feature_extractor = processor.feature_extractor
tokenizer = processor.tokenizer
print(f"Processor loaded: {MODEL_ID}")


# ── CELL 4 — Audio decode helper (same proven logic as Notebook 1) ───────────
def decode_audio(audio_field):
    """Accept HF Audio dict, bytes dict, or raw bytes → 16 kHz float32 array."""
    # Case 1: HF Audio feature already decoded to numpy
    if isinstance(audio_field, dict) and "array" in audio_field:
        arr = np.array(audio_field["array"], dtype=np.float32)
        sr = audio_field.get("sampling_rate", 16000)
        if sr != 16000:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        return arr

    # Case 2: bytes wrapped in a dict ({"bytes": b"...", "path": "..."})
    raw = audio_field.get("bytes") if isinstance(audio_field, dict) else audio_field
    if isinstance(raw, bytes) and len(raw) > 0:
        try:
            arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            if sr != 16000:
                import librosa
                arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
            return arr
        except Exception:
            pass
        from pydub import AudioSegment
        seg = (AudioSegment.from_file(io.BytesIO(raw))
               .set_frame_rate(16000).set_channels(1))
        return np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0

    raise ValueError(f"Cannot decode audio — got type {type(audio_field)}, "
                     f"dict keys: {list(audio_field.keys()) if isinstance(audio_field, dict) else 'N/A'}")


# ── CELL 5 — Inspect manifest + audio tar structure (share output before continuing)
import json, tarfile
from huggingface_hub import hf_hub_download

# Step 1: read a manifest to see field names and audio key format
print("Downloading manifest_0.jsonl ...")
manifest_path = hf_hub_download(
    repo_id="DigitalUmuganda/Afrivoice_Swahili",
    filename="agriculture_swahili_train/manifest_0.jsonl",
    repo_type="dataset",
    token=HF_TOKEN,
)
with open(manifest_path) as f:
    lines = f.readlines()
print(f"Entries in manifest_0: {len(lines)}")
print("\nFirst entry fields:")
entry = json.loads(lines[0])
for k, v in entry.items():
    print(f"  {k:35s}: {str(v)[:80]}")

# Step 2: inspect the matching audio tar to see how files are named inside
print("\nDownloading agriculture_swahili_train/audio/audio_0.tar.xz ...")
audio_tar_path = hf_hub_download(
    repo_id="DigitalUmuganda/Afrivoice_Swahili",
    filename="agriculture_swahili_train/audio/audio_0.tar.xz",
    repo_type="dataset",
    token=HF_TOKEN,
)
with tarfile.open(audio_tar_path, "r:xz") as tar:
    members = tar.getnames()
print(f"Files inside audio_0.tar.xz: {len(members)}")
print("First 5 filenames:", members[:5])


# ── CELL 6 — Load Swahili training data (manifest + tar.xz, no loading script) ─
# Structure confirmed in Cell 5:
#   manifest_{shard}.jsonl  → key + normalized_transcription (7000 entries/shard)
#   audio_{shard}.tar.xz    → audio_{shard}/{key}.webm

import json, tarfile, os
from huggingface_hub import hf_hub_download

N_SAMPLES = 5000   # clips to use for training — shard 0 has 7000 so one download is enough
SHARD     = 0

print(f"Loading {N_SAMPLES} Swahili clips from shard {SHARD}...")

# 1. Download manifest (small, ~6 MB)
manifest_path = hf_hub_download(
    repo_id="DigitalUmuganda/Afrivoice_Swahili",
    filename=f"agriculture_swahili_train/manifest_{SHARD}.jsonl",
    repo_type="dataset",
    token=HF_TOKEN,
)
with open(manifest_path) as f:
    all_entries = [json.loads(l) for l in f]

# Build key → text lookup for the first N_SAMPLES valid entries
wanted = {}
for entry in all_entries:
    text = (entry.get("normalized_transcription") or "").strip()
    if text:
        wanted[entry["key"]] = text
    if len(wanted) >= N_SAMPLES:
        break
print(f"  {len(wanted)} valid manifest entries selected.")

# 2. Download audio shard (~1.26 GB — one download covers all 7000 clips)
audio_tar_path = hf_hub_download(
    repo_id="DigitalUmuganda/Afrivoice_Swahili",
    filename=f"agriculture_swahili_train/audio/audio_{SHARD}.tar.xz",
    repo_type="dataset",
    token=HF_TOKEN,
)
print(f"  Audio tar downloaded. Processing...")

# 3. Iterate tar, decode and preprocess matching clips
swa_records = []
tokenizer.set_prefix_tokens(language="swahili", task="transcribe")

with tarfile.open(audio_tar_path, "r:xz") as tar:
    for member in tar:
        if not member.name.endswith(".webm"):
            continue
        key = os.path.basename(member.name).replace(".webm", "")
        if key not in wanted:
            continue
        try:
            webm_bytes = tar.extractfile(member).read()
            arr = decode_audio({"bytes": webm_bytes})
            arr = arr[:480_000]   # cap at 30 s
        except Exception as e:
            print(f"  SKIP {key}: {e}")
            continue

        input_features = feature_extractor(arr, sampling_rate=16000).input_features[0]
        labels = tokenizer(wanted[key]).input_ids

        swa_records.append({"input_features": input_features, "labels": labels})

        if len(swa_records) % 500 == 0:
            print(f"  {len(swa_records)} / {N_SAMPLES} clips processed", flush=True)
        if len(swa_records) >= N_SAMPLES:
            break

print(f"\nSwahili: {len(swa_records)} clips ready.")


# ── CELL 7 — Inspect + load Somali from DigitalUmuganda/Afrivoice (raw files) ─
from huggingface_hub import list_repo_files, hf_hub_download
import json, tarfile, os

# Step 1: list only Somali files so we know the exact structure
print("Listing Somali files in DigitalUmuganda/Afrivoice...")
all_afrivoice_files = list(list_repo_files(
    "DigitalUmuganda/Afrivoice",
    repo_type="dataset",
    token=HF_TOKEN,
))
somali_files = [f for f in all_afrivoice_files if f.startswith("Somali/")]
print(f"Found {len(somali_files)} Somali files:")
for f in sorted(somali_files)[:30]:
    print(" ", f)

# Step 2: identify manifest and audio tar files
manifest_files = sorted([f for f in somali_files if f.endswith(".jsonl")])
audio_tar_files = sorted([f for f in somali_files if f.endswith(".tar.xz") and "/audio" in f])
print(f"\nManifest files: {len(manifest_files)}")
print(f"Audio tar files: {len(audio_tar_files)}")

som_records = []

if manifest_files and audio_tar_files:
    # Same manifest + tar approach as Swahili
    print(f"\nDownloading {manifest_files[0]} ...")
    manifest_path = hf_hub_download(
        repo_id="DigitalUmuganda/Afrivoice",
        filename=manifest_files[0],
        repo_type="dataset",
        token=HF_TOKEN,
    )
    with open(manifest_path) as f:
        all_entries = [json.loads(l) for l in f]
    print(f"  Entries: {len(all_entries)}")
    print("  First entry fields:", list(all_entries[0].keys()))

    # Detect text field
    first = all_entries[0]
    text_col = ("normalized_transcription" if "normalized_transcription" in first
                else "transcription" if "transcription" in first else None)
    print(f"  Text field: {text_col!r}")
    print(f"  Sample text: {str(first.get(text_col, ''))[:80]}")

    # Build key → text lookup
    wanted_som = {}
    for entry in all_entries:
        text = (entry.get(text_col) or "").strip()
        if text and "key" in entry:
            wanted_som[entry["key"]] = text
        if len(wanted_som) >= N_SAMPLES:
            break
    print(f"  {len(wanted_som)} valid entries selected.")

    # Download audio tar
    print(f"\nDownloading {audio_tar_files[0]} ...")
    audio_tar_path = hf_hub_download(
        repo_id="DigitalUmuganda/Afrivoice",
        filename=audio_tar_files[0],
        repo_type="dataset",
        token=HF_TOKEN,
    )
    # Inspect filenames inside tar
    with tarfile.open(audio_tar_path, "r:xz") as tar:
        sample_names = [m.name for m in tar if not m.isdir()][:5]
    print(f"  Sample filenames in tar: {sample_names}")

    # Process
    tokenizer.set_prefix_tokens(language="somali", task="transcribe")
    with tarfile.open(audio_tar_path, "r:xz") as tar:
        for member in tar:
            if member.isdir():
                continue
            name = member.name
            # key is the filename without extension
            key = os.path.splitext(os.path.basename(name))[0]
            if key not in wanted_som:
                continue
            try:
                raw = tar.extractfile(member).read()
                arr = decode_audio({"bytes": raw})
                arr = arr[:480_000]
            except Exception as e:
                print(f"  SKIP {key}: {e}")
                continue

            input_features = feature_extractor(arr, sampling_rate=16000).input_features[0]
            labels = tokenizer(wanted_som[key]).input_ids
            som_records.append({"input_features": input_features, "labels": labels})

            if len(som_records) % 500 == 0:
                print(f"  {len(som_records)} / {N_SAMPLES} Somali clips processed", flush=True)
            if len(som_records) >= N_SAMPLES:
                break

    print(f"\nSomali: {len(som_records)} clips ready.")

else:
    # No manifest files — the tar may contain both audio + transcription files
    print("\nNo manifest files found — inspecting audio tar directly...")
    if audio_tar_files:
        audio_tar_path = hf_hub_download(
            repo_id="DigitalUmuganda/Afrivoice",
            filename=audio_tar_files[0],
            repo_type="dataset",
            token=HF_TOKEN,
        )
        with tarfile.open(audio_tar_path, "r:xz") as tar:
            members = tar.getnames()
        print(f"Files in {audio_tar_files[0]}: {len(members)}")
        print("First 10:", members[:10])
        print("Share this output so we can adjust the loader.")
    else:
        print("No audio tars found either. Share the file listing above.")


# ── CELL 8 — Build combined dataset + train/eval split ───────────────────────
from torch.utils.data import Dataset as TorchDataset

class WhisperDataset(TorchDataset):
    def __init__(self, records):
        self.records = records
    def __len__(self):
        return len(self.records)
    def __getitem__(self, i):
        return self.records[i]

all_records = swa_records + som_records + anv_records
np.random.shuffle(all_records)

split = int(0.95 * len(all_records))
train_ds = WhisperDataset(all_records[:split])
eval_ds  = WhisperDataset(all_records[split:])
print(f"Train: {len(train_ds)}  Eval: {len(eval_ds)}")


# ── CELL 9 — Data collator + WER metric ──────────────────────────────────────
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Mel features are all (80, 3000) — just stack
        input_features = torch.tensor(
            np.stack([f["input_features"] for f in features]), dtype=torch.float32
        )

        # Labels need padding
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # Remove decoder_start_token if it was added by set_prefix_tokens
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        return {"input_features": input_features, "labels": labels}


wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": round(wer, 4)}


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>"),
)
print("Data collator ready.")


# ── CELL 10 — Load model + configure for multilingual fine-tuning ─────────────
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

# These three lines are critical for multilingual fine-tuning
model.config.forced_decoder_ids = None          # don't hardwire any language
model.config.suppress_tokens    = []            # don't suppress anything
model.generation_config.forced_decoder_ids = None

print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M parameters")

training_args = Seq2SeqTrainingArguments(
    output_dir                  = CHECKPOINT_DIR,
    per_device_train_batch_size = 16,
    gradient_accumulation_steps = 2,       # effective batch = 32
    learning_rate               = 1e-5,
    warmup_steps                = 100,
    max_steps                   = 1500,    # ~2 epochs over 11k clips at effective batch 32
    gradient_checkpointing      = True,
    fp16                        = True,
    eval_strategy               = "steps",
    per_device_eval_batch_size  = 8,
    predict_with_generate       = True,
    generation_max_length       = 225,
    save_steps                  = 250,
    eval_steps                  = 250,
    logging_steps               = 25,
    report_to                   = ["tensorboard"],
    load_best_model_at_end      = True,
    metric_for_best_model       = "wer",
    greater_is_better           = False,
    push_to_hub                 = False,
    dataloader_num_workers      = 2,
)
print("Training args configured.")


# ── CELL 11 — Train ──────────────────────────────────────────────────────────
trainer = Seq2SeqTrainer(
    model         = model,
    args          = training_args,
    train_dataset = train_ds,
    eval_dataset  = eval_ds,
    data_collator = data_collator,
    compute_metrics = compute_metrics,
    processing_class = feature_extractor,
)

print("Starting fine-tuning...")
print(f"  {len(train_ds)} train clips  |  {len(eval_ds)} eval clips")
print(f"  max_steps=1000, effective batch=32  →  ~{1000*32//len(train_ds)} epochs")
trainer.train()
print("Training done.")

# Save final model + processor
trainer.save_model(CHECKPOINT_DIR)
processor.save_pretrained(CHECKPOINT_DIR)
print(f"Model + processor saved to {CHECKPOINT_DIR}")


# ── CELL 12 — Reload fine-tuned model ────────────────────────────────────────
import torch, gc
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Free trainer + training model before loading inference copy
try:
    del trainer
except NameError:
    pass
try:
    del model
except NameError:
    pass
gc.collect()
torch.cuda.empty_cache()
print(f"GPU memory freed: {torch.cuda.memory_reserved()/1e9:.1f} GB reserved")

ft_processor = WhisperProcessor.from_pretrained(CHECKPOINT_DIR)
ft_model = WhisperForConditionalGeneration.from_pretrained(
    CHECKPOINT_DIR, torch_dtype=torch.float16
).to(device)
ft_model.eval()
ft_model.config.forced_decoder_ids = None
ft_model.generation_config.forced_decoder_ids = None
print(f"Fine-tuned model loaded from {CHECKPOINT_DIR}")


# ── CELL 13 — Transcribe test set with fine-tuned model ──────────────────────
# Identical to Notebook 1 Cell 6, but uses ft_model + ft_processor.
# The model now has better Swahili + Somali; other languages are unchanged.

import kagglehub, glob, os, io, numpy as np, pandas as pd, soundfile as sf

test_path = kagglehub.dataset_download("digitalumuganda/anv-test-data-nt")
print("Test data at:", test_path)

BATCH_SIZE      = 8   # reduced from 16 to avoid OOM after training
CHECKPOINT_FILE = "/kaggle/working/submission_ft_checkpoint.csv"

# decode_audio defined here so this cell runs safely after a kernel restart
def decode_audio(audio_field):
    """Accept HF Audio dict, bytes dict, or raw bytes → 16 kHz float32 array."""
    if isinstance(audio_field, dict) and "array" in audio_field:
        arr = np.array(audio_field["array"], dtype=np.float32)
        sr = audio_field.get("sampling_rate", 16000)
        if sr != 16000:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        return arr
    raw = audio_field.get("bytes") if isinstance(audio_field, dict) else audio_field
    if isinstance(raw, bytes) and len(raw) > 0:
        try:
            arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            if sr != 16000:
                import librosa
                arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
            return arr
        except Exception:
            pass
        from pydub import AudioSegment
        seg = (AudioSegment.from_file(io.BytesIO(raw))
               .set_frame_rate(16000).set_channels(1))
        return np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0
    raise ValueError(f"Cannot decode audio — got type {type(audio_field)}, "
                     f"dict keys: {list(audio_field.keys()) if isinstance(audio_field, dict) else 'N/A'}")

def transcribe_batch_ft(arrays, language=None):
    """Transcribe with the fine-tuned model.
    Pass language ('sw', 'so', etc.) to force correct decoder prefix."""
    arrays = [a[:480_000] for a in arrays]
    inputs = ft_processor(
        arrays, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device).to(torch.float16)

    gen_kwargs = {"max_new_tokens": 225}
    if language:
        # Force the right language token so the model knows which language
        lang_token_id = ft_processor.tokenizer.convert_tokens_to_ids(f"<|{language}|>")
        task_token_id = ft_processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")
        notimestamp_id = ft_processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        gen_kwargs["forced_decoder_ids"] = [[1, lang_token_id],
                                             [2, task_token_id],
                                             [3, notimestamp_id]]

    with torch.no_grad():
        ids = ft_model.generate(input_features=inputs, **gen_kwargs)
    return ft_processor.batch_decode(ids, skip_special_tokens=True)

# Language code mapping: test parquet uses 3-letter codes
LANG_TO_WHISPER = {
    "swa": "sw",   # Swahili    — fine-tuned ✅ (Whisper has native token)
    "som": "so",   # Somali     — fine-tuned ✅ (Whisper has native token)
    "kik": None,   # Kikuyu     — fine-tuned ✅ (not in Whisper vocab, no forced token)
    "luo": None,   # Luo        — fine-tuned ✅ (not in Whisper vocab, no forced token)
    "mas": None,   # Maasai     — fine-tuned ✅ (not in Whisper vocab, no forced token)
    "kln": None,   # Kalenjin   — fine-tuned ✅ (not in Whisper vocab, no forced token)
}

if os.path.exists(CHECKPOINT_FILE):
    existing   = pd.read_csv(CHECKPOINT_FILE)
    empty_pct  = (existing["transcription"].isna() |
                  (existing["transcription"].str.strip() == "")).mean()
    if empty_pct > 0.5:
        os.remove(CHECKPOINT_FILE)
        print(f"Deleted corrupted checkpoint ({empty_pct:.0%} empty). Starting fresh.")
        results, done_ids = [], set()
    else:
        results  = existing.to_dict("records")
        done_ids = set(existing["id"])
        print(f"Resuming: {len(results)} clips done.")
else:
    results, done_ids = [], set()
    print("Starting fresh.")

all_parquet_files = sorted(
    glob.glob(os.path.join(test_path, "**", "*.parquet"), recursive=True)
)
print(f"{len(all_parquet_files)} parquet files. Batch size: {BATCH_SIZE}.\n")

for pf_idx, pq_file in enumerate(all_parquet_files):
    df = pd.read_parquet(pq_file)
    df = df[~df["id"].isin(done_ids)]
    if len(df) == 0:
        print(f"[{pf_idx+1}/{len(all_parquet_files)}] already done — skip")
        continue

    lang3   = df["language"].iloc[0]     # e.g. "swa"
    wh_lang = LANG_TO_WHISPER.get(lang3) # e.g. "sw" or None
    print(f"[{pf_idx+1}/{len(all_parquet_files)}] {os.path.basename(pq_file)} "
          f"lang={lang3} (whisper={wh_lang}) rows={len(df)}", flush=True)

    rows = list(df.itertuples(index=False))
    for i in range(0, len(rows), BATCH_SIZE):
        chunk = rows[i: i + BATCH_SIZE]
        arrays, batch_ids, batch_langs = [], [], []

        for row in chunk:
            try:
                arrays.append(decode_audio(row.audio))
                batch_ids.append(row.id)
                batch_langs.append(row.language)
            except Exception as e:
                print(f"  DECODE ERROR {row.id}: {e}")
                results.append({"id": row.id, "language": row.language, "transcription": "."})
                done_ids.add(row.id)

        if arrays:
            try:
                texts = transcribe_batch_ft(arrays, language=wh_lang)
                for id_, lang_, text in zip(batch_ids, batch_langs, texts):
                    results.append({"id": id_, "language": lang_,
                                    "transcription": text.strip() or "."})
                    done_ids.add(id_)
            except Exception as e:
                print(f"  BATCH ERROR ({e}) — one-by-one")
                for id_, lang_, arr in zip(batch_ids, batch_langs, arrays):
                    try:
                        text = transcribe_batch_ft([arr], language=wh_lang)[0].strip() or "."
                    except Exception as e2:
                        print(f"    FAILED {id_}: {e2}")
                        text = "."
                    results.append({"id": id_, "language": lang_, "transcription": text})
                    done_ids.add(id_)

    pd.DataFrame(results).to_csv(CHECKPOINT_FILE, index=False)
    print(f"  → checkpoint saved ({len(results)} total)", flush=True)

print(f"\nDone. Total: {len(results)}")


# ── CELL 14 — Build final submission file ────────────────────────────────────
if not results and os.path.exists(CHECKPOINT_FILE):
    submission_df = pd.read_csv(CHECKPOINT_FILE)
else:
    submission_df = pd.DataFrame(results)

mask = (submission_df["transcription"].isna() |
        (submission_df["transcription"].str.strip() == ""))
if mask.sum() > 0:
    print(f"Replacing {mask.sum()} empty rows with '.'")
    submission_df.loc[mask, "transcription"] = "."

submission_df = submission_df[["id", "language", "transcription"]]
submission_df.to_csv("submission.csv", index=False)

sub = pd.read_csv("submission.csv")
print(f"NaN transcription : {sub['transcription'].isna().sum()}")
print(f"Empty transcription: {(sub['transcription'].str.strip() == '').sum()}")
print(f"NaN language       : {sub['language'].isna().sum()}")
print(sub.head())
print(f"\nsubmission.csv written — {len(sub)} rows, 3 columns.")
print("Go to competition page → Submit Prediction → upload submission.csv")
