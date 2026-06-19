"""
Kaggle Notebook 1 — Zero-shot baseline submission
==================================================
"""

# ── CELL 1 — Install dependencies ───────────────────────────────────────────
# !pip install -q -U transformers accelerate jiwer kagglehub soundfile librosa


# ── CELL 2 — Imports + secrets ──────────────────────────────────────────────
import os
import io
import json
import glob
import numpy as np
import pandas as pd
import soundfile as sf
from kaggle_secrets import UserSecretsClient
import kagglehub

# Pull the HF token from Kaggle Secrets (set this in Add-ons -> Secrets first)
try:
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = HF_TOKEN
    print("HF_TOKEN loaded from Kaggle secrets.")
except Exception as e:
    print(f"No HF_TOKEN secret found ({e}) — fine for this notebook, "
          f"the test set comes from Kaggle, not HF.")


# ── CELL 3 — Download the official test set ────────────────────────────────
print("Downloading test set via kagglehub ...")
test_path = kagglehub.dataset_download("digitalumuganda/anv-test-data-nt")
print("Downloaded to:", test_path)

print("\nDirectory contents:")
for root, dirs, files in os.walk(test_path):
    level = root.replace(test_path, "").count(os.sep)
    indent = "  " * level
    print(f"{indent}{os.path.basename(root)}/")
    for f in files:
        fp = os.path.join(root, f)
        size = os.path.getsize(fp)
        print(f"{indent}  {f}  ({size:,} bytes)")


# ── CELL 4 — Inspect whatever structured file we find (csv/json/parquet) ──
# This cell is exploratory — run it and READ the printed output before
# continuing. We genuinely don't know the test set's exact schema yet.
candidate_files = (
    glob.glob(os.path.join(test_path, "**", "*.csv"), recursive=True)
    + glob.glob(os.path.join(test_path, "**", "*.json"), recursive=True)
    + glob.glob(os.path.join(test_path, "**", "*.jsonl"), recursive=True)
    + glob.glob(os.path.join(test_path, "**", "*.parquet"), recursive=True)
    + glob.glob(os.path.join(test_path, "**", "*.tsv"), recursive=True)
)
print(f"Found {len(candidate_files)} structured file(s):\n")

test_df = None
for f in candidate_files:
    print(f"--- {f} ---")
    try:
        if f.endswith(".csv") or f.endswith(".tsv"):
            sep = "\t" if f.endswith(".tsv") else ","
            df = pd.read_csv(f, sep=sep)
        elif f.endswith(".parquet"):
            df = pd.read_parquet(f)
        elif f.endswith(".jsonl"):
            df = pd.read_json(f, lines=True)
        else:
            with open(f) as fh:
                print(json.load(fh))
            continue
        print(df.head(3))
        print("columns:", list(df.columns))
        print("n_rows:", len(df))
        if test_df is None:
            test_df = df  # assume the first structured file is the manifest
    except Exception as e:
        print(f"  could not parse: {e}")
    print()

audio_files = (
    glob.glob(os.path.join(test_path, "**", "*.wav"), recursive=True)
    + glob.glob(os.path.join(test_path, "**", "*.flac"), recursive=True)
    + glob.glob(os.path.join(test_path, "**", "*.mp3"), recursive=True)
)
print(f"Found {len(audio_files)} raw audio file(s) on disk (first 5): {audio_files[:5]}")


# ── CELL 5 — Load Whisper-small (zero-shot, no fine-tuning) ────────────────
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODEL_ID = "openai/whisper-small"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
print(f"Loading {MODEL_ID} on {device} ({dtype}) ...")

processor = WhisperProcessor.from_pretrained(MODEL_ID)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
model.eval()
# Clear forced English — without this, Whisper tries to transcribe everything
# in English and outputs empty strings for unknown languages
model.generation_config.forced_decoder_ids = None
print("Model loaded.")


# ── CELL 6 — Transcribe every test audio (audio embedded in parquet bytes) ──
# Uses model.generate() directly — no pipeline, no config conflicts.
# Expect ~1-2 hours for ~40 000 clips on GPU T4 x2.

import numpy as np

BATCH_SIZE = 16
CHECKPOINT_FILE = "/kaggle/working/submission_checkpoint.csv"

def decode_audio_bytes(audio_field):
    """Decode parquet audio bytes → 16 kHz float32 numpy array."""
    raw = audio_field["bytes"] if isinstance(audio_field, dict) else bytes(audio_field)
    try:
        arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != 16_000:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16_000)
        return arr
    except Exception:
        pass
    from pydub import AudioSegment
    seg = AudioSegment.from_file(io.BytesIO(raw)).set_frame_rate(16_000).set_channels(1)
    return np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0

def transcribe_batch(arrays):
    """Transcribe a list of 16 kHz float32 arrays using model.generate() directly."""
    # Truncate to 30 s max — avoids the "> 3000 mel features" error
    arrays = [arr[:480_000] for arr in arrays]
    inputs = processor(
        arrays, sampling_rate=16_000, return_tensors="pt",
    ).input_features.to(device).to(dtype)
    with torch.no_grad():
        ids = model.generate(input_features=inputs, max_new_tokens=225)
    return processor.batch_decode(ids, skip_special_tokens=True)

# ── Resume from checkpoint (auto-deletes if corrupted from a previous bad run) ─
if os.path.exists(CHECKPOINT_FILE):
    existing = pd.read_csv(CHECKPOINT_FILE)
    empty_pct = (existing["transcription"].isna() | (existing["transcription"].str.strip() == "")).mean()
    if empty_pct > 0.5:
        os.remove(CHECKPOINT_FILE)
        print(f"Deleted corrupted checkpoint ({empty_pct:.0%} empty transcriptions). Starting fresh.")
        results, done_ids = [], set()
    else:
        results  = existing.to_dict("records")
        done_ids = set(existing["id"])
        print(f"Resuming from checkpoint: {len(results)} clips already done.")
else:
    results, done_ids = [], set()
    print("Starting fresh.")

all_parquet_files = sorted(
    glob.glob(os.path.join(test_path, "**", "*.parquet"), recursive=True)
)
print(f"{len(all_parquet_files)} parquet files total. Batch size: {BATCH_SIZE}.\n")

for pf_idx, pq_file in enumerate(all_parquet_files):
    df = pd.read_parquet(pq_file)
    df = df[~df["id"].isin(done_ids)]
    if len(df) == 0:
        print(f"[{pf_idx+1}/{len(all_parquet_files)}] already done — skip")
        continue
    lang = df["language"].iloc[0]
    print(f"[{pf_idx+1}/{len(all_parquet_files)}] {os.path.basename(pq_file)}  "
          f"lang={lang}  rows={len(df)}", flush=True)

    rows = list(df.itertuples(index=False))
    for i in range(0, len(rows), BATCH_SIZE):
        chunk = rows[i : i + BATCH_SIZE]
        arrays, batch_ids, batch_langs = [], [], []
        for row in chunk:
            try:
                arrays.append(decode_audio_bytes(row.audio))
                batch_ids.append(row.id)
                batch_langs.append(row.language)
            except Exception as e:
                print(f"  DECODE ERROR {row.id}: {e}")
                results.append({"id": row.id, "language": row.language, "transcription": "."})
                done_ids.add(row.id)

        if arrays:
            try:
                texts = transcribe_batch(arrays)
                for id_, lang_, text in zip(batch_ids, batch_langs, texts):
                    results.append({"id": id_, "language": lang_, "transcription": text.strip() or "."})
                    done_ids.add(id_)
            except Exception as e:
                print(f"  BATCH ERROR ({e}) — one-by-one")
                for id_, lang_, arr in zip(batch_ids, batch_langs, arrays):
                    try:
                        text = transcribe_batch([arr])[0].strip() or "."
                    except Exception as e2:
                        print(f"    FAILED {id_}: {e2}")
                        text = "."
                    results.append({"id": id_, "language": lang_, "transcription": text})
                    done_ids.add(id_)

    pd.DataFrame(results).to_csv(CHECKPOINT_FILE, index=False)
    print(f"  -> checkpoint saved ({len(results)} total)", flush=True)

print(f"\nDone. Total: {len(results)}")


# ── CELL 7 — Build submission file ──────────────────────────────────────────
# Submission format: id, language, transcription  (3 columns, exactly)

# Safety: if the kernel reset after Cell 6 saved the checkpoint, load from disk
if not results and os.path.exists(CHECKPOINT_FILE):
    print(f"Loading results from checkpoint: {CHECKPOINT_FILE}")
    submission_df = pd.read_csv(CHECKPOINT_FILE)
else:
    submission_df = pd.DataFrame(results)

# Replace any remaining null/empty with "." so Kaggle won't reject the file
mask = submission_df["transcription"].isna() | (submission_df["transcription"].str.strip() == "")
if mask.sum() > 0:
    print(f"Replacing {mask.sum()} null/empty rows with '.'")
    submission_df.loc[mask, "transcription"] = "."

# Add language column if missing (checkpoint saved before language was tracked)
if "language" not in submission_df.columns:
    print("Building id→language map from parquet files...")
    lang_map = {}
    for pq_file in sorted(glob.glob(os.path.join(test_path, "**", "*.parquet"), recursive=True)):
        df_tmp = pd.read_parquet(pq_file, columns=["id", "language"])
        lang_map.update(dict(zip(df_tmp["id"], df_tmp["language"])))
    submission_df["language"] = submission_df["id"].map(lang_map)
    print(f"Language column added. Missing: {submission_df['language'].isna().sum()}")

# Reorder to required format: id, language, transcription
submission_df = submission_df[["id", "language", "transcription"]]
submission_df.to_csv("submission.csv", index=False)

# Verify — all three must look clean before submitting
sub = pd.read_csv("submission.csv")
print(f"NaN transcription: {sub['transcription'].isna().sum()}")
print(f"Empty transcription: {(sub['transcription'].str.strip() == '').sum()}")
print(f"NaN language: {sub['language'].isna().sum()}")
print(sub.head())
print(f"\nWrote submission.csv with {len(sub)} rows.")
