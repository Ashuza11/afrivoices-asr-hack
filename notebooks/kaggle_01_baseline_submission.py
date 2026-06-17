"""
Kaggle Notebook 1 — Zero-shot baseline submission
==================================================
Goal: get ONE valid submission on the leaderboard today, with zero training.
This validates the submission pipeline end-to-end before we invest in the
heavy multilingual fine-tuning work (Notebook 2).

HOW TO USE ON KAGGLE:
  1. kaggle.com -> Create -> New Notebook
  2. Settings (right sidebar): Accelerator = GPU T4 x2 (or P100), Internet = ON
  3. Add-ons -> Secrets -> add HF_TOKEN (your Hugging Face token, read access)
  4. Copy each "# ── CELL n ──" block below into its own notebook cell, in order
  5. Run all

Each cell is self-contained and prints what it found — read the printed
output before moving to the next cell, since we don't yet know the exact
test-set file layout or submission schema.
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
print(f"Loading {MODEL_ID} on {device} ...")

processor = WhisperProcessor.from_pretrained(MODEL_ID)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
model.eval()
print("Model loaded.")


# ── CELL 6 — Transcribe every test audio file ───────────────────────────────
# NOTE: we do NOT force a language — this is a unified multilingual model,
# the test set gives no language label, so we let Whisper auto-detect.
# Swahili is in Whisper's pretraining; the other 5 languages are not, so
# expect high WER on those — that's exactly the gap fine-tuning will close.

def transcribe(audio_path_or_array, sr=None):
    if isinstance(audio_path_or_array, str):
        audio_array, sr = sf.read(audio_path_or_array, dtype="float32")
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
    else:
        audio_array = audio_path_or_array

    inputs = processor(audio_array, sampling_rate=16_000, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    with torch.no_grad():
        predicted_ids = model.generate(input_features, max_new_tokens=225)
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return text.strip()

results = []
if test_df is not None:
    # Adjust these column-name guesses once Cell 4's output tells you the real names
    id_col = next((c for c in test_df.columns if "id" in c.lower() or "path" in c.lower()), test_df.columns[0])
    print(f"Using '{id_col}' as the file/id column. Verify this is correct!")

    for i, row in test_df.iterrows():
        file_ref = row[id_col]
        # try to resolve to an actual file on disk
        matches = [f for f in audio_files if os.path.basename(f).startswith(str(file_ref))
                   or str(file_ref) in f]
        if not matches:
            print(f"  [{i}] WARNING: no audio file found for '{file_ref}' — skipping")
            continue
        hyp = transcribe(matches[0])
        results.append({"id": file_ref, "transcription": hyp})
        if i < 3:
            print(f"  [{i}] {file_ref} -> {hyp}")
else:
    # fallback: just transcribe every audio file found directly
    for f in audio_files:
        hyp = transcribe(f)
        results.append({"id": os.path.basename(f), "transcription": hyp})

print(f"\nTotal transcriptions: {len(results)}")


# ── CELL 7 — Build submission file ──────────────────────────────────────────
# TODO: confirm exact column names against sample_submission.csv on Kaggle's
# "Data" tab before submitting — this is a best guess (id, transcription).
submission_df = pd.DataFrame(results)
submission_df.to_csv("submission.csv", index=False)
print(submission_df.head())
print(f"\nWrote submission.csv with {len(submission_df)} rows.")
print("Go to the competition page -> 'Submit Prediction' and upload submission.csv")
