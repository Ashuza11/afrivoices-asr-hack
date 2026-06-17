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


# ── CELL 6 — Transcribe every test audio (audio embedded in parquet bytes) ──
# The test set stores audio as bytes inside each parquet row — no separate
# audio files exist on disk. We read every parquet, decode the 'audio' column
# bytes with soundfile, and transcribe via HF pipeline which handles clips
# longer than 30 sec by chunking automatically.
# Expect ~2-4 hours for ~40 000 clips on GPU T4 x2.

from transformers import pipeline as hf_pipeline

pipe = hf_pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=0 if torch.cuda.is_available() else -1,
    chunk_length_s=30,   # clips can be 3+ minutes — split into 30 s chunks
    stride_length_s=5,   # 5 s overlap between chunks for smoother joins
)

def decode_audio_bytes(audio_field):
    """Convert parquet audio field (dict with 'bytes') to numpy array at 16 kHz.
    Primary: soundfile (WAV/FLAC). Fallback: pydub via ffmpeg (webm/mp3/opus/aac).
    """
    import numpy as np
    raw = audio_field["bytes"] if isinstance(audio_field, dict) else bytes(audio_field)
    try:
        arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != 16_000:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16_000)
        return {"array": arr, "sampling_rate": 16_000}
    except Exception:
        pass
    from pydub import AudioSegment
    audio = AudioSegment.from_file(io.BytesIO(raw)).set_frame_rate(16_000).set_channels(1)
    arr = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    return {"array": arr, "sampling_rate": 16_000}

all_parquet_files = sorted(
    glob.glob(os.path.join(test_path, "**", "*.parquet"), recursive=True)
)
print(f"Transcribing {len(all_parquet_files)} parquet files (~40 000 clips total).")
print("This will take 2-4 hours on GPU. Progress is printed per file.\n")

results = []
for pf_idx, pq_file in enumerate(all_parquet_files):
    df = pd.read_parquet(pq_file)
    lang = df["language"].iloc[0] if len(df) > 0 else "?"
    print(f"[{pf_idx+1}/{len(all_parquet_files)}] {os.path.basename(pq_file)}  "
          f"lang={lang}  rows={len(df)}", flush=True)
    for _, row in df.iterrows():
        try:
            audio_input = decode_audio_bytes(row["audio"])
            hyp = pipe(audio_input)["text"].strip()
        except Exception as e:
            hyp = ""
            print(f"  ERROR {row['id']}: {e}")
        results.append({"id": row["id"], "transcription": hyp})
    print(f"  -> cumulative: {len(results)} transcriptions", flush=True)

print(f"\nTotal transcriptions: {len(results)}")


# ── CELL 7 — Build submission file ──────────────────────────────────────────
# Confirmed column names from test parquet inspection:
#   id           — wav filename, e.g. "QeUUZNkacY_10Jun2025...wav"
#   transcription — our predicted text
submission_df = pd.DataFrame(results)
submission_df.to_csv("submission.csv", index=False)
print(submission_df.head())
print(f"\nWrote submission.csv with {len(submission_df)} rows.")
print("Go to the competition page -> 'Submit Prediction' and upload submission.csv")
