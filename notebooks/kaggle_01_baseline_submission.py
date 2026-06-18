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
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
print(f"Loading {MODEL_ID} on {device} ({dtype}) ...")

processor = WhisperProcessor.from_pretrained(MODEL_ID)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
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
    generate_kwargs={"task": "transcribe"},  # transcribe in source language, not translate to English
    # return_timestamps=True is intentionally NOT set here — it breaks batching
    # by forcing single-clip long-form mode. Test clips are ≤30s so no truncation risk.
)

BATCH_SIZE = 16  # clips processed in parallel on GPU — ~5-10x faster than one-by-one
CHECKPOINT_FILE = "/kaggle/working/submission_checkpoint.csv"

def decode_audio_bytes(audio_field):
    """Decode parquet audio bytes → 16 kHz numpy array.
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

# ── Resume from checkpoint if the session was interrupted ───────────────────
if os.path.exists(CHECKPOINT_FILE):
    existing = pd.read_csv(CHECKPOINT_FILE)
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
    df = df[~df["id"].isin(done_ids)]      # skip rows already transcribed
    if len(df) == 0:
        print(f"[{pf_idx+1}/{len(all_parquet_files)}] already done — skip")
        continue
    lang = df["language"].iloc[0]
    print(f"[{pf_idx+1}/{len(all_parquet_files)}] {os.path.basename(pq_file)}  "
          f"lang={lang}  rows={len(df)}", flush=True)

    rows = list(df.itertuples(index=False))
    for i in range(0, len(rows), BATCH_SIZE):
        chunk = rows[i : i + BATCH_SIZE]
        batch_audio, batch_ids = [], []
        for row in chunk:
            try:
                batch_audio.append(decode_audio_bytes(row.audio))
                batch_ids.append(row.id)
            except Exception as e:
                print(f"  DECODE ERROR {row.id}: {e}")
                results.append({"id": row.id, "transcription": ""})
                done_ids.add(row.id)

        if batch_audio:
            try:
                outputs = pipe(batch_audio, batch_size=len(batch_audio))
                for id_, out in zip(batch_ids, outputs):
                    results.append({"id": id_, "transcription": out["text"].strip()})
                    done_ids.add(id_)
            except Exception as e:
                # batch failed — fall back to one-by-one for this chunk
                print(f"  BATCH ERROR ({e}) — falling back to one-by-one")
                for id_, audio in zip(batch_ids, batch_audio):
                    try:
                        results.append({"id": id_, "transcription": pipe(audio)["text"].strip()})
                    except Exception:
                        results.append({"id": id_, "transcription": ""})
                    done_ids.add(id_)

    # Save after every parquet file — if session dies, resume from here
    pd.DataFrame(results).to_csv(CHECKPOINT_FILE, index=False)
    print(f"  -> checkpoint saved ({len(results)} total)", flush=True)

print(f"\nDone. Total: {len(results)}")


# ── CELL 7 — Build submission file ──────────────────────────────────────────
# Confirmed column names from test parquet inspection:
#   id           — wav filename, e.g. "QeUUZNkacY_10Jun2025...wav"
#   transcription — our predicted text

# Safety: if the kernel reset after Cell 6 saved the checkpoint, load from disk
if not results and os.path.exists(CHECKPOINT_FILE):
    print(f"Loading results from checkpoint file: {CHECKPOINT_FILE}")
    submission_df = pd.read_csv(CHECKPOINT_FILE)
else:
    submission_df = pd.DataFrame(results)

submission_df.to_csv("submission.csv", index=False)
print(submission_df.head())
print(f"\nWrote submission.csv with {len(submission_df)} rows.")
print("Go to the competition page -> 'Submit Prediction' and upload submission.csv")
