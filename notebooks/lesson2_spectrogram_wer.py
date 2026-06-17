"""
Lesson 2 — Hands-on: Log-Mel Spectrograms + First WER
  1. Stream one Swahili audio clip from FLEURS (no full download needed)
  2. Visualize: raw waveform vs. log-mel spectrogram side-by-side
  3. Run Whisper-tiny zero-shot
  4. Compute WER with jiwer
"""
import io
import os
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")           # no display needed — saves to PNG
import matplotlib.pyplot as plt
import whisper
from jiwer import wer
from datasets import load_dataset, Audio

os.makedirs("data", exist_ok=True)

# ── 1. Load one Swahili clip from FLEURS (streaming = no bulk download) ────────
print("Step 1 — Streaming one Swahili sample from google/fleurs ...")
ds = load_dataset("google/fleurs", "sw_ke", split="validation", streaming=True)
# Keep audio as raw bytes (avoids the torchcodec dependency introduced in datasets 3.x)
ds = ds.cast_column("audio", Audio(decode=False))
sample = next(iter(ds))

# Decode audio bytes with soundfile
audio_bytes = sample["audio"]["bytes"]
audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
if audio_array.ndim > 1:            # convert stereo → mono
    audio_array = audio_array.mean(axis=1)
reference_text = sample["transcription"].strip()
duration = len(audio_array) / sample_rate

print(f"  Duration      : {duration:.1f} s")
print(f"  Sample rate   : {sample_rate} Hz")
print(f"  # of samples  : {len(audio_array):,}  (one number per 1/{sample_rate}s)")
print(f"  Reference     : {reference_text}\n")

# ── 2. Save the audio so Whisper can read it ───────────────────────────────────
audio_path = "data/swahili_sample.wav"
sf.write(audio_path, audio_array, sample_rate)
print(f"Step 2 — Audio saved to {audio_path}\n")

# ── 3. Compute log-mel spectrogram ─────────────────────────────────────────────
print("Step 3 — Computing log-mel spectrogram ...")
#   n_fft=400   → 25 ms window  (400 samples at 16 kHz = 0.025 s)
#   hop_length=160 → 10 ms step (160 samples at 16 kHz = 0.010 s)
#   n_mels=80   → same as Whisper
mel = librosa.feature.melspectrogram(
    y=audio_array, sr=sample_rate,
    n_mels=80, n_fft=400, hop_length=160,
    fmin=0, fmax=8000,
)
log_mel = librosa.power_to_db(mel, ref=np.max)   # convert energy → decibels

print(f"  log_mel shape : {log_mel.shape}  (mel_bands × time_frames)")
print(f"  = {log_mel.shape[0]} frequency bands × {log_mel.shape[1]} frames")
print(f"  Each frame covers {160/sample_rate*1000:.0f} ms of audio\n")

# ── 4. Plot waveform + spectrogram side-by-side ───────────────────────────────
print("Step 4 — Plotting waveform + spectrogram ...")
fig, axes = plt.subplots(2, 1, figsize=(13, 8))

time_axis = np.linspace(0, duration, len(audio_array))
axes[0].plot(time_axis, audio_array, linewidth=0.4, color="steelblue")
axes[0].set_title("Raw Waveform  —  what the microphone records (amplitude over time)")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Amplitude")
axes[0].set_xlim(0, duration)

img = librosa.display.specshow(
    log_mel, sr=sample_rate, hop_length=160,
    x_axis="time", y_axis="mel", fmax=8000,
    ax=axes[1], cmap="magma",
)
axes[1].set_title("Log-Mel Spectrogram  —  what the ASR model sees (80 mel bands)")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Mel frequency (Hz)")
fig.colorbar(img, ax=axes[1], format="%+2.0f dB", label="Energy (dB)")

plt.suptitle(f'Swahili FLEURS sample  |  {duration:.1f}s  |  reference: "{reference_text[:60]}..."',
             fontsize=9, y=1.01)
plt.tight_layout()
spectrogram_path = "data/spectrogram.png"
plt.savefig(spectrogram_path, dpi=150, bbox_inches="tight")
print(f"  Saved → {spectrogram_path}")
print( "  Open with:  explorer.exe data/spectrogram.png\n")

# ── 5. Run Whisper-tiny zero-shot ──────────────────────────────────────────────
print("Step 5 — Loading Whisper-tiny (39M params) ...")
model = whisper.load_model("tiny")
print("  Running zero-shot transcription (no fine-tuning yet) ...")
# Pass numpy array directly — avoids the ffmpeg dependency for loading
# Whisper expects float32 audio at 16 kHz, which is exactly what FLEURS gives us
result = model.transcribe(audio_array, language="sw", fp16=False)
hypothesis = result["text"].strip()

print(f"\n  Whisper-tiny output : {hypothesis}")
print(f"  Reference text      : {reference_text}")

# ── 6. Compute WER ────────────────────────────────────────────────────────────
error = wer(reference_text.lower(), hypothesis.lower())
print(f"\n  WER = {error:.1%}")
print()
if error > 0.8:
    print("  High WER is expected — this is zero-shot (untrained on these languages).")
    print("  Fine-tuning will drop this dramatically, often from 80-100% → 20-40%.")
elif error > 0.4:
    print("  Moderate WER — Whisper has seen some Swahili in pretraining but not enough.")
else:
    print("  Surprisingly low — Whisper's Swahili pretraining data helped here.")

print("\n=== Lesson 2 complete ===")
print("Key takeaway: the model never sees raw audio. It sees the log-mel spectrogram.")
print("Fine-tuning teaches it to map THOSE patterns to the target language's text.")
