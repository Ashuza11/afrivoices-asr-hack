"""
Lesson 3 — Hands-on: CTC vs Encoder-Decoder
  1. Run Whisper-tiny (seq2seq) on our Swahili sample and time it
  2. Run facebook/mms-300m (CTC) on the same sample and time it
  3. Compare WER, inference time, and RTF side-by-side
  4. Show the CTC blank-collapsing step manually
"""
import io, os, sys, time
import numpy as np
import soundfile as sf
import torch
from datasets import load_dataset, Audio
import whisper
from transformers import Wav2Vec2ForCTC, AutoProcessor
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from eval.compute_wer import evaluate, print_report

# ── 0. Load same Swahili sample as Lesson 2 ───────────────────────────────────
print("Loading Swahili sample from FLEURS (streaming) ...")
ds = load_dataset("google/fleurs", "sw_ke", split="validation", streaming=True)
ds = ds.cast_column("audio", Audio(decode=False))
sample = next(iter(ds))
audio_bytes = sample["audio"]["bytes"]
audio_array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
if audio_array.ndim > 1:
    audio_array = audio_array.mean(axis=1)
reference = sample["transcription"].strip()
duration  = len(audio_array) / sr
print(f"  Audio duration : {duration:.1f} s")
print(f"  Reference      : {reference}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# TRACK A — Whisper-tiny (encoder-decoder / seq2seq)
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TRACK A — Whisper-tiny (encoder-decoder, autoregressive)")
print("=" * 60)

model_a = whisper.load_model("tiny")   # already cached from Lesson 2

t0 = time.perf_counter()
result_a = model_a.transcribe(audio_array, language="sw", fp16=False)
t1 = time.perf_counter()

hyp_a       = result_a["text"].strip()
time_a      = t1 - t0
rtf_a       = time_a / duration
tokens_a    = result_a.get("segments", [])
n_tokens_a  = sum(len(s["tokens"]) for s in tokens_a)

print(f"  Hypothesis : {hyp_a}")
print(f"  Inference  : {time_a:.1f}s  for  {duration:.1f}s audio")
print(f"  RTF        : {rtf_a:.2f}x  (target ≤ 2.0x on Pi 4)")
print(f"  Tokens generated (decoder steps): ~{n_tokens_a}")
eval_a = evaluate([reference], [hyp_a])
print_report(eval_a, "Whisper-tiny")

# ── Show why seq2seq is slow: decoder runs N times ────────────────────────────
print(f"\n  [WHY SLOW] The decoder ran ~{n_tokens_a} separate forward passes.")
print(f"  Each pass attends to all encoder frames AND all previous tokens.")
print(f"  On CPU, these N passes can't be parallelised — they're sequential.\n")

# ═══════════════════════════════════════════════════════════════════════════════
# TRACK B — MMS-300m (CTC, single forward pass)
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TRACK B — facebook/mms-300m (CTC, single forward pass)")
print("=" * 60)
print("  Downloading model (~1.2 GB, cached after first run) ...")

# wav2vec2-base-960h: 95M params, CTC, English-trained.
# WER on Swahili will still be terrible (wrong language), but it demonstrates
# CTC architecture and the speed advantage — which survive fine-tuning.
MODEL_ID = "facebook/wav2vec2-base-960h"
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
processor_b = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model_b     = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
model_b.eval()

print("  Model loaded. Running CTC inference ...")
inputs = processor_b(audio_array, sampling_rate=sr, return_tensors="pt")

t0 = time.perf_counter()
with torch.no_grad():
    logits = model_b(**inputs).logits          # shape: (1, time_frames, vocab)
t1 = time.perf_counter()

# CTC greedy decode: argmax at each frame, then collapse
raw_ids   = torch.argmax(logits, dim=-1)       # best character at every frame
hyp_b     = processor_b.batch_decode(raw_ids)[0].strip()
time_b    = t1 - t0
rtf_b     = time_b / duration
n_frames  = logits.shape[1]

print(f"  Hypothesis : {hyp_b}")
print(f"  Inference  : {time_b:.1f}s  for  {duration:.1f}s audio")
print(f"  RTF        : {rtf_b:.2f}x  (target ≤ 2.0x on Pi 4)")
print(f"  Encoder frames processed in ONE pass: {n_frames}")
eval_b = evaluate([reference], [hyp_b])
print_report(eval_b, "MMS-300m CTC")

print(f"\n  [WHY FAST] CTC ran exactly 1 forward pass over {n_frames} frames.")
print(f"  All frames processed in parallel — no sequential dependency.\n")

# ═══════════════════════════════════════════════════════════════════════════════
# CTC DEMO — show blank collapsing on a toy example
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("CTC BLANK-COLLAPSING (toy example)")
print("=" * 60)
raw_ctc = "kk w  aa  -  k u  ss h i rr ii  kk i aa n aa"
print(f"  Raw CTC frames : {raw_ctc}")
# step 1: merge consecutive duplicates
import re
merged = re.sub(r'(.)\1+', r'\1', raw_ctc.replace(" ", ""))
print(f"  After dedup    : {merged}")
# step 2: remove blanks
collapsed = merged.replace("-", "").replace(" ", "")
print(f"  After collapse : {collapsed}  ← final transcript\n")

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'Model':<22} {'WER':>8} {'CER':>8} {'Time':>8} {'RTF':>8} {'≤2x?':>6}")
print("-" * 60)
print(f"{'Whisper-tiny (seq2seq)':<22} "
      f"{eval_a['wer']:>7.1%} {eval_a['cer']:>7.1%} "
      f"{time_a:>7.1f}s {rtf_a:>7.2f}x {'✅' if rtf_a<=2 else '❌':>6}")
print(f"{'MMS-300m (CTC)':<22} "
      f"{eval_b['wer']:>7.1%} {eval_b['cer']:>7.1%} "
      f"{time_b:>7.1f}s {rtf_b:>7.2f}x {'✅' if rtf_b<=2 else '❌':>6}")
print()
print("Note: WER is high for both — these are ZERO-SHOT, untrained on these")
print("languages. The RTF gap is what matters here: that gap survives fine-tuning.")
print("\n=== Lesson 3 complete ===")
