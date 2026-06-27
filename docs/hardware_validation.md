---
layout: default
title: Hardware Validation Report
---

# Hardware Validation Report

*Required by AfriVoices East Africa ASR Hackathon rules: "Submissions must include a hardware validation report showing inference latency for all the test set."*

**Model:** [Ash11/afrivoices-whisper-small-all6](https://huggingface.co/Ash11/afrivoices-whisper-small-all6)  
**Format tested:** CTranslate2 int8 quantized (CPU-optimized)  
**Date:** *to be completed before July 12, 2026*

---

## Hardware Specifications

### Primary validation platform (Raspberry Pi 4 proxy)

| Spec | Value |
|---|---|
| Device | Development machine via WSL2 (Intel CPU, Pi 4 proxy) |
| CPU | *to be filled* |
| Cores used | 4 (matching Pi 4's Cortex-A72 quad-core) |
| RAM available | *to be filled* |
| OS | Ubuntu 22.04 (WSL2) |
| GPU | None (CPU-only inference) |

### Target platform

| Spec | Value |
|---|---|
| Device | Raspberry Pi 4 Model B |
| CPU | Broadcom BCM2711, Cortex-A72 @ 1.8 GHz, 4 cores |
| RAM | 4 GB LPDDR4 |
| OS | Raspberry Pi OS (64-bit) |

---

## Model Specifications

| Metric | Value |
|---|---|
| Architecture | Whisper-small (encoder-decoder) |
| Total parameters | 244M |
| Original size (fp16 safetensors) | ~967 MB |
| Quantized size (CTranslate2 int8) | *to be measured* |
| Peak RAM during inference | *to be measured* |

---

## Inference Latency Results

### Full Test Set (41,733 clips, 94 parquet files)

| Metric | Value |
|---|---|
| Total audio duration (estimated) | *to be measured* |
| Total inference time (CPU, int8) | *to be measured* |
| **Real-Time Factor (RTF)** | *to be measured* |
| Peak RAM usage | *to be measured* |
| Passes RTF ≤ 2× constraint | *to be confirmed* |
| Passes RAM ≤ 8 GB constraint | *to be confirmed* |

### Per-Language Latency Breakdown

| Language | Clips | Avg clip duration | Avg inference time/clip | RTF |
|---|---|---|---|---|
| swa | 12,553 | *pending* | *pending* | *pending* |
| kik | 9,192 | *pending* | *pending* | *pending* |
| luo | 7,437 | *pending* | *pending* | *pending* |
| kln | 4,837 | *pending* | *pending* | *pending* |
| som | 3,925 | *pending* | *pending* | *pending* |
| mas | 3,789 | *pending* | *pending* | *pending* |

---

## Quantization Process

```bash
# Install CTranslate2
pip install ctranslate2

# Convert fine-tuned model to int8 CTranslate2 format
ct2-whisper-converter \
  --model Ash11/afrivoices-whisper-small-all6 \
  --output_dir afrivoices-whisper-small-ct2-int8 \
  --quantization int8 \
  --force

# Verify model size
du -sh afrivoices-whisper-small-ct2-int8/
```

## Inference Benchmark Script

```python
import time
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

model = WhisperModel(
    "afrivoices-whisper-small-ct2-int8",
    device="cpu",
    compute_type="int8",
    cpu_threads=4,   # match Raspberry Pi 4 core count
    num_workers=1,
)

def benchmark_clip(audio_array, sample_rate=16000):
    t0 = time.perf_counter()
    segments, _ = model.transcribe(audio_array, beam_size=1)
    text = " ".join(s.text for s in segments)
    elapsed = time.perf_counter() - t0
    audio_duration = len(audio_array) / sample_rate
    rtf = elapsed / audio_duration
    return text, elapsed, rtf

# Run over full test set and collect RTF stats
# ... (full benchmark script at bench/run_cpu_benchmark.py)
```

---

## WER vs Latency Trade-off (Quantization Impact)

| Model format | WER | RTF (CPU) | RAM |
|---|---|---|---|
| fp16 (GPU, training) | 0.89330 | N/A | ~2 GB VRAM |
| fp32 (CPU, unquantized) | *to be measured* | *to be measured* | *to be measured* |
| int8 (CPU, CTranslate2) | *to be measured* | *to be measured* | *to be measured* |

---

## Compliance Summary

| Competition Requirement | Status |
|---|---|
| Parameters < 1B | ✅ 244M |
| CPU-only inference | ✅ CTranslate2 int8, no GPU |
| RAM ≤ 8 GB | *to be confirmed* |
| RTF ≤ 2× on Pi 4-class hardware | *to be confirmed* |
| Offline (no cloud dependency) | ✅ model runs fully locally |
| Apache-2.0 license | ✅ |
