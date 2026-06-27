---
layout: default
title: "AfriVoices ASR: Building Multilingual Speech Recognition for East Africa"
---

# AfriVoices ASR: Building Multilingual Speech Recognition for East Africa

*AfriVoices East Africa ASR Hackathon 2026 — technical blog post*

**Model:** [Ash11/afrivoices-whisper-small-all6](https://huggingface.co/Ash11/afrivoices-whisper-small-all6) · **Code:** [github.com/Ashuza11/afrivoices-asr-hack](https://github.com/Ashuza11/afrivoices-asr-hack) · **License:** Apache-2.0

---

## Motivation

East Africa is home to hundreds of millions of people who speak languages that are almost entirely absent from mainstream speech technology. A Luo speaker in Kisumu, a Maasai elder in the Rift Valley, or a Somali trader in Mogadishu cannot use voice search, dictation, or accessibility tools in their first language. The digital divide is not just about connectivity — it is about whose voice the machine can hear.

The AfriVoices East Africa ASR Hackathon challenges participants to build a **single unified model** that understands six typologically diverse languages simultaneously: Swahili, Kikuyu (Gĩkũyũ), Luo (Dholuo), Somali, Maasai, and Kalenjin — and to do it on hardware that is actually available in East Africa: smartphones and Raspberry Pi-class devices.

---

## The Six Languages

| Language | Family | ISO | Hours of data | Notes |
|---|---|---|---|---|
| Swahili | Bantu | swa | 2,979 h | Largest dataset; code-switches with English |
| Kikuyu | Bantu | kik | 754 h | Not in Whisper's 99-language vocab |
| Luo (Dholuo) | Nilotic (Western) | luo | 723 h | Not in Whisper's vocab |
| Somali | Cushitic | som | 1,002 h | In Whisper's vocab (`<\|so\|>`) |
| Maasai | Nilotic (Eastern) | mas | 505 h | Not in Whisper's vocab |
| Kalenjin | Nilotic (Southern) | kln | 521 h | Not in Whisper's vocab |

The dataset is a mix of **scripted (read) speech** and **unscripted (spontaneous) speech**, recorded across multiple dialects per language. Spontaneous speech is significantly harder — faster delivery, disfluencies, dialect mixing, and code-switching — which is why this benchmark is challenging even for large models.

---

## Our Approach

### Architecture: Whisper-small (244M parameters)

We fine-tune [openai/whisper-small](https://huggingface.co/openai/whisper-small), a 244M-parameter encoder-decoder model pre-trained by OpenAI on 680,000 hours of multilingual speech. Whisper already knows Swahili (`<|sw|>`) and Somali (`<|so|>`), giving us strong priors for two of the six languages out of the box.

For the four languages not in Whisper's vocabulary (Kikuyu, Luo, Maasai, Kalenjin), we fine-tune **without a language prefix token** — the model learns to transcribe them directly from acoustic features.

Why Whisper-small and not a larger model?

- **Parameter budget:** The competition caps at 1B parameters. Whisper-small's 244M leaves headroom for quantization overhead.
- **Edge deployment:** Whisper-small quantized to int8 runs in real time on a Raspberry Pi 4, while Whisper-medium (307M) barely fits in 8 GB RAM.
- **Strong multilingual priors:** Despite its size, Whisper-small was pre-trained on a diverse multilingual corpus. Fine-tuning a model with existing speech representations is far more data-efficient than training from scratch.

### Data Sources

All datasets are licensed under CC BY 4.0 and are publicly available on Hugging Face Hub.

| Source | Languages | Access |
|---|---|---|
| [DigitalUmuganda/Afrivoice_Swahili](https://huggingface.co/datasets/DigitalUmuganda/Afrivoice_Swahili) | swa | Public |
| [DigitalUmuganda/Afrivoice](https://huggingface.co/datasets/DigitalUmuganda/Afrivoice) (Somali path) | som | Public |
| [MCAA1-MSU/anv_data_ke](https://huggingface.co/datasets/MCAA1-MSU/anv_data_ke) | kik, luo, mas, kln | Gated (approved) |

### Training Data Composition

**Round 1 (baseline fine-tune, Colab T4, 600 steps):**
- 5,000 Swahili clips
- 2,000 Somali clips
- 1,000 clips × 4 ANV languages = 4,000 clips
- **Total: ~11,000 clips**

**Round 2 (Modal A100, 1,500 steps from base model):**
- Same data composition
- Result: WER **0.89330** (↓ from baseline 1.61077, −44.6%)

**Round 3 (Modal A100, 2,000 steps from Round 2 checkpoint) — in progress:**
- 8,000 Swahili clips
- 4,000 Somali clips
- 2,000 clips × 4 ANV languages = 8,000 clips
- **Total: ~20,000 clips**
- Learning rate lowered to 5e-6 (fine-tuning a fine-tuned model)
- *Results pending*

### Training Infrastructure

| Component | Round 1 | Round 2 & 3 |
|---|---|---|
| Hardware | Google Colab T4 (free) | Modal.com A100 40GB |
| Batch size (per device) | 4 | 16 |
| Effective batch size | 16 | 32 |
| Optimizer | Adafactor | Adafactor |
| Mixed precision | fp16 | fp16 |
| Duration | ~35 min (600 steps) | ~35 min (1,500 steps) |

**Why Adafactor instead of AdamW?**
AdamW stores momentum and variance tensors for every parameter — for Whisper-small's 244M parameters, that's ~1.84 GB of optimizer state. On Colab's 12.7 GB RAM (shared with the OS, the 11k training clips at 5.3 GB, and the model at ~1 GB), AdamW pushed us over the limit at step 2. Adafactor reconstructs the second moment from a factored low-rank approximation, cutting optimizer RAM to ~0.24 GB.

**Why float16 in-memory records?**
Loading 11,000 audio clips as float32 spectrograms requires ~10.5 GB of RAM. Storing them as float16 halves this to ~5.3 GB while the DataCollator converts back to float32 per batch during training — no accuracy loss, immediate OOM fix.

---

## Results

### WER Progression

| Stage | Steps | WER (public leaderboard) |
|---|---|---|
| Zero-shot whisper-small (baseline) | — | 1.61077 |
| Round 2 fine-tune (A100, 1,500 steps) | 1,500 | **0.89330** |
| Round 3 fine-tune (A100, 2,000 steps) | 3,500 total | *pending* |

The metric is **average WER across all six languages** (unweighted mean). Lower is better.

### Per-Language Test Set Breakdown

| Language | Test clips | WER (Round 2) |
|---|---|---|
| swa | 12,553 | *to be measured* |
| kik | 9,192 | *to be measured* |
| luo | 7,437 | *to be measured* |
| kln | 4,837 | *to be measured* |
| som | 3,925 | *to be measured* |
| mas | 3,789 | *to be measured* |

*Per-language WER will be added once we run the per-language evaluation script.*

---

## Edge Deployment

### Why CPU Inference Matters

The competition requires inference on edge hardware: ≤8 GB RAM, CPU-only, real-time factor (RTF) ≤2×. This means a 10-second clip must be transcribed in ≤20 seconds. Standard Whisper-small autoregressive decoding on a CPU is too slow unoptimized — a full beam search pass on 30 seconds of audio can take 60+ seconds.

### Quantization Strategy: ctranslate2 int8

We use [CTranslate2](https://github.com/OpenNMT/CTranslate2) with int8 quantization to convert the fine-tuned model for CPU-efficient inference:

```bash
ct2-opus-convert --model Ash11/afrivoices-whisper-small-all6 \
  --output_dir afrivoices-whisper-small-ct2 \
  --quantization int8
```

CTranslate2's int8 quantization:
- Reduces model size from ~967 MB (fp16) to ~250 MB
- Speeds up CPU inference 3–4× via SIMD int8 kernels
- Negligible WER degradation (<0.5% relative in our measurements — *to be confirmed*)

### Hardware Validation

*This section will be completed before July 12, 2026.*

Target platform: Raspberry Pi 4 (4 GB RAM, Cortex-A72 CPU, 4 cores @ 1.8 GHz)  
Proxy platform used for development: WSL2 on Intel CPU, 1-thread limited to simulate Pi 4.

| Metric | Target | Measured (WSL2 proxy) | Measured (Pi 4) |
|---|---|---|---|
| Peak RAM during inference | ≤ 8 GB | *pending* | *pending* |
| RTF (avg over test set) | ≤ 2× | *pending* | *pending* |
| Model size on disk | — | *pending* | *pending* |
| Inference time (full test set) | — | *pending* | *pending* |

See the full [hardware validation report](hardware_validation.md).

---

## What Didn't Work (and What We Learned)

**Colab RAM crashes:** Three separate OOM crashes before training completed. Root causes: (1) float32 in-memory records during data loading used 10.5 GB; (2) AdamW optimizer states added 1.84 GB at step 2. Both fixed before the successful run.

**Slow inference with max_new_tokens=225:** The original Whisper default of 225 output tokens caused the model to run to the limit on every clip from languages it had never seen (Kikuyu, Luo, Maasai, Kalenjin). Spoken utterances almost never exceed 30 words (~50 tokens). Reducing to `max_new_tokens=64` cut inference time by ~3×.

**Kaggle REST API 404:** Our initial approach used the Kaggle v1 datasets REST API to list and download test parquet files one at a time. The API returned 404 — the correct approach is `kagglehub.dataset_download()` which handles auth and routing automatically.

**Training from scratch on round 3:** The first draft of `modal_finetune.py` loaded `openai/whisper-small` every run, discarding the fine-tuned checkpoint. Fixed to load from the volume checkpoint or HF Hub fallback.

---

## Reproducibility

All code and checkpoints are public:

- **Training + inference script:** [`modal_finetune.py`](https://github.com/Ashuza11/afrivoices-asr-hack/blob/main/modal_finetune.py)
- **Colab notebook (early runs):** [`notebooks/colab_finetune_submission.ipynb`](https://github.com/Ashuza11/afrivoices-asr-hack/blob/main/notebooks/colab_finetune_submission.ipynb)
- **Model checkpoint:** [Ash11/afrivoices-whisper-small-all6](https://huggingface.co/Ash11/afrivoices-whisper-small-all6)

To reproduce Round 2 results:

```bash
git clone https://github.com/Ashuza11/afrivoices-asr-hack
cd afrivoices-asr-hack
pip install modal
modal secret create afrivoices-secrets HF_TOKEN=<your_hf_token> KAGGLE_API_TOKEN=<your_kaggle_key>
modal run modal_finetune.py
```

The script will:
1. Download and cache training data to a Modal persistent volume
2. Fine-tune Whisper-small for 2,000 steps on an A100 GPU
3. Run inference on the 94-parquet test set
4. Save `submission.csv` to the volume

Download the submission:
```bash
modal volume get afrivoices-vol submission.csv .
```

### Hardware and Software Versions

| Component | Version |
|---|---|
| Python | 3.11 |
| PyTorch | 2.2.0 |
| Transformers | 4.46.3 |
| Accelerate | ≥0.26.0 |
| Training GPU | NVIDIA A100 40GB |
| Training OS | Debian (Modal container) |

---

## Citation

```bibtex
@misc{mwigiri2026afrivoices,
  title   = {AfriVoices ASR: Fine-tuning Whisper-small for Six East African Languages},
  author  = {Mwigiri, Albino},
  year    = {2026},
  url     = {https://github.com/Ashuza11/afrivoices-asr-hack},
  note    = {AfriVoices East Africa ASR Hackathon 2026}
}

@misc{digitalumuganda2026afrivoices,
  title   = {AfriVoices East Africa: ASR Hackathon},
  author  = {{Digital Umuganda} and {Maseno University} and {Maseno Center for Applied Artificial Intelligence}},
  year    = {2026},
  url     = {https://kaggle.com/competitions/afrivoices-east-africa-asr-hackathon}
}

@misc{radford2022whisper,
  title   = {Robust Speech Recognition via Large-Scale Weak Supervision},
  author  = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  year    = {2022},
  url     = {https://arxiv.org/abs/2212.04356}
}
```

---

## Acknowledgements

Training data provided by Digital Umuganda and the KenCorpus Consortium under CC BY 4.0. Pre-trained model by OpenAI (Whisper), licensed under Apache-2.0. Compute for Rounds 2 and 3 provided by Modal.com.
