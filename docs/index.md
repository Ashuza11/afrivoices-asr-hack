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

**Round 1 (Colab T4, 600 steps):**
- 5,000 Swahili + 2,000 Somali + 1,000 × 4 ANV = **~11,000 clips**
- Inference crashed (Colab disk full at 37.5 GB test set download)
- Eval WER on held-out set: **1.933** at step 400 (best checkpoint)

**Round 2 (Modal A100, 1,500 steps from base whisper-small):**
- Same data composition as Round 1
- Result: WER **0.89330** on public leaderboard (−44.6% vs baseline)

**Round 3 (Modal A100, 2,000 steps from Round 2 checkpoint) — in progress:**
- 8,000 Swahili + 4,000 Somali + 2,000 × 4 ANV = **~20,000 clips**
- Learning rate lowered to 5e-6 (fine-tuning a fine-tuned model)
- *Results pending*

### Training Infrastructure

| Component | Round 1 (Colab) | Round 2 & 3 (Modal) |
|---|---|---|
| Hardware | Google Colab T4 (free tier) | Modal.com A100 40GB (~$5–15/run) |
| Batch size (per device) | 4 | 16 |
| Effective batch size | 16 | 32 |
| Optimizer | Adafactor | Adafactor |
| Mixed precision | fp16 | fp16 |
| Duration | ~86 min (600 steps) | ~35 min (1,500 steps) |
| Inference | Crashed (disk full) | ✅ 45 min on A100 |

**Why Adafactor instead of AdamW?**
AdamW stores momentum and variance tensors for every parameter — for Whisper-small's 244M parameters, that's ~1.84 GB of optimizer state. On Colab's 12.7 GB RAM (shared with the OS, the 11k training clips at 5.3 GB, and the model at ~1 GB), AdamW pushed us over the limit at step 2. Adafactor reconstructs the second moment from a factored low-rank approximation, cutting optimizer RAM to ~0.24 GB.

**Why float16 in-memory records?**
Loading 11,000 audio clips as float32 spectrograms requires ~10.5 GB of RAM. Storing them as float16 halves this to ~5.3 GB while the DataCollator converts back to float32 per batch during training — no accuracy loss, immediate OOM fix.

---

## Results

### WER Progression

| Stage | Hardware | Steps | Train Loss | Leaderboard WER | Δ vs previous |
|---|---|---|---|---|---|
| Zero-shot whisper-small | — | — | — | 1.61077 | baseline |
| Round 1 fine-tune | Colab T4 | 600 | — | *(inference crashed)* | — |
| Round 2 fine-tune | Modal A100 | 1,500 | — | **0.89330** | −44.6% |
| Round 3 fine-tune | Modal A100 | +2,000 from R2 ckpt | 0.5442 | 0.91618 | **+2.6% (regression)** |
| Inference fix attempt | Modal A100 | — (R2 weights) | — | 0.93813 | **+5.0% (normalisation backfired)** |
| Kaggle experiment A | Kaggle T4 | TBD | — | *pending* | — |

The metric is the **unweighted mean of the per-language WER** — WER is computed independently for each of the six languages, then averaged. Each language contributes exactly **1/6 (16.7%)** of the final score, regardless of how many clips it has in the test set. Lower is better. Round 3 regressed despite more data and more steps — see the post-mortem analysis below.

### Per-Language Test Set Distribution

| Language | Test clips | Share of test set | Share of score |
|---|---|---|---|
| swa | 12,553 | 30.1% | 16.7% |
| kik | 9,192 | 22.0% | 16.7% |
| luo | 7,437 | 17.8% | 16.7% |
| kln | 4,837 | 11.6% | 16.7% |
| som | 3,925 | 9.4% | 16.7% |
| mas | 3,789 | 9.1% | 16.7% |

Because scoring is a macro-average, **clip volume is irrelevant to the score** — every language weighs 1/6. Swahili and Somali are the only two languages Whisper knows from pre-training; Kikuyu, Luo, Maasai, and Kalenjin are out-of-vocabulary. **Those four OOV languages therefore account for 4/6 = 67% of the leaderboard score.** They — not the high-volume Swahili set — are where the competition is won or lost.

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

### Round 3 Regression Post-Mortem

Round 3 made things worse (0.89330 → 0.91618). This is the most instructive failure of the project. After a thorough code audit, we identified five compounding causes:

**1. Data distribution shift — but not in the way we first thought.**

| Language | Round 2 clips | Round 2 % | Round 3 clips | Round 3 % |
|---|---|---|---|---|
| Swahili | 5,951 | **49.8%** | 5,951 | **33.2%** |
| Somali | 2,000 | 16.7% | 4,000 | 22.3% |
| ANV × 4 | 4,000 | 33.5% | 8,000 | 44.5% |

Our first reading blamed the drop in Swahili's share (half → a third). That reasoning was wrong: because the leaderboard is a **macro-average** (each language 1/6), shifting gradient toward the four OOV languages is the *correct* direction — Swahili is only 1/6 of the score and Whisper already handles it well. The real problem was not *which* way the mix shifted but that it shifted **while continuing to train a converged model at a low LR** (see cause 2), and that the eval/inference mismatch (cause 4) then selected a checkpoint that was not actually best at inference settings. The lesson for Round 4 is the opposite of our first instinct: deliberately weight the four OOV languages *up*, not protect Swahili.

**2. Over-training a model that had already converged.**

Round 2 found a good solution in 1,500 steps from a cold start. Round 3 applied 2,000 more steps to that solution at a lower LR (5e-6) on a different data composition. The final training loss was 0.5442 — very low, suggesting the model memorised the training distribution rather than generalising. More steps + different data pushed the model away from the Round 2 solution without finding a better one.

**3. No text normalisation in training labels or inference output.**

Training transcriptions use different conventions across the three data sources: `normalized_transcription` (Swahili/Somali) vs raw `transcription`/`actualSentence` columns (ANV). The model learns to reproduce the output style of each source inconsistently. During inference, the model output is decoded and stripped of special tokens but otherwise not normalised — mixed case, punctuation, and spacing variations are all passed to the submission CSV as-is. If the competition's reference transcriptions use a different normalisation, every such mismatch is counted as a substitution or insertion in WER.

**4. Eval metric during training uses a different setting than inference.**

`Seq2SeqTrainingArguments` sets `generation_max_length=225` for the eval loop in `compute_metrics`. But during inference, the code uses `max_new_tokens=64`. The `load_best_model_at_end=True` flag selects the checkpoint with the best WER at 225 tokens, then deploys it at 64 tokens. These are not equivalent for a model that has learned to produce longer sequences. A checkpoint that looks best at 225 tokens may be sub-optimal when hard-capped at 64.

**5. No language token for four of six test languages.**

For kik, luo, mas, and kln, `LANG_TO_WHISPER` maps to `None`, so inference runs without a forced language prefix. Without this constraint, the Whisper decoder can freely choose its output language based on acoustic similarity. Since Swahili was the dominant training language in both rounds, the model may silently switch to Swahili-like output for ambiguous ANV utterances, inflating WER on those four languages.

**What this tells us about the remaining road:**

Because the score is a macro-average, **67% of it comes from the four OOV languages** (kik, luo, mas, kln) — so that is where effort belongs, not on the already-strong Swahili. The highest-value levers, ranked: (a) adding Whisper language-id tokens for the four OOV languages (vocab extension, Paza-style) so the model conditions on each one separately; (b) inverting the data mix to upsample the four OOV languages; (c) SpecAugment during training; (d) model-soup weight averaging to undo the regression at near-zero cost; and (e) aligning `max_new_tokens` between eval and inference. Note that text normalisation of the inference output was tested and **made WER worse** (0.89330 → 0.93813) — see "What Didn't Work" — so it is *not* a quick win. The best current submission is still Round 2 (WER 0.89330).

---

### Earlier Issues

**Colab RAM crashes:** Three separate OOM crashes before training completed. Root causes: (1) float32 in-memory records during data loading used 10.5 GB; (2) AdamW optimizer states added 1.84 GB at step 2. Both fixed before the successful run.

**Slow inference with max_new_tokens=225:** The original Whisper default of 225 output tokens caused the model to run to the limit on every clip from languages it had never seen (Kikuyu, Luo, Maasai, Kalenjin). Spoken utterances almost never exceed 30 words (~50 tokens). Reducing to `max_new_tokens=64` cut inference time by ~3×.

**Kaggle REST API 404:** Our initial approach used the Kaggle v1 datasets REST API to list and download test parquet files one at a time. The API returned 404 — the correct approach is `kagglehub.dataset_download()` which handles auth and routing automatically.

**Training from scratch on round 3:** The first draft of `modal_finetune.py` loaded `openai/whisper-small` every run, discarding the fine-tuned checkpoint. Fixed to load from the volume checkpoint or HF Hub fallback.

**Text normalisation backfired (WER 0.89330 → 0.93813):** Based on research showing WAXAL-NET applies lowercase + punctuation removal before computing WER, we applied the same normalisation to our inference output. This made WER significantly worse. The conclusion: the competition evaluates WER against references stored in their original format (mixed-case, with punctuation — matching Whisper's output style). Applying lowercase on our side turned every sentence-initial capital into a substitution error and every missing period into a deletion. At 12,553 Swahili clips alone (30% of the leaderboard), that is thousands of extra errors. The lesson: normalisation must be applied to both sides identically or not at all. Since we cannot inspect the competition's reference format, we reverted to raw Whisper output, which was already in the same format as the references.

**max_new_tokens 64 → 128 also hurt:** Whisper hallucinates or loops when given more token budget than the utterance needs, especially on low-resource languages (kik, luo, mas, kln) where EOS prediction is unreliable. 64 tokens naturally bounded hallucinations; 128 doubled the room for spurious insertions. Reverted to 64.

**Inference checkpoint reuse:** After Round 3 training, the inference script found 41,733 rows in the old checkpoint file and skipped all 94 parquet files, producing an identical submission to Round 2. Fixed by comparing model file modification time to checkpoint modification time — if the model is newer, the checkpoint is discarded and inference runs fresh.

**37.5 GB re-download on every run:** `kagglehub` cached the test archive to `/root/.cache/`, which is ephemeral per container. Fixed by copying the extracted parquet files (a few GB) to the Modal persistent volume after the first download — subsequent runs load from the volume directly.

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
  author  = {Digital Umuganda and Maseno University and Maseno Center for Applied Artificial Intelligence},
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
