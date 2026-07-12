---
layout: default
title: Round 6 Colab Runs
---

# Round 6 Colab Runs

Round 6 is split into two parallel Colab Pro experiments. Both use the same
low-RAM cache design: audio is stored as FLAC files plus a JSONL manifest in
Google Drive, and Whisper log-mel features are computed inside the collator per
batch. This avoids the Kaggle memory failure caused by holding all feature
tensors in RAM.

## Runtime

Use Google Colab Pro with:

| Setting | Value |
|---|---|
| Hardware accelerator | A100 GPU |
| High-RAM | Enabled |
| Runtime | Python 3 |
| Fallback GPU | H100, then L4 |
| Avoid | TPU, CPU-only, T4 for full runs |

A100 High-RAM is the default recommendation. H100 is acceptable if available.
L4 can run the notebooks with lower batch size, but it is not the preferred
configuration for these full Round 6 experiments.

## Notebook A: Round 6A

Notebook: `notebooks/colab_round6_low_ram_whisper_small.ipynb`

Purpose:

- Re-run the current Whisper-small Round 6 plan on stable Colab Pro hardware.
- Continue from the strongest existing Whisper-small checkpoint when available.
- Preserve the low-RAM FLAC manifest cache design.
- Keep stronger Maasai/Kalenjin coverage than the earlier Modal runs.
- Save raw and post-processed submission variants.

Default targets:

| Type | swa | som | kik | luo | mas | kln |
|---|---:|---:|---:|---:|---:|---:|
| Scripted | 3000 | 2500 | 3500 | 3000 | 4000 | 4000 |
| Unscripted | - | 900 | 900 | 900 | 1200 | 1200 |

## Notebook B: Round 6B

Notebook: `notebooks/colab_round6b_unscripted_somali_whisper_small.ipynb`

Purpose:

- Test the same model family with a more aggressive spontaneous-speech mix.
- Shift Somali toward ANV/Maxatire instead of DigitalUmuganda Mogadishu Somali.
- Increase unscripted coverage for Somali, Maasai, and Kalenjin, which are the
  likely highest-loss languages under macro WER.
- Use a separate Drive directory and checkpoint so it can run in parallel with
  Round 6A.

Default targets:

| Type | swa | som | kik | luo | mas | kln |
|---|---:|---:|---:|---:|---:|---:|
| Scripted | 2000 | 4500 | 3000 | 3000 | 3500 | 3500 |
| Unscripted | - | 2500 | 1200 | 1200 | 2200 | 2200 |

## Run Order

For each notebook:

1. Set Colab secrets: `HF_TOKEN` and `KAGGLE_KEY` or `KAGGLE_API_TOKEN`.
2. Run Cell 1 to mount Drive and install dependencies. After Cell 1 finishes,
   restart the runtime before continuing. Colab Python 3.12 uses NumPy-2 ABI
   wheels; the notebooks reinstall a consistent NumPy/Pandas/PyArrow/SciPy
   stack together. Running Cell 2 without restarting can leave old compiled
   extensions in memory and fail later at `import evaluate` with
   `numpy.dtype size changed`.
3. Run once with `RUN_PREPARE_DATA=True`, `RUN_TRAINING=False`,
   `RUN_INFERENCE=False`.
4. Restart the runtime after data preparation completes.
5. Set `RUN_PREPARE_DATA=False`, `RUN_TRAINING=True`,
   `RUN_INFERENCE=False`.
6. After training, set `RUN_TRAINING=False`, `RUN_INFERENCE=True`.

Both notebooks write submission variants:

- primary lower-strip-punctuation CSV
- raw
- whitespace-only
- strip-final-punctuation
- strip-all-punctuation

This is intentional. Earlier evidence showed normalization could hurt, while
later Round 6 evidence suggested lowercasing plus punctuation removal improved
public WER. Saving all variants lets leaderboard attempts resolve that
uncertainty instead of hard-coding one assumption.

## Crash Recovery

The notebooks are designed to preserve useful progress if Colab disconnects or
restarts:

- Data preparation writes FLAC files directly to Google Drive and appends
  manifest rows every 250 accepted clips.
- Training checkpoints are written under the notebook-specific Drive checkpoint
  directory. Re-running the training cell resumes from the latest
  `checkpoint-*` directory when one exists.
- Inference writes a CSV checkpoint every 10 batches and again after each test
  parquet. Re-running inference reloads that checkpoint and skips completed
  `id` values.
- Final submission variants are written only after inference completes, but the
  checkpoint CSV is enough to resume and regenerate them.

In the worst case, a crash can lose the currently in-memory training step,
current data-prep mini-buffer, or up to 10 inference batches. It should not lose
completed cached audio, completed manifest rows, saved training checkpoints, or
completed inference rows.

## Success Criteria

Round 6A is the safer recovery run. Round 6B is the higher-upside run. Compare:

- public leaderboard WER
- per-language dev WER
- Somali repetition/hallucination rate
- Maasai and Kalenjin WER
- whether normalization variants differ materially

If Round 6B beats 6A, the next experiment should not simply train longer. It
should segment long unscripted clips instead of discarding them, because the
AfriVoices-KE data is mostly spontaneous speech and many useful clips exceed
Whisper's 30-second training window.

## Round 6A Post-Mortem

Round 6A successfully fixed the Kaggle/Colab memory path and trained without
data-loader crashes, but the WER curve does not justify spending more compute
on full inference as a leaderboard push.

Observed validation metrics:

| Step | Loss | Eval loss | Macro WER | swa | som | kik | luo | mas | kln |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 1.6452 | 1.5862 | 0.8282 | 0.8964 | 0.6753 | 0.8707 | 0.5345 | 0.9280 | 0.9823 |
| 1000 | 1.4790 | 1.5009 | 0.8191 | 0.7913 | 0.7174 | 0.8546 | 0.5872 | 0.9354 | 0.9934 |
| 1500 | 1.4574 | 1.4530 | 0.8200 | 0.7673 | 0.6942 | 0.8762 | 0.5883 | 0.9360 | 1.0203 |
| 2000 | 1.3544 | 1.4268 | 0.8273 | 0.6826 | 0.7234 | 0.9027 | 0.6240 | 0.9424 | 1.0819 |
| 2500 | 1.2441 | 1.4127 | 0.8239 | 0.6965 | 0.7694 | 0.8975 | 0.5880 | 0.9571 | 1.0225 |
| 3000 | 1.2446 | 1.3962 | **0.8168** | 0.6878 | 0.7910 | 0.8949 | 0.5683 | **0.9178** | 1.0339 |
| 3500 | 1.1740 | 1.3950 | 0.8417 | 0.6976 | 0.8291 | 0.9070 | 0.6073 | 0.9476 | 1.0682 |
| 4000 | 1.1890 | 1.3873 | 0.8353 | **0.6805** | 0.8387 | 0.8973 | 0.5750 | 0.9428 | 1.0914 |
| 4500 | 1.1134 | 1.3889 | 0.8633 | 0.6927 | 0.8799 | 0.9139 | 0.6351 | 0.9715 | 1.1150 |

Decision:

- Do not run full inference for Round 6A as a serious 0.31 attempt.
- The best single checkpoint is step 3000 with macro WER 0.8168, only a small
  gain over the existing ~0.75-0.82 range.
- Available checkpoints after the old `save_total_limit=3` pruning were only
  `checkpoint-3000`, `checkpoint-4000`, and `checkpoint-4500`. The high-value
  early checkpoints for som/luo/kln were deleted by Trainer before the notebook
  was patched to `save_total_limit=20`.
- With only remaining checkpoints, per-language checkpoint selection is roughly
  0.8144 dev macro WER, not enough to justify a full inference run when the goal
  is 0.31-0.32.

Precise bottlenecks:

- **Kalenjin is the largest blocker:** best observed kln WER was 0.9823 at step
  500, and it degraded to 1.1150 by step 4500.
- **Maasai remains a blocker:** best observed mas WER was 0.9178 at step 3000.
- **Loss and WER diverged:** eval loss improved from 1.5862 to 1.3889 while
  macro WER degraded after step 3000. Continuing Whisper fine-tuning lowered
  teacher-forced loss without improving decoded text.
- **Swahili improved most, but Swahili is not the path to 0.31:** swa improved
  from 0.8964 to 0.6805, while kln/mas stayed near or above 0.9.
- **Unscripted data is underused:** AfriVoices-KE is mostly spontaneous speech
  (about 2,250h spontaneous vs 750h scripted), but Round 6A used only 1,152
  unscripted rows for mas/kln and skipped long clips instead of segmenting them.

Grounding from external work:

- AfriVoices-KE reports roughly 3,000h across Dholuo, Kikuyu, Kalenjin, Maasai,
  and Somali, with 750h scripted and 2,250h spontaneous speech. This supports
  prioritizing spontaneous-speech handling over scripted-only scaling.
- WAXAL-NET reports that compact domain-specialized African ASR models reached
  38.0% macro WER compared with 64.9% for the best zero-shot baseline on
  conversational African speech. This supports domain/language-specific
  specialization rather than blindly continuing a general Whisper checkpoint.
- Recent African ASR benchmarking reports that MMS and W2v-BERT can be more
  data-efficient in very-low-resource African settings, while Whisper is better
  in some mid-resource conditions. This supports adding a CTC/MMS track for
  kln/mas instead of relying only on Whisper-small.

## Next Plan Toward 0.31-0.32

Do not spend the next compute block on Round 6A inference. Use the remaining
Round 6A checkpoints only for a small diagnostic if submission attempts are
abundant. The next serious run should target the languages blocking macro WER:
kln and mas first, then kik/som.

Required changes before the next training run:

1. **Run a kln/mas-heavy Round 6C**, not just the existing Somali-heavy 6B.
   Suggested targets:

   | Type | swa | som | kik | luo | mas | kln |
   |---|---:|---:|---:|---:|---:|---:|
   | Scripted | 1200 | 2500 | 3000 | 2500 | 6000 | 6000 |
   | Unscripted | - | 1500 | 1200 | 1000 | 3500 | 3500 |

2. **Preserve all useful checkpoints.** Keep `save_total_limit=20` and do not
   start a run from an older notebook with `save_total_limit=3`.
3. **Stop by WER, not loss.** Stop if macro WER fails to beat the best checkpoint
   for two consecutive evals, or if kln/mas degrade while only swa improves.
4. **Segment long unscripted clips.** Skipping >30s spontaneous clips removes
   exactly the domain AfriVoices-KE says dominates the data. The next data
   improvement should chunk long mas/kln/som unscripted audio into Whisper-sized
   windows using weak alignment or conservative proportional transcript splits.
5. **Add a CTC/MMS-300M track for kln/mas.** If CTC beats Whisper on kln/mas dev
   WER, use it for per-language submission generation or use the result to
   diagnose Whisper decoder hallucination versus acoustic modeling failure.

## Round 6C: Kalenjin/Maasai-Heavy Audit Run

Notebook: `notebooks/colab_round6c_kln_mas_duration_audit_whisper_small.ipynb`

Round 6C is prepared as the next serious Whisper run. It is not another generic
longer fine-tune. It targets the measured Round 6A blockers:

| Type | swa | som | kik | luo | mas | kln |
|---|---:|---:|---:|---:|---:|---:|
| Scripted | 1200 | 2500 | 3000 | 2500 | 6000 | 6000 |
| Unscripted | - | 1500 | 1200 | 1000 | 3500 | 3500 |

Round 6C adds a duration audit before training. The audit writes:

- `round6c_duration_audit.csv`
- `round6c_duration_audit_summary.csv`

The audit measures duration distributions by language and subtype, including
the percentage of clips over 30s, 45s, and 60s. This is required before writing
segmentation logic. The question to answer is factual: are kln/mas/som losing
more usable spontaneous data because their clips exceed Whisper's 30-second
window?

Segmentation should be implemented only after the audit confirms where long
clips are concentrated. The expected first segmentation targets are kln and mas,
but this must be confirmed from the audit CSV, not assumed.

## Normalization Policy

Normalization is decided after inference, not before training. The repository
already has evidence in both directions:

- Older raw-output submissions beat some normalized variants.
- Later Round 6 evidence showed lowercasing plus punctuation removal improved
  public WER from roughly 0.77 to 0.75.

Therefore every inference run must save multiple submission variants:

- raw / whitespace collapsed
- strip final punctuation
- strip all punctuation
- lower + strip punctuation

The leaderboard decides which variant is better. Do not hard-code a single
normalization assumption before seeing submission scores.

## Round 6D: Segmented Long-Unscripted Run

Notebook: `notebooks/colab_round6d_segmented_unscripted_whisper_small.ipynb`

The spread-shard duration audit was added to `data/`:

- `data/round6c_duration_audit_spread.csv`
- `data/round6c_duration_audit_spread_summary.csv`

The audit confirms that long unscripted clips are a major bottleneck:

| Language | Unscripted >30s | Unscripted p90 sec | Mean sec |
|---|---:|---:|---:|
| som | 92.75% | 90.24 | 77.08 |
| kik | 85.57% | 79.26 | 59.73 |
| kln | 81.03% | 83.97 | 65.07 |
| luo | 74.11% | 87.41 | 53.77 |
| mas | 62.98% | 75.90 | 49.65 |

Scripted clips are mostly short, so the segmentation fix is targeted only at
unscripted clips. Round 6D segments long unscripted clips instead of skipping
them:

- segment length: 25 seconds
- maximum source duration for proportional segmentation: 180 seconds
- maximum segments per source clip: 8
- minimum segment duration: 4 seconds
- minimum transcript words per segment: 3
- transcript splitting: proportional by word count

This is intentionally conservative. It does not claim timestamp-level alignment.
It converts a previously discarded long clip into several approximate training
pairs only when the transcript has enough words to support the number of audio
segments. The goal is to recover spontaneous-domain signal without flooding the
model with obviously empty or tiny text chunks.

Round 6D targets:

| Type | swa | som | kik | luo | mas | kln |
|---|---:|---:|---:|---:|---:|---:|
| Scripted | 1200 | 2500 | 3000 | 2500 | 6000 | 6000 |
| Short unscripted | - | 1500 | 1200 | 1000 | 3500 | 3500 |
| Segmented bonus | - | 1200 | 1200 | 1000 | 2500 | 2500 |

Estimated training size is about 42,700 rows after scripted speed perturbation
for mas/kln. With batch size 32 and a 4% eval split, one epoch is about 1,281
optimizer steps. The notebook uses `MAX_STEPS=3000`, or about 2.34 epochs.
This is intentional: Round 6A improved until step 3000 and then degraded by WER
even though eval loss continued to fall. Round 6D should be stopped by WER, not
by loss. If macro WER or kln/mas WER degrades for two consecutive evals after
the best checkpoint, stop the run.

Expected impact is not enough by itself to reach 0.31. A realistic target for a
successful Whisper-only segmented run is a 0.08-0.18 macro WER improvement if
the proportional chunks are usable. If the run does not materially improve kln
and mas, the next major lever must be a CTC/MMS or W2v-BERT track for those
languages.

### Round 6D Result

Round 6D completed 3000 steps in Colab after segmented long-unscripted data
preparation.

Observed validation metrics:

| Step | Loss | Eval loss | WER | swa | som | kik | luo | mas | kln |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 1.9251 | 1.8265 | 0.8181 | 0.8437 | 0.7941 | 0.8781 | 0.5194 | 0.8482 | 0.8805 |
| 1000 | 1.7417 | 1.7340 | 0.7994 | 0.7708 | 0.7734 | 0.8776 | 0.4991 | 0.8447 | 0.8424 |
| 1500 | 1.7046 | 1.6792 | **0.7867** | 0.7426 | **0.7555** | **0.8747** | **0.4902** | **0.8431** | **0.8145** |
| 2000 | 1.6814 | 1.6448 | 0.7958 | 0.7472 | 0.7603 | 0.8812 | 0.4979 | 0.8534 | 0.8275 |
| 2500 | 1.6658 | 1.6287 | 0.8037 | **0.7170** | 0.7666 | 0.8807 | 0.5046 | 0.8707 | 0.8369 |
| 3000 | 1.6075 | 1.6256 | 0.8010 | 0.7308 | 0.7559 | 0.8810 | 0.5043 | 0.8631 | 0.8381 |

Decision:

- The best overall checkpoint is step 1500. `load_best_model_at_end=True`
  means the model saved to `CHECKPOINT_DIR` after training should already be the
  best checkpoint by the configured WER metric.
- Segmentation improved the intended bottlenecks relative to Round 6A:
  - kln improved from 0.9823 best observed in Round 6A to 0.8145.
  - mas improved from 0.9178 best observed in Round 6A to 0.8431.
- The run is a real improvement over Round 6A, but still far from the 0.31-0.32
  target. It is suitable for one inference/submission only if compute budget is
  available; it is not a final top-leaderboard solution.
- The displayed `WER` is the trainer's pooled eval WER, not a confirmed
  competition macro-average. For leaderboard planning, per-language WER remains
  the more important diagnostic.

### Round 6D Inference Handoff

Hugging Face model repo:

- `Ash11/afrivoices-whisper-small-colab-round6d`

The autogenerated model card text is not enough to prove inference readiness.
The required evidence is the file list in the repository. The dedicated Modal
script checks for:

- `config.json`
- `generation_config.json`
- `model.safetensors`
- `preprocessor_config.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `tokenizer.json` or `vocab.json`

Inference script:

- `modal_inference_round6d.py`

Recommended execution:

```bash
modal run modal_inference_round6d.py --check-only
modal run modal_inference_round6d.py --fresh --beam-size 3
modal volume get afrivoices-vol round6d_outputs .
```

If inference is interrupted, rerun without `--fresh`:

```bash
modal run modal_inference_round6d.py --beam-size 3
```

The script writes a checkpoint CSV after every five parquet files and emits
multiple normalization variants. The primary file is lowercased with punctuation
removed because that matched the previous public-WER gain from about 0.77 to
0.75, but raw and lighter-normalized variants are also saved for leaderboard
comparison.

Public leaderboard result:

| Submission | Public WER | Comment |
|---|---:|---|
| `submission_round6d_beam3.csv` | **0.72534** | segmented long-unscripted training, best Colab checkpoint, Modal beam=3 inference, lowercase + punctuation normalization |
| `submission_round6d_beam3_strip_all_punct.csv` | 0.72540 | same beam=3 inference, strip punctuation only, keep original casing |
| `submission_round6d_beam3_strip_final_punct.csv` | 0.72541 | same beam=3 inference, keep casing and internal punctuation, strip final punctuation only |

This is an absolute improvement of roughly 0.0247 WER versus the previous
0.75-level submission. The gain confirms that long-unscripted segmentation was
useful, especially because the validation run had already shown large kln/mas
improvements. The result is still far from the 0.31-0.32 target, so Round 6D
should be treated as a confirmed incremental fix, not as the main path to the
top of the leaderboard.

Normalization decision:

- Lowercase plus punctuation stripping remains the best observed normalization:
  `0.72534`.
- Punctuation stripping while preserving original casing was slightly worse:
  `0.72540`.
- Final-punctuation stripping while preserving casing and internal punctuation was
  also slightly worse: `0.72541`.
- The difference is small, but the public leaderboard evidence supports keeping
  the lowercase variant as the Round 6D best submission.

Round 6D critique:

- The main fix was correct: recovering long unscripted clips improved public WER
  from the previous 0.75-level submission to 0.72534.
- The gain is too small for the 0.31-0.32 target. Round 6D solved a data
  coverage bug, not the central modeling problem.
- The proportional transcript segmentation is weak supervision. It recovers
  signal from long clips, but it almost certainly introduces segment/text
  misalignment because no timestamp model was used.
- Training still used a single Whisper-small model for all six languages.
  Validation showed very uneven language behavior, with kln/mas still much
  worse than luo and swa despite improvement.
- The best validation WER was at step 1500, while later training reduced eval
  loss but worsened WER. Future runs should prioritize WER-based early stopping
  and avoid spending compute after degradation is visible.
- Inference normalization is now saturated for this model. Three variants landed
  within 0.00007 public WER, so post-processing is not the next high-leverage
  path.

Next-run direction:

The cluster-ready execution proposal is documented in
[`round7_cluster_strategy.md`](round7_cluster_strategy.md).

- Build language-aware models or adapters instead of one shared Whisper-small
  model for everything. At minimum, isolate kln/mas in a focused run because they
  remain the largest error contributors.
- Replace proportional segmentation with stronger long-audio supervision:
  VAD/speech segmentation first, then pseudo-alignment or transcription
  filtering per segment.
- Evaluate a non-Whisper acoustic track for kln/mas, especially MMS/wav2vec2 or
  W2v-BERT style CTC fine-tuning, because the Whisper-small multilingual decoder
  has not shown a path from 0.72 to 0.31 on this data.
- Use pseudo-labeling only after selecting a stronger teacher checkpoint or
  ensemble. Round 6D alone is not accurate enough to trust all pseudo-labels.
- Keep submission variants, but do not spend major effort on casing/punctuation
  until model WER is much lower.

## Storage Plan

The free Google Drive quota is too small for repeated FLAC caches and checkpoint
folders. Colab also provides local VM storage, including local scratch, but that
storage is ephemeral and disappears when the runtime is recycled.

Use this policy:

- Store durable artifacts in Drive: manifest JSONL, final selected checkpoints,
  final submissions, and logs.
- Store temporary Hugging Face downloads, parquet extraction, and intermediate
  shard files on local scratch (`/content` or the mounted local scratch path).
- After a run, keep only selected checkpoints: macro-best, per-language-best,
  and final if needed. Delete dominated checkpoints.
- Keep one Drive cache per active experiment. Do not keep 6A, 6B, and 6C audio
  caches simultaneously on a 15GB Drive quota.
- If Drive is above 85%, pause before inference or training. A full Drive can
  silently break checkpoint saves and submission CSV writes.
