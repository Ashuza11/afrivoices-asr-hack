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
