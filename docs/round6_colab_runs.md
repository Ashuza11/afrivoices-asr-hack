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
2. Run once with `RUN_PREPARE_DATA=True`, `RUN_TRAINING=False`,
   `RUN_INFERENCE=False`.
3. Restart the runtime.
4. Set `RUN_PREPARE_DATA=False`, `RUN_TRAINING=True`,
   `RUN_INFERENCE=False`.
5. After training, set `RUN_TRAINING=False`, `RUN_INFERENCE=True`.

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
