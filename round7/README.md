# Round 7 Implementation

This directory contains the deadline-safe XLS-R pipeline. Only completed and
tested stages are exposed by the command-line interface.

## Implemented stages

1. `preflight`: validates configuration, disk, PyTorch, CUDA, and GPU memory.
2. `prepare-sources`: downloads and extracts two Digital Umuganda Swahili
   shards and six evenly spread ANV shards per language/subtype for the pilot.
   Somali is locked to AfriVoices-KE/ANV; Digital Umuganda Somali is rejected by
   configuration validation. Original long recordings are preserved for later
   forced alignment and are never proportionally split.
3. `build-splits`: canonicalizes JSONL/CSV input, rejects malformed rows,
   verifies audio metadata, normalizes text, creates deterministic
   speaker/source-grouped splits, builds the shared vocabulary, checks leakage,
   and writes an audit report.
4. `seed-smoke`: loads Apache-licensed XLS-R 300M with one shared CTC head and a
   language embedding, trains exactly 100 steps, computes per-language and macro
   WER, then saves, reloads, and verifies the checkpoint.
5. `train-seed`: continues from the smoke checkpoint on naturally short clean
   audio, balances languages using accepted hours, unfreezes the feature encoder
   after 10% of optimizer updates, selects by macro WER, stops after four
   non-improving evaluations, and retains only best/latest resumable checkpoints.
6. `align-pilot`: extracts chunked CTC emissions from long unscripted training
   recordings, uses blank-aware forced alignment to obtain word timestamps,
   creates 2-25 second segments at word boundaries, and writes accepted/rejected
   manifests with token-posterior and speech-occupancy evidence. Thresholds are
   provisional until the clean-only versus clean-plus-aligned gate is run.
   One atomic result is saved per source recording, so interrupted alignment
   resumes without repeating completed sources. Stale scratch-audio references
   invalidate stage completion automatically.
7. `alignment-gate`: trains a clean-plus-pilot candidate from the same smoke
   checkpoint and validation split as the clean baseline. Coverage, macro WER,
   language breadth, and maximum regression determine whether work proceeds.
8. `align-full`: uses the gated candidate to align every eligible long recording
   in the prepared manifest, with the same per-record recovery.
9. `train-final`: initializes from the gated checkpoint and trains on clean plus
   all accepted aligned segments, selecting by macro language WER.
10. `infer-test`: runs resumable language-conditioned greedy CTC inference and
    refuses to produce an incomplete submission or hide failed audio rows.

ONNX export, int8 quantization, and edge RAM/RTF benchmarking are intentionally
deferred until a final checkpoint exists. Optional language-model decoding is
also outside the critical training-to-submission path.

## Input manifest

Set `paths.source_manifest` in `config.yaml`. JSONL and CSV are supported. Input
column aliases are accepted, but each valid row must resolve to:

- `id`
- `source_id` or recording ID
- `audio_path`
- `language`
- `transcription`
- optional `speaker_id`, `dialect`, `subtype`, and `duration`

## Run

Install the pinned environment, then run:

```bash
python3 -m pip install -r requirements-round7.txt
bash scripts/run_round7.sh
```

To run a single stage:

```bash
python3 -m round7.pipeline --stage build-splits
python3 -m round7.pipeline --stage seed-smoke
python3 -m round7.pipeline --stage train-seed
python3 -m round7.pipeline --stage align-pilot
python3 -m round7.pipeline --stage alignment-gate
python3 -m round7.pipeline --stage align-full
python3 -m round7.pipeline --stage train-final
python3 -m round7.pipeline --stage infer-test
```

Completed stages are skipped. Pass `--force` to rerun one. Outputs are written
atomically under `outputs/round7/<stage>`.

## Cluster configuration

Copy `cluster.yaml.example` to the ignored local path `round7/cluster.yaml` only
after the cluster operator supplies the values. The preflight refuses to run if
any required `FILL_ME` value remains.

Do not commit `.env` or `round7/cluster.yaml`.

### UCREL Hex

Hex-specific settings are complete in `config.hex.yaml` and
`cluster.hex.yaml`. They request one RTX A5000 because the current trainer is
single-GPU; requesting three cards would reserve resources without accelerating
the job or combining VRAM.

On Hex, from the repository root:

```bash
bash scripts/setup_hex_round7.sh
cp .env.example .env
# Fill HF_TOKEN and KAGGLE_API_TOKEN directly on Hex; do not commit .env.
bash scripts/submit_hex_preflight.sh
# Inspect the five-minute job log before reserving the 48-hour worker.
bash scripts/submit_hex_round7.sh
```

Monitor with `squeue --me` and read logs under `outputs/round7/logs`. A job that
reaches the 48-hour limit can be submitted again; completed stages, seed
checkpoints, source shards, and per-record alignments resume from home storage.
The preflight uses `a5000-5m`; the pipeline job uses `a5000-48h`. Both request
one A5000, 16 CPU cores, and 32 GiB system RAM. The first data pass can be slow
because Hex populates its local read cache, while subsequent reads benefit from
the NVMe-backed cache.

## Tests

```bash
python3 -m unittest discover -s round7/tests -v
```
