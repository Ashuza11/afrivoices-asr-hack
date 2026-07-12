# Round 7 Implementation

This directory contains the deadline-safe XLS-R pipeline. Only completed and
tested stages are exposed by the command-line interface.

## Implemented stages

1. `prepare-sources`: downloads and extracts two Digital Umuganda Swahili
   shards and six evenly spread ANV shards per language/subtype for the pilot.
   Somali is locked to AfriVoices-KE/ANV; Digital Umuganda Somali is rejected by
   configuration validation. Original long recordings are preserved for later
   forced alignment and are never proportionally split.
2. `preflight`: validates configuration, source manifest, disk, PyTorch, CUDA,
   and GPU memory.
3. `build-splits`: canonicalizes JSONL/CSV input, rejects malformed rows,
   verifies audio metadata, normalizes text, creates deterministic
   speaker/source-grouped splits, builds the shared vocabulary, checks leakage,
   and writes an audit report.
4. `seed-smoke`: loads Apache-licensed XLS-R 300M with one shared CTC head and a
   language embedding, trains exactly 100 steps, computes per-language and macro
   WER, then saves, reloads, and verifies the checkpoint.

Full seed training, forced alignment, export, and inference remain intentionally
unavailable until their smoke tests are implemented.

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
```

Completed stages are skipped. Pass `--force` to rerun one. Outputs are written
atomically under `outputs/round7/<stage>`.

## Cluster configuration

Copy `cluster.yaml.example` to the ignored local path `round7/cluster.yaml` only
after the cluster operator supplies the values. The preflight refuses to run if
any required `FILL_ME` value remains.

Do not commit `.env` or `round7/cluster.yaml`.

## Tests

```bash
python3 -m unittest discover -s round7/tests -v
```
