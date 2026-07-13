# Round 7 Cluster Strategy

## Decision

Round 7 should not be another longer Whisper-small run. The primary experiment
should be a shared `facebook/wav2vec2-xls-r-300m` acoustic encoder trained with
CTC on correctly aligned AfriVoices segments, with language conditioning and a
shared Latin-character vocabulary. A repaired Whisper-small run on exactly the
same aligned data is the secondary benchmark, not the main bet.

This plan does **not** guarantee a public WER of 0.31. Moving from 0.72534 to
0.31 requires a 57.3% relative error reduction; reaching 0.20 requires 72.4%.
Round 6D delivered only a 3.3% relative reduction from the previous 0.75-level
result. The cluster opportunity is valuable because it lets us repair the data
pipeline and evaluate a stronger acoustic architecture, not because more GPUs
make the target certain.

## Competition Constraints

The final package must remain one unified ASR system, have fewer than 1 billion
parameters, use no more than 8 GB RAM during inference, run CPU-only, and meet
an RTF of at most 2 on the target edge hardware. The model, code, weights, and
training recipe must be public under an accepted permissive license.

Source: [official AfriVoices EAC Hackathon page](https://afrivoice.github.io/afrivoice_eac_hackathon/).

The recommended base model is
[`facebook/wav2vec2-xls-r-300m`](https://huggingface.co/facebook/wav2vec2-xls-r-300m),
which has 300M parameters and an Apache-2.0 license. CTC decoding is parallel
and is a better edge starting point than Whisper's autoregressive decoder. Edge
compliance still has to be measured after ONNX int8 export; it must not be
claimed from parameter count alone.

MMS is a useful alignment or research reference, but not the final model for
this deadline. The released `facebook/mms-300m` checkpoint is CC-BY-NC, while
the strongest MMS ASR checkpoint is a 1B model. That creates both license and
strict parameter-limit risk. W2v-BERT 2.0 and larger Whisper variants are also
secondary research options until their exact final package, license, RAM, and
Pi latency are validated.

## What Round 6 Proved

- Round 6D improved public WER from approximately 0.75 to **0.72534** after
  recovering long unscripted audio.
- The three normalization variants scored 0.72534, 0.72540, and 0.72541. Text
  normalization is therefore saturated for the current model.
- Most sampled unscripted recordings are longer than Whisper's 30-second input:
  92.75% for Somali, 85.57% for Kikuyu, 81.03% for Kalenjin, 74.11% for Luo,
  and 62.98% for Maasai.
- Round 6D divided each transcript proportionally across fixed audio windows.
  This recovered coverage but did not establish word timestamps. It can assign
  words to the wrong segment and train the model on contradictory supervision.
- Validation improved until step 1500 and then WER deteriorated while validation
  loss continued to improve. WER-based early stopping is mandatory.
- The training manifest contained only tens of thousands of rows despite an
  official dataset measured in hundreds of hours per language. Accessible data
  volume is not the primary limitation; trustworthy segmentation and filtering
  are.

The AfriVoices-KE paper describes about 3,000 hours across five languages, with
2,250 hours spontaneous, from 4,777 speakers. See
[AfriVoices-KE](https://arxiv.org/abs/2604.08448). A recent African ASR scaling
study also reports that transcription-quality problems accounted for 38.6% of
high-error cases, reinforcing that label quality can dominate raw hours. See
[Akera et al.](https://arxiv.org/abs/2510.07221).

## Data Pipeline

Source policy is fixed from the Round 6 evidence and test-dialect match:
Digital Umuganda supplies Swahili only. Somali, Kikuyu, Luo, Maasai, and
Kalenjin come from `MCAA1-MSU/anv_data_ke`; Digital Umuganda Mogadishu Somali is
not mixed into the primary run. This is recorded in every manifest row through
`dataset`, `source_repo`, and `source_shard` fields. The completed evidence does
not isolate Somali source as the sole cause of the Round 6D gain, so provenance
is retained for a controlled comparison after the primary run.

### 1. Split before segmentation

Create train, development, and diagnostic-test splits by original recording and
speaker when speaker IDs exist. If speaker IDs are absent, group by source clip,
shard, dialect, and recording session. Every child segment from one long clip
must remain in one split. This prevents segment leakage and makes validation WER
credible.

Development must be stratified by language, scripted/unscripted source, dialect,
and duration bucket. Preserve a frozen development set for every model and never
select checkpoints on the Kaggle public score.

### 2. Normalize references once

Use one auditable normalization function for train labels, validation references,
CTC vocabulary construction, language-model text, and submission output:
Unicode NFC, lowercase, normalized whitespace, and the punctuation policy already
validated by the 0.72534 submission. Log the percentage of rows changed and all
characters removed. Do not transliterate meaningful orthographic characters.

### 3. Train an alignment seed

Fine-tune XLS-R 300M CTC first on trustworthy examples:

- scripted recordings under 25 seconds;
- unscripted recordings under 25 seconds whose transcript duration and token
  rate pass sanity checks;
- no proportional transcript slices from Round 6D.

Use a shared character vocabulary plus explicit language identity. The
implemented architecture is one shared encoder, one learned language embedding,
and one shared CTC output head. This avoids ambiguity around whether six output
heads count as separate language models while still conditioning the acoustic
representation on the known language field.

### 4. Forced-align long recordings

Run the seed CTC model over each full transcript/audio pair and use CTC forced
alignment to obtain token or word boundaries. Combine these boundaries with VAD
to create 10-25 second speech segments. Do not split transcripts by word-count
ratio.

Accept an aligned segment only if all of the following are true:

- duration is 2-28 seconds;
- speech occupancy is at least 70%;
- every target token is represented by the vocabulary;
- alignment covers at least 90% of normalized transcript characters;
- mean token posterior and CTC loss pass language-specific thresholds derived
  from the clean development distribution;
- token rate is within the language-specific 1st-99th percentile from scripted
  data;
- no overlap exists across train and development source groups.

Write accepted and rejected manifests with reason codes. Report accepted hours
by language, subtype, dialect, and confidence decile before full training.

### 5. Iterative refinement

Train the first full XLS-R model on clean plus accepted aligned segments, then
realign once with that stronger model. One refinement pass is justified; repeated
self-training before the deadline increases confirmation bias and operational
risk.

## Primary Model

Train one XLS-R 300M CTC encoder with language-balanced batches. Sampling should
be based on accepted **hours**, not row counts, using temperature sampling
`p_l proportional to h_l^0.5`. Cap repeated exposure so a low-resource language
is not memorized through extreme oversampling.

Recommended starting configuration:

| Setting | Value |
|---|---|
| Audio | mono 16 kHz, 2-28 s |
| Precision | bf16 on Ampere/Hopper |
| Optimizer | AdamW |
| Peak learning rate | 2e-5 full encoder; 1e-4 heads |
| Warmup | 5% of updates |
| Feature encoder | frozen for first 10% of updates |
| Regularization | time masking; no synthetic speed/noise until clean baseline |
| Effective batch | at least 20 minutes of audio globally |
| Evaluation | every 1,000 updates or at most every 30 minutes |
| Selection | lowest macro language WER, with per-language WER logged |
| Early stopping | stop after 4 evaluations without macro-WER improvement |
| Seeds | one seed for screening, second seed only after a clear win |

These are starting values, not evidence that one exact learning rate is optimal.
Run a 5-10% training-budget pilot and continue only if the frozen development
WER is materially below Round 6D for at least four of six languages and does not
regress badly on any language.

CTC models benefit from text decoding when acoustic predictions are ambiguous.
Train a compact character 5-gram language model from **training text only** for
each language and tune LM weight and insertion penalty on the frozen development
set. Keep greedy, beam-only, and LM-decoded WER separate. Include the LMs in the
parameter/storage and latency report even though they are not neural parameters.

The architecture choice is supported by the
[XLS-R paper](https://arxiv.org/abs/2111.09296), which reports 14-34% relative
ASR improvements over prior systems across multilingual benchmarks, and by the
[official Transformers XLS-R recipe](https://github.com/huggingface/transformers/tree/main/examples/pytorch/speech-recognition).
The result is not directly transferable to this competition, so our frozen
AfriVoices validation remains the decision authority.

## Secondary Model

If cluster time permits, continue the existing Whisper-small checkpoint on the
same forced-aligned manifest. Use no Round 6D proportional segments, use
WER-based early stopping, and stop after two non-improving evaluations. This
isolates whether the gain comes from corrected labels or the CTC architecture.

Do not deploy both full models or select per-language models for the final entry
unless the organizers explicitly approve that interpretation of a unified model
and the combined package passes edge validation. The secondary model is primarily
an ablation and possible teacher for confidence filtering.

## Cluster Execution Order

The run should be delivered as one resumable pipeline so the person operating
the cluster does not have to edit notebooks:

1. `preflight`: print GPU, CUDA, storage, and environment checks.
2. `prepare-sources`: download the source pilot with resumable per-shard output.
3. `build-splits`: create immutable grouped split manifests and leakage report.
4. `seed-smoke`: run 100 microbatches and verify checkpoint reload.
5. `train-seed`: train the clean short-audio alignment seed.
6. `align-pilot`: align the pilot by language/source, resume per recording, and
   write accepted/rejected manifests.
7. `alignment-gate`: audit samples and compare clean-only with clean-plus-aligned
   training under the same validation split.
8. `align-long-audio`: shard by language/source, resume per shard, and write
   accepted/rejected manifests.
9. `audit-alignment`: produce accepted hours, confidence distributions, and 20
   random audio/text examples per language for inspection. Inspection changes
   thresholds only; it must not manually transcribe data.
10. `train-full-ctc`: training with checkpoint and log upload after
   every evaluation.
11. `decode-dev`: greedy, beam, and LM decoding with overall and per-language WER.
12. `export-edge`: ONNX int8 export and numerical comparison against PyTorch.
13. `benchmark-edge`: peak RSS and RTF by language and duration bucket on the
   actual Raspberry Pi 4 or accepted equivalent.
14. `infer-test`: run only after validation and edge gates pass; write resumable
    raw and normalized submissions.

The repository implements stages 1-7, stage 8 over every eligible recording in
the prepared pilot manifest, stage 10 final training, and stage 14 greedy test
inference. Expanding preparation to every source-repository shard, optional LM
decoding, ONNX export, and edge benchmarking remain separate work. The prepared
manifest scope must be reported accurately; `align-full` does not imply that
all approximately 3,000 source hours were downloaded.

All stages must be restartable. Hex has no separate scratch filesystem, so
manifests, extracted audio, checkpoints, metrics, and logs are kept under the
shared home directory. The workers' aggressive read cache accelerates repeated
reads after the first pass. Never place credentials in the job script or logs.

### Deadline schedule

With the July 15 deadline, reserve the last 10 hours for test inference,
submission validation, upload, and recovery from one failed job. A practical
72-hour critical path is:

| Time remaining | Work |
|---|---|
| 72-60 h | preflight, stage data, immutable grouped splits |
| 60-48 h | seed CTC training and alignment pilot on all six languages |
| 48 h gate | stop if alignment coverage/quality or seed WER is unacceptable |
| 48-28 h | parallel forced alignment, prioritizing kln/mas then kik/som/luo/swa |
| 28-12 h | full CTC training, frequent WER evaluation, early stop |
| 12-10 h | dev decoding, LM tuning, int8 export smoke test |
| final 10 h | test inference, CSV checks, upload, preserve artifacts |

If the operator receives the package with less than 48 hours remaining, skip
the secondary Whisper run and the second alignment pass. Do not skip grouped
splitting, alignment confidence filtering, or resumable test inference.

## Go/No-Go Gates

The next model is a submission candidate only when:

- split leakage count is zero;
- every accepted segment passes the predefined confidence rules and the report
  shows at least 50 high-confidence training hours per language; 150 hours per
  language is the target where the source data permits it;
- macro development WER beats Round 6D under the same normalization and decoding;
- kln and mas improve without sacrificing the other four languages enough to
  erase the macro gain;
- ONNX int8 WER degradation is at most 0.01 absolute;
- measured peak RAM is at most 8 GB and measured RTF is at most 2;
- the released model and every required dependency have competition-compatible
  licenses.

For a realistic attempt at 0.31, the internal development result should be near
that range before test inference. A development WER of 0.55 is useful research
progress but is not evidence that leaderboard WER will be 0.31.

## UCREL Hex Configuration

The supplied Hex profile uses Slurm partition `a5000-48h`, one RTX A5000 with
24 GiB VRAM, 16 CPU cores, 32 GiB RAM, and a 48-hour limit. The trainer is
single-GPU, so reserving a three-GPU A5000 worker would not accelerate it and
would not automatically combine the cards' memory. Batch size 1, accumulation
8, BF16, and gradient checkpointing are set for the 24 GiB card.

Python 3.12.3 and a repository-local venv are used. Compute-node access to
Hugging Face and Kaggle is available. The default shared-home quota is 512 GiB;
there is no independent scratch path. Job arrays are limited to five unless the
operator raises the limit. A dedicated `a5000-5m` preflight should pass before
the 48-hour job is submitted. The actual 100-step model smoke test remains the
first empirical memory check; static configuration cannot prove that a full
25-second batch fits.

## Evidence Base

- [AfriVoices-KE dataset paper](https://arxiv.org/abs/2604.08448)
- [African ASR model and data-scaling benchmark](https://arxiv.org/abs/2512.10968)
- [African-language data quality and scaling study](https://arxiv.org/abs/2510.07221)
- [XLS-R paper](https://arxiv.org/abs/2111.09296)
- [MMS paper](https://arxiv.org/abs/2305.13516)
- [MMS forced-alignment implementation](https://github.com/facebookresearch/fairseq/tree/main/examples/mms)
- [Auxiliary language-conditioned CTC](https://arxiv.org/abs/2302.12829)
- [GigaSpeech alignment and segmentation pipeline](https://arxiv.org/abs/2106.06909)
