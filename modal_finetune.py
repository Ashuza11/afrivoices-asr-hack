"""
AfriVoices ASR — Modal training + inference script
"""

import modal

import os

# Must be set BEFORE huggingface_hub is imported anywhere (it reads this at
# import time) — setting it inside train() after the import was a no-op.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

# ── App + Image ───────────────────────────────────────────────────────────────
app = modal.App("afrivoices-asr")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.2.0",
        "transformers==4.46.3",
        "accelerate>=0.26.0",
        "datasets==2.20.0",
        "evaluate",
        "jiwer",
        "soundfile",
        "librosa",
        "pydub",
        "huggingface_hub>=0.21",
        "pandas",
        "pyarrow",
        "numpy<2",
        "requests",
        "kagglehub",
    )
)

# Persistent volume — stores cached records, checkpoint, submission.csv
vol      = modal.Volume.from_name("afrivoices-vol", create_if_missing=True)
VOL_PATH = "/vol"

RECORDS_DIR    = f"{VOL_PATH}/records"
CHECKPOINT_DIR = f"{VOL_PATH}/whisper-small-all6"
HF_REPO_ID     = "Ash11/afrivoices-whisper-small-all6"

secrets = [modal.Secret.from_name("afrivoices-secrets")]


# ── Shared helpers ────────────────────────────────────────────────────────────

def _decode_audio(audio_field):
    import io
    import numpy as np
    import soundfile as sf

    if isinstance(audio_field, dict) and "array" in audio_field:
        arr = np.array(audio_field["array"], dtype=np.float32)
        sr  = audio_field.get("sampling_rate", 16000)
        if sr != 16000:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        return arr
    raw = audio_field.get("bytes") if isinstance(audio_field, dict) else audio_field
    if isinstance(raw, bytes) and len(raw) > 0:
        try:
            arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            if sr != 16000:
                import librosa
                arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
            return arr
        except Exception:
            pass
        from pydub import AudioSegment
        seg = (AudioSegment.from_file(io.BytesIO(raw))
               .set_frame_rate(16000).set_channels(1))
        return np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0
    raise ValueError(f"Cannot decode audio: type={type(audio_field)}")


def _save_records(records, path):
    import pickle
    import numpy as np
    packed = [{"input_features": r["input_features"].astype(np.float16, copy=False),
               "labels": r["labels"]} for r in records]
    with open(path, "wb") as f:
        pickle.dump(packed, f, protocol=4)
    print(f"  → {len(records)} records saved ({os.path.getsize(path)/1e6:.0f} MB)")


def _load_records(path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)  # stays float16 in RAM; DataCollator converts per batch


def _kaggle_download_dataset(dataset: str, kaggle_key: str, cache_dir: str) -> str:
    """Download a Kaggle dataset using kagglehub (new key-only auth).
    Files are cached in cache_dir so re-runs skip the download."""
    import kagglehub
    os.environ["KAGGLE_KEY"]          = kaggle_key
    os.environ["KAGGLE_CACHE_FOLDER"] = cache_dir
    return kagglehub.dataset_download(dataset)


# ── Train ─────────────────────────────────────────────────────────────────────
# _train_impl holds the full pipeline; two thin Modal wrappers below give it two
# billing profiles: prepare_data() runs the data harvest on a cheap CPU container
# (a stalled download burns cents), train() runs on the A100 only once the caches
# exist (the GPU never idles on network I/O — that mistake cost us ~$15).
def _train_impl(resume: bool = False, fresh: bool = False, data_only: bool = False):
    import io, json, tarfile, lzma, shutil
    import numpy as np
    import pandas as pd
    import glob
    import torch
    from dataclasses import dataclass
    from typing import Any, Dict, List
    from torch.utils.data import Dataset as TorchDataset
    import evaluate
    from transformers import (
        WhisperProcessor, WhisperForConditionalGeneration,
        Seq2SeqTrainingArguments, Seq2SeqTrainer,
    )
    from huggingface_hub import login, hf_hub_download, list_repo_files, HfApi

    HF_TOKEN = os.environ["HF_TOKEN"]
    login(token=HF_TOKEN)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(RECORDS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Processor always from base model (tokenizer/feature extractor don't change)
    processor         = WhisperProcessor.from_pretrained("openai/whisper-small")
    feature_extractor = processor.feature_extractor
    tokenizer         = processor.tokenizer

    # ── Language tokens for the 4 out-of-vocab languages (Paza approach) ──────
    # kik/luo/mas/kln are NOT in Whisper's vocab. Giving each its own language
    # token lets the model condition on the language instead of guessing it from
    # acoustics. These 4 languages are 4/6 = 67% of the macro-averaged score.
    NEW_LANG_TOKENS = ["<|kik|>", "<|luo|>", "<|mas|>", "<|kln|>"]
    n_added = tokenizer.add_tokens(NEW_LANG_TOKENS, special_tokens=True)
    print(f"Added {n_added} new language tokens (vocab now {len(tokenizer)}).")

    SOT          = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    TRANSCRIBE   = tokenizer.convert_tokens_to_ids("<|transcribe|>")
    NOTIMESTAMPS = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    EOT          = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    # ISO-639-3 → Whisper language token. swa/som reuse Whisper's native tokens;
    # the 4 OOV languages use the tokens we just added.
    LANG_TOKEN = {"swa": "<|sw|>", "som": "<|so|>",
                  "kik": "<|kik|>", "luo": "<|luo|>",
                  "mas": "<|mas|>", "kln": "<|kln|>"}

    def build_labels(text, lang3):
        """Label ids = [sot, <|lang|>, transcribe, notimestamps, ...text..., eot].
        The exact layout Whisper expects — but works for OOV languages too, and
        puts the language token at a fixed position so we can read it back for
        per-language WER. Identical output to the old set_prefix_tokens path for
        swa/som, so their caches stay valid."""
        lang_id  = tokenizer.convert_tokens_to_ids(LANG_TOKEN[lang3])
        text_ids = tokenizer(text, add_special_tokens=False).input_ids
        return [SOT, lang_id, TRANSCRIBE, NOTIMESTAMPS] + text_ids + [EOT]

    # ── Swahili (cache-first, multi-shard) ───────────────────────────────────
    # Swahili is only 1/6 of the score and Whisper already handles it well —
    # keep it modest so the OOV languages dominate the gradient.
    SWA_TARGET = 3000
    SWA_CACHE  = f"{RECORDS_DIR}/swa_records.pkl"
    if os.path.exists(SWA_CACHE):
        swa_records = _load_records(SWA_CACHE)
        print(f"Swahili: {len(swa_records)} clips from cache.")
    else:
        swa_records = []
        for shard in range(10):
            if len(swa_records) >= SWA_TARGET:
                break
            try:
                manifest_path = hf_hub_download(
                    repo_id="DigitalUmuganda/Afrivoice_Swahili",
                    filename=f"agriculture_swahili_train/manifest_{shard}.jsonl",
                    repo_type="dataset", token=HF_TOKEN,
                )
            except Exception:
                break
            with open(manifest_path) as f:
                all_entries = [json.loads(l) for l in f]
            if shard == 0 and all_entries:
                print(f"  manifest sample: {list(all_entries[0].keys())} | first key={all_entries[0].get('key','?')!r}", flush=True)
            wanted = {}
            for entry in all_entries:
                text = (
                    entry.get("normalized_transcription") or
                    entry.get("transcription") or
                    entry.get("transcript") or
                    entry.get("text") or ""
                ).strip()
                if text:
                    k = entry.get("key", "")
                    wanted[k] = text
                    wanted[os.path.basename(k)] = text
            try:
                audio_tar = hf_hub_download(
                    repo_id="DigitalUmuganda/Afrivoice_Swahili",
                    filename=f"agriculture_swahili_train/audio/audio_{shard}.tar.xz",
                    repo_type="dataset", token=HF_TOKEN,
                )
            except Exception:
                break
            print(f"  shard {shard}: {len(wanted)} manifest entries", flush=True)
            with tarfile.open(audio_tar, "r:xz") as tar:
                members = list(tar.getmembers())
                if shard == 0 and members:
                    sample = [m.name for m in members[:3]]
                    print(f"  tar sample members: {sample}", flush=True)
                for member in members:
                    ext = os.path.splitext(member.name)[1].lower()
                    if ext not in (".webm", ".wav", ".mp3", ".flac", ".ogg", ".opus"):
                        continue
                    base = os.path.splitext(os.path.basename(member.name))[0]
                    if base not in wanted:
                        continue
                    try:
                        arr = _decode_audio({"bytes": tar.extractfile(member).read()})
                        arr = arr[:480_000]
                    except Exception:
                        continue
                    feats  = feature_extractor(arr, sampling_rate=16000).input_features[0].astype(np.float16)
                    labels = build_labels(wanted[base], "swa")
                    swa_records.append({"input_features": feats, "labels": labels})
                    if len(swa_records) % 500 == 0:
                        print(f"  {len(swa_records)}/{SWA_TARGET}", flush=True)
                    if len(swa_records) >= SWA_TARGET:
                        break
        print(f"Swahili: {len(swa_records)} clips ready.")
        _save_records(swa_records, SWA_CACHE)
        vol.commit()

    # ── ANV: Kikuyu, Luo, Maasai, Kalenjin, Somali (Maxatire) ─────────────────
    # All 5 non-Swahili languages come from anv_data_ke. Two grounded changes:
    #   1. Include UNSCRIPTED (spontaneous) speech, not just scripted. The test set
    #      is spontaneous-dominated (official hours e.g. mas 51h read / 454h spont);
    #      training scripted-only was a train/test mismatch.
    #   2. Somali sourced here (Maxatire dialect) instead of DigitalUmuganda
    #      (Mogadishu) — the ANV test Somali is Maxatire.
    # Whisper caps at 30s (1500 frames); unscripted clips average ~34s, so we SKIP
    # any clip >30s (its transcript would not align to a truncated 30s window).
    MAX_SAMPLES = 480_000                               # 30s @ 16kHz — hard Whisper limit
    # (scripted, unscripted) targets. Unscripted spontaneous clips average ~34s and
    # MANY exceed Whisper's 30s limit (measured: skip rate climbs past 80% on later
    # shards). Harvesting to a high unscripted target would download 100s of GB of
    # 250-600MB shards mostly to discard >30s clips — so unscripted is bounded by
    # MAX_UNSCRIPTED_SHARDS below (takes whatever ≤30s clips the early shards yield).
    ANV_TARGETS = {
        "kik": (3000, 2500), "luo": (3000, 2500), "kln": (3000, 2500),
        "mas": (2500, 2500), "som": (3000, 2500),   # som = Maxatire dialect
    }
    MAX_UNSCRIPTED_SHARDS = 30
    ANV_LANGS  = list(ANV_TARGETS.keys())
    # One cache PER LANGUAGE, committed as soon as that language finishes — a hang
    # or crash mid-harvest now costs at most one language, not the whole run.
    # (Delete a language's pkl to force its re-harvest.)
    def _lang_cache(lang):
        return f"{RECORDS_DIR}/anv_{lang}_v2.pkl"

    anv_records = []
    missing = []
    for l in ANV_LANGS:
        if os.path.exists(_lang_cache(l)):
            recs = _load_records(_lang_cache(l))
            anv_records += recs
            print(f"ANV {l}: {len(recs)} clips from cache.")
        else:
            missing.append(l)
    if missing:
        print(f"Harvesting: {missing}")
        print("Listing MCAA1-MSU/anv_data_ke on HF Hub...")
        all_files     = list(list_repo_files("MCAA1-MSU/anv_data_ke", repo_type="dataset", token=HF_TOKEN))
        parquet_files = sorted(f for f in all_files if f.endswith(".parquet"))
        # path = {lang}/train/{scripted|unscripted}/audios/*.parquet
        shards = {(l, sub): [] for l in ANV_LANGS for sub in ("scripted", "unscripted")}
        for f in parquet_files:
            p = f.split("/")
            if len(p) >= 3 and p[0] in ANV_LANGS and p[1] == "train" and p[2] in ("scripted", "unscripted"):
                shards[(p[0], p[2])].append(f)
        for k in sorted(shards):
            print(f"  {k[0]}/{k[1]}: {len(shards[k])} shards")

        import tempfile, shutil as _sh
        import soundfile as _sf

        import concurrent.futures as _cf

        def _dl_shard(shard_path, td):
            """Download with a HARD 10-min wall-clock cap per attempt. A trickling
            transfer (bytes dripping too slowly to ever hit a socket read-timeout)
            is the failure mode that froze two runs — only a wall-clock cap catches
            it. Each attempt uses a fresh subdir so a zombie attempt's file locks
            can't block the retry. After 3 failed attempts the shard is skipped."""
            for attempt in range(3):
                sub = os.path.join(td, f"try{attempt}")
                ex  = _cf.ThreadPoolExecutor(max_workers=1)
                try:
                    fut = ex.submit(hf_hub_download, repo_id="MCAA1-MSU/anv_data_ke",
                                    filename=shard_path, repo_type="dataset",
                                    token=HF_TOKEN, local_dir=sub)
                    return fut.result(timeout=600)
                except Exception as e:
                    print(f"    download attempt {attempt+1} failed ({type(e).__name__}): {str(e)[:80]}", flush=True)
                finally:
                    ex.shutdown(wait=False)
            return None

        def _too_long(audio_field):
            """Duration from the audio HEADER when possible — skipping a >30s clip
            then costs ~nothing instead of a full waveform decode (≈2/3 of
            unscripted clips are >30s, so this roughly halves harvest time)."""
            if isinstance(audio_field, dict):
                arr = audio_field.get("array")
                if arr is not None:
                    sr = audio_field.get("sampling_rate") or 16000
                    return len(arr) / sr > 30.0
                raw = audio_field.get("bytes")
                if raw:
                    try:
                        info = _sf.info(io.BytesIO(raw))
                        return info.frames / info.samplerate > 30.0
                    except Exception:
                        return None   # header unreadable — decide after decode
            return None

        def harvest(lang, subtype, target, max_shards=None):
            recs, skipped_long = [], 0
            slist = shards[(lang, subtype)]
            if max_shards:
                slist = slist[:max_shards]   # bound cost; unscripted shards are huge
            for shard_path in slist:
                if len(recs) >= target:
                    break
                td = tempfile.mkdtemp()
                try:
                    pq_path = _dl_shard(shard_path, td)
                    if pq_path is None:
                        print(f"    skipping shard after 3 failed downloads", flush=True)
                        continue
                    df   = pd.read_parquet(pq_path)
                    tcol = ("transcription"  if "transcription"  in df.columns else
                            "actualSentence" if "actualSentence" in df.columns else
                            "transcript"     if "transcript"     in df.columns else None)
                    if tcol is None:
                        continue
                    for _, row in df.iterrows():
                        if len(recs) >= target:
                            break
                        text = (row.get(tcol) or "").strip()
                        if not text:
                            continue
                        long = _too_long(row["audio"])
                        if long:                        # cheap header check, no decode
                            skipped_long += 1
                            continue
                        try:
                            arr = _decode_audio(row["audio"])
                        except Exception:
                            continue
                        if len(arr) > MAX_SAMPLES:      # >30s: transcript won't align to 30s window
                            skipped_long += 1
                            continue
                        feats = feature_extractor(arr, sampling_rate=16000).input_features[0].astype(np.float16)
                        recs.append({"input_features": feats, "labels": build_labels(text, lang)})
                finally:
                    _sh.rmtree(td, ignore_errors=True)   # audio-heavy parquets — free disk each shard
                print(f"  {lang}/{subtype}: {len(recs)}/{target}  (skipped >30s: {skipped_long})", flush=True)
            return recs

        for lang in missing:
            st, ut = ANV_TARGETS[lang]
            print(f"\nLoading {lang}: scripted<={st}, unscripted<={ut}")
            sc = harvest(lang, "scripted", st)
            un = harvest(lang, "unscripted", ut, max_shards=MAX_UNSCRIPTED_SHARDS)
            lang_recs = sc + un
            print(f"  {lang}: {len(sc)} scripted + {len(un)} unscripted = {len(lang_recs)}")
            _save_records(lang_recs, _lang_cache(lang))
            vol.commit()                                 # survive stalls/crashes per language
            anv_records += lang_recs
    print(f"ANV total: {len(anv_records)} clips ready.")

    if data_only:
        print("Data preparation complete — all caches committed to the volume.")
        return

    # ── Combined dataset + train/eval split ───────────────────────────────────
    class WhisperDataset(TorchDataset):
        def __init__(self, records):
            self.records = records
        def __len__(self):
            return len(self.records)
        def __getitem__(self, i):
            return self.records[i]

    # Whisper's decoder hard-caps labels at 448 tokens (max_target_positions).
    # Spontaneous transcripts + byte-level tokenization of the OOV languages can
    # exceed it (crashed at step 188 with 471). Drop those clips — truncating the
    # transcript would teach the model to stop mid-utterance.
    MAX_LABEL_LEN = 448
    def _drop_overlong(recs, name):
        kept = [r for r in recs if len(r["labels"]) <= MAX_LABEL_LEN]
        if len(kept) < len(recs):
            print(f"  {name}: dropped {len(recs) - len(kept)} clips with >"
                  f"{MAX_LABEL_LEN}-token labels")
        return kept
    swa_records = _drop_overlong(swa_records, "swa")
    anv_records = _drop_overlong(anv_records, "anv")

    np.random.seed(42)

    def split_source(recs, cap=None):
        """Shuffle, optionally cap to enforce the data mix, then 95/5 split.
        Splitting per source guarantees every language is present in the dev set —
        essential because the leaderboard is a macro-average (one bad language
        tanks the mean)."""
        recs = list(recs)
        np.random.shuffle(recs)
        if cap:
            recs = recs[:cap]
        n = int(0.95 * len(recs))
        return recs[:n], recs[n:]

    swa_tr, swa_ev = split_source(swa_records, cap=SWA_TARGET)
    anv_tr, anv_ev = split_source(anv_records)  # kik/luo/mas/kln/som, already capped at load

    train_records = swa_tr + anv_tr
    eval_records  = swa_ev + anv_ev
    np.random.shuffle(train_records)
    train_ds = WhisperDataset(train_records)
    eval_ds  = WhisperDataset(eval_records)
    all_records = train_records + eval_records
    print(f"\nData mix — swa:{len(swa_tr)} anv(5 langs):{len(anv_tr)}")
    print(f"Total: {len(all_records)}  Train: {len(train_ds)}  Eval: {len(eval_ds)}")

    # ── Data collator + WER metric ────────────────────────────────────────────
    @dataclass
    class DataCollator:
        processor: Any
        decoder_start_token_id: int
        def __call__(self, features: List[Dict[str, Any]]):
            input_features = torch.tensor(
                np.stack([f["input_features"] for f in features]), dtype=torch.float32
            )
            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch   = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]
            return {"input_features": input_features, "labels": labels}

    wer_metric = evaluate.load("wer")
    LANG_TOKEN_ID = {tokenizer.convert_tokens_to_ids(tok): l3
                     for l3, tok in LANG_TOKEN.items()}

    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids
        # Language = first label token (the collator strips <|sot|>, so column 0
        # is the <|lang|> token). Capture it before -100 gets overwritten.
        first_tok = label_ids[:, 0].copy()
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        metrics = {"wer": round(wer_metric.compute(predictions=pred_str, references=label_str), 4)}
        parts = []
        for tok_id, l3 in LANG_TOKEN_ID.items():
            idx = [i for i, t in enumerate(first_tok) if t == tok_id]
            if idx:
                w = round(wer_metric.compute(
                    predictions=[pred_str[i] for i in idx],
                    references =[label_str[i] for i in idx]), 4)
                metrics[f"wer_{l3}"] = w
                parts.append(f"{l3}={w:.3f}")
        print("  per-language WER:  " + "  ".join(parts), flush=True)
        return metrics

    data_collator = DataCollator(
        processor=processor,
        decoder_start_token_id=processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>"),
    )

    # ── Model (continue from fine-tuned checkpoint, not base model) ──────────
    vol.reload()
    # --fresh forces a clean fine-tune from base whisper-small (correct when the
    # data composition changes materially, as in the scripted+unscripted overhaul).
    # The embedding-resize/seed block below adds our 4 OOV tokens on top of base.
    if fresh:
        load_from = "openai/whisper-small"
    else:
        load_from = CHECKPOINT_DIR if os.path.exists(f"{CHECKPOINT_DIR}/config.json") else HF_REPO_ID
    print(f"Loading model from: {load_from}")
    model = WhisperForConditionalGeneration.from_pretrained(load_from, token=HF_TOKEN)

    # Grow the embedding table for the 4 new language tokens. New rows are seeded
    # from the Swahili (<|sw|>) embedding — a sensible point in "language-token
    # space" that trains far faster than random init. Skipped (no-op) if the
    # checkpoint already has these tokens, so we never clobber learned embeddings.
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        old_vocab = model.get_input_embeddings().num_embeddings
        model.resize_token_embeddings(len(tokenizer))
        with torch.no_grad():
            emb   = model.get_input_embeddings().weight
            sw_id = tokenizer.convert_tokens_to_ids("<|sw|>")
            for tok in NEW_LANG_TOKENS:
                emb[tokenizer.convert_tokens_to_ids(tok)] = emb[sw_id].clone()
        print(f"Resized embeddings {old_vocab} -> {len(tokenizer)}; "
              f"seeded new language tokens from <|sw|>.")

    # SpecAugment — mask random time/frequency bands during training only
    # (Whisper gates this on model.training, so inference is unaffected).
    model.config.apply_spec_augment  = True
    model.config.mask_time_prob      = 0.05
    model.config.mask_time_length    = 10
    model.config.mask_feature_prob   = 0.05
    model.config.mask_feature_length = 10

    model.config.forced_decoder_ids            = None
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens    = []
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.0f}M parameters")

    # Match inference decoding so best-checkpoint selection is honest (see run_inference).
    model.generation_config.no_repeat_ngram_size = 3

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir                  = CHECKPOINT_DIR,
        per_device_train_batch_size = 16,
        gradient_accumulation_steps = 2,    # effective batch = 32
        learning_rate               = 1e-5, # higher than R3's 5e-6: new random embeddings must move
        warmup_steps                = 400,
        max_steps                   = 3000, # ~3 epochs over ~29k clips; load_best picks the peak
        gradient_checkpointing      = True,
        fp16                        = True,
        optim                       = "adafactor",
        eval_strategy               = "steps",
        per_device_eval_batch_size  = 8,
        predict_with_generate       = True,
        generation_max_length       = 64,   # match inference max_new_tokens=64 so best-checkpoint selection is honest
        generation_num_beams        = 3,     # match inference beam search
        save_steps                  = 750,
        eval_steps                  = 750,   # 4 evals over 3000 steps — beam evals cost A100 time
        save_total_limit            = 3,
        logging_steps               = 50,
        load_best_model_at_end      = True,
        metric_for_best_model       = "wer",
        greater_is_better           = False,
        report_to                   = [],
        push_to_hub                 = False,
        dataloader_num_workers      = 0,
    )

    trainer = Seq2SeqTrainer(
        model            = model,
        args             = training_args,
        train_dataset    = train_ds,
        eval_dataset     = eval_ds,
        data_collator    = data_collator,
        compute_metrics  = compute_metrics,
        processing_class = feature_extractor,
    )

    # Resume is now OPT-IN. A completed run leaves checkpoint-{max_steps} behind;
    # silently resuming it makes the next run do ZERO steps (a wasted A100). And
    # after a vocab change the old optimizer state is shape-incompatible anyway —
    # so unless --resume is passed, wipe stale checkpoints and start the optimizer
    # fresh from the loaded weights.
    checkpoint_dirs = sorted(glob.glob(f"{CHECKPOINT_DIR}/checkpoint-*"))
    if resume and checkpoint_dirs:
        resume_from = checkpoint_dirs[-1]
        print(f"Resuming from {resume_from}")
    else:
        resume_from = None
        for d in checkpoint_dirs:
            shutil.rmtree(d, ignore_errors=True)
        if checkpoint_dirs:
            print(f"Cleared {len(checkpoint_dirs)} stale checkpoint dir(s) — fresh optimizer.")
    print("\nStarting training...")
    trainer.train(resume_from_checkpoint=resume_from)
    print("Training done.")

    trainer.save_model(CHECKPOINT_DIR)
    processor.save_pretrained(CHECKPOINT_DIR)
    vol.commit()
    print(f"Checkpoint saved to Modal volume: {CHECKPOINT_DIR}")

    # Push to HF Hub as backup
    api     = HfApi(token=HF_TOKEN)
    user    = api.whoami()
    repo_id = f"{user['name']}/afrivoices-whisper-small-all6"
    api.create_repo(repo_id, exist_ok=True, private=True)
    model.push_to_hub(repo_id, token=HF_TOKEN)
    processor.push_to_hub(repo_id, token=HF_TOKEN)
    print(f"Pushed to HF Hub: https://huggingface.co/{repo_id}")


@app.function(
    image=image,
    timeout=int(4 * 3600),
    volumes={VOL_PATH: vol},
    secrets=secrets,
    cpu=4,
    memory=32768,
)
def prepare_data():
    """Data harvest on a CPU-only container (~$1.3/hr vs ~$3+/hr for the A100
    stack). Populates the per-language caches on the volume, then exits."""
    _train_impl(data_only=True)


@app.function(
    gpu="A100",
    image=image,
    timeout=int(5.5 * 3600),
    volumes={VOL_PATH: vol},
    secrets=secrets,
    memory=65536,
)
def train(resume: bool = False, fresh: bool = False):
    _train_impl(resume=resume, fresh=fresh)


# ── Inference ─────────────────────────────────────────────────────────────────
@app.function(
    gpu="A100",
    image=image,
    timeout=int(4 * 3600),
    volumes={VOL_PATH: vol},
    secrets=secrets,
    memory=65536,
)
def run_inference(fresh: bool = False):
    import time
    import numpy as np
    import pandas as pd
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from huggingface_hub import login

    HF_TOKEN         = os.environ["HF_TOKEN"]
    KAGGLE_API_TOKEN = os.environ["KAGGLE_API_TOKEN"]  # just the key string
    login(token=HF_TOKEN)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vol.reload()  # pick up checkpoint written by train()

    # Load from volume if available, else HF Hub
    load_from = CHECKPOINT_DIR if os.path.exists(f"{CHECKPOINT_DIR}/config.json") else HF_REPO_ID
    print(f"Loading model from: {load_from}")

    ft_processor = WhisperProcessor.from_pretrained(load_from, token=HF_TOKEN)
    ft_model     = WhisperForConditionalGeneration.from_pretrained(
        load_from, torch_dtype=torch.float16, token=HF_TOKEN
    ).to(device)
    ft_model.eval()
    ft_model.config.forced_decoder_ids            = None
    ft_model.generation_config.forced_decoder_ids = None
    ft_model.generation_config.suppress_tokens    = []
    ft_model.generation_config.max_length         = None
    print(f"Model ready on {device}.")

    # Now that the model has dedicated tokens for the 4 OOV languages, force each
    # one at inference (previously None → the decoder guessed and drifted to swa).
    LANG_TO_WHISPER = {"swa": "sw", "som": "so",
                       "kik": "kik", "luo": "luo", "mas": "mas", "kln": "kln"}

    def transcribe_batch(arrays, language=None):
        arrays = [a[:480_000] for a in arrays]
        inputs = ft_processor(arrays, sampling_rate=16000, return_tensors="pt")\
                   .input_features.to(device).to(torch.float16)
        # Anti-hallucination decoding. The Round-4 CSV showed ~6% of Somali rows were
        # degenerate repetition loops ("oo oo oo …" ×50), each scoring WER 5-8 and
        # inflating Somali's macro WER. Beam search + no-repeat-3gram kill these loops.
        # Both are edge-safe (no_repeat is free on CPU; use num_beams=1 for the Pi report).
        gen_kw = {"max_new_tokens": 64, "num_beams": 3, "no_repeat_ngram_size": 3}
        if language:
            lid = ft_processor.tokenizer.convert_tokens_to_ids(f"<|{language}|>")
            tid = ft_processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")
            nid = ft_processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
            gen_kw["forced_decoder_ids"] = [[1, lid], [2, tid], [3, nid]]
        with torch.no_grad():
            ids = ft_model.generate(input_features=inputs, **gen_kw)
        return ft_processor.batch_decode(ids, skip_special_tokens=True)

    def safe_decode(row):
        try:
            return row.id, row.language, _decode_audio(row.audio)
        except Exception:
            return row.id, row.language, None

    import glob, shutil
    TEST_DATA_DIR = f"{VOL_PATH}/test_parquets"
    all_parquet_files = sorted(glob.glob(os.path.join(TEST_DATA_DIR, "**", "*.parquet"), recursive=True))

    if all_parquet_files:
        print(f"Test data on volume: {len(all_parquet_files)} parquet files — skipping download.")
    else:
        print("Downloading Kaggle test data (first time only — parquets cached to volume)...")
        os.environ["KAGGLE_KEY"] = KAGGLE_API_TOKEN
        import kagglehub
        test_path = kagglehub.dataset_download("digitalumuganda/anv-test-data-nt")
        downloaded = sorted(glob.glob(os.path.join(test_path, "**", "*.parquet"), recursive=True))
        print(f"Copying {len(downloaded)} parquet files to volume...")
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        for pf in downloaded:
            rel = os.path.relpath(pf, test_path)
            dst = os.path.join(TEST_DATA_DIR, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(pf, dst)
        vol.commit()
        all_parquet_files = sorted(glob.glob(os.path.join(TEST_DATA_DIR, "**", "*.parquet"), recursive=True))
        print(f"{len(all_parquet_files)} parquet files saved to volume.")

    CHECKPOINT_FILE = f"{VOL_PATH}/submission_checkpoint.csv"
    BATCH_SIZE      = 32
    SAVE_EVERY      = 5

    model_mtime = os.path.getmtime(f"{CHECKPOINT_DIR}/config.json") \
                  if os.path.exists(f"{CHECKPOINT_DIR}/config.json") else 0
    ckpt_mtime  = os.path.getmtime(CHECKPOINT_FILE) \
                  if os.path.exists(CHECKPOINT_FILE) else 0
    model_is_fresh = model_mtime > ckpt_mtime

    if (model_is_fresh or fresh) and os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        reason = "explicit --fresh flag" if fresh else "new model detected"
        print(f"Starting fresh inference ({reason}).")

    if os.path.exists(CHECKPOINT_FILE):
        existing  = pd.read_csv(CHECKPOINT_FILE)
        empty_pct = (existing["transcription"].isna() |
                     (existing["transcription"].str.strip() == "")).mean()
        if empty_pct > 0.5:
            os.remove(CHECKPOINT_FILE)
            results, done_ids = [], set()
            print("Corrupted checkpoint removed. Starting fresh.")
        else:
            results  = existing.to_dict("records")
            done_ids = set(existing["id"])
            print(f"Resuming: {len(results)} clips done.")
    else:
        results, done_ids = [], set()
        print("Starting fresh.")

    t0 = time.time()
    for pf_idx, pq_path in enumerate(all_parquet_files):
        fname   = os.path.basename(pq_path)
        elapsed = (time.time() - t0) / 60
        eta     = (elapsed / max(pf_idx, 1)) * (len(all_parquet_files) - pf_idx)
        print(f"[{pf_idx+1}/{len(all_parquet_files)}] {fname}  "
              f"({elapsed:.1f} min elapsed, ETA {eta:.0f} min)", flush=True)

        df = pd.read_parquet(pq_path)

        df = df[~df["id"].isin(done_ids)]
        if len(df) == 0:
            print("  already done — skip")
            continue

        lang3   = df["language"].iloc[0]
        wh_lang = LANG_TO_WHISPER.get(lang3)
        print(f"  lang={lang3}  rows={len(df)}", flush=True)

        rows = list(df.itertuples(index=False))
        for i in range(0, len(rows), BATCH_SIZE):
            chunk = rows[i: i + BATCH_SIZE]
            with ThreadPoolExecutor(max_workers=4) as ex:
                decoded = list(ex.map(safe_decode, chunk))

            arrays, batch_ids, batch_langs = [], [], []
            for id_, lang_, arr in decoded:
                if arr is None:
                    results.append({"id": id_, "language": lang_, "transcription": "."})
                    done_ids.add(id_)
                else:
                    arrays.append(arr)
                    batch_ids.append(id_)
                    batch_langs.append(lang_)

            if arrays:
                try:
                    texts = transcribe_batch(arrays, language=wh_lang)
                    for id_, lang_, text in zip(batch_ids, batch_langs, texts):
                        results.append({"id": id_, "language": lang_,
                                        "transcription": text or "."})
                        done_ids.add(id_)
                except Exception as e:
                    print(f"  BATCH ERROR: {e} — one-by-one fallback")
                    for id_, lang_, arr in zip(batch_ids, batch_langs, arrays):
                        try:
                            text = transcribe_batch([arr], language=wh_lang)[0] or "."
                        except Exception:
                            text = "."
                        results.append({"id": id_, "language": lang_, "transcription": text})
                        done_ids.add(id_)

        if (pf_idx + 1) % SAVE_EVERY == 0 or (pf_idx + 1) == len(all_parquet_files):
            pd.DataFrame(results).to_csv(CHECKPOINT_FILE, index=False)
            vol.commit()
            print(f"  → checkpoint saved ({len(results)} total)", flush=True)

    # Build final submission
    sub  = pd.DataFrame(results)
    mask = sub["transcription"].isna() | (sub["transcription"].str.strip() == "")
    sub.loc[mask, "transcription"] = "."
    sub  = sub[["id", "language", "transcription"]]
    sub.to_csv(f"{VOL_PATH}/submission.csv", index=False)
    vol.commit()

    total_min = (time.time() - t0) / 60
    print(f"\nDone! {len(sub)} rows in {total_min:.1f} min")
    print(sub["language"].value_counts().to_string())
    print("\nDownload your submission with:")
    print("  modal volume get afrivoices-vol submission.csv .")


# ── Local entrypoint ─────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(skip_train: bool = False, fresh: bool = False, resume: bool = False):
    if not skip_train:
        print("=== Step 0: Data preparation (CPU container — no GPU billing) ===")
        prepare_data.remote()
        print("=== Step 1/2: Training (A100 starts only now) ===")
        train.remote(resume=resume, fresh=fresh)
    print("\n=== Inference ===")
    run_inference.remote(fresh=fresh)
    print("\nAll done! Download your submission:")
    print("  modal volume get afrivoices-vol submission.csv .")
