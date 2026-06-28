"""
AfriVoices ASR — Modal training + inference script
"""

import modal

import os

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
@app.function(
    gpu="A100",
    image=image,
    timeout=int(5.5 * 3600),
    volumes={VOL_PATH: vol},
    secrets=secrets,
    memory=65536,
)
def train():
    import io, json, tarfile, lzma
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

    # ── Swahili (cache-first, multi-shard) ───────────────────────────────────
    SWA_TARGET = 8000
    SWA_CACHE  = f"{RECORDS_DIR}/swa_records.pkl"
    if os.path.exists(SWA_CACHE):
        swa_records = _load_records(SWA_CACHE)
        print(f"Swahili: {len(swa_records)} clips from cache.")
    else:
        swa_records = []
        tokenizer.set_prefix_tokens(language="swahili", task="transcribe")
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
                    labels = tokenizer(wanted[base]).input_ids
                    swa_records.append({"input_features": feats, "labels": labels})
                    if len(swa_records) % 500 == 0:
                        print(f"  {len(swa_records)}/{SWA_TARGET}", flush=True)
                    if len(swa_records) >= SWA_TARGET:
                        break
        print(f"Swahili: {len(swa_records)} clips ready.")
        _save_records(swa_records, SWA_CACHE)
        vol.commit()

    # ── Somali (cache-first, target-aware) ───────────────────────────────────
    SOM_CACHE  = f"{RECORDS_DIR}/som_records.pkl"
    SOM_TARGET = 4000
    _cached = _load_records(SOM_CACHE) if os.path.exists(SOM_CACHE) else []
    if len(_cached) >= SOM_TARGET:
        som_records = _cached
        print(f"Somali: {len(som_records)} clips from cache.")
    else:
        if _cached:
            print(f"Cache has {len(_cached)} clips, need {SOM_TARGET}. Re-downloading...")
            os.remove(SOM_CACHE)
        print("Scanning Somali manifests from HF Hub...")
        wanted_by_shard = {}
        total_wanted    = 0
        for shard in range(171):
            manifest_path = hf_hub_download(
                repo_id="DigitalUmuganda/Afrivoice",
                filename=f"Somali/manifest_{shard}.json",
                repo_type="dataset", token=HF_TOKEN,
            )
            with open(manifest_path) as f:
                entries = [json.loads(line) for line in f if line.strip()]
            shard_wanted = {}
            for entry in entries:
                text = (entry.get("transcription") or
                        entry.get("normalized_transcription") or "").strip()
                key  = os.path.splitext(
                    os.path.basename(str(entry.get("audio_filepath", "")))
                )[0]
                if text and key:
                    shard_wanted[key] = text
            if shard_wanted:
                wanted_by_shard[shard] = shard_wanted
                total_wanted += len(shard_wanted)
                print(f"  shard {shard}: {len(shard_wanted)} valid (total: {total_wanted})", flush=True)
            if total_wanted >= SOM_TARGET:
                break

        som_records = []
        tokenizer.set_prefix_tokens(language="somali", task="transcribe")
        for shard, wanted in wanted_by_shard.items():
            if len(som_records) >= SOM_TARGET:
                break
            for attempt in range(2):
                audio_tar = hf_hub_download(
                    repo_id="DigitalUmuganda/Afrivoice",
                    filename=f"Somali/audio_shards/audio_{shard}.tar.xz",
                    repo_type="dataset", token=HF_TOKEN,
                    force_download=(attempt > 0),
                )
                try:
                    with tarfile.open(audio_tar, "r:xz") as tar:
                        for member in tar:
                            if member.isdir():
                                continue
                            key = os.path.splitext(os.path.basename(member.name))[0]
                            if key not in wanted:
                                continue
                            try:
                                arr = _decode_audio({"bytes": tar.extractfile(member).read()})
                                arr = arr[:480_000]
                            except Exception:
                                continue
                            feats  = feature_extractor(arr, sampling_rate=16000).input_features[0].astype(np.float16)
                            labels = tokenizer(wanted[key]).input_ids
                            som_records.append({"input_features": feats, "labels": labels})
                            if len(som_records) % 200 == 0:
                                print(f"  {len(som_records)} Somali clips", flush=True)
                            if len(som_records) >= SOM_TARGET:
                                break
                    break
                except (lzma.LZMAError, tarfile.TarError) as e:
                    print(f"  Shard {shard} corrupt (attempt {attempt+1}): {e}")

        print(f"Somali: {len(som_records)} clips ready.")
        _save_records(som_records, SOM_CACHE)
        vol.commit()

    # ── ANV: Kikuyu, Luo, Maasai, Kalenjin (cache-first, target-aware) ───────
    ANV_CACHE    = f"{RECORDS_DIR}/anv_records.pkl"
    TARGET_LANGS = ["kik", "luo", "mas", "kln"]
    ANV_TARGET   = 2000  # per language
    _cached = _load_records(ANV_CACHE) if os.path.exists(ANV_CACHE) else []
    if len(_cached) >= ANV_TARGET * len(TARGET_LANGS):
        anv_records = _cached
        print(f"ANV: {len(anv_records)} clips from cache.")
    else:
        if _cached:
            print(f"Cache has {len(_cached)} clips, need {ANV_TARGET * len(TARGET_LANGS)}. Re-downloading...")
            os.remove(ANV_CACHE)
        print("Listing MCAA1-MSU/anv_data_ke on HF Hub...")
        all_files     = list(list_repo_files("MCAA1-MSU/anv_data_ke", repo_type="dataset", token=HF_TOKEN))
        parquet_files = sorted([f for f in all_files if f.endswith(".parquet")])
        train_files   = {lang: [] for lang in TARGET_LANGS}
        for f in parquet_files:
            parts = f.split("/")
            if parts[0] in TARGET_LANGS and parts[1] == "train":
                train_files[parts[0]].append(f)
        for lang, files in train_files.items():
            print(f"  {lang}: {len(files)} train shards")

        # These languages are not in Whisper's vocab — train without language prefix
        tokenizer.set_prefix_tokens(language=None, task="transcribe")
        buckets = {lang: [] for lang in TARGET_LANGS}
        for lang in TARGET_LANGS:
            print(f"\nLoading {lang}...")
            for shard_path in train_files[lang]:
                if len(buckets[lang]) >= ANV_TARGET:
                    break
                pq_path = hf_hub_download(
                    repo_id="MCAA1-MSU/anv_data_ke",
                    filename=shard_path,
                    repo_type="dataset", token=HF_TOKEN,
                )
                df   = pd.read_parquet(pq_path)
                tcol = ("transcription"  if "transcription"  in df.columns else
                        "actualSentence" if "actualSentence" in df.columns else
                        "transcript"     if "transcript"     in df.columns else None)
                if tcol is None:
                    continue
                for _, row in df.iterrows():
                    if len(buckets[lang]) >= ANV_TARGET:
                        break
                    text = (row.get(tcol) or "").strip()
                    if not text:
                        continue
                    try:
                        arr = _decode_audio(row["audio"])
                        arr = arr[:480_000]
                    except Exception:
                        continue
                    feats  = feature_extractor(arr, sampling_rate=16000).input_features[0].astype(np.float16)
                    labels = tokenizer(text).input_ids
                    buckets[lang].append({"input_features": feats, "labels": labels})
                print(f"  {lang}: {len(buckets[lang])}/{ANV_TARGET}", flush=True)

        anv_records = []
        for lang, recs in buckets.items():
            print(f"  {lang}: {len(recs)} clips")
            anv_records.extend(recs)
        print(f"ANV total: {len(anv_records)} clips ready.")
        _save_records(anv_records, ANV_CACHE)
        vol.commit()

    # ── Combined dataset + train/eval split ───────────────────────────────────
    class WhisperDataset(TorchDataset):
        def __init__(self, records):
            self.records = records
        def __len__(self):
            return len(self.records)
        def __getitem__(self, i):
            return self.records[i]

    all_records = swa_records + som_records + anv_records
    np.random.seed(42)
    np.random.shuffle(all_records)
    split    = int(0.95 * len(all_records))
    train_ds = WhisperDataset(all_records[:split])
    eval_ds  = WhisperDataset(all_records[split:])
    print(f"\nTotal: {len(all_records)}  Train: {len(train_ds)}  Eval: {len(eval_ds)}")

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

    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": round(wer_metric.compute(predictions=pred_str, references=label_str), 4)}

    data_collator = DataCollator(
        processor=processor,
        decoder_start_token_id=processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>"),
    )

    # ── Model (continue from fine-tuned checkpoint, not base model) ──────────
    vol.reload()
    load_from = CHECKPOINT_DIR if os.path.exists(f"{CHECKPOINT_DIR}/config.json") else HF_REPO_ID
    print(f"Loading model from: {load_from}")
    model = WhisperForConditionalGeneration.from_pretrained(load_from, token=HF_TOKEN)
    model.config.forced_decoder_ids            = None
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens    = []
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.0f}M parameters")

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir                  = CHECKPOINT_DIR,
        per_device_train_batch_size = 16,
        gradient_accumulation_steps = 2,    # effective batch = 32
        learning_rate               = 5e-6, # lower LR: fine-tuning a fine-tuned model
        warmup_steps                = 100,
        max_steps                   = 2000, # ~60 min on A100 with larger dataset
        gradient_checkpointing      = True,
        fp16                        = True,
        optim                       = "adafactor",
        eval_strategy               = "steps",
        per_device_eval_batch_size  = 8,
        predict_with_generate       = True,
        generation_max_length       = 225,
        save_steps                  = 300,
        eval_steps                  = 300,
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

    checkpoint_dirs = sorted(glob.glob(f"{CHECKPOINT_DIR}/checkpoint-*"))
    resume_from     = checkpoint_dirs[-1] if checkpoint_dirs else None
    if resume_from:
        print(f"Resuming from {resume_from}")
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


# ── Inference ─────────────────────────────────────────────────────────────────
@app.function(
    gpu="A100",
    image=image,
    timeout=int(4 * 3600),
    volumes={VOL_PATH: vol},
    secrets=secrets,
    memory=65536,
)
def run_inference():
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

    LANG_TO_WHISPER = {"swa": "sw", "som": "so",
                       "kik": None, "luo": None, "mas": None, "kln": None}

    def transcribe_batch(arrays, language=None):
        arrays = [a[:480_000] for a in arrays]
        inputs = ft_processor(arrays, sampling_rate=16000, return_tensors="pt")\
                   .input_features.to(device).to(torch.float16)
        gen_kw = {"max_new_tokens": 64}
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

    # Download test data via kagglehub (new key-only auth, no username needed).
    # Files are cached on the volume so re-runs are instant.
    import glob
    DATASET    = "digitalumuganda/anv-test-data-nt"
    CACHE_DIR  = f"{VOL_PATH}/kaggle_cache"
    os.makedirs(CACHE_DIR, exist_ok=True)
    print("Downloading Kaggle test data (cached on volume after first run)...")
    test_path = _kaggle_download_dataset(DATASET, KAGGLE_API_TOKEN, CACHE_DIR)
    vol.commit()  # persist the downloaded files in case we need to re-run
    all_parquet_files = sorted(glob.glob(os.path.join(test_path, "**", "*.parquet"), recursive=True))
    print(f"{len(all_parquet_files)} parquet files found at {test_path}")

    CHECKPOINT_FILE = f"{VOL_PATH}/submission_checkpoint.csv"
    BATCH_SIZE      = 32
    SAVE_EVERY      = 5

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
                                        "transcription": text.strip() or "."})
                        done_ids.add(id_)
                except Exception as e:
                    print(f"  BATCH ERROR: {e} — one-by-one fallback")
                    for id_, lang_, arr in zip(batch_ids, batch_langs, arrays):
                        try:
                            text = transcribe_batch([arr], language=wh_lang)[0].strip() or "."
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
def main(skip_train: bool = False):
    if not skip_train:
        print("=== Step 1/2: Training ===")
        train.remote()
    print("\n=== Inference ===")
    run_inference.remote()
    print("\nAll done! Download your submission:")
    print("  modal volume get afrivoices-vol submission.csv .")
