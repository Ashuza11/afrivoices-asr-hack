# %% [markdown]
# # AfriVoices ASR — MMS-300M CTC Fine-tune (Kaggle P100)
# Experiment B: wav2vec2-based CTC model. Single forward pass decoding — no autoregressive loop.
# 5-10x faster CPU inference than Whisper. Directly addresses the RTF ≤ 2x edge constraint.

# %% [code]
import subprocess
subprocess.run([
    "pip", "install", "-q",
    "evaluate", "jiwer", "kagglehub", "pydub",
], check=True)

# %% [code]
import os, io, json, tarfile, lzma, glob, time, pickle, re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import soundfile as sf
import librosa
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login, hf_hub_download, list_repo_files, HfApi
from transformers import (
    Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor,
    Wav2Vec2ForCTC, TrainingArguments, Trainer,
)
from torch.utils.data import Dataset as TorchDataset
import evaluate
import kagglehub

secrets  = UserSecretsClient()
HF_TOKEN = secrets.get_secret("HF_TOKEN")
login(token=HF_TOKEN)

MODEL_ID   = "facebook/mms-300m"
HF_REPO_ID = "Ash11/afrivoices-mms-300m-all6"
CACHE_DIR  = "/kaggle/working/records_mms"
CKPT_DIR   = "/kaggle/working/mms-300m-all6"
VOCAB_PATH = "/kaggle/working/vocab_mms.json"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,  exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device} | GPU: {torch.cuda.get_device_name(0) if device == 'cuda' else 'none'}")

# %% [code]
def decode_audio(field):
    if isinstance(field, dict) and "array" in field:
        arr = np.array(field["array"], dtype=np.float32)
        sr  = field.get("sampling_rate", 16000)
        if sr != 16000:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        return arr
    raw = field.get("bytes") if isinstance(field, dict) else field
    if isinstance(raw, bytes) and raw:
        try:
            arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            if sr != 16000:
                arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
            return arr
        except Exception:
            pass
        from pydub import AudioSegment
        seg = (AudioSegment.from_file(io.BytesIO(raw))
               .set_frame_rate(16000).set_channels(1))
        return np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0
    raise ValueError("unreadable audio")


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\sÀ-ɏ̀-ͯḀ-ỿ]", "", text, flags=re.UNICODE)
    return " ".join(text.split())


def save_records(records, path):
    with open(path, "wb") as f:
        pickle.dump(records, f, protocol=4)


def load_records(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# %% [markdown]
# ## Data loading (raw audio + text — no feature extraction yet)
# MMS/CTC stores raw audio arrays; feature extraction happens in the collator.

# %% [code]
SWA_CACHE  = f"{CACHE_DIR}/swa_raw.pkl"
SWA_TARGET = 8000

_cached = load_records(SWA_CACHE) if os.path.exists(SWA_CACHE) else []
if len(_cached) >= SWA_TARGET:
    swa_raw = _cached
    print(f"Swahili: {len(swa_raw)} clips from cache.")
else:
    manifest_path = hf_hub_download(
        repo_id="DigitalUmuganda/Afrivoice_Swahili",
        filename="agriculture_swahili_train/manifest_0.jsonl",
        repo_type="dataset", token=HF_TOKEN,
    )
    with open(manifest_path) as f:
        all_entries = [json.loads(l) for l in f]
    wanted = {}
    for entry in all_entries:
        text = (entry.get("normalized_transcription") or "").strip()
        if text:
            wanted[entry["key"]] = normalize_text(text)
        if len(wanted) >= SWA_TARGET:
            break

    audio_tar = hf_hub_download(
        repo_id="DigitalUmuganda/Afrivoice_Swahili",
        filename="agriculture_swahili_train/audio/audio_0.tar.xz",
        repo_type="dataset", token=HF_TOKEN,
    )
    swa_raw = []
    with tarfile.open(audio_tar, "r:xz") as tar:
        for member in tar:
            if not member.name.endswith(".webm"):
                continue
            key = os.path.basename(member.name).replace(".webm", "")
            if key not in wanted:
                continue
            try:
                arr = decode_audio({"bytes": tar.extractfile(member).read()})
                arr = arr[:480_000]
            except Exception:
                continue
            swa_raw.append({"audio": arr, "text": wanted[key]})
            if len(swa_raw) % 500 == 0:
                print(f"  {len(swa_raw)}/{SWA_TARGET}", flush=True)
            if len(swa_raw) >= SWA_TARGET:
                break
    print(f"Swahili: {len(swa_raw)} clips ready.")
    save_records(swa_raw, SWA_CACHE)

# %% [code]
SOM_CACHE  = f"{CACHE_DIR}/som_raw.pkl"
SOM_TARGET = 4000

_cached = load_records(SOM_CACHE) if os.path.exists(SOM_CACHE) else []
if len(_cached) >= SOM_TARGET:
    som_raw = _cached
    print(f"Somali: {len(som_raw)} clips from cache.")
else:
    wanted_by_shard = {}
    total_wanted    = 0
    for shard in range(171):
        manifest_path = hf_hub_download(
            repo_id="DigitalUmuganda/Afrivoice",
            filename=f"Somali/manifest_{shard}.json",
            repo_type="dataset", token=HF_TOKEN,
        )
        with open(manifest_path) as f:
            entries = [json.loads(l) for l in f if l.strip()]
        shard_wanted = {}
        for entry in entries:
            text = (entry.get("transcription") or entry.get("normalized_transcription") or "").strip()
            key  = os.path.splitext(os.path.basename(str(entry.get("audio_filepath", ""))))[0]
            if text and key:
                shard_wanted[key] = normalize_text(text)
        if shard_wanted:
            wanted_by_shard[shard] = shard_wanted
            total_wanted          += len(shard_wanted)
        if total_wanted >= SOM_TARGET:
            break

    som_raw = []
    for shard, wanted in wanted_by_shard.items():
        if len(som_raw) >= SOM_TARGET:
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
                            arr = decode_audio({"bytes": tar.extractfile(member).read()})
                            som_raw.append({"audio": arr[:480_000], "text": wanted[key]})
                        except Exception:
                            continue
                        if len(som_raw) >= SOM_TARGET:
                            break
                break
            except (lzma.LZMAError, tarfile.TarError):
                if attempt == 1:
                    print(f"  shard {shard} corrupt, skipping.")
    print(f"Somali: {len(som_raw)} clips ready.")
    save_records(som_raw, SOM_CACHE)

# %% [code]
ANV_CACHE    = f"{CACHE_DIR}/anv_raw.pkl"
TARGET_LANGS = ["kik", "luo", "mas", "kln"]
ANV_TARGET   = 2000

_cached = load_records(ANV_CACHE) if os.path.exists(ANV_CACHE) else []
if len(_cached) >= ANV_TARGET * len(TARGET_LANGS):
    anv_raw = _cached
    print(f"ANV: {len(anv_raw)} clips from cache.")
else:
    all_files     = list(list_repo_files("MCAA1-MSU/anv_data_ke", repo_type="dataset", token=HF_TOKEN))
    parquet_files = sorted([f for f in all_files if f.endswith(".parquet")])
    train_files   = {lang: [] for lang in TARGET_LANGS}
    for f in parquet_files:
        parts = f.split("/")
        if parts[0] in TARGET_LANGS and parts[1] == "train":
            train_files[parts[0]].append(f)

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
            tcol = next((c for c in ["transcription", "actualSentence", "transcript"] if c in df.columns), None)
            if tcol is None:
                continue
            for _, row in df.iterrows():
                if len(buckets[lang]) >= ANV_TARGET:
                    break
                text = normalize_text((row.get(tcol) or ""))
                if not text:
                    continue
                try:
                    arr = decode_audio(row["audio"])
                    buckets[lang].append({"audio": arr[:480_000], "text": text})
                except Exception:
                    continue
            print(f"  {lang}: {len(buckets[lang])}/{ANV_TARGET}", flush=True)

    anv_raw = [r for lang in TARGET_LANGS for r in buckets[lang]]
    print(f"ANV total: {len(anv_raw)} clips.")
    save_records(anv_raw, ANV_CACHE)

# %% [markdown]
# ## Vocabulary — build shared character set from all transcripts

# %% [code]
all_raw = swa_raw + som_raw + anv_raw

if os.path.exists(VOCAB_PATH):
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)
    print(f"Vocab loaded: {len(vocab)} tokens")
else:
    chars = Counter()
    for r in all_raw:
        chars.update(r["text"].replace(" ", "|"))
    vocab = {c: i for i, c in enumerate(sorted(chars.keys()))}
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)
    with open(VOCAB_PATH, "w") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Vocab built: {len(vocab)} tokens")

tokenizer_ctc = Wav2Vec2CTCTokenizer(
    VOCAB_PATH,
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|",
)
feature_extractor_mms = Wav2Vec2FeatureExtractor.from_pretrained(
    MODEL_ID, return_attention_mask=True,
)
processor_mms = Wav2Vec2Processor(
    feature_extractor=feature_extractor_mms,
    tokenizer=tokenizer_ctc,
)

# %% [markdown]
# ## Dataset + collator

# %% [code]
class CTCDataset(TorchDataset):
    def __init__(self, records):
        self.records = records
    def __len__(self):
        return len(self.records)
    def __getitem__(self, i):
        r      = self.records[i]
        inputs = processor_mms(r["audio"], sampling_rate=16000)
        labels = processor_mms.tokenizer(r["text"]).input_ids
        return {
            "input_values":     inputs.input_values[0],
            "attention_mask":   inputs.attention_mask[0],
            "labels":           labels,
        }


np.random.seed(42)
np.random.shuffle(all_raw)
split    = int(0.95 * len(all_raw))
train_ds = CTCDataset(all_raw[:split])
eval_ds  = CTCDataset(all_raw[split:])
print(f"Total: {len(all_raw)}  Train: {len(train_ds)}  Eval: {len(eval_ds)}")


@dataclass
class CTCCollator:
    processor: Any
    padding: bool = True

    def __call__(self, features: List[Dict[str, Any]]):
        input_values = [{"input_values": f["input_values"]} for f in features]
        label_ids    = [f["labels"] for f in features]

        batch = self.processor.pad(input_values, padding=self.padding, return_tensors="pt")

        max_len = max(len(l) for l in label_ids)
        labels  = torch.full((len(label_ids), max_len), -100, dtype=torch.long)
        for i, l in enumerate(label_ids):
            labels[i, :len(l)] = torch.tensor(l, dtype=torch.long)

        batch["labels"] = labels
        return batch


wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    logits    = pred.predictions
    pred_ids  = np.argmax(logits, axis=-1)
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor_mms.tokenizer.pad_token_id
    pred_str  = processor_mms.batch_decode(pred_ids)
    label_str = processor_mms.batch_decode(label_ids, group_tokens=False)
    return {"wer": round(wer_metric.compute(predictions=pred_str, references=label_str), 4)}


collator = CTCCollator(processor=processor_mms)

# %% [markdown]
# ## Model

# %% [code]
model_mms = Wav2Vec2ForCTC.from_pretrained(
    MODEL_ID,
    ctc_loss_reduction="mean",
    pad_token_id=processor_mms.tokenizer.pad_token_id,
    vocab_size=len(processor_mms.tokenizer),
    ignore_mismatched_sizes=True,
)
model_mms.freeze_feature_encoder()
print(f"Model: {sum(p.numel() for p in model_mms.parameters())/1e6:.0f}M parameters")

# %% [markdown]
# ## Training

# %% [code]
training_args = TrainingArguments(
    output_dir                  = CKPT_DIR,
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 4,
    learning_rate               = 3e-4,
    warmup_steps                = 200,
    max_steps                   = 1000,
    gradient_checkpointing      = True,
    fp16                        = True,
    eval_strategy               = "steps",
    per_device_eval_batch_size  = 8,
    save_steps                  = 250,
    eval_steps                  = 250,
    save_total_limit            = 2,
    logging_steps               = 50,
    load_best_model_at_end      = True,
    metric_for_best_model       = "wer",
    greater_is_better           = False,
    report_to                   = [],
    dataloader_num_workers      = 0,
)

trainer = Trainer(
    model         = model_mms,
    args          = training_args,
    train_dataset = train_ds,
    eval_dataset  = eval_ds,
    data_collator = collator,
    compute_metrics = compute_metrics,
    processing_class = feature_extractor_mms,
)

trainer.train()
trainer.save_model(CKPT_DIR)
processor_mms.save_pretrained(CKPT_DIR)

api     = HfApi(token=HF_TOKEN)
user    = api.whoami()
repo_id = f"{user['name']}/afrivoices-mms-300m-all6"
api.create_repo(repo_id, exist_ok=True, private=True)
model_mms.push_to_hub(repo_id, token=HF_TOKEN)
processor_mms.push_to_hub(repo_id, token=HF_TOKEN)
print(f"Pushed to: https://huggingface.co/{repo_id}")

# %% [markdown]
# ## Inference — CTC greedy decode (no autoregressive loop)

# %% [code]
ft_model_mms     = Wav2Vec2ForCTC.from_pretrained(CKPT_DIR).to(device)
ft_processor_mms = Wav2Vec2Processor.from_pretrained(CKPT_DIR)
ft_model_mms.eval()


def transcribe_batch_ctc(arrays):
    inputs = ft_processor_mms(
        arrays, sampling_rate=16000, return_tensors="pt", padding=True,
    ).to(device)
    with torch.no_grad():
        logits = ft_model_mms(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return ft_processor_mms.batch_decode(pred_ids)


def safe_decode(row):
    try:
        return row.id, row.language, decode_audio(row.audio)
    except Exception:
        return row.id, row.language, None


test_path         = kagglehub.dataset_download("digitalumuganda/anv-test-data-nt")
all_parquet_files = sorted(glob.glob(os.path.join(test_path, "**", "*.parquet"), recursive=True))
print(f"{len(all_parquet_files)} parquet files found.")

CHECKPOINT_FILE = "/kaggle/working/submission_mms_checkpoint.csv"
BATCH_SIZE      = 32
SAVE_EVERY      = 5

if os.path.exists(CHECKPOINT_FILE):
    existing  = pd.read_csv(CHECKPOINT_FILE)
    empty_pct = (existing["transcription"].isna() | (existing["transcription"].str.strip() == "")).mean()
    if empty_pct > 0.5:
        os.remove(CHECKPOINT_FILE)
        results, done_ids = [], set()
    else:
        results  = existing.to_dict("records")
        done_ids = set(existing["id"])
        print(f"Resuming: {len(results)} done.")
else:
    results, done_ids = [], set()

t0 = time.time()
for pf_idx, pq_path in enumerate(all_parquet_files):
    fname   = os.path.basename(pq_path)
    elapsed = (time.time() - t0) / 60
    eta     = (elapsed / max(pf_idx, 1)) * (len(all_parquet_files) - pf_idx)
    print(f"[{pf_idx+1}/{len(all_parquet_files)}] {fname}  ({elapsed:.1f}m, ETA {eta:.0f}m)", flush=True)

    df = pd.read_parquet(pq_path)
    df = df[~df["id"].isin(done_ids)]
    if df.empty:
        continue

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
                texts = transcribe_batch_ctc(arrays)
                for id_, lang_, text in zip(batch_ids, batch_langs, texts):
                    results.append({"id": id_, "language": lang_, "transcription": text.strip() or "."})
                    done_ids.add(id_)
            except Exception as e:
                print(f"  batch error: {e} — one-by-one")
                for id_, lang_, arr in zip(batch_ids, batch_langs, arrays):
                    try:
                        text = transcribe_batch_ctc([arr])[0].strip() or "."
                    except Exception:
                        text = "."
                    results.append({"id": id_, "language": lang_, "transcription": text})
                    done_ids.add(id_)

    if (pf_idx + 1) % SAVE_EVERY == 0 or (pf_idx + 1) == len(all_parquet_files):
        pd.DataFrame(results).to_csv(CHECKPOINT_FILE, index=False)
        print(f"  checkpoint saved ({len(results)} total)", flush=True)

sub = pd.DataFrame(results)
sub.loc[sub["transcription"].isna() | (sub["transcription"].str.strip() == ""), "transcription"] = "."
sub[["id", "language", "transcription"]].to_csv("/kaggle/working/submission_mms.csv", index=False)
print(f"\nDone: {len(sub)} rows in {(time.time()-t0)/60:.1f} min")
print(sub["language"].value_counts().to_string())
