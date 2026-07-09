# %% [markdown]
# # AfriVoices ASR — Round 6 Kaggle Whisper-small
#
# Practical Round 6 plan:
# - Keep the Modal Whisper-small pipeline shape.
# - Train with the leaderboard-validated normalization:
#   lowercase + punctuation removal + whitespace collapse.
# - Restore/expand clean scripted Maasai and Kalenjin.
# - Keep spontaneous speech and Maxatire Somali.
# - Use targeted speed perturbation for Maasai/Kalenjin only.
# - Run one strong Kaggle training job first; optional CV is off by default.

# %% [code]
# ============================================================
# 0. Install packages
# ============================================================

import subprocess
import sys

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "transformers==4.46.3",
        "datasets==2.20.0",
        "accelerate>=0.26.0",
        "evaluate",
        "jiwer",
        "soundfile",
        "librosa",
        "pydub",
        "huggingface_hub>=0.21",
        "requests",
        "kagglehub",
    ],
    check=True,
)

print("Packages installed.")


# %% [code]
# ============================================================
# 1. Imports and configuration
# ============================================================

import gc
import glob
import io
import json
import os
import pickle
import resource
import random
import re
import shutil
import tarfile
import tempfile
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List

import evaluate
import kagglehub
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from huggingface_hub import HfApi, hf_hub_download, list_repo_files, login
from kaggle_secrets import UserSecretsClient
from pydub import AudioSegment
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

secrets = UserSecretsClient()
HF_TOKEN = secrets.get_secret("HF_TOKEN")
login(token=HF_TOKEN)

MODEL_ID = "openai/whisper-small"
PREVIOUS_REPO_ID = "Ash11/afrivoices-whisper-small-all6"
ROUND_REPO_NAME = "afrivoices-whisper-small-round6"

WORK_DIR = "/kaggle/working"
CACHE_DIR = f"{WORK_DIR}/records_round6_raw_v2"
CHECKPOINT_DIR = f"{WORK_DIR}/whisper-small-round6"
OUTPUT_DIR = f"{WORK_DIR}/outputs_round6"
TEST_CACHE = f"{WORK_DIR}/test_parquets"

for folder in (CACHE_DIR, CHECKPOINT_DIR, OUTPUT_DIR, TEST_CACHE):
    os.makedirs(folder, exist_ok=True)

SAMPLE_RATE = 16000
MAX_AUDIO_SAMPLES = 480_000
MAX_LABEL_LEN = 448

# Start from base by default because labels are now normalized. If you are short
# on GPU time, set this to PREVIOUS_REPO_ID to adapt from Round 5 instead.
START_FROM = MODEL_ID

# Kaggle RAM is the bottleneck. Do not cache Whisper feature tensors for every
# clip; keep compact audio bytes and compute features inside the collator.
# This prevents the data-prep phase from filling system RAM before training.
SCRIPTED_TARGETS = {
    "swa": 2500,
    "som": 2500,
    "kik": 3000,
    "luo": 2500,
    "mas": 3000,
    "kln": 3000,
}

UNSCRIPTED_TARGETS = {
    "som": 1000,
    "kik": 1000,
    "luo": 1000,
    "mas": 1000,
    "kln": 1000,
}

MAX_UNSCRIPTED_SHARDS = 12
MEMORY_STOP_GB = 24.0

ENABLE_SPEED_PERTURBATION = False
SPEED_PERTURB_LANGS = {"mas", "kln"}
SPEED_FACTORS = [0.9, 1.1]
MAX_AUGMENTED_PER_LANG = 1000

MAX_STEPS = 3000
LEARNING_RATE = 1e-5
WARMUP_STEPS = 500
PER_DEVICE_TRAIN_BATCH = 16
GRADIENT_ACCUMULATION = 2
PER_DEVICE_EVAL_BATCH = 4
GEN_MAX_LENGTH = 64
GEN_BEAMS = 3

RUN_PREPARE_DATA = True
RUN_TRAINING = False
RUN_INFERENCE = False
PUSH_TO_HUB = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# %% [code]
# ============================================================
# 2. Normalization and audio helpers
# ============================================================

ALL_PUNCT_RE = re.compile(r"[^\w\sÀ-ɏ̀-ͯḀ-ỿ'’ŋŊ]", flags=re.UNICODE)


def normalize_text(text: str) -> str:
    """Leaderboard-validated conservative normalization.

    Round 5 submission variants proved that lowercase + punctuation removal
    beat raw Whisper-style text by 0.04255 public WER.
    """
    text = "" if text is None else str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = ALL_PUNCT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def postprocess_submission_text(text: str) -> str:
    return normalize_text(text) or "."


def decode_audio(audio_field):
    if isinstance(audio_field, dict) and "array" in audio_field:
        arr = np.array(audio_field["array"], dtype=np.float32)
        sr = audio_field.get("sampling_rate", 16000)
        if sr != 16000:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        return arr

    raw = audio_field.get("bytes") if isinstance(audio_field, dict) else audio_field
    if isinstance(raw, bytes) and len(raw) > 0:
        try:
            arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            if sr != 16000:
                arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
            return arr
        except Exception:
            pass
        seg = AudioSegment.from_file(io.BytesIO(raw)).set_frame_rate(16000).set_channels(1)
        return np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0

    raise ValueError(f"Cannot decode audio: type={type(audio_field)}")


def too_long(audio_field):
    if isinstance(audio_field, dict):
        arr = audio_field.get("array")
        if arr is not None:
            sr = audio_field.get("sampling_rate") or 16000
            return len(arr) / sr > 30.0
        raw = audio_field.get("bytes")
        if raw:
            try:
                info = sf.info(io.BytesIO(raw))
                return info.frames / info.samplerate > 30.0
            except Exception:
                return None
    return None


def speed_perturb(arr: np.ndarray, factor: float) -> np.ndarray:
    y = librosa.effects.time_stretch(arr.astype(np.float32), rate=factor)
    if len(y) > MAX_AUDIO_SAMPLES:
        y = y[:MAX_AUDIO_SAMPLES]
    return y.astype(np.float32, copy=False)


def current_rss_gb() -> float:
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss_kb / 1e6


def memory_guard(context: str):
    rss = current_rss_gb()
    if rss >= MEMORY_STOP_GB:
        raise MemoryError(
            f"RAM guard tripped during {context}: max RSS {rss:.1f} GB >= {MEMORY_STOP_GB:.1f} GB. "
            "Reduce targets or run training with the caches already prepared."
        )


def save_records(records, path):
    packed = [
        {
            "audio_bytes": r.get("audio_bytes"),
            "audio_array": r.get("audio_array"),
            "labels": r["labels"],
            "lang": r["lang"],
            "source": r["source"],
        }
        for r in records
    ]
    with open(path, "wb") as f:
        pickle.dump(packed, f, protocol=4)
    print(f"Saved {len(records)} records -> {path} ({os.path.getsize(path)/1e6:.0f} MB)")


def load_records(path):
    with open(path, "rb") as f:
        records = pickle.load(f)
    if records and "input_features" in records[0]:
        raise RuntimeError(
            f"{path} is an old feature-tensor cache. Delete it or use the raw_v2 cache path."
        )
    return records


# %% [code]
# ============================================================
# 3. Whisper processor, tokenizer, and language tokens
# ============================================================

processor = WhisperProcessor.from_pretrained(MODEL_ID)
feature_extractor = processor.feature_extractor
tokenizer = processor.tokenizer

NEW_LANG_TOKENS = ["<|kik|>", "<|luo|>", "<|mas|>", "<|kln|>"]
tokenizer.add_tokens(NEW_LANG_TOKENS, special_tokens=True)

SOT = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
TRANSCRIBE = tokenizer.convert_tokens_to_ids("<|transcribe|>")
NOTIMESTAMPS = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
EOT = tokenizer.convert_tokens_to_ids("<|endoftext|>")

LANG_TOKEN = {
    "swa": "<|sw|>",
    "som": "<|so|>",
    "kik": "<|kik|>",
    "luo": "<|luo|>",
    "mas": "<|mas|>",
    "kln": "<|kln|>",
}


def build_labels(text: str, lang3: str):
    text = normalize_text(text)
    if not text:
        return None
    lang_id = tokenizer.convert_tokens_to_ids(LANG_TOKEN[lang3])
    text_ids = tokenizer(text, add_special_tokens=False).input_ids
    labels = [SOT, lang_id, TRANSCRIBE, NOTIMESTAMPS] + text_ids + [EOT]
    if len(labels) > MAX_LABEL_LEN:
        return None
    return labels


def make_record_from_bytes(audio_bytes: bytes, text: str, lang: str, source: str):
    labels = build_labels(text, lang)
    if labels is None:
        return None
    return {
        "audio_bytes": audio_bytes,
        "audio_array": None,
        "labels": labels,
        "lang": lang,
        "source": source,
    }


def make_record_from_array(arr: np.ndarray, text: str, lang: str, source: str):
    labels = build_labels(text, lang)
    if labels is None:
        return None
    return {
        "audio_bytes": None,
        "audio_array": arr[:MAX_AUDIO_SAMPLES].astype(np.float16, copy=False),
        "labels": labels,
        "lang": lang,
        "source": source,
    }


def make_record(audio_field, text: str, lang: str, source: str):
    raw = audio_field.get("bytes") if isinstance(audio_field, dict) else audio_field
    if isinstance(raw, bytes) and raw:
        return make_record_from_bytes(raw, text, lang, source)
    arr = decode_audio(audio_field)
    return make_record_from_array(arr, text, lang, source)


print(f"Tokenizer size: {len(tokenizer)}")


# %% [code]
# ============================================================
# 4. Swahili loader
# ============================================================


def load_swahili():
    cache = f"{CACHE_DIR}/swa_records_norm.pkl"
    target = SCRIPTED_TARGETS["swa"]
    if os.path.exists(cache):
        recs = load_records(cache)
        print(f"Swahili from cache: {len(recs)}", flush=True)
        return recs[:target]

    records = []
    print(f"Loading Swahili target={target}", flush=True)
    for shard in range(10):
        if len(records) >= target:
            break
        print(f"Swahili shard {shard}: downloading manifest/audio if not cached", flush=True)
        try:
            manifest_path = hf_hub_download(
                repo_id="DigitalUmuganda/Afrivoice_Swahili",
                filename=f"agriculture_swahili_train/manifest_{shard}.jsonl",
                repo_type="dataset",
                token=HF_TOKEN,
            )
            audio_tar = hf_hub_download(
                repo_id="DigitalUmuganda/Afrivoice_Swahili",
                filename=f"agriculture_swahili_train/audio/audio_{shard}.tar.xz",
                repo_type="dataset",
                token=HF_TOKEN,
            )
        except Exception as e:
            print(f"Swahili shard {shard} unavailable: {e}", flush=True)
            break

        with open(manifest_path) as f:
            entries = [json.loads(line) for line in f if line.strip()]

        wanted = {}
        for entry in entries:
            text = (
                entry.get("normalized_transcription")
                or entry.get("transcription")
                or entry.get("transcript")
                or entry.get("text")
                or ""
            ).strip()
            if text:
                key = entry.get("key", "")
                wanted[key] = text
                wanted[os.path.basename(key)] = text

        with tarfile.open(audio_tar, "r:xz") as tar:
            members = tar.getmembers()
            print(
                f"Swahili shard {shard}: scanning {len(members)} tar members "
                f"against {len(wanted)} manifest keys",
                flush=True,
            )
            for member_idx, member in enumerate(members):
                if len(records) >= target:
                    break
                if member_idx and member_idx % 5000 == 0:
                    print(
                        f"Swahili shard {shard}: scanned {member_idx}/{len(members)} "
                        f"members, records={len(records)}/{target}",
                        flush=True,
                    )
                ext = os.path.splitext(member.name)[1].lower()
                if ext not in (".webm", ".wav", ".mp3", ".flac", ".ogg", ".opus"):
                    continue
                base = os.path.splitext(os.path.basename(member.name))[0]
                if base not in wanted:
                    continue
                try:
                    raw = tar.extractfile(member).read()
                    rec = make_record_from_bytes(raw, wanted[base], "swa", "scripted")
                except Exception:
                    rec = None
                if rec:
                    records.append(rec)
                if len(records) % 500 == 0 and len(records) > 0:
                    print(f"Swahili: {len(records)}/{target}", flush=True)
                    memory_guard("Swahili harvest")

    save_records(records, cache)
    return records


# %% [code]
# ============================================================
# 5. ANV loader: kik, luo, mas, kln, som
# ============================================================


def download_shard(shard_path: str, td: str):
    for attempt in range(3):
        sub = os.path.join(td, f"try{attempt}")
        ex = ThreadPoolExecutor(max_workers=1)
        try:
            fut = ex.submit(
                hf_hub_download,
                repo_id="MCAA1-MSU/anv_data_ke",
                filename=shard_path,
                repo_type="dataset",
                token=HF_TOKEN,
                local_dir=sub,
            )
            return fut.result(timeout=600)
        except Exception as e:
            print(f"Download attempt {attempt + 1} failed for {shard_path}: {type(e).__name__} {str(e)[:100]}")
        finally:
            ex.shutdown(wait=False)
    return None


def list_anv_shards():
    all_files = list(list_repo_files("MCAA1-MSU/anv_data_ke", repo_type="dataset", token=HF_TOKEN))
    parquet_files = sorted(f for f in all_files if f.endswith(".parquet"))
    langs = ["kik", "luo", "mas", "kln", "som"]
    shards = {(lang, sub): [] for lang in langs for sub in ("scripted", "unscripted")}
    for f in parquet_files:
        parts = f.split("/")
        if len(parts) >= 4 and parts[0] in langs and parts[1] == "train" and parts[2] in ("scripted", "unscripted"):
            shards[(parts[0], parts[2])].append(f)
    for key in sorted(shards):
        print(f"{key[0]}/{key[1]}: {len(shards[key])} shards")
    return shards


def harvest_anv(lang: str, subtype: str, target: int, shard_list: List[str], max_augmented: int = 0):
    records = []
    augmented = []
    skipped_long = 0
    per_factor_limit = max_augmented // max(len(SPEED_FACTORS), 1) if max_augmented else 0
    made_by_factor = {f: 0 for f in SPEED_FACTORS}

    for shard_path in shard_list:
        if len(records) >= target:
            break
        td = tempfile.mkdtemp()
        try:
            pq_path = download_shard(shard_path, td)
            if pq_path is None:
                continue
            df = pd.read_parquet(pq_path)
            tcol = next((c for c in ("transcription", "actualSentence", "transcript") if c in df.columns), None)
            if tcol is None or "audio" not in df.columns:
                continue

            for _, row in df.iterrows():
                if len(records) >= target:
                    break
                text = (row.get(tcol) or "").strip()
                if not text:
                    continue
                long = too_long(row["audio"])
                if long:
                    skipped_long += 1
                    continue
                try:
                    rec = make_record(row["audio"], text, lang, subtype)
                except Exception:
                    rec = None
                if rec:
                    records.append(rec)
                    if (
                        max_augmented
                        and subtype == "scripted"
                        and lang in SPEED_PERTURB_LANGS
                        and len(augmented) < max_augmented
                    ):
                        for factor in SPEED_FACTORS:
                            if made_by_factor[factor] >= per_factor_limit:
                                continue
                            try:
                                arr = decode_audio(row["audio"])
                                aug_arr = speed_perturb(arr, factor)
                                aug_rec = make_record_from_array(aug_arr, text, lang, f"scripted_speed_{factor}")
                            except Exception:
                                aug_rec = None
                            if aug_rec:
                                augmented.append(aug_rec)
                                made_by_factor[factor] += 1
                            if len(augmented) >= max_augmented:
                                break
        finally:
            shutil.rmtree(td, ignore_errors=True)
        msg = f"{lang}/{subtype}: {len(records)}/{target} skipped_long={skipped_long}"
        if max_augmented:
            msg += f" augmented={len(augmented)}/{max_augmented}"
        print(msg)
        del df
        gc.collect()
        memory_guard(f"{lang}/{subtype} harvest")
    return records + augmented


def load_anv_language(lang: str, shards):
    cache = f"{CACHE_DIR}/anv_{lang}_round6_norm.pkl"
    if os.path.exists(cache):
        recs = load_records(cache)
        print(f"ANV {lang} from cache: {len(recs)}")
        return recs

    scripted_target = SCRIPTED_TARGETS.get(lang, 0)
    unscripted_target = UNSCRIPTED_TARGETS.get(lang, 0)
    scripted_shards = shards[(lang, "scripted")]
    unscripted_shards = shards[(lang, "unscripted")][:MAX_UNSCRIPTED_SHARDS]

    aug_cap = MAX_AUGMENTED_PER_LANG if ENABLE_SPEED_PERTURBATION and lang in SPEED_PERTURB_LANGS else 0
    scripted = harvest_anv(lang, "scripted", scripted_target, scripted_shards, max_augmented=aug_cap)
    unscripted = harvest_anv(lang, "unscripted", unscripted_target, unscripted_shards)
    records = scripted + unscripted

    save_records(records, cache)
    return records


def load_all_records():
    print("=== Data loading starts ===", flush=True)
    records = []
    records.extend(load_swahili())
    print(f"After Swahili: records={len(records)} max_rss={current_rss_gb():.1f} GB", flush=True)
    memory_guard("load_all_records/Swahili")
    shards = list_anv_shards()
    for lang in ["som", "kik", "luo", "mas", "kln"]:
        print(f"=== Loading ANV language: {lang} ===", flush=True)
        records.extend(load_anv_language(lang, shards))
        print(f"After {lang}: records={len(records)} max_rss={current_rss_gb():.1f} GB", flush=True)
        memory_guard(f"load_all_records/{lang}")
    print(f"Total records before split: {len(records)}", flush=True)
    print(pd.Series([r["lang"] for r in records]).value_counts().sort_index().to_string(), flush=True)
    print(pd.Series([r["source"] for r in records]).value_counts().to_string(), flush=True)
    return records


def prepare_data_caches():
    """Create per-language record caches without retaining all languages in RAM."""
    print("=== Data cache preparation starts ===", flush=True)

    recs = load_swahili()
    print(f"Cached Swahili records: {len(recs)}", flush=True)
    del recs
    gc.collect()
    print(f"RAM after Swahili cache: max_rss={current_rss_gb():.1f} GB", flush=True)
    memory_guard("prepare_data_caches/Swahili")

    shards = list_anv_shards()
    for lang in ["som", "kik", "luo", "mas", "kln"]:
        print(f"=== Preparing ANV cache for: {lang} ===", flush=True)
        recs = load_anv_language(lang, shards)
        print(f"Cached {lang} records: {len(recs)}", flush=True)
        del recs
        gc.collect()
        print(f"RAM after {lang} cache: max_rss={current_rss_gb():.1f} GB", flush=True)
        memory_guard(f"prepare_data_caches/{lang}")

    print("=== Data cache preparation complete ===", flush=True)


# %% [code]
# ============================================================
# 6. Dataset split, collator, and metrics
# ============================================================


class WhisperDataset(TorchDataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def stratified_split(records, eval_frac=0.05):
    by_bucket = {}
    for r in records:
        key = (r["lang"], "unscripted" if "unscripted" in r["source"] else "scripted")
        by_bucket.setdefault(key, []).append(r)

    train, eval_ = [], []
    rng = np.random.default_rng(SEED)
    for key, bucket in sorted(by_bucket.items()):
        bucket = list(bucket)
        rng.shuffle(bucket)
        n_eval = max(1, int(round(len(bucket) * eval_frac)))
        eval_.extend(bucket[:n_eval])
        train.extend(bucket[n_eval:])
        print(f"{key}: train={len(bucket[n_eval:])} eval={len(bucket[:n_eval])}")

    rng.shuffle(train)
    rng.shuffle(eval_)
    return train, eval_


@dataclass
class DataCollator:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]):
        arrays = []
        for f in features:
            if f.get("audio_bytes"):
                arr = decode_audio({"bytes": f["audio_bytes"]})
            elif f.get("audio_array") is not None:
                arr = np.asarray(f["audio_array"], dtype=np.float32)
            else:
                raise ValueError("Record has neither audio_bytes nor audio_array")
            arrays.append(arr[:MAX_AUDIO_SAMPLES])
        input_features = self.processor.feature_extractor(
            arrays,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        ).input_features
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        return {"input_features": input_features, "labels": labels}


wer_metric = evaluate.load("wer")
LANG_TOKEN_ID = {tokenizer.convert_tokens_to_ids(tok): lang for lang, tok in LANG_TOKEN.items()}


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    first_tok = label_ids[:, 0].copy()
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = [postprocess_submission_text(s) for s in tokenizer.batch_decode(pred_ids, skip_special_tokens=True)]
    label_str = [postprocess_submission_text(s) for s in tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

    metrics = {"wer": round(wer_metric.compute(predictions=pred_str, references=label_str), 4)}
    parts = []
    for tok_id, lang in LANG_TOKEN_ID.items():
        idx = [i for i, t in enumerate(first_tok) if t == tok_id]
        if idx:
            w = round(
                wer_metric.compute(
                    predictions=[pred_str[i] for i in idx],
                    references=[label_str[i] for i in idx],
                ),
                4,
            )
            metrics[f"wer_{lang}"] = w
            parts.append(f"{lang}={w:.3f}")
    print("per-language WER: " + " ".join(parts), flush=True)
    return metrics


# %% [code]
# ============================================================
# 7. Model and training
# ============================================================


def load_model():
    print(f"Loading model from: {START_FROM}")
    model = WhisperForConditionalGeneration.from_pretrained(START_FROM, token=HF_TOKEN)

    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        old_vocab = model.get_input_embeddings().num_embeddings
        model.resize_token_embeddings(len(tokenizer))
        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            sw_id = tokenizer.convert_tokens_to_ids("<|sw|>")
            for tok in NEW_LANG_TOKENS:
                emb[tokenizer.convert_tokens_to_ids(tok)] = emb[sw_id].clone()
        print(f"Resized embeddings {old_vocab} -> {len(tokenizer)}")

    model.config.apply_spec_augment = True
    model.config.mask_time_prob = 0.05
    model.config.mask_time_length = 10
    model.config.mask_feature_prob = 0.05
    model.config.mask_feature_length = 10

    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    model.generation_config.no_repeat_ngram_size = 3
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    return model


def train_round6(records=None):
    print("=== Round 6 training wrapper starts ===", flush=True)
    if records is None:
        print("No prepared records were passed; loading cached/preprocessed records first", flush=True)
        records = load_all_records()
    print("=== Data loading complete; building stratified split ===", flush=True)
    train_records, eval_records = stratified_split(records, eval_frac=0.05)
    train_ds = WhisperDataset(train_records)
    eval_ds = WhisperDataset(eval_records)
    print(f"Train={len(train_ds)} Eval={len(eval_ds)}", flush=True)

    data_collator = DataCollator(
        processor=processor,
        decoder_start_token_id=tokenizer.convert_tokens_to_ids("<|startoftranscript|>"),
    )
    model = load_model()

    training_args = Seq2SeqTrainingArguments(
        output_dir=CHECKPOINT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        lr_scheduler_type="cosine",
        label_smoothing_factor=0.05,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        fp16=True,
        optim="adafactor",
        eval_strategy="steps",
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
        predict_with_generate=True,
        generation_max_length=GEN_MAX_LENGTH,
        generation_num_beams=GEN_BEAMS,
        save_steps=1000,
        eval_steps=1000,
        save_total_limit=3,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=[],
        push_to_hub=False,
        dataloader_num_workers=0,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=feature_extractor,
    )

    print("=== Actual GPU training starts now ===", flush=True)
    trainer.train()
    trainer.save_model(CHECKPOINT_DIR)
    processor.save_pretrained(CHECKPOINT_DIR)
    print(f"Saved best model to {CHECKPOINT_DIR}")

    if PUSH_TO_HUB:
        api = HfApi(token=HF_TOKEN)
        user = api.whoami()
        repo_id = f"{user['name']}/{ROUND_REPO_NAME}"
        api.create_repo(repo_id, exist_ok=True, private=True)
        trainer.model.push_to_hub(repo_id, token=HF_TOKEN)
        processor.push_to_hub(repo_id, token=HF_TOKEN)
        print(f"Pushed to https://huggingface.co/{repo_id}")

    del trainer
    gc.collect()
    torch.cuda.empty_cache()


# %% [code]
# ============================================================
# 8. Prepare data only
# ============================================================


PREPARED_RECORDS = None

if RUN_PREPARE_DATA:
    prepare_data_caches()


# %% [code]
# ============================================================
# 9. Train only
# ============================================================


if RUN_TRAINING:
    if PREPARED_RECORDS is None:
        PREPARED_RECORDS = load_all_records()
    train_round6(PREPARED_RECORDS)


# %% [code]
# ============================================================
# 10. Test duration probe
# ============================================================


def cache_test_parquets():
    existing = sorted(glob.glob(os.path.join(TEST_CACHE, "**", "*.parquet"), recursive=True))
    if existing:
        print(f"Test parquets already cached: {len(existing)}")
        return existing

    test_path = kagglehub.dataset_download("digitalumuganda/anv-test-data-nt")
    downloaded = sorted(glob.glob(os.path.join(test_path, "**", "*.parquet"), recursive=True))
    print(f"Downloaded test parquets: {len(downloaded)}")

    for pf in downloaded:
        rel = os.path.relpath(pf, test_path)
        dst = os.path.join(TEST_CACHE, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(pf, dst)

    return sorted(glob.glob(os.path.join(TEST_CACHE, "**", "*.parquet"), recursive=True))


def audio_duration_seconds(audio_field):
    if isinstance(audio_field, dict):
        arr = audio_field.get("array")
        if arr is not None:
            sr = audio_field.get("sampling_rate") or 16000
            return len(arr) / sr
        raw = audio_field.get("bytes")
        if raw:
            try:
                info = sf.info(io.BytesIO(raw))
                return info.frames / info.samplerate
            except Exception:
                pass
    try:
        arr = decode_audio(audio_field)
        return len(arr) / 16000
    except Exception:
        return np.nan


def run_test_duration_probe(parquet_files):
    rows = []
    for idx, pq in enumerate(parquet_files):
        df = pd.read_parquet(pq)
        lang = df["language"].iloc[0]
        durations = [audio_duration_seconds(a) for a in df["audio"]]
        for d in durations:
            rows.append({"language": lang, "duration": d})
        print(f"duration probe {idx + 1}/{len(parquet_files)} {lang} rows={len(df)}")

    dur = pd.DataFrame(rows)
    summary = dur.groupby("language")["duration"].agg(["count", "mean", "median", "sum"])
    summary["pct_under_10s"] = dur.assign(under=dur["duration"] < 10).groupby("language")["under"].mean() * 100
    summary["pct_over_20s"] = dur.assign(over=dur["duration"] > 20).groupby("language")["over"].mean() * 100
    out = f"{OUTPUT_DIR}/test_duration_summary.csv"
    summary.to_csv(out)
    print(summary)
    print(f"Saved {out}")
    return summary


# Run this before training in a separate CPU session if you want the exact test mix.
# parquet_files = cache_test_parquets()
# duration_summary = run_test_duration_probe(parquet_files)


# %% [code]
# ============================================================
# 9. Inference and submission generation
# ============================================================


def load_inference_model():
    load_from = CHECKPOINT_DIR if os.path.exists(f"{CHECKPOINT_DIR}/config.json") else PREVIOUS_REPO_ID
    print(f"Loading inference model from: {load_from}")
    ft_processor = WhisperProcessor.from_pretrained(load_from, token=HF_TOKEN)
    ft_model = WhisperForConditionalGeneration.from_pretrained(
        load_from,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        token=HF_TOKEN,
    ).to(DEVICE)
    ft_model.eval()
    ft_model.config.forced_decoder_ids = None
    ft_model.generation_config.forced_decoder_ids = None
    ft_model.generation_config.suppress_tokens = []
    ft_model.generation_config.max_length = None
    return ft_processor, ft_model


LANG_TO_WHISPER = {
    "swa": "sw",
    "som": "so",
    "kik": "kik",
    "luo": "luo",
    "mas": "mas",
    "kln": "kln",
}


def run_inference(beam_size=3, output_name="submission.csv"):
    parquet_files = cache_test_parquets()
    ft_processor, ft_model = load_inference_model()

    def transcribe_batch(arrays, language=None):
        arrays = [a[:MAX_AUDIO_SAMPLES] for a in arrays]
        inputs = ft_processor(arrays, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE)
        if DEVICE == "cuda":
            inputs = inputs.to(torch.float16)
        gen_kw = {
            "max_new_tokens": 64,
            "num_beams": beam_size,
            "no_repeat_ngram_size": 3,
        }
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
            return row.id, row.language, decode_audio(row.audio)
        except Exception:
            return row.id, row.language, None

    results = []
    done_ids = set()
    checkpoint_file = f"{OUTPUT_DIR}/{output_name.replace('.csv', '')}_checkpoint.csv"
    if os.path.exists(checkpoint_file):
        existing = pd.read_csv(checkpoint_file)
        results = existing.to_dict("records")
        done_ids = set(existing["id"])
        print(f"Resuming inference: {len(results)} rows")

    batch_size = 32
    save_every = 5
    t0 = time.time()
    for pf_idx, pq_path in enumerate(parquet_files):
        df = pd.read_parquet(pq_path)
        df = df[~df["id"].isin(done_ids)]
        if df.empty:
            continue
        lang3 = df["language"].iloc[0]
        wh_lang = LANG_TO_WHISPER.get(lang3)
        rows = list(df.itertuples(index=False))
        print(f"[{pf_idx + 1}/{len(parquet_files)}] {os.path.basename(pq_path)} lang={lang3} rows={len(rows)}")

        for i in range(0, len(rows), batch_size):
            chunk = rows[i : i + batch_size]
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
                except Exception as e:
                    print(f"Batch error: {e}; falling back one-by-one")
                    texts = []
                    for arr in arrays:
                        try:
                            texts.append(transcribe_batch([arr], language=wh_lang)[0])
                        except Exception:
                            texts.append(".")

                for id_, lang_, text in zip(batch_ids, batch_langs, texts):
                    results.append(
                        {
                            "id": id_,
                            "language": lang_,
                            "transcription": postprocess_submission_text(text),
                        }
                    )
                    done_ids.add(id_)

        if (pf_idx + 1) % save_every == 0 or (pf_idx + 1) == len(parquet_files):
            pd.DataFrame(results).to_csv(checkpoint_file, index=False)
            print(f"Checkpoint saved: {len(results)} rows")

    sub = pd.DataFrame(results)
    sub["transcription"] = sub["transcription"].map(postprocess_submission_text)
    sub = sub[["id", "language", "transcription"]]
    out = f"{WORK_DIR}/{output_name}"
    sub.to_csv(out, index=False)
    print(f"Saved {out} rows={len(sub)} time={(time.time() - t0) / 60:.1f} min")
    print(sub["language"].value_counts().to_string())
    return out


if RUN_INFERENCE:
    run_inference(beam_size=3, output_name="submission_round6_beam3.csv")
    # Run beam 5 only after you have a beam 3 score and enough GPU time.
    # run_inference(beam_size=5, output_name="submission_round6_beam5.csv")
