"""Modal inference for Round 6D Whisper-small checkpoint.

This script does not train. It loads the private Hugging Face model
Ash11/afrivoices-whisper-small-colab-round6d and generates Kaggle submission
variants.
"""

import os

import modal


os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

APP_NAME = "afrivoices-round6d-inference"
VOL_PATH = "/vol"
HF_REPO_ID = "Ash11/afrivoices-whisper-small-colab-round6d"
TEST_DATA_DIR = f"{VOL_PATH}/test_parquets"
OUTPUT_DIR = f"{VOL_PATH}/round6d_outputs"
CHECKPOINT_FILE = f"{OUTPUT_DIR}/submission_round6d_checkpoint.csv"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.2.0",
        "transformers==4.46.3",
        "accelerate>=0.26.0",
        "soundfile",
        "librosa",
        "pydub",
        "huggingface_hub>=0.21",
        "pandas",
        "pyarrow",
        "numpy<2",
        "kagglehub",
    )
)

vol = modal.Volume.from_name("afrivoices-vol", create_if_missing=True)
secrets = [modal.Secret.from_name("afrivoices-secrets")]


def _decode_audio(audio_field):
    import io

    import librosa
    import numpy as np
    import soundfile as sf
    from pydub import AudioSegment

    if isinstance(audio_field, dict) and "array" in audio_field and audio_field["array"] is not None:
        arr = np.asarray(audio_field["array"], dtype=np.float32)
        sr = audio_field.get("sampling_rate") or 16000
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != 16000:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        return arr.astype(np.float32, copy=False)

    raw = audio_field.get("bytes") if isinstance(audio_field, dict) else audio_field
    if isinstance(raw, bytes) and raw:
        try:
            arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            if sr != 16000:
                arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
            return arr.astype(np.float32, copy=False)
        except Exception:
            seg = AudioSegment.from_file(io.BytesIO(raw)).set_frame_rate(16000).set_channels(1)
            return (np.asarray(seg.get_array_of_samples(), dtype=np.float32) / 32768.0).astype(
                np.float32, copy=False
            )
    raise ValueError(f"Cannot decode audio: {type(audio_field)}")


def _collapse_ws(text: str) -> str:
    import re

    return re.sub(r"\s+", " ", str(text)).strip() or "."


def _strip_final_punct(text: str) -> str:
    import re

    text = _collapse_ws(text)
    text = re.sub(r"[\s\.\,\!\?\:\;\u0964\u061f]+$", "", text).strip()
    return text or "."


def _strip_all_punct(text: str) -> str:
    import re

    punct_re = re.compile(r"[^\w\sÀ-ɏ̀-ͯḀ-ỿ'’ŋŊ]", flags=re.UNICODE)
    return _collapse_ws(punct_re.sub(" ", str(text)))


def _lower_strip_punct(text: str) -> str:
    return _strip_all_punct(str(text).lower())


def _write_submission_variants(df, primary_path: str):
    import os

    variants = {
        "raw": _collapse_ws,
        "ws": _collapse_ws,
        "strip_final_punct": _strip_final_punct,
        "strip_all_punct": _strip_all_punct,
        "lower_strip_punct": _lower_strip_punct,
    }

    base, ext = os.path.splitext(primary_path)
    written = []
    for suffix, fn in variants.items():
        out = primary_path if suffix == "lower_strip_punct" else f"{base}_{suffix}{ext}"
        variant = df.copy()
        variant["transcription"] = variant["transcription"].map(fn)
        variant[["id", "language", "transcription"]].to_csv(out, index=False)
        written.append(out)
    return written


@app.function(image=image, secrets=secrets, timeout=10 * 60)
def check_hf_model():
    from huggingface_hub import HfApi

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    info = api.model_info(HF_REPO_ID)
    files = sorted(s.rfilename for s in info.siblings)
    required_any_tokenizer = {"tokenizer.json", "vocab.json"}
    required = {
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    }
    missing = sorted(required - set(files))
    has_tokenizer_payload = bool(required_any_tokenizer & set(files))

    print(f"FOUND {HF_REPO_ID}")
    print(f"private={info.private} last_modified={info.last_modified}")
    print("Files:")
    for f in files:
        print(f"  {f}")
    if missing:
        raise RuntimeError(f"Missing required model files: {missing}")
    if not has_tokenizer_payload:
        raise RuntimeError("Missing tokenizer payload: expected tokenizer.json or vocab.json")
    print("HF model has the required files for Transformers inference.")


@app.function(
    gpu="A100",
    image=image,
    timeout=int(4 * 3600),
    volumes={VOL_PATH: vol},
    secrets=secrets,
    memory=65536,
)
def run_round6d_inference(fresh: bool = False, beam_size: int = 3):
    import glob
    import shutil
    import time
    from concurrent.futures import ThreadPoolExecutor

    import pandas as pd
    import torch
    from huggingface_hub import login
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    hf_token = os.environ["HF_TOKEN"]
    kaggle_key = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")
    if not kaggle_key:
        raise RuntimeError("Missing Kaggle credential: set KAGGLE_API_TOKEN or KAGGLE_KEY in the Modal secret.")
    login(token=hf_token)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    vol.reload()

    print(f"Loading model from {HF_REPO_ID}")
    processor = WhisperProcessor.from_pretrained(HF_REPO_ID, token=hf_token)
    model = WhisperForConditionalGeneration.from_pretrained(
        HF_REPO_ID,
        torch_dtype=torch.float16,
        token=hf_token,
    ).to("cuda")
    model.eval()
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    model.generation_config.max_length = None

    lang_to_token = {"swa": "sw", "som": "so", "kik": "kik", "luo": "luo", "mas": "mas", "kln": "kln"}

    all_parquet_files = sorted(glob.glob(os.path.join(TEST_DATA_DIR, "**", "*.parquet"), recursive=True))
    if all_parquet_files:
        print(f"Using cached test parquet files: {len(all_parquet_files)}")
    else:
        print("Downloading Kaggle test data to Modal volume...")
        os.environ["KAGGLE_KEY"] = kaggle_key
        import kagglehub

        test_path = kagglehub.dataset_download("digitalumuganda/anv-test-data-nt")
        downloaded = sorted(glob.glob(os.path.join(test_path, "**", "*.parquet"), recursive=True))
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        for pf in downloaded:
            rel = os.path.relpath(pf, test_path)
            dst = os.path.join(TEST_DATA_DIR, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(pf, dst)
        vol.commit()
        all_parquet_files = sorted(glob.glob(os.path.join(TEST_DATA_DIR, "**", "*.parquet"), recursive=True))
        print(f"Cached test parquet files: {len(all_parquet_files)}")

    if fresh and os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("Removed old inference checkpoint because fresh=True.")

    if os.path.exists(CHECKPOINT_FILE):
        existing = pd.read_csv(CHECKPOINT_FILE)
        results = existing.to_dict("records")
        done_ids = set(existing["id"].astype(str))
        print(f"Resuming inference from {len(results)} completed rows.")
    else:
        results = []
        done_ids = set()
        print("Starting fresh inference.")

    def transcribe_batch(arrays, language_token=None):
        arrays = [a[:480_000] for a in arrays]
        inputs = processor(arrays, sampling_rate=16000, return_tensors="pt").input_features.to("cuda").to(torch.float16)
        gen_kw = {"max_new_tokens": 72, "num_beams": beam_size, "no_repeat_ngram_size": 3}
        if language_token:
            lid = processor.tokenizer.convert_tokens_to_ids(f"<|{language_token}|>")
            tid = processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")
            nid = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
            gen_kw["forced_decoder_ids"] = [[1, lid], [2, tid], [3, nid]]
        with torch.no_grad():
            ids = model.generate(input_features=inputs, **gen_kw)
        return processor.batch_decode(ids, skip_special_tokens=True)

    def safe_decode(row):
        try:
            return str(row.id), row.language, _decode_audio(row.audio)
        except Exception:
            return str(row.id), row.language, None

    t0 = time.time()
    batch_size = 32
    checkpoint_every_files = 5

    for pf_idx, pq_path in enumerate(all_parquet_files):
        df = pd.read_parquet(pq_path)
        df["id"] = df["id"].astype(str)
        df = df[~df["id"].isin(done_ids)]
        if df.empty:
            continue

        lang3 = df["language"].iloc[0]
        language_token = lang_to_token.get(lang3)
        elapsed = (time.time() - t0) / 60
        print(f"[{pf_idx + 1}/{len(all_parquet_files)}] {os.path.basename(pq_path)} lang={lang3} rows={len(df)} elapsed={elapsed:.1f}m")

        rows = list(df.itertuples(index=False))
        for i in range(0, len(rows), batch_size):
            chunk = rows[i : i + batch_size]
            with ThreadPoolExecutor(max_workers=4) as ex:
                decoded = list(ex.map(safe_decode, chunk))

            arrays, ids, langs = [], [], []
            for id_, lang_, arr in decoded:
                if arr is None:
                    results.append({"id": id_, "language": lang_, "transcription": "."})
                    done_ids.add(id_)
                else:
                    arrays.append(arr)
                    ids.append(id_)
                    langs.append(lang_)

            if arrays:
                try:
                    texts = transcribe_batch(arrays, language_token=language_token)
                except Exception as e:
                    print(f"Batch failed: {e}; falling back one-by-one")
                    texts = []
                    for arr in arrays:
                        try:
                            texts.append(transcribe_batch([arr], language_token=language_token)[0])
                        except Exception:
                            texts.append(".")
                for id_, lang_, text in zip(ids, langs, texts):
                    results.append({"id": id_, "language": lang_, "transcription": _collapse_ws(text)})
                    done_ids.add(id_)

        if (pf_idx + 1) % checkpoint_every_files == 0 or (pf_idx + 1) == len(all_parquet_files):
            pd.DataFrame(results).to_csv(CHECKPOINT_FILE, index=False)
            vol.commit()
            print(f"Checkpoint saved: {len(results)} rows")

    sub = pd.DataFrame(results)[["id", "language", "transcription"]]
    primary = f"{OUTPUT_DIR}/submission_round6d_beam{beam_size}.csv"
    written = _write_submission_variants(sub, primary)
    vol.commit()

    print(f"Done rows={len(sub)} time={(time.time() - t0) / 60:.1f}m")
    print(sub["language"].value_counts().to_string())
    print("Wrote:")
    for path in written:
        print(f"  {path}")
    print("Download with:")
    print("  modal volume get afrivoices-vol round6d_outputs .")


@app.local_entrypoint()
def main(check_only: bool = False, fresh: bool = False, beam_size: int = 3):
    check_hf_model.remote()
    if not check_only:
        run_round6d_inference.remote(fresh=fresh, beam_size=beam_size)
