"""Create post-processing variants of a Kaggle ASR submission.

Usage:
    python make_punctuation_variants.py submission.csv

Outputs are written next to the input file:
    submission_ws.csv
    submission_strip_final_punct.csv
    submission_strip_all_punct.csv
    submission_lower_strip_punct.csv
"""

import argparse
import re
from pathlib import Path

import pandas as pd


FINAL_PUNCT_RE = re.compile(r"[\s\.\,\!\?\:\;\u0964\u061f]+$")
ALL_PUNCT_RE = re.compile(r"[^\w\sÀ-ɏ̀-ͯḀ-ỿ'’ŋŊ]", flags=re.UNICODE)


def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip() or "."


def strip_final_punct(text: str) -> str:
    text = collapse_ws(text)
    text = FINAL_PUNCT_RE.sub("", text).strip()
    return text or "."


def strip_all_punct(text: str) -> str:
    text = ALL_PUNCT_RE.sub(" ", str(text))
    return collapse_ws(text)


def lower_strip_punct(text: str) -> str:
    return strip_all_punct(str(text).lower())


def write_variant(df: pd.DataFrame, src: Path, suffix: str, fn) -> Path:
    out = src.with_name(f"{src.stem}_{suffix}{src.suffix}")
    variant = df.copy()
    variant["transcription"] = variant["transcription"].map(fn)
    variant[["id", "language", "transcription"]].to_csv(out, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.submission)
    required = {"id", "language", "transcription"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    outputs = [
        write_variant(df, args.submission, "ws", collapse_ws),
        write_variant(df, args.submission, "strip_final_punct", strip_final_punct),
        write_variant(df, args.submission, "strip_all_punct", strip_all_punct),
        write_variant(df, args.submission, "lower_strip_punct", lower_strip_punct),
    ]

    print("Wrote:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
