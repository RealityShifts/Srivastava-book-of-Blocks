#!/usr/bin/env python3
"""Download a truncated Book 1 corpus for tiny_transformer_lm training.

Uses the Hugging Face Datasets Server (no extra pip deps).

    python models/examples/tiny_transformer_lm/dataset/prepare.py
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = DATASET_DIR / "harry_potter_book1.txt"
BOOK1_FILENAME = "1-Harry-Potter-and-the-Sorcerer\u2019s-Stone.txt"
HF_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=elricwan/HarryPotter&config=default&split=train&offset=0&length=8"
)


def _fetch_book1_text() -> str:
    req = urllib.request.Request(HF_ROWS_URL, headers={"User-Agent": "bob-tiny-lm/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.load(resp)
    for row in payload["rows"]:
        rec = row["row"]
        name = rec.get("filename", "")
        if "Sorcerer" in name or "Stone" in name or name == BOOK1_FILENAME:
            return rec["content"]
    # fallback: first row is book 1 in this dataset
    return payload["rows"][0]["row"]["content"]


def prepare(out: Path, max_chars: int) -> Path:
    print("Fetching Book 1 from Hugging Face (elricwan/HarryPotter)…")
    text = _fetch_book1_text()
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars]
        print(f"Truncated to {max_chars:,} characters (~early chapters).")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {len(text):,} characters → {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Harry Potter Book 1 corpus")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output .txt path (default: {DEFAULT_OUT.name})",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=200_000,
        help="Keep only the first N characters (0 = full book)",
    )
    args = parser.parse_args()
    prepare(args.out, args.max_chars)


if __name__ == "__main__":
    main()
