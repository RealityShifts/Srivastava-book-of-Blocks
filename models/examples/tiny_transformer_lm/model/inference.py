"""Load a trained TinyTransformerLM checkpoint and generate text.

Checkpoint (written by ``tiny_transformer_lm_train.py``)::

    models/examples/tiny_transformer_lm/checkpoints/tiny_lm.pt

Prepare corpus, train, then infer::

    python models/examples/tiny_transformer_lm/dataset/prepare.py
    python models/examples/tiny_transformer_lm/model/tiny_transformer_lm_train.py
    python models/examples/tiny_transformer_lm/model/inference.py --prompt "Harry"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_MODEL_DIR = Path(__file__).resolve().parent
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

from tiny_transformer_lm_train import (  # noqa: E402
    CHECKPOINT_PATH,
    _device,
    load_checkpoint,
)


def encode(text: str) -> torch.Tensor:
    """Byte-level ids in [0, 255]."""
    return torch.tensor([list(text.encode("latin-1"))], dtype=torch.long)


def decode(ids: list[int]) -> str:
    return bytes(ids).decode("latin-1", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny LM inference")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PATH,
        help=f"Path to .pt file (default: {CHECKPOINT_PATH})",
    )
    parser.add_argument("--prompt", type=str, default="Harry")
    parser.add_argument("--max-new", type=int, default=64)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(
            f"No checkpoint at {args.checkpoint}\n"
            "Train first: python models/examples/tiny_transformer_lm/model/tiny_transformer_lm_train.py"
        )

    device = _device()
    model, cfg = load_checkpoint(args.checkpoint, device)

    prompt = encode(args.prompt).to(device)
    out = model.generate(prompt, max_new=args.max_new)
    ids = out[0].tolist()

    print(f"checkpoint: {args.checkpoint}")
    print(f"prompt:     {args.prompt!r}")
    print(f"output:     {decode(ids)!r}")


if __name__ == "__main__":
    main()
