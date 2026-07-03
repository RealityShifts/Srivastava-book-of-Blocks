"""Tiny character LM — model + minimal training loop.

Blocks from ``models/blocks`` (``pytorch_blocks``). Matches
``diagram/tiny_transformer_lm_spec.py``:
  TokenEmbedding → LearnedPositionalEmbedding → 2× TransformerEncoderBlock
  → LayerNorm → tied vocab projection.

Prepare corpus (Book 1 excerpt)::

    python models/examples/tiny_transformer_lm/dataset/prepare.py

Train::

    python models/examples/tiny_transformer_lm/model/tiny_transformer_lm_train.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Book-of-Blocks submodule: models/blocks/pytorch_blocks/
_BLOCKS_ROOT = Path(__file__).resolve().parents[3] / "blocks"
if str(_BLOCKS_ROOT) not in sys.path:
    sys.path.insert(0, str(_BLOCKS_ROOT))

from pytorch_blocks.embedding_blocks import (  # noqa: E402
    LearnedPositionalEmbedding,
    TokenEmbedding,
)
from pytorch_blocks.transformer_blocks import TransformerEncoderBlock  # noqa: E402

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = EXAMPLE_DIR / "dataset"
CHECKPOINT_PATH = EXAMPLE_DIR / "checkpoints" / "tiny_lm.pt"
DEFAULT_CORPUS = DATASET_DIR / "harry_potter_book1.txt"


@dataclass
class TinyLMConfig:
    vocab_size: int = 256
    dim: int = 128
    max_len: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.0


class TinyTransformerLM(nn.Module):
    """ids (B, T) → logits (B, T, V)."""

    def __init__(self, cfg: TinyLMConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or TinyLMConfig()
        self.cfg = cfg

        self.embed = TokenEmbedding(cfg.vocab_size, cfg.dim)
        self.pos = LearnedPositionalEmbedding(cfg.max_len, cfg.dim)
        self.blocks = nn.ModuleList(
            TransformerEncoderBlock(cfg.dim, cfg.num_heads, dropout=cfg.dropout)
            for _ in range(cfg.num_layers)
        )
        self.ln = nn.LayerNorm(cfg.dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        x = self.pos(x)

        t = ids.shape[1]
        mask = torch.triu(
            torch.ones(t, t, device=ids.device, dtype=torch.bool), diagonal=1
        )
        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.ln(x)
        return x @ self.embed.weight.T

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, max_new: int = 32) -> torch.Tensor:
        """Greedy decode from a (1, T) prompt."""
        self.eval()
        out = ids
        for _ in range(max_new):
            ctx = out[:, -self.cfg.max_len :]
            logits = self(ctx)
            next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
            out = torch.cat([out, next_id], dim=1)
        return out


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_corpus(
    dataset_dir: Path = DATASET_DIR,
    fallback: Path = DEFAULT_CORPUS,
) -> bytes:
    """Load all ``.txt`` files in *dataset_dir* as latin-1 bytes."""
    parts: list[str] = []
    if dataset_dir.is_dir():
        for path in sorted(dataset_dir.glob("*.txt")):
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
    if parts:
        return "\n\n".join(parts).encode("latin-1", errors="replace")

    if fallback.is_file():
        return fallback.read_bytes()

    raise FileNotFoundError(
        f"No .txt corpus in {dataset_dir} and missing {fallback.name}.\n"
        "Run: python models/examples/tiny_transformer_lm/dataset/prepare.py"
    )


def _encode_corpus(data: bytes, device: torch.device) -> torch.Tensor:
    """(N,) byte ids on *device*."""
    return torch.tensor(list(data), dtype=torch.long, device=device)


def sample_batch(
    corpus_ids: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    """Random contiguous windows from the byte stream."""
    n = corpus_ids.numel()
    need = seq_len + 1
    if n < need:
        raise ValueError(f"Corpus too short ({n} bytes); need at least {need}.")
    starts = torch.randint(0, n - need, (batch_size,), device=corpus_ids.device)
    rows = [corpus_ids[s : s + seq_len] for s in starts.tolist()]
    return torch.stack(rows)


def save_checkpoint(
    model: TinyTransformerLM,
    cfg: TinyLMConfig,
    path: Path = CHECKPOINT_PATH,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": asdict(cfg), "state_dict": model.state_dict()}, path)
    return path


def load_checkpoint(
    path: Path = CHECKPOINT_PATH,
    device: torch.device | None = None,
) -> tuple[TinyTransformerLM, TinyLMConfig]:
    device = device or _device()
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = TinyLMConfig(**ckpt["cfg"])
    model = TinyTransformerLM(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def train_on_corpus(
    steps: int = 2000,
    seq_len: int = 64,
    batch_size: int = 16,
    lr: float = 3e-3,
    dataset_dir: Path = DATASET_DIR,
    prompt: str = "Harry",
) -> None:
    """Train next-byte prediction on text under ``dataset/``."""
    device = _device()
    data = load_corpus(dataset_dir)
    corpus_ids = _encode_corpus(data, device)
    print(f"corpus: {len(data):,} bytes from {dataset_dir}")

    cfg = TinyLMConfig()
    model = TinyTransformerLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for step in range(1, steps + 1):
        ids = sample_batch(corpus_ids, batch_size, seq_len)
        logits = model(ids)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, cfg.vocab_size),
            ids[:, 1:].reshape(-1),
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 200 == 0 or step == 1:
            print(f"step {step:5d}  loss {loss.item():.4f}")

    ckpt_path = save_checkpoint(model, cfg)
    print(f"saved checkpoint → {ckpt_path}")

    model.eval()
    prompt_ids = torch.tensor(
        [list(prompt.encode("latin-1"))], dtype=torch.long, device=device
    )
    out = model.generate(prompt_ids, max_new=80)[0].tolist()
    text = bytes(out).decode("latin-1", errors="replace")
    print(f"sample ({prompt!r}): {text!r}")


def train_tiny_demo(steps: int = 200, seq_len: int = 32, lr: float = 3e-3) -> None:
    """Train on random bytes — stack smoke test only."""
    device = _device()
    cfg = TinyLMConfig()
    model = TinyTransformerLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for step in range(1, steps + 1):
        ids = torch.randint(0, cfg.vocab_size, (8, seq_len), device=device)
        logits = model(ids)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, cfg.vocab_size),
            ids[:, 1:].reshape(-1),
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == 1:
            print(f"step {step:4d}  loss {loss.item():.4f}")

    ckpt_path = save_checkpoint(model, cfg)
    print(f"saved checkpoint → {ckpt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TinyTransformerLM")
    parser.add_argument(
        "--random",
        action="store_true",
        help="Train on random bytes (old smoke test) instead of dataset/",
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--prompt", type=str, default="Harry")
    args = parser.parse_args()

    if args.random:
        train_tiny_demo(steps=args.steps, seq_len=args.seq_len, lr=args.lr)
    else:
        train_on_corpus(
            steps=args.steps,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            lr=args.lr,
            dataset_dir=args.dataset_dir,
            prompt=args.prompt,
        )


if __name__ == "__main__":
    main()
