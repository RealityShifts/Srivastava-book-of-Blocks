"""Runnable examples: ``python -m Blocks.visualized.examples [outdir]``.

Writes one HTML file per example and prints where they landed.
"""

from __future__ import annotations

import os
import sys

import jax
from flax import nnx

from Blocks.flax_blocks.core_blocks import (
    Activation, ConvBlock, Norm, ResidualBlock,
)
from .visualize import summary, visualize


class TinyEncoder(nnx.Module):
    """A small conv encoder - shows nesting, downsampling and residuals."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.stem = ConvBlock(3, 32, 7, norm=Norm.BATCH,
                              activation=Activation.LEAKY_RELU, rngs=rngs)
        self.stage1 = ResidualBlock(32, 64, strides=2, norm=Norm.BATCH,
                                    rngs=rngs)
        self.stage2 = ResidualBlock(64, 128, strides=2, norm=Norm.BATCH,
                                    rngs=rngs)
        self.head = ConvBlock(128, 64, 1, norm=Norm.NONE,
                              activation=Activation.IDENTITY, rngs=rngs)

    def __call__(self, x):
        return self.head(self.stage2(self.stage1(self.stem(x))))


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    outdir = argv[0] if argv else "visualized_examples"
    os.makedirs(outdir, exist_ok=True)
    key = jax.random.key(0)

    # 1. a single residual block
    block = ResidualBlock(8, 16, strides=2, norm=Norm.BATCH, rngs=nnx.Rngs(0))
    x = jax.random.normal(key, (2, 16, 16, 8))
    print(summary(block, x), "\n")
    print(visualize(block, x, path=os.path.join(outdir, "residual_block.html"),
                    title="ResidualBlock"))

    # 2. a full encoder
    model = TinyEncoder(rngs=nnx.Rngs(0))
    img = jax.random.normal(key, (1, 64, 64, 3))
    print(visualize(model, img, path=os.path.join(outdir, "tiny_encoder.html"),
                    title="TinyEncoder"))

    # 3. a model that fails mid-forward: the graph still shows how far it got
    class Broken(nnx.Module):
        def __init__(self, *, rngs: nnx.Rngs) -> None:
            self.a = ConvBlock(3, 16, norm=Norm.BATCH, rngs=rngs)
            self.b = ConvBlock(99, 32, norm=Norm.BATCH, rngs=rngs)  # wrong in_ch

        def __call__(self, x):
            return self.b(self.a(x))

    print(visualize(Broken(rngs=nnx.Rngs(0)),
                    jax.random.normal(key, (1, 32, 32, 3)),
                    path=os.path.join(outdir, "broken_model.html"),
                    title="Broken (partial graph)"))

    print(f"\nopen the files in {os.path.abspath(outdir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
