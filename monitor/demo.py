"""A small training loop with the monitor attached.

    python -m Blocks.monitor.demo                 # runs until Ctrl-C
    python -m Blocks.monitor.demo --port 8891 --steps 500

Open the printed address, open the Monitor menu, switch "Select" on, click
some boxes, press Collect. The loop trains a tiny encoder-decoder to
reproduce its input; nothing about the model matters except that it nests.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..visualized.tracer_torch import trace_model
from .server import Monitor


class ResBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm = nn.GroupNorm(4, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(F.silu(self.norm(self.conv1(x))))


class Stage(nn.Module):
    def __init__(self, cin: int, cout: int, down: bool) -> None:
        super().__init__()
        self.proj = nn.Conv2d(cin, cout, 3, stride=2 if down else 1, padding=1)
        self.block1 = ResBlock(cout)
        self.block2 = ResBlock(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block2(self.block1(self.proj(x)))


class DemoNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(Stage(3, 16, True), Stage(16, 32, True))
        self.middle = ResBlock(32)
        self.decoder = nn.Sequential(Stage(32, 16, False), Stage(16, 16, False))
        self.to_rgb = nn.Conv2d(16, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.middle(self.encoder(x))
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.decoder[0](h)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        return torch.tanh(self.to_rgb(self.decoder[1](h)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8891)
    ap.add_argument("--steps", type=int, default=0, help="0 = until Ctrl-C")
    ap.add_argument("--out", default=None, help="folder for result JSON")
    ap.add_argument("--delay", type=float, default=0.05,
                    help="pause per step, so the page is watchable")
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    model = DemoNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    monitor = Monitor(
        model, port=args.port, out_dir=args.out, title="DemoNet",
        graph=lambda: trace_model(model, torch.randn(1, 3, 32, 32)))
    print(f"monitor: {monitor.url}", flush=True)

    step = 0
    try:
        while not args.steps or step < args.steps:
            x = torch.rand(8, 3, 32, 32) * 2 - 1
            loss = F.mse_loss(model(x), x)           # the training forward
            opt.zero_grad()
            loss.backward()
            opt.step()
            # Idle: returns at once. Request active: one more forward of this
            # batch, with hooks on the picked modules.
            monitor.step(step, probe=lambda: model(x))
            if step % 100 == 0:
                print(f"step {step:6d}  loss {loss.item():.4f}", flush=True)
            step += 1
            time.sleep(args.delay)
    except KeyboardInterrupt:
        pass
    monitor.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
