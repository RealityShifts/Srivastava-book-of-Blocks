"""Section 14 - Optimization & training blocks.

Optimizers (Lion, Sophia-G), schedulers (cosine, warmup, step decay,
one-cycle alias), gradient clipping, exponential moving average,
mixed-precision training step, gradient checkpointing helper.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Callable, Iterable

import torch
import torch.nn as nn
from jaxtyping import Float
from ._typecheck import typecheck
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


# ---------------------------------------------------------------------------
# Lion (Chen et al. 2023 - "EvoLved Sign Momentum")
# ---------------------------------------------------------------------------

class Lion(Optimizer):
    """Lion: ``update = sign(beta1 * m + (1 - beta1) * g)``."""

    def __init__(self, params: Iterable[nn.Parameter], lr: float = 1e-4,
                 betas: tuple[float, float] = (0.9, 0.99),
                 weight_decay: float = 0.0) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid lr {lr}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):                       # type: ignore[override]
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                m = state["exp_avg"]
                if wd != 0:
                    p.mul_(1 - lr * wd)
                update = (b1 * m + (1 - b1) * p.grad).sign_()
                p.add_(update, alpha=-lr)
                m.mul_(b2).add_(p.grad, alpha=1 - b2)
        return loss


# ---------------------------------------------------------------------------
# Sophia-G (Liu et al. 2023) - simplified version
# ---------------------------------------------------------------------------

class Sophia(Optimizer):
    """Sophia optimizer. Uses a Hutchinson-style Hessian diagonal estimate."""

    def __init__(self, params: Iterable[nn.Parameter], lr: float = 1e-4,
                 betas: tuple[float, float] = (0.965, 0.99),
                 rho: float = 0.04, weight_decay: float = 0.0,
                 eps: float = 1e-12) -> None:
        defaults = dict(lr=lr, betas=betas, rho=rho,
                        weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def update_hessian(self) -> None:
        """Update the diagonal Hessian estimate ``h`` from current ``grad ** 2``."""
        for group in self.param_groups:
            b2 = group["betas"][1]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                state.setdefault("hessian", torch.zeros_like(p))
                state["hessian"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):                       # type: ignore[override]
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, rho, eps = group["lr"], group["rho"], group["eps"]
            b1, _ = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                state.setdefault("exp_avg", torch.zeros_like(p))
                state.setdefault("hessian", torch.zeros_like(p))
                m, h = state["exp_avg"], state["hessian"]
                m.mul_(b1).add_(p.grad, alpha=1 - b1)
                if wd:
                    p.mul_(1 - lr * wd)
                ratio = m / (rho * h + eps)
                p.add_(-lr * ratio.clamp(-1.0, 1.0))
        return loss


# ---------------------------------------------------------------------------
# Schedulers
# ---------------------------------------------------------------------------

def cosine_warmup_scheduler(optimizer: Optimizer, warmup_steps: int,
                            total_steps: int, min_lr_ratio: float = 0.0) -> LambdaLR:
    """Linear warmup -> cosine decay to ``min_lr_ratio * base_lr``."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def step_decay_scheduler(optimizer: Optimizer, step_size: int,
                         gamma: float = 0.1) -> LambdaLR:
    """Multiply lr by ``gamma`` every ``step_size`` steps."""
    return LambdaLR(optimizer, lambda s: gamma ** (s // step_size))


# ``OneCycleLR`` already lives in :mod:`torch.optim.lr_scheduler`; we
# re-export it for symmetry.
from torch.optim.lr_scheduler import OneCycleLR  # noqa: E402,F401


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------

@typecheck
def clip_grad_norm(
    parameters: Iterable[nn.Parameter], max_norm: float = 1.0
) -> Float[Tensor, ""]:
    """Thin alias for :func:`torch.nn.utils.clip_grad_norm_`."""
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm)


# ---------------------------------------------------------------------------
# Exponential Moving Average
# ---------------------------------------------------------------------------

class EMA:
    """Maintains an EMA copy of model parameters: ``theta_ema = d * theta_ema + (1-d) * theta``."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = deepcopy(model)
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)
        for ema_b, b in zip(self.shadow.buffers(), model.buffers()):
            ema_b.data.copy_(b.data)

    def state_dict(self) -> dict:
        return self.shadow.state_dict()


# ---------------------------------------------------------------------------
# Mixed precision helper
# ---------------------------------------------------------------------------

class MixedPrecisionTrainer:
    """One-call training step with autocast + GradScaler."""

    def __init__(self, model: nn.Module, optimizer: Optimizer,
                 device_type: str = "cuda", dtype: torch.dtype = torch.float16,
                 max_grad_norm: float | None = 1.0) -> None:
        self.model = model
        self.optim = optimizer
        self.scaler = torch.cuda.amp.GradScaler(enabled=device_type == "cuda")
        self.device_type = device_type
        self.dtype = dtype
        self.max_grad_norm = max_grad_norm

    def step(
        self, loss_fn: Callable[[], Float[Tensor, ""]]
    ) -> Float[Tensor, ""]:
        self.optim.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device_type, dtype=self.dtype):
            loss = loss_fn()
        self.scaler.scale(loss).backward()
        if self.max_grad_norm:
            self.scaler.unscale_(self.optim)
            clip_grad_norm(self.model.parameters(), self.max_grad_norm)
        self.scaler.step(self.optim)
        self.scaler.update()
        return loss.detach()


# ---------------------------------------------------------------------------
# Gradient checkpointing helper
# ---------------------------------------------------------------------------

def checkpointed(module: nn.Module, *args, use_reentrant: bool = False, **kwargs):
    """Forward through ``module`` with activation checkpointing."""
    return torch.utils.checkpoint.checkpoint(
        module, *args, use_reentrant=use_reentrant, **kwargs
    )


class CheckpointedSequential(nn.Sequential):
    """Sequential that runs each submodule under :func:`torch.utils.checkpoint`."""

    @typecheck
    def forward(                                                           # type: ignore[override]
        self, x: Float[Tensor, "..."]
    ) -> Float[Tensor, "..."]:
        for m in self:
            x = torch.utils.checkpoint.checkpoint(m, x, use_reentrant=False)
        return x
