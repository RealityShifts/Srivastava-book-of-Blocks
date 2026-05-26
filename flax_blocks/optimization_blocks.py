"""Section 14 - Optimization & training blocks.

JAX-native optimizers (SGD, Adam, AdamW, Lion, Sophia-G, RMSProp),
scheduler functions, gradient clipping, exponential moving average,
mixed-precision and gradient checkpointing helpers.

Each optimizer follows a tiny functional protocol:

    state = init_fn(params)
    new_params, new_state = update_fn(grads, params, state, lr=...)

This avoids depending on optax while still being trivially compatible
with Flax NNX state (any pytree of ``nnx.Param`` values).
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Int, PyTree
from ._typecheck import typecheck


Params = PyTree[Float[Array, "..."]]     # nested jax pytree of float arrays
Grads = PyTree[Float[Array, "..."]]
State = Any


# ---------------------------------------------------------------------------
# SGD / momentum
# ---------------------------------------------------------------------------

class SGDState(NamedTuple):
    momentum: Any


@typecheck
def sgd_init(params: Params, momentum: bool = False) -> SGDState:
    return SGDState(jax.tree.map(jnp.zeros_like, params) if momentum else None)


@typecheck
def sgd_update(grads: Grads, params: Params, state: SGDState,
               lr: float = 1e-2, momentum: float = 0.0,
               weight_decay: float = 0.0) -> tuple[Params, SGDState]:
    if state.momentum is None:
        new_p = jax.tree.map(
            lambda p, g: p - lr * (g + weight_decay * p), params, grads)
        return new_p, state
    new_v = jax.tree.map(
        lambda v, g: momentum * v + g, state.momentum, grads)
    new_p = jax.tree.map(
        lambda p, v: p - lr * (v + weight_decay * p), params, new_v)
    return new_p, SGDState(new_v)


# ---------------------------------------------------------------------------
# Adam / AdamW
# ---------------------------------------------------------------------------

class AdamState(NamedTuple):
    step: Int[Array, ""]
    m: Any
    v: Any


@typecheck
def adam_init(params: Params) -> AdamState:
    zeros = jax.tree.map(jnp.zeros_like, params)
    return AdamState(jnp.array(0), zeros, deepcopy(zeros))


@typecheck
def adam_update(grads: Grads, params: Params, state: AdamState,
                lr: float = 1e-3,
                betas: tuple[float, float] = (0.9, 0.999),
                eps: float = 1e-8,
                weight_decay: float = 0.0,
                decoupled: bool = False) -> tuple[Params, AdamState]:
    """Adam (``decoupled=False``) / AdamW (``decoupled=True``)."""
    b1, b2 = betas
    step = state.step + 1
    m = jax.tree.map(lambda mi, g: b1 * mi + (1 - b1) * g, state.m, grads)
    v = jax.tree.map(lambda vi, g: b2 * vi + (1 - b2) * g * g, state.v, grads)
    m_hat = jax.tree.map(lambda x: x / (1 - b1 ** step), m)
    v_hat = jax.tree.map(lambda x: x / (1 - b2 ** step), v)
    if decoupled:
        new_p = jax.tree.map(
            lambda p, mh, vh: p * (1 - lr * weight_decay) - lr * mh / (jnp.sqrt(vh) + eps),
            params, m_hat, v_hat)
    else:
        new_p = jax.tree.map(
            lambda p, mh, vh: p - lr * (mh / (jnp.sqrt(vh) + eps) + weight_decay * p),
            params, m_hat, v_hat)
    return new_p, AdamState(step, m, v)


# ---------------------------------------------------------------------------
# Lion (Chen et al. 2023)
# ---------------------------------------------------------------------------

class LionState(NamedTuple):
    m: Any


@typecheck
def lion_init(params: Params) -> LionState:
    return LionState(jax.tree.map(jnp.zeros_like, params))


@typecheck
def lion_update(grads: Grads, params: Params, state: LionState,
                lr: float = 1e-4,
                betas: tuple[float, float] = (0.9, 0.99),
                weight_decay: float = 0.0) -> tuple[Params, LionState]:
    b1, b2 = betas
    update = jax.tree.map(
        lambda mi, g: jnp.sign(b1 * mi + (1 - b1) * g), state.m, grads)
    new_p = jax.tree.map(
        lambda p, u: (1 - lr * weight_decay) * p - lr * u, params, update)
    new_m = jax.tree.map(
        lambda mi, g: b2 * mi + (1 - b2) * g, state.m, grads)
    return new_p, LionState(new_m)


# ---------------------------------------------------------------------------
# Sophia-G (Liu et al. 2023)
# ---------------------------------------------------------------------------

class SophiaState(NamedTuple):
    m: Any
    h: Any


@typecheck
def sophia_init(params: Params) -> SophiaState:
    zeros = jax.tree.map(jnp.zeros_like, params)
    return SophiaState(zeros, deepcopy(zeros))


@typecheck
def sophia_update_hessian(grads: Grads, state: SophiaState,
                          beta2: float = 0.99) -> SophiaState:
    new_h = jax.tree.map(
        lambda h, g: beta2 * h + (1 - beta2) * g * g, state.h, grads)
    return SophiaState(state.m, new_h)


@typecheck
def sophia_update(grads: Grads, params: Params, state: SophiaState,
                  lr: float = 1e-4, beta1: float = 0.965,
                  rho: float = 0.04, weight_decay: float = 0.0,
                  eps: float = 1e-12) -> tuple[Params, SophiaState]:
    new_m = jax.tree.map(
        lambda mi, g: beta1 * mi + (1 - beta1) * g, state.m, grads)
    update = jax.tree.map(
        lambda mi, hi: jnp.clip(mi / (rho * hi + eps), -1.0, 1.0),
        new_m, state.h)
    new_p = jax.tree.map(
        lambda p, u: (1 - lr * weight_decay) * p - lr * u, params, update)
    return new_p, SophiaState(new_m, state.h)


# ---------------------------------------------------------------------------
# RMSProp
# ---------------------------------------------------------------------------

class RMSPropState(NamedTuple):
    v: Any


@typecheck
def rmsprop_init(params: Params) -> RMSPropState:
    return RMSPropState(jax.tree.map(jnp.zeros_like, params))


@typecheck
def rmsprop_update(grads: Grads, params: Params, state: RMSPropState,
                   lr: float = 1e-3, alpha: float = 0.99,
                   eps: float = 1e-8) -> tuple[Params, RMSPropState]:
    new_v = jax.tree.map(
        lambda vi, g: alpha * vi + (1 - alpha) * g * g, state.v, grads)
    new_p = jax.tree.map(
        lambda p, g, v: p - lr * g / (jnp.sqrt(v) + eps),
        params, grads, new_v)
    return new_p, RMSPropState(new_v)


# ---------------------------------------------------------------------------
# Schedulers
# ---------------------------------------------------------------------------

def cosine_warmup_schedule(warmup_steps: int, total_steps: int,
                           base_lr: float, min_lr_ratio: float = 0.0
                           ) -> Callable[[int], float]:
    def schedule(step: int) -> float:
        if step < warmup_steps:
            return base_lr * step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return base_lr * (min_lr_ratio
                          + 0.5 * (1 - min_lr_ratio)
                          * (1 + math.cos(math.pi * progress)))
    return schedule


def step_decay_schedule(base_lr: float, step_size: int,
                        gamma: float = 0.1) -> Callable[[int], float]:
    return lambda step: base_lr * gamma ** (step // step_size)


def one_cycle_schedule(max_lr: float, total_steps: int,
                       div_factor: float = 25.0,
                       final_div_factor: float = 1e4
                       ) -> Callable[[int], float]:
    init_lr = max_lr / div_factor
    final_lr = init_lr / final_div_factor
    peak = total_steps // 2
    def schedule(step: int) -> float:
        if step <= peak:
            return init_lr + (max_lr - init_lr) * step / max(1, peak)
        return max_lr - (max_lr - final_lr) * (step - peak) / max(1, total_steps - peak)
    return schedule


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------

@typecheck
def clip_grad_norm(
    grads: Grads, max_norm: float = 1.0,
) -> tuple[Grads, Float[Array, ""]]:
    """Globally clip gradients so their L2 norm is at most ``max_norm``."""
    leaves = jax.tree.leaves(grads)
    total_sq = sum(jnp.sum(g * g) for g in leaves)
    total_norm = jnp.sqrt(total_sq)
    scale = jnp.minimum(1.0, max_norm / (total_norm + 1e-12))
    return jax.tree.map(lambda g: g * scale, grads), total_norm


# ---------------------------------------------------------------------------
# Exponential Moving Average
# ---------------------------------------------------------------------------

class EMA:
    """Maintains an EMA copy of model parameters.

    ``theta_ema = decay * theta_ema + (1 - decay) * theta``.
    """

    def __init__(self, model: nnx.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = deepcopy(model)

    def update(self, model: nnx.Module) -> None:
        src = nnx.state(model, nnx.Param)
        tgt = nnx.state(self.shadow, nnx.Param)
        new = jax.tree.map(
            lambda t, s: self.decay * t + (1 - self.decay) * s, tgt, src)
        nnx.update(self.shadow, new)


# ---------------------------------------------------------------------------
# Mixed precision helper
# ---------------------------------------------------------------------------

def cast_pytree(tree: Any, dtype: jnp.dtype) -> Any:
    return jax.tree.map(
        lambda x: x.astype(dtype) if isinstance(x, jax.Array) else x, tree)


class LossScaler:
    """Dynamic loss scaler used with float16 mixed-precision training."""

    def __init__(self, init_scale: float = 2 ** 15, growth_interval: int = 2000,
                 growth_factor: float = 2.0, backoff: float = 0.5) -> None:
        self.scale = init_scale
        self.steps_since_grow = 0
        self.growth_interval = growth_interval
        self.growth_factor = growth_factor
        self.backoff = backoff

    def scale_loss(self, loss: Float[Array, ""]) -> Float[Array, ""]:
        return loss * self.scale

    def unscale_grads(self, grads: Grads) -> tuple[Grads, bool]:
        unscaled = jax.tree.map(lambda g: g / self.scale, grads)
        finite = bool(all(jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(unscaled)))
        return unscaled, finite

    def update(self, finite: bool) -> None:
        if not finite:
            self.scale = max(self.scale * self.backoff, 1.0)
            self.steps_since_grow = 0
        else:
            self.steps_since_grow += 1
            if self.steps_since_grow >= self.growth_interval:
                self.scale *= self.growth_factor
                self.steps_since_grow = 0


# ---------------------------------------------------------------------------
# Gradient checkpointing
# ---------------------------------------------------------------------------

def checkpointed(fn: Callable, *args, **kwargs):
    """Forward through ``fn`` with rematerialization of activations."""
    return jax.checkpoint(fn)(*args, **kwargs)


def checkpoint_module(module: nnx.Module) -> nnx.Module:
    """Re-materialize the forward pass of ``module`` (NNX wrapper)."""
    return nnx.remat(module)
