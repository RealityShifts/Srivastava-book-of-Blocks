"""Section 11 - Reinforcement Learning blocks.

Policy / value / Q networks, an actor-critic combo, a circular replay
buffer (host-side, not jax-traceable) and a soft-update target network.
"""

from __future__ import annotations

import random
from collections import deque
from copy import deepcopy
from typing import Iterable, Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, PRNGKeyArray
from ._typecheck import typecheck
from .core_blocks import Activation, ActivationLike, get_activation


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

class _MLP(nnx.Module):
    def __init__(self, dims: Sequence[int],
                 activation: ActivationLike = Activation.TANH,
                 *, rngs: nnx.Rngs) -> None:
        self.act = get_activation(activation)
        self.layers = [nnx.Linear(dims[i], dims[i + 1], rngs=rngs)
                       for i in range(len(dims) - 1)]

    @typecheck
    def __call__(
        self, x: Float[Array, "*B D_in"]
    ) -> Float[Array, "*B D_out"]:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.act(x)
        return x


# ---------------------------------------------------------------------------
# Policy / Value / Actor-Critic
# ---------------------------------------------------------------------------

class PolicyNetwork(nnx.Module):
    """Stochastic policy outputting categorical or Gaussian-action parameters."""

    def __init__(self, state_dim: int, action_dim: int,
                 hidden: Sequence[int] = (128, 128),
                 discrete: bool = True, *, rngs: nnx.Rngs) -> None:
        self.discrete = discrete
        self.body = _MLP([state_dim, *hidden, action_dim], rngs=rngs)
        if not discrete:
            self.log_std = nnx.Param(jnp.zeros((action_dim,)))

    def logits_or_mean(
        self, s: Float[Array, "B D_state"]
    ) -> Float[Array, "B D_action"]:
        return self.body(s)

    def sample(
        self,
        s: Float[Array, "B D_state"],
        key: PRNGKeyArray,
    ) -> Float[Array, "B *action"]:
        out = self.logits_or_mean(s)
        if self.discrete:
            return jax.random.categorical(key, out)
        std = jnp.exp(self.log_std.value)
        return out + std * jax.random.normal(key, out.shape)

    @typecheck
    def __call__(
        self,
        s: Float[Array, "B D_state"],
        key: PRNGKeyArray,
    ) -> Float[Array, "B *action"]:
        return self.sample(s, key)


class ValueNetwork(nnx.Module):
    """State value ``V_phi(s)``."""

    def __init__(self, state_dim: int, hidden: Sequence[int] = (128, 128),
                 *, rngs: nnx.Rngs) -> None:
        self.net = _MLP([state_dim, *hidden, 1], rngs=rngs)

    @typecheck
    def __call__(
        self, s: Float[Array, "B D_state"]
    ) -> Float[Array, "B"]:
        return self.net(s).squeeze(-1)


class QNetwork(nnx.Module):
    """Action-value ``Q(s, a)`` with a discrete output head."""

    def __init__(self, state_dim: int, action_dim: int,
                 hidden: Sequence[int] = (128, 128),
                 *, rngs: nnx.Rngs) -> None:
        self.net = _MLP([state_dim, *hidden, action_dim], rngs=rngs)

    @typecheck
    def __call__(
        self, s: Float[Array, "B D_state"]
    ) -> Float[Array, "B D_action"]:
        return self.net(s)


class ActorCritic(nnx.Module):
    """Combined actor-critic network sharing a torso."""

    def __init__(self, state_dim: int, action_dim: int,
                 hidden: Sequence[int] = (128, 128), discrete: bool = True,
                 *, rngs: nnx.Rngs) -> None:
        self.discrete = discrete
        self.torso = _MLP([state_dim, *hidden], rngs=rngs)
        self.actor_head = nnx.Linear(hidden[-1], action_dim, rngs=rngs)
        self.critic_head = nnx.Linear(hidden[-1], 1, rngs=rngs)
        if not discrete:
            self.log_std = nnx.Param(jnp.zeros((action_dim,)))

    @typecheck
    def __call__(
        self, s: Float[Array, "B D_state"]
    ) -> tuple[Float[Array, "B D_action"], Float[Array, "B"]]:
        h = nnx.tanh(self.torso(s))
        return self.actor_head(h), self.critic_head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Replay buffer (host-side, not jax-traceable)
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-capacity FIFO replay buffer with random sampling."""

    def __init__(self, capacity: int = 100_000) -> None:
        self.buffer: deque[tuple] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def extend(self, transitions: Iterable[tuple]) -> None:
        self.buffer.extend(transitions)

    def sample(self, batch: int) -> tuple[Float[Array, "..."], ...]:
        items = random.sample(self.buffer, batch)
        cols = list(zip(*items))
        return tuple(jnp.asarray(c) for c in cols)


# ---------------------------------------------------------------------------
# Target network helper
# ---------------------------------------------------------------------------

class TargetNetwork(nnx.Module):
    """Wrap a network and keep a slowly-updated copy of its parameters."""

    def __init__(self, source: nnx.Module, tau: float = 0.005) -> None:
        self.target = deepcopy(source)
        self.tau = tau

    def soft_update(self, source: nnx.Module) -> None:
        """``theta' = tau * theta + (1 - tau) * theta'``."""
        src_state = nnx.state(source, nnx.Param)
        tgt_state = nnx.state(self.target, nnx.Param)
        new_state = jax.tree.map(
            lambda t, s: (1 - self.tau) * t + self.tau * s,
            tgt_state, src_state)
        nnx.update(self.target, new_state)

    @typecheck
    def __call__(self, *args, **kwargs):
        return self.target(*args, **kwargs)
