"""Section 11 - Reinforcement Learning blocks.

Policy / value / actor-critic networks, a circular replay buffer and
a soft-update target network helper.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


# ---------------------------------------------------------------------------
# Policy / Value / Actor-Critic
# ---------------------------------------------------------------------------

def _mlp(dims: Sequence[int], activation: type[nn.Module] = nn.Tanh) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class PolicyNetwork(nn.Module):
    """Stochastic policy ``pi_theta(a|s)`` for either discrete or continuous actions."""

    def __init__(self, state_dim: int, action_dim: int,
                 hidden: Sequence[int] = (128, 128), discrete: bool = True) -> None:
        super().__init__()
        self.discrete = discrete
        self.body = _mlp([state_dim, *hidden, action_dim])
        if not discrete:
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(
        self, s: Float[Tensor, "B D_s"]
    ) -> torch.distributions.Distribution:
        out = self.body(s)
        if self.discrete:
            return torch.distributions.Categorical(logits=out)
        return torch.distributions.Normal(out, self.log_std.exp().expand_as(out))


class ValueNetwork(nn.Module):
    """State value ``V_phi(s)``."""

    def __init__(self, state_dim: int, hidden: Sequence[int] = (128, 128)) -> None:
        super().__init__()
        self.net = _mlp([state_dim, *hidden, 1])

    def forward(
        self, s: Float[Tensor, "B D_s"]
    ) -> Float[Tensor, "B"]:
        return self.net(s).squeeze(-1)


class QNetwork(nn.Module):
    """Action-value ``Q(s, a)`` with a discrete output head."""

    def __init__(self, state_dim: int, action_dim: int,
                 hidden: Sequence[int] = (128, 128)) -> None:
        super().__init__()
        self.net = _mlp([state_dim, *hidden, action_dim])

    def forward(
        self, s: Float[Tensor, "B D_s"]
    ) -> Float[Tensor, "B D_a"]:
        return self.net(s)


class ActorCritic(nn.Module):
    """Combined actor-critic network sharing a torso."""

    def __init__(self, state_dim: int, action_dim: int,
                 hidden: Sequence[int] = (128, 128), discrete: bool = True) -> None:
        super().__init__()
        self.torso = _mlp([state_dim, *hidden])
        self.actor_head = nn.Linear(hidden[-1], action_dim)
        self.critic_head = nn.Linear(hidden[-1], 1)
        self.discrete = discrete
        if not discrete:
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(
        self, s: Float[Tensor, "B D_s"]
    ) -> tuple[torch.distributions.Distribution, Float[Tensor, "B"]]:
        h = self.torso(s)
        logits = self.actor_head(h)
        v = self.critic_head(h).squeeze(-1)
        if self.discrete:
            dist = torch.distributions.Categorical(logits=logits)
        else:
            dist = torch.distributions.Normal(logits, self.log_std.exp().expand_as(logits))
        return dist, v


# ---------------------------------------------------------------------------
# Replay buffer
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

    def sample(self, batch: int) -> tuple[Tensor, ...]:
        items = random.sample(self.buffer, batch)
        cols = list(zip(*items))
        return tuple(torch.as_tensor(c) for c in cols)


# ---------------------------------------------------------------------------
# Target network helper
# ---------------------------------------------------------------------------

class TargetNetwork(nn.Module):
    """Wrap a network and provide a frozen, slowly-updated copy.

    Soft update: ``theta' = tau * theta + (1 - tau) * theta'``.
    """

    def __init__(self, source: nn.Module, tau: float = 0.005) -> None:
        super().__init__()
        import copy
        self.target = copy.deepcopy(source)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.tau = tau

    @torch.no_grad()
    def soft_update(self, source: nn.Module) -> None:
        for tp, sp in zip(self.target.parameters(), source.parameters()):
            tp.data.mul_(1 - self.tau).add_(self.tau * sp.data)

    def forward(self, *args, **kwargs):
        return self.target(*args, **kwargs)
