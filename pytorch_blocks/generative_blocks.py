"""Section 10 - Probabilistic / generative blocks.

VAE encoder/decoder + reparameterization, autoregressive masked-conv
block (PixelCNN), affine coupling layer (RealNVP / Glow), an
energy-based-model wrapper, and DDPM/DDIM noise schedulers.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from ._typecheck import typecheck
from torch import Tensor


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------

@typecheck
def reparameterize(
    mu: Float[Tensor, "..."], logvar: Float[Tensor, "..."]
) -> Float[Tensor, "..."]:
    """``z = mu + sigma * eps``,  ``eps ~ N(0, I)``."""
    std = (0.5 * logvar).exp()
    return mu + std * torch.randn_like(std)


class VAE(nn.Module):
    """Minimal convolutional VAE; designed for 64x64 inputs by default."""

    def __init__(self, in_ch: int = 3, latent: int = 64,
                 channels: Sequence[int] = (32, 64, 128, 256)) -> None:
        super().__init__()
        enc, c = [], in_ch
        for ch in channels:
            enc += [nn.Conv2d(c, ch, 4, 2, 1), nn.GroupNorm(8, ch), nn.SiLU()]
            c = ch
        self.encoder = nn.Sequential(*enc)
        self.fc_mu = nn.Conv2d(c, latent, 1)
        self.fc_lv = nn.Conv2d(c, latent, 1)

        dec = [nn.Conv2d(latent, c, 1)]
        for ch in reversed(channels[:-1]):
            dec += [nn.ConvTranspose2d(c, ch, 4, 2, 1), nn.GroupNorm(8, ch), nn.SiLU()]
            c = ch
        dec += [nn.ConvTranspose2d(c, in_ch, 4, 2, 1)]
        self.decoder = nn.Sequential(*dec)

    def encode(
        self, x: Float[Tensor, "B C H W"]
    ) -> tuple[Float[Tensor, "B Z H_z W_z"], Float[Tensor, "B Z H_z W_z"]]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_lv(h)

    def decode(
        self, z: Float[Tensor, "B Z H_z W_z"]
    ) -> Float[Tensor, "B C H W"]:
        return self.decoder(z)

    @typecheck
    def forward(
        self, x: Float[Tensor, "B C H W"]
    ) -> tuple[
        Float[Tensor, "B C H W"],
        Float[Tensor, "B Z H_z W_z"],
        Float[Tensor, "B Z H_z W_z"],
    ]:
        mu, lv = self.encode(x)
        z = reparameterize(mu, lv)
        return self.decode(z), mu, lv

    @staticmethod
    def kl_loss(
        mu: Float[Tensor, "..."], logvar: Float[Tensor, "..."]
    ) -> Float[Tensor, ""]:
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()


# ---------------------------------------------------------------------------
# Autoregressive (PixelCNN-style masked conv)
# ---------------------------------------------------------------------------

class MaskedConv2d(nn.Conv2d):
    """Masked 2D conv used by PixelCNN.

    ``mask_type='A'`` blocks the centre pixel; ``'B'`` includes it.
    """

    def __init__(self, mask_type: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if mask_type not in {"A", "B"}:
            raise ValueError("mask_type must be 'A' or 'B'")
        _, _, kH, kW = self.weight.shape
        mask = torch.ones_like(self.weight)
        mask[:, :, kH // 2, kW // 2 + (mask_type == "B"):] = 0
        mask[:, :, kH // 2 + 1:] = 0
        self.register_buffer("mask", mask, persistent=False)

    @typecheck
    def forward(                                                  # type: ignore[override]
        self, x: Float[Tensor, "B C_in H W"]
    ) -> Float[Tensor, "B C_out H W"]:
        self.weight.data.mul_(self.mask)
        return super().forward(x)


class AutoregressiveBlock(nn.Module):
    """A residual gated PixelCNN block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = MaskedConv2d("B", channels, 2 * channels, 3, padding=1)
        self.proj = MaskedConv2d("B", channels, channels, 1)

    @typecheck
    def forward(
        self, x: Float[Tensor, "B C H W"]
    ) -> Float[Tensor, "B C H W"]:
        a, b = self.conv(x).chunk(2, dim=1)
        return x + self.proj(torch.tanh(a) * torch.sigmoid(b))


# ---------------------------------------------------------------------------
# Normalizing flows
# ---------------------------------------------------------------------------

class AffineCouplingLayer(nn.Module):
    """RealNVP affine coupling: half the dims are scaled/shifted by the other half."""

    def __init__(self, dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.dim = dim
        self.half = dim // 2
        self.net = nn.Sequential(
            nn.Linear(self.half, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * (dim - self.half)),
        )

    @typecheck
    def forward(
        self, x: Float[Tensor, "B D"]
    ) -> tuple[Float[Tensor, "B D"], Float[Tensor, "B"]]:
        x1, x2 = x[:, :self.half], x[:, self.half:]
        s, t = self.net(x1).chunk(2, dim=-1)
        s = torch.tanh(s)
        y2 = x2 * s.exp() + t
        return torch.cat([x1, y2], dim=-1), s.sum(-1)             # log|det J| = sum s

    def inverse(
        self, y: Float[Tensor, "B D"]
    ) -> Float[Tensor, "B D"]:
        y1, y2 = y[:, :self.half], y[:, self.half:]
        s, t = self.net(y1).chunk(2, dim=-1)
        x2 = (y2 - t) * (-torch.tanh(s)).exp()
        return torch.cat([y1, x2], dim=-1)


# ---------------------------------------------------------------------------
# Energy-Based Model
# ---------------------------------------------------------------------------

class EnergyBasedModel(nn.Module):
    """Wraps any net so its scalar output is the energy ``E_theta(x)``."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    @typecheck
    def forward(
        self, x: Float[Tensor, "B ..."]
    ) -> Float[Tensor, "B"]:
        return self.backbone(x).flatten(1).sum(-1)

    def langevin_sample(
        self,
        x: Float[Tensor, "B ..."],
        steps: int = 60,
        step_size: float = 10.0,
        noise: float = 0.005,
    ) -> Float[Tensor, "B ..."]:
        """Stochastic gradient Langevin dynamics for sampling."""
        x = x.detach().requires_grad_(True)
        for _ in range(steps):
            E = self(x).sum()
            grad = torch.autograd.grad(E, x)[0]
            x = (x - step_size * grad + noise * torch.randn_like(x)).detach()
            x.requires_grad_(True)
        return x.detach()


# ---------------------------------------------------------------------------
# Diffusion schedulers
# ---------------------------------------------------------------------------

class DDPMScheduler(nn.Module):
    """Standard DDPM linear-beta scheduler with q-sample / step utilities."""

    def __init__(self, num_steps: int = 1000,
                 beta_start: float = 1e-4, beta_end: float = 0.02) -> None:
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, num_steps)
        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, 0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.num_steps = num_steps

    def add_noise(
        self,
        x0: Float[Tensor, "B C H W"],
        t: Int[Tensor, "B"],
        noise: Optional[Float[Tensor, "B C H W"]] = None,
    ) -> tuple[Float[Tensor, "B C H W"], Float[Tensor, "B C H W"]]:
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.alpha_bar[t][:, None, None, None]
        return a.sqrt() * x0 + (1 - a).sqrt() * noise, noise

    @torch.no_grad()
    def step(
        self,
        eps: Float[Tensor, "B C H W"],
        t: int,
        x_t: Float[Tensor, "B C H W"],
    ) -> Float[Tensor, "B C H W"]:
        beta = self.betas[t]
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bar[t]
        coef = beta / (1 - alpha_bar).sqrt()
        mean = (1 / alpha.sqrt()) * (x_t - coef * eps)
        if t > 0:
            return mean + beta.sqrt() * torch.randn_like(x_t)
        return mean


class DDIMScheduler(DDPMScheduler):
    """Deterministic DDIM step (Song et al. 2020)."""

    @torch.no_grad()
    def step(                                                                # type: ignore[override]
        self,
        eps: Float[Tensor, "B C H W"],
        t: int,
        x_t: Float[Tensor, "B C H W"],
        eta: float = 0.0,
        prev_t: Optional[int] = None,
    ) -> Float[Tensor, "B C H W"]:
        prev_t = max(t - 1, 0) if prev_t is None else prev_t
        a_t = self.alpha_bar[t]
        a_p = self.alpha_bar[prev_t] if prev_t >= 0 else torch.tensor(1.0,
                                                                      device=eps.device)
        x0 = (x_t - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        sigma = eta * ((1 - a_p) / (1 - a_t)).sqrt() * (1 - a_t / a_p).sqrt()
        dir_t = (1 - a_p - sigma ** 2).clamp(min=0).sqrt() * eps
        z = torch.randn_like(x_t) if eta > 0 else 0.0
        return a_p.sqrt() * x0 + dir_t + sigma * z
