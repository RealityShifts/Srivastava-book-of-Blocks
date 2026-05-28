"""Section 10 - Probabilistic / generative blocks.

VAE encoder/decoder + reparameterization, autoregressive masked-conv
block (PixelCNN), affine coupling layer (RealNVP / Glow), an EBM
wrapper, and DDPM/DDIM noise schedulers.
"""

from __future__ import annotations

from typing import Optional, Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Int, PRNGKeyArray
from ._typecheck import typecheck


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------

@typecheck
def reparameterize(
    mu: Float[Array, "..."],
    logvar: Float[Array, "..."],
    key: PRNGKeyArray,
) -> Float[Array, "..."]:
    """``z = mu + sigma * eps``,  ``eps ~ N(0, I)``."""
    std = jnp.exp(0.5 * logvar)
    return mu + std * jax.random.normal(key, mu.shape)


class VAE(nnx.Module):
    """Minimal convolutional VAE (NHWC); designed for 64x64 inputs by default."""

    def __init__(self, in_ch: int = 3, latent: int = 64,
                 channels: Sequence[int] = (32, 64, 128, 256),
                 *, rngs: nnx.Rngs) -> None:
        self.enc_convs = []
        c = in_ch
        for ch in channels:
            self.enc_convs.append(nnx.Conv(c, ch, (4, 4), strides=2,
                                           padding="SAME", rngs=rngs))
            self.enc_convs.append(nnx.GroupNorm(ch, num_groups=8, rngs=rngs))
            c = ch
        self.fc_mu = nnx.Conv(c, latent, (1, 1), rngs=rngs)
        self.fc_lv = nnx.Conv(c, latent, (1, 1), rngs=rngs)

        self.dec_first = nnx.Conv(latent, c, (1, 1), rngs=rngs)
        self.dec_layers = []
        for ch in reversed(channels[:-1]):
            self.dec_layers.append(nnx.ConvTranspose(c, ch, (4, 4), strides=2,
                                                     padding="SAME", rngs=rngs))
            self.dec_layers.append(nnx.GroupNorm(ch, num_groups=8, rngs=rngs))
            c = ch
        self.dec_final = nnx.ConvTranspose(c, in_ch, (4, 4), strides=2,
                                           padding="SAME", rngs=rngs)

    def encode(
        self, x: Float[Array, "B H W C"]
    ) -> tuple[Float[Array, "B H_z W_z D_z"], Float[Array, "B H_z W_z D_z"]]:
        for layer in self.enc_convs:
            x = nnx.silu(layer(x))
        return self.fc_mu(x), self.fc_lv(x)

    def decode(
        self, z: Float[Array, "B H_z W_z D_z"]
    ) -> Float[Array, "B H W C"]:
        h = self.dec_first(z)
        for layer in self.dec_layers:
            h = nnx.silu(layer(h))
        return self.dec_final(h)

    @typecheck
    def __call__(
        self,
        x: Float[Array, "B H W C"],
        key: PRNGKeyArray,
    ) -> tuple[Float[Array, "B H W C"],
               Float[Array, "B H_z W_z D_z"],
               Float[Array, "B H_z W_z D_z"]]:
        mu, lv = self.encode(x)
        z = reparameterize(mu, lv, key)
        return self.decode(z), mu, lv

    @staticmethod
    def kl_loss(
        mu: Float[Array, "..."],
        logvar: Float[Array, "..."],
    ) -> Float[Array, ""]:
        return -0.5 * jnp.mean(1 + logvar - mu ** 2 - jnp.exp(logvar))


# ---------------------------------------------------------------------------
# Autoregressive (PixelCNN-style masked conv)
# ---------------------------------------------------------------------------

class MaskedConv2d(nnx.Module):
    """Masked 2D conv used by PixelCNN.

    ``mask_type='A'`` blocks the centre pixel; ``'B'`` includes it.
    """

    def __init__(self, mask_type: str, in_ch: int, out_ch: int,
                 kernel_size: int = 3, use_bias: bool = True, *, rngs: nnx.Rngs) -> None:
        if mask_type not in {"A", "B"}:
            raise ValueError("mask_type must be 'A' or 'B'")
        self.conv = nnx.Conv(in_ch, out_ch, (kernel_size, kernel_size),
                             padding="SAME", use_bias=use_bias, rngs=rngs)
        kH, kW = kernel_size, kernel_size
        mask = jnp.ones((kH, kW, in_ch, out_ch))
        mask = mask.at[kH // 2, kW // 2 + (mask_type == "B"):].set(0)
        mask = mask.at[kH // 2 + 1:].set(0)
        self.mask = mask

    @typecheck
    def __call__(
        self, x: Float[Array, "B H W C_in"]
    ) -> Float[Array, "B H W C_out"]:
        self.conv.kernel.value = self.conv.kernel.value * self.mask
        return self.conv(x)


class AutoregressiveBlock(nnx.Module):
    """A residual gated PixelCNN block."""

    def __init__(self, channels: int, *, rngs: nnx.Rngs) -> None:
        self.conv = MaskedConv2d("B", channels, 2 * channels, 3, rngs=rngs)
        self.proj = MaskedConv2d("B", channels, channels, 1, rngs=rngs)

    @typecheck
    def __call__(
        self, x: Float[Array, "B H W C"]
    ) -> Float[Array, "B H W C"]:
        a, b = jnp.split(self.conv(x), 2, axis=-1)
        return x + self.proj(jnp.tanh(a) * nnx.sigmoid(b))


# ---------------------------------------------------------------------------
# Normalizing flows
# ---------------------------------------------------------------------------

class AffineCouplingLayer(nnx.Module):
    """RealNVP affine coupling: half the dims are scaled/shifted by the other half."""

    def __init__(self, dim: int, hidden: int = 128,
                 *, rngs: nnx.Rngs) -> None:
        self.dim = dim
        self.half = dim // 2
        self.fc1 = nnx.Linear(self.half, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, hidden, rngs=rngs)
        self.fc3 = nnx.Linear(hidden, 2 * (dim - self.half), rngs=rngs)

    def _net(
        self, x1: Float[Array, "B D_half"]
    ) -> Float[Array, "B D_st"]:
        return self.fc3(nnx.relu(self.fc2(nnx.relu(self.fc1(x1)))))

    @typecheck
    def __call__(
        self, x: Float[Array, "B D"]
    ) -> tuple[Float[Array, "B D"], Float[Array, "B"]]:
        x1, x2 = x[:, :self.half], x[:, self.half:]
        s, t = jnp.split(self._net(x1), 2, axis=-1)
        s = jnp.tanh(s)
        y2 = x2 * jnp.exp(s) + t
        return jnp.concatenate([x1, y2], axis=-1), jnp.sum(s, axis=-1)

    def inverse(
        self, y: Float[Array, "B D"]
    ) -> Float[Array, "B D"]:
        y1, y2 = y[:, :self.half], y[:, self.half:]
        s, t = jnp.split(self._net(y1), 2, axis=-1)
        x2 = (y2 - t) * jnp.exp(-jnp.tanh(s))
        return jnp.concatenate([y1, x2], axis=-1)


# ---------------------------------------------------------------------------
# Energy-based model
# ---------------------------------------------------------------------------

class EnergyBasedModel(nnx.Module):
    """Wraps any net so its scalar output is the energy ``E_theta(x)``."""

    def __init__(self, backbone: nnx.Module) -> None:
        self.backbone = backbone

    @typecheck
    def __call__(
        self, x: Float[Array, "B *spatial"]
    ) -> Float[Array, "B"]:
        return jnp.sum(self.backbone(x).reshape(x.shape[0], -1), axis=-1)

    def langevin_sample(
        self,
        x: Float[Array, "B *spatial"],
        key: PRNGKeyArray,
        steps: int = 60,
        step_size: float = 10.0,
        noise: float = 0.005,
    ) -> Float[Array, "B *spatial"]:
        """Stochastic gradient Langevin dynamics for sampling."""
        grad_fn = jax.grad(lambda y: jnp.sum(self(y)))
        for _ in range(steps):
            key, sub = jax.random.split(key)
            x = (x - step_size * grad_fn(x)
                 + noise * jax.random.normal(sub, x.shape))
        return x


# ---------------------------------------------------------------------------
# Diffusion schedulers
# ---------------------------------------------------------------------------

class DDPMScheduler:
    """Standard DDPM linear-beta scheduler with q-sample / step utilities."""

    def __init__(self, num_steps: int = 1000,
                 beta_start: float = 1e-4, beta_end: float = 0.02) -> None:
        self.num_steps = num_steps
        self.betas = jnp.linspace(beta_start, beta_end, num_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = jnp.cumprod(self.alphas, axis=0)

    def add_noise(
        self,
        x0: Float[Array, "B H W C"],
        t: Int[Array, "B"],
        key: PRNGKeyArray,
        noise: Optional[Float[Array, "B H W C"]] = None,
    ) -> tuple[Float[Array, "B H W C"], Float[Array, "B H W C"]]:
        noise = jax.random.normal(key, x0.shape) if noise is None else noise
        a = self.alpha_bar[t][:, None, None, None]
        return jnp.sqrt(a) * x0 + jnp.sqrt(1 - a) * noise, noise

    def step(
        self,
        eps: Float[Array, "B H W C"],
        t: int,
        x_t: Float[Array, "B H W C"],
        key: PRNGKeyArray,
    ) -> Float[Array, "B H W C"]:
        beta = self.betas[t]
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bar[t]
        coef = beta / jnp.sqrt(1 - alpha_bar)
        mean = (1 / jnp.sqrt(alpha)) * (x_t - coef * eps)
        if t > 0:
            return mean + jnp.sqrt(beta) * jax.random.normal(key, x_t.shape)
        return mean


class DDIMScheduler(DDPMScheduler):
    """Deterministic DDIM step (Song et al. 2020)."""

    def step(                                                              # type: ignore[override]
        self,
        eps: Float[Array, "B H W C"],
        t: int,
        x_t: Float[Array, "B H W C"],
        key: PRNGKeyArray,
        eta: float = 0.0,
        prev_t: Optional[int] = None,
    ) -> Float[Array, "B H W C"]:
        prev_t = max(t - 1, 0) if prev_t is None else prev_t
        a_t = self.alpha_bar[t]
        a_p = self.alpha_bar[prev_t] if prev_t >= 0 else jnp.array(1.0)
        x0 = (x_t - jnp.sqrt(1 - a_t) * eps) / jnp.sqrt(a_t)
        sigma = (eta * jnp.sqrt((1 - a_p) / (1 - a_t))
                 * jnp.sqrt(1 - a_t / a_p))
        dir_t = jnp.sqrt(jnp.maximum(1 - a_p - sigma ** 2, 0)) * eps
        z = (jax.random.normal(key, x_t.shape) if eta > 0
             else jnp.zeros_like(x_t))
        return jnp.sqrt(a_p) * x0 + dir_t + sigma * z
