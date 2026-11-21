"""Summary statistics of one tensor: mean, median, mode, min, max, histogram.

Everything is computed on the tensor's own device; only Python numbers and
the histogram's bin counts leave it.

Definitions:

* **mean / std / min / max** over the finite values. Non-finite ones are
  counted in ``nonfinite`` and left out, so one NaN does not blank the row
  that would have told you where it is.
* **median**: the midpoint of the two middle values of the sorted data (the
  textbook median). Exact up to ``median_cap`` elements; above it, taken on a
  fixed random subsample of that many and flagged ``median_sampled`` - sorting
  25M values per tensor per iteration is the one cost here that scales badly.
* **mode**: floating-point values rarely repeat, so the most frequent *value*
  says nothing. It is the centre of the fullest of ``bins`` equal-width
  histogram bins between min and max; the histogram is returned so the reader
  can see what that bin is. For a constant tensor it is that constant.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

BINS = 64
MEDIAN_CAP = 1_000_000
#: How many of the tensor's first values (flattened) are kept as a preview.
HEAD = 16


def _sample(x: Tensor, cap: int) -> Tensor:
    """``cap`` elements of flat ``x`` at fixed pseudo-random positions (the
    same ones for the same length, so iterations are comparable)."""
    gen = torch.Generator().manual_seed(x.numel())
    idx = torch.randint(0, x.numel(), (cap,), generator=gen)
    return x[idx.to(x.device)]


def tensor_stats(t: Tensor, bins: int = BINS,
                 median_cap: int = MEDIAN_CAP) -> dict:
    """Statistics of ``t`` as plain Python values (JSON-ready)."""
    out = {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""),
           "numel": t.numel()}
    x = t.detach()
    if x.is_complex():
        x = torch.view_as_real(x)
    x = x.reshape(-1).to(torch.float32)
    # The values themselves, as far as a preview goes: the first HEAD of the
    # flattened tensor - all of it for a small vector.
    out["head"] = [v if math.isfinite(v) else None
                   for v in x[:HEAD].tolist()]
    finite = torch.isfinite(x)
    nonfinite = int((~finite).sum())
    out["nonfinite"] = nonfinite
    if nonfinite:
        x = x[finite]
    n = x.numel()
    if n == 0:
        out.update(mean=None, std=None, median=None, mode=None, min=None,
                   max=None, median_sampled=False, hist=[], hist_range=None)
        return out
    lo, hi = float(x.min()), float(x.max())
    out["mean"] = float(x.mean())
    out["std"] = float(x.std()) if n > 1 else 0.0
    out["min"], out["max"] = lo, hi

    sampled = n > median_cap
    m = _sample(x, median_cap) if sampled else x
    s = torch.sort(m).values
    k = s.numel()
    out["median"] = float((s[(k - 1) // 2] + s[k // 2]) / 2)
    out["median_sampled"] = sampled

    if hi > lo:
        hist = torch.histc(x, bins=bins, min=lo, max=hi)
        top = int(hist.argmax())
        width = (hi - lo) / bins
        out["mode"] = lo + (top + 0.5) * width
        out["hist"] = [int(c) for c in hist.tolist()]
    else:
        out["mode"] = lo
        out["hist"] = [n]
    out["hist_range"] = [lo, hi]
    return out


def summarize(values: list) -> dict | None:
    """Mean, min and max of one statistic across iterations (``None``
    entries - an empty tensor that iteration - are skipped)."""
    vals = [v for v in values if v is not None and not (
        isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None
    return {"mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals)}
