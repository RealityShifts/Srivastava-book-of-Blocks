"""Hooks that measure selected modules during one forward pass.

A :class:`Collector` is attached for a single *probe* forward and removed
again; with none attached the model is exactly as it was. Per selected module
it records, for every call of its ``forward`` in that pass:

* statistics of every tensor going in and coming out (:mod:`.stats`),
* wall time, with the device synchronised at entry and exit so asynchronous
  GPU work is charged to the module that queued it.

and, once per pass, statistics of the module's own parameters (and their
gradients, when present) and of all its parameters together.

Timing excludes the monitor's own work: the time spent computing statistics
inside a module's window is accumulated and subtracted, for that module and
for every selected ancestor around it. ``self`` time is the inclusive time
minus the inclusive time of the *selected* modules called directly inside -
unselected children stay in their parent's self time.
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from .stats import BINS, MEDIAN_CAP, summarize, tensor_stats

#: The numbers summarised across iterations for every tensor.
STAT_KEYS = ("mean", "std", "median", "mode", "min", "max")


def named_tensors(obj: Any, prefix: str = "") -> list:
    """``(name, tensor)`` for every tensor inside nested tuples, lists and
    dicts: ``"0"``, ``"1.2"``, ``"kw.mask"``. Anything else is skipped."""
    if isinstance(obj, Tensor):
        return [(prefix or "0", obj)]
    out = []
    if isinstance(obj, (tuple, list)):
        for i, item in enumerate(obj):
            out += named_tensors(item, f"{prefix}.{i}" if prefix else str(i))
    elif isinstance(obj, dict):
        for key, item in obj.items():
            out += named_tensors(item, f"{prefix}.{key}" if prefix else str(key))
    return out


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def resolve(model: nn.Module, path: str) -> nn.Module:
    """The module at ``path`` (``""`` is ``model`` itself)."""
    return model.get_submodule(path) if path else model


class Collector:
    """Measure ``paths`` of ``model`` for one forward pass.

    Use as a context manager around the probe forward, then call
    :meth:`record`::

        with Collector(model, ["encoder", "encoder.0"]) as c:
            model(x)
        data = c.record()
    """

    def __init__(self, model: nn.Module, paths: list, *, stats: bool = True,
                 timing: bool = True, bins: int = BINS,
                 median_cap: int = MEDIAN_CAP) -> None:
        self.model = model
        self.modules = {p: resolve(model, p) for p in paths}
        self.stats, self.timing = stats, timing
        self.bins, self.median_cap = bins, median_cap
        first = next(model.parameters(), None)
        self.device = first.device if first is not None else torch.device("cpu")
        self._handles: list = []
        self._stack: list = []
        self._overhead = 0.0
        self._calls: dict = {p: [] for p in paths}

    # ------------------------------------------------------------------ hooks
    def _tensor_rows(self, obj: Any) -> list:
        return [{"name": name,
                 **tensor_stats(t, self.bins, self.median_cap)}
                for name, t in named_tensors(obj)]

    def _pre(self, path: str):
        def hook(module, args, kwargs):
            _sync(self.device)                 # the caller's queued work is its own
            start = time.perf_counter()
            call = {"inputs": [], "outputs": [], "time": None}
            if self.stats:
                call["inputs"] = self._tensor_rows(args)
                if kwargs:
                    call["inputs"] += self._tensor_rows({"kw": kwargs})
            self._calls[path].append(call)
            t0 = time.perf_counter()
            self._overhead += t0 - start
            self._stack.append({"path": path, "call": call, "t0": t0,
                                "overhead0": self._overhead, "children": 0.0})
        return hook

    def _post(self, path: str):
        def hook(module, args, kwargs, output):
            _sync(self.device)
            t1 = time.perf_counter()
            frame = self._stack.pop()
            if frame["path"] != path:          # an exception unwound past a hook
                self._stack.clear()
                return
            inclusive = (t1 - frame["t0"]) - (self._overhead - frame["overhead0"])
            if self.timing:
                frame["call"]["time"] = {
                    "inclusive": inclusive,
                    "self": inclusive - frame["children"]}
            if self._stack:
                self._stack[-1]["children"] += inclusive
            if self.stats:
                frame["call"]["outputs"] = self._tensor_rows(output)
            self._overhead += time.perf_counter() - t1
        return hook

    def __enter__(self) -> "Collector":
        for path, module in self.modules.items():
            self._handles.append(module.register_forward_pre_hook(
                self._pre(path), with_kwargs=True))
            self._handles.append(module.register_forward_hook(
                self._post(path), with_kwargs=True))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._stack.clear()

    # ----------------------------------------------------------------- record
    def _params(self, module: nn.Module) -> tuple[list, dict]:
        rows = []
        for name, p in module._parameters.items():
            if p is None:
                continue
            row = {"name": name, **tensor_stats(p, self.bins, self.median_cap)}
            row["grad"] = (tensor_stats(p.grad, self.bins, self.median_cap)
                           if p.grad is not None else None)
            rows.append(row)
        count, total, lo, hi = 0, 0.0, None, None
        for p in module.parameters():
            if p.numel() == 0:
                continue
            x = p.detach().float()
            count += x.numel()
            total += float(x.sum())
            a, b = float(x.min()), float(x.max())
            lo = a if lo is None else min(lo, a)
            hi = b if hi is None else max(hi, b)
        whole = {"params": count, "mean": total / count if count else None,
                 "min": lo, "max": hi}
        return rows, whole

    def record(self) -> dict:
        """This pass's measurements, JSON-ready: ``{path: {...}}``."""
        out = {}
        for path, module in self.modules.items():
            entry = {"cls": type(module).__name__, "calls": self._calls[path]}
            if self.stats:
                entry["params"], entry["whole"] = self._params(module)
            out[path] = entry
        return out


def _summ_rows(per_iter: list) -> list:
    """One row per tensor name: each statistic's mean/min/max across
    iterations, plus the last iteration's shape, dtype and histogram."""
    names: list = []
    for rows in per_iter:
        for row in rows:
            if row["name"] not in names:
                names.append(row["name"])
    out = []
    for name in names:
        seen = [r for rows in per_iter for r in rows if r["name"] == name]
        last = seen[-1]
        row = {"name": name, "shape": last["shape"], "dtype": last["dtype"],
               "numel": last["numel"], "hist": last["hist"],
               "hist_range": last["hist_range"],
               "median_sampled": last["median_sampled"],
               "nonfinite": max(r["nonfinite"] for r in seen),
               "shapes_vary": any(r["shape"] != last["shape"] for r in seen)}
        for key in STAT_KEYS:
            row[key] = summarize([r[key] for r in seen])
        if "grad" in last:
            grads = [[{"name": "grad", **r["grad"]}] for r in seen
                     if r.get("grad") is not None]
            row["grad"] = _summ_rows(grads)[0] if grads else None
        out.append(row)
    return out


def summarize_iterations(iterations: list) -> dict:
    """Roll per-iteration ``Collector.record()`` dicts up to one summary per
    module: ``{path: {cls, calls: [...], params: [...], whole: {...}}}``."""
    out: dict = {}
    for path in (iterations[0] if iterations else {}):
        mods = [it[path] for it in iterations if path in it]
        ncalls = max((len(m["calls"]) for m in mods), default=0)
        calls = []
        for k in range(ncalls):
            ks = [m["calls"][k] for m in mods if len(m["calls"]) > k]
            times = [c["time"] for c in ks if c["time"]]
            calls.append({
                "seen": len(ks),
                "time": ({"inclusive": summarize([t["inclusive"] for t in times]),
                          "self": summarize([t["self"] for t in times])}
                         if times else None),
                "inputs": _summ_rows([c["inputs"] for c in ks]),
                "outputs": _summ_rows([c["outputs"] for c in ks])})
        entry = {"cls": mods[0]["cls"], "calls": calls}
        if "params" in mods[0]:
            entry["params"] = _summ_rows([m["params"] for m in mods])
            wholes = [m["whole"] for m in mods]
            entry["whole"] = {
                "params": wholes[-1]["params"],
                **{k: summarize([w[k] for w in wholes])
                   for k in ("mean", "min", "max")}}
        out[path] = entry
    return out
