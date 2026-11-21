"""Scalar curves and the alerts raised on them.

A :class:`ScalarBoard` holds scalar series - ``(step, value)`` per *run* and
*tag* - from two sources:

* **TensorBoard logs**: :meth:`ScalarBoard.open` reads every event file under
  a directory (one run per folder that has any), and re-reads only what was
  appended since.
* **the training loop**: :meth:`ScalarBoard.log` takes values directly, for a
  run that writes no TensorBoard logs.

and checks every series against a small set of rules (:data:`RULES`):
non-finite values, spikes, steady growth, vanishing, a loss turning upward, a
flat line, a run gone quiet. What they find are :class:`alerts <Alert>`, each
``critical``, ``warning`` or ``info``.

The rules read the *name* of a series to decide which apply: growth is a
concern for an activation, a weight or a gradient norm, not for a step
counter; "vanishing" is about gradients. The patterns are in
:data:`DEFAULTS` and are yours to change.

Reading TensorBoard files needs the ``tensorboard`` package (any training
environment that writes them has it); everything else here needs only numpy.

    python -m Blocks.monitor.scalars runs/exp1            # print the alerts
    python -m Blocks.monitor.scalars runs/exp1 --serve    # and open the page
"""

from __future__ import annotations

import dataclasses
import math
import os
import re
import threading
import time
from typing import Callable, Optional

import numpy as np

LEVELS = ("critical", "warning", "info")
_RANK = {"critical": 0, "warning": 1, "info": 2}

#: Every limit the rules use. Override any of them:
#: ``ScalarBoard(limits={"grow_factor": 5, "loss_tags": r"loss|nll"})``.
DEFAULTS = {
    # Which series each rule looks at, as case-insensitive regexes on the tag.
    "grow_tags": r"act|weight|param|norm|grad|max|logit|scale",
    "grad_tags": r"grad",
    "loss_tags": r"loss|total|recon|lpips|\bl1\b|\bl2\b|nll|\bkl\b|error",
    "ignore_tags": r"^(epoch|step|gen_step|progress/|lr\b|lr/)|/lr$|learning_rate",
    # A rule needs this many points before it says anything about a trend.
    "min_points": 20,
    # spike: a point further than spike_k robust deviations from the median of
    # the spike_window points before it.
    "spike_k": 12.0,
    "spike_window": 50,
    # growing: the recent level over the early level, in magnitude.
    "grow_factor": 10.0,          # warning
    "grow_factor_critical": 100.0,
    # vanishing (gradient tags): below vanish_abs, or fallen by vanish_factor.
    "vanish_abs": 1e-8,
    "vanish_factor": 1e3,
    # diverging (loss tags): recent mean above the best earlier mean by this
    # fraction of it, and still rising.
    "diverge_fraction": 0.5,
    # flat: this many identical trailing points.
    "flat_points": 30,
    # stale: a run whose newest point is older than this many seconds, while
    # younger than stale_ignore_after (an old finished run is not "stale").
    "stale_seconds": 15 * 60,
    "stale_ignore_after": 24 * 3600,
}


@dataclasses.dataclass
class Alert:
    """One concern about one series."""
    run: str
    tag: str
    rule: str            # a key of RULES
    level: str           # "critical" | "warning" | "info"
    step: int            # where it was seen (the latest occurrence)
    value: Optional[float]
    message: str
    count: int = 1       # occurrences, for rules that can fire repeatedly

    @property
    def key(self) -> tuple:
        return (self.run, self.tag, self.rule)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        if d["value"] is not None and not math.isfinite(d["value"]):
            d["value"] = None
        return d


def _fmt(v: float) -> str:
    return f"{v:.3g}"


def _windows(y: np.ndarray) -> tuple:
    """(early, recent): the first tenth after a 5% warm-up, and the last
    tenth - at least 5 points each."""
    n = len(y)
    w = max(5, n // 10)
    skip = min(n // 20, max(0, n - 2 * w))
    return y[skip:skip + w], y[-w:]


def rule_nonfinite(steps, y, tag, L) -> Optional[dict]:
    bad = ~np.isfinite(y)
    if not bad.any():
        return None
    at = int(np.nonzero(bad)[0][-1])
    kind = "NaN" if np.isnan(y[at]) else "Inf"
    return {"level": "critical", "step": int(steps[at]), "value": None,
            "count": int(bad.sum()),
            "message": f"{int(bad.sum())} non-finite value(s); the latest "
                       f"is {kind} at step {int(steps[at])}"}


def rule_spike(steps, y, tag, L) -> Optional[dict]:
    w = int(L["spike_window"])
    if len(y) < max(w + 1, L["min_points"]):
        return None
    # Rolling median / MAD of the w points before each point.
    view = np.lib.stride_tricks.sliding_window_view(y[:-1], w)
    med = np.median(view, axis=1)
    mad = np.median(np.abs(view - med[:, None]), axis=1)
    cur = y[w:]
    scale = 1.4826 * mad + 1e-3 * np.abs(med) + 1e-12
    dev = np.abs(cur - med) / scale
    hits = np.nonzero(dev > L["spike_k"])[0]
    if len(hits) == 0:
        return None
    at = int(hits[-1]) + w
    worst = int(hits[np.argmax(dev[hits])]) + w
    return {"level": "warning", "step": int(steps[at]), "value": float(y[at]),
            "count": int(len(hits)),
            "message": f"{len(hits)} spike(s): latest {_fmt(y[at])} at step "
                       f"{int(steps[at])}; largest {_fmt(y[worst])} at step "
                       f"{int(steps[worst])} against a local level of "
                       f"{_fmt(med[worst - w])}"}


def rule_growing(steps, y, tag, L) -> Optional[dict]:
    if len(y) < L["min_points"] or not re.search(L["grow_tags"], tag, re.I):
        return None
    early, recent = _windows(np.abs(y))
    a, b = float(np.median(early)), float(np.median(recent))
    if a <= 0 or b <= 0:
        return None
    ratio = b / a
    # Still rising, not a step that happened long ago and settled.
    half = np.abs(y[len(y) // 2:])
    rising = np.median(half[len(half) // 2:]) > 1.05 * np.median(
        half[:max(1, len(half) // 2)])
    if ratio < L["grow_factor"] or not rising:
        return None
    level = "critical" if ratio >= L["grow_factor_critical"] else "warning"
    return {"level": level, "step": int(steps[-1]), "value": float(y[-1]),
            "message": f"growing: magnitude {_fmt(a)} to {_fmt(b)} "
                       f"({_fmt(ratio)}x) and still rising"}


def rule_vanishing(steps, y, tag, L) -> Optional[dict]:
    if len(y) < L["min_points"] or not re.search(L["grad_tags"], tag, re.I):
        return None
    early, recent = _windows(np.abs(y))
    a, b = float(np.median(early)), float(np.median(recent))
    if b < L["vanish_abs"]:
        return {"level": "critical" if b == 0 else "warning",
                "step": int(steps[-1]), "value": float(y[-1]),
                "message": f"vanishing: magnitude {_fmt(b)} is below "
                           f"{_fmt(L['vanish_abs'])}"}
    if a > 0 and a / b >= L["vanish_factor"]:
        return {"level": "warning", "step": int(steps[-1]),
                "value": float(y[-1]),
                "message": f"vanishing: magnitude fell {_fmt(a)} to {_fmt(b)} "
                           f"({_fmt(a / b)}x smaller)"}
    return None


def rule_diverging(steps, y, tag, L) -> Optional[dict]:
    if len(y) < L["min_points"] or not re.search(L["loss_tags"], tag, re.I):
        return None
    w = max(5, len(y) // 10)
    recent = float(np.mean(y[-w:]))
    # The best (lowest) level any earlier window of the same length reached.
    means = np.convolve(y[:-w], np.ones(w) / w, mode="valid")
    if len(means) == 0:
        return None
    best_at = int(np.argmin(means))
    best = float(means[best_at])
    before = float(np.mean(y[-2 * w:-w])) if len(y) >= 2 * w else best
    gap = recent - best
    if gap <= L["diverge_fraction"] * max(abs(best), 1e-12) or recent <= before:
        return None
    return {"level": "warning", "step": int(steps[-1]), "value": float(y[-1]),
            "message": f"rising: recent mean {_fmt(recent)} against a best of "
                       f"{_fmt(best)} around step {int(steps[best_at + w - 1])}"}


def rule_flat(steps, y, tag, L) -> Optional[dict]:
    k = int(L["flat_points"])
    if len(y) < k:
        return None
    tail = y[-k:]
    if not np.all(tail == tail[0]):
        return None
    run = len(y) - 1
    while run > 0 and y[run - 1] == y[-1]:
        run -= 1
    whole = run == 0
    return {"level": "info", "step": int(steps[-1]), "value": float(y[-1]),
            "message": (f"constant: every one of {len(y)} points is "
                        f"{_fmt(y[-1])}" if whole else
                        f"flat: unchanged at {_fmt(y[-1])} since step "
                        f"{int(steps[run])} ({len(y) - run} points)")}


#: name -> (function, what it catches). Applied to every series, in order.
RULES = {
    "nonfinite": (rule_nonfinite, "NaN or Inf logged"),
    "spike": (rule_spike, "a point far outside its recent neighbourhood"),
    "growing": (rule_growing, "magnitude growing steadily"),
    "vanishing": (rule_vanishing, "a gradient shrinking away"),
    "diverging": (rule_diverging, "a loss turning upward"),
    "flat": (rule_flat, "a value that stopped changing"),
}


class _Series:
    __slots__ = ("steps", "values", "wall")

    def __init__(self) -> None:
        self.steps: list = []
        self.values: list = []
        self.wall: list = []


class ScalarBoard:
    """Scalar series from TensorBoard logs and/or the loop, with alerts.

    ``limits`` overrides entries of :data:`DEFAULTS`. ``on_alert(alert)`` is
    called once for each alert the first time it appears and again when its
    level gets worse - the place to send a message somewhere. ``root``, when
    set, is the only directory tree :meth:`open` will read from.
    """

    def __init__(self, *, limits: Optional[dict] = None,
                 on_alert: Optional[Callable[[Alert], None]] = None,
                 root: Optional[str] = None) -> None:
        unknown = set(limits or {}) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"limits: unknown {sorted(unknown)}; known: "
                             f"{sorted(DEFAULTS)}")
        self.limits = {**DEFAULTS, **(limits or {})}
        self.on_alert = on_alert
        self.root = os.path.realpath(root) if root else None
        self.logdir: Optional[str] = None
        self._lock = threading.RLock()
        self._series: dict = {}            # (run, tag) -> _Series
        self._accumulators: dict = {}      # run -> EventAccumulator
        self._cursor: dict = {}            # (run, tag) -> events consumed
        self._alerts: dict = {}            # key -> Alert
        self._notified: dict = {}          # key -> level already announced
        self._newest: dict = {}            # run -> wall time of its last point
        self._refreshed_at = 0.0
        self._dirty = True
        self.error: Optional[str] = None

    # ------------------------------------------------------------- sources
    def log(self, step: int, values: dict, run: str = "live") -> None:
        """Add ``{tag: number}`` at ``step`` straight from the loop."""
        now = time.time()
        with self._lock:
            for tag, value in values.items():
                s = self._series.setdefault((run, str(tag)), _Series())
                s.steps.append(int(step))
                s.values.append(float(value))
                s.wall.append(now)
            self._dirty = True

    def open(self, logdir: str) -> dict:
        """Read the TensorBoard logs under ``logdir`` (replacing any
        directory opened before; series from :meth:`log` are kept)."""
        real = os.path.realpath(os.path.expanduser(logdir))
        if self.root and os.path.commonpath([real, self.root]) != self.root:
            raise ValueError(f"only paths under {self.root} can be opened")
        if not os.path.isdir(real):
            raise ValueError(f"not a directory: {logdir}")
        runs = _find_runs(real)
        if not runs:
            raise ValueError(f"no TensorBoard event files under {logdir}")
        with self._lock:
            for key in [k for k in self._series if k[0] in self._accumulators]:
                del self._series[key]
                self._cursor.pop(key, None)
            self._accumulators = {}
            self.logdir = real
            self._dirty = True
        self.refresh()
        return self.summary()

    def refresh(self) -> None:
        """Pick up whatever the open log directory gained since last time."""
        if self.logdir is None:
            return
        self._refreshed_at = time.time()
        try:
            from tensorboard.backend.event_processing.event_accumulator import (
                EventAccumulator)
        except ImportError as exc:
            self.error = ("reading TensorBoard logs needs the 'tensorboard' "
                          f"package: {exc}")
            raise RuntimeError(self.error) from exc
        with self._lock:
            for run, path in _find_runs(self.logdir).items():
                acc = self._accumulators.get(run)
                if acc is None:
                    # scalars=0: keep every point. The default keeps a random
                    # 10,000, which is how a single NaN goes missing.
                    acc = EventAccumulator(path, size_guidance={"scalars": 0})
                    self._accumulators[run] = acc
                acc.Reload()
                for tag in acc.Tags().get("scalars", []):
                    events = acc.Scalars(tag)
                    key = (run, tag)
                    done = self._cursor.get(key, 0)
                    if len(events) <= done:
                        continue
                    s = self._series.setdefault(key, _Series())
                    for e in events[done:]:
                        s.steps.append(int(e.step))
                        s.values.append(float(e.value))
                        s.wall.append(float(e.wall_time))
                    self._cursor[key] = len(events)
                    self._dirty = True
            self.error = None

    def refresh_if_due(self, seconds: float = 10.0) -> None:
        """:meth:`refresh`, at most once per ``seconds`` (what the page's
        polling calls)."""
        if self.logdir and time.time() - self._refreshed_at >= seconds:
            self.refresh()

    # -------------------------------------------------------------- alerts
    def check(self) -> list:
        """Run the rules over every series; returns the alerts, worst first.
        New alerts (and ones that got worse) go to ``on_alert``."""
        with self._lock:
            if self._dirty:
                found: dict = {}
                newest: dict = {}
                for (run, tag), s in self._series.items():
                    if s.wall:
                        newest[run] = max(newest.get(run, 0.0), s.wall[-1])
                    if re.search(self.limits["ignore_tags"], tag, re.I):
                        continue
                    steps = np.asarray(s.steps)
                    y = np.asarray(s.values, dtype=np.float64)
                    order = np.argsort(steps, kind="stable")
                    steps, y = steps[order], y[order]
                    hit = rule_nonfinite(steps, y, tag, self.limits)
                    if hit:
                        found[(run, tag, "nonfinite")] = Alert(
                            run, tag, "nonfinite", hit["level"], hit["step"],
                            hit["value"], hit["message"], hit.get("count", 1))
                    ok = np.isfinite(y)
                    steps, y = steps[ok], y[ok]
                    for name, (fn, _) in RULES.items():
                        if name == "nonfinite" or len(y) == 0:
                            continue
                        hit = fn(steps, y, tag, self.limits)
                        if hit:
                            found[(run, tag, name)] = Alert(
                                run, tag, name, hit["level"], hit["step"],
                                hit["value"], hit["message"],
                                hit.get("count", 1))
                self._alerts = found
                self._newest = newest
                self._dirty = False
            alerts = dict(self._alerts)
            # stale depends on the clock, so it is decided on every call.
            now = time.time()
            for run, last in self._newest.items():
                age = now - last
                if self.limits["stale_seconds"] <= age < \
                        self.limits["stale_ignore_after"]:
                    alerts[(run, "", "stale")] = Alert(
                        run, "", "stale", "info", 0, None,
                        f"no new points for {age / 60:.0f} min: the run has "
                        f"stopped, stalled, or logs rarely")
            fresh = []
            for key, alert in alerts.items():
                seen = self._notified.get(key)
                if seen is None or _RANK[alert.level] < _RANK[seen]:
                    self._notified[key] = alert.level
                    fresh.append(alert)
        if self.on_alert:
            for alert in fresh:
                try:
                    self.on_alert(alert)
                except Exception:                           # noqa: BLE001
                    pass
        return sorted(alerts.values(),
                      key=lambda a: (_RANK[a.level], a.run, a.tag, a.rule))

    # ------------------------------------------------------------- reading
    def summary(self) -> dict:
        """Runs, tags and each tag's worst alert level - what the page lists."""
        alerts = self.check()
        worst: dict = {}
        for a in alerts:
            k = (a.run, a.tag)
            if k not in worst or _RANK[a.level] < _RANK[worst[k]]:
                worst[k] = a.level
        with self._lock:
            tags = [{"run": run, "tag": tag, "points": len(s.steps),
                     "last_step": s.steps[-1] if s.steps else None,
                     "last": (s.values[-1] if s.values and
                              math.isfinite(s.values[-1]) else None),
                     "level": worst.get((run, tag))}
                    for (run, tag), s in sorted(self._series.items())]
        counts = {lv: sum(1 for a in alerts if a.level == lv) for lv in LEVELS}
        return {"logdir": self.logdir, "error": self.error,
                "runs": sorted({t["run"] for t in tags}), "tags": tags,
                "alerts": [a.to_dict() for a in alerts], "counts": counts,
                "limits": self.limits,
                "rules": {k: v[1] for k, v in RULES.items()}}

    def series(self, run: str, tag: str, max_points: int = 1500) -> dict:
        """One curve for drawing. Above ``max_points`` it is bucketed, and
        each bucket keeps its minimum *and* its maximum, so a one-step spike
        survives being drawn small. Non-finite points are returned separately
        (``bad``: their steps), since they cannot be drawn."""
        with self._lock:
            s = self._series.get((run, tag))
            if s is None:
                raise KeyError(f"no series {run!r} / {tag!r}")
            steps = np.asarray(s.steps)
            y = np.asarray(s.values, dtype=np.float64)
        order = np.argsort(steps, kind="stable")
        steps, y = steps[order], y[order]
        ok = np.isfinite(y)
        bad = [int(v) for v in steps[~ok][-200:]]
        steps, y = steps[ok], y[ok]
        n = len(y)
        if n > max_points:
            buckets = max(1, max_points // 2)
            edges = np.linspace(0, n, buckets + 1).astype(int)
            xs, ys = [], []
            for a, b in zip(edges[:-1], edges[1:]):
                if b <= a:
                    continue
                lo, hi = a + int(np.argmin(y[a:b])), a + int(np.argmax(y[a:b]))
                for i in sorted({lo, hi}):
                    xs.append(int(steps[i]))
                    ys.append(float(y[i]))
        else:
            xs, ys = [int(v) for v in steps], [float(v) for v in y]
        return {"run": run, "tag": tag, "points": n, "steps": xs,
                "values": ys, "bad": bad}


def _find_runs(logdir: str) -> dict:
    """``{run name: directory}`` for every folder under ``logdir`` holding a
    TensorBoard event file. The top folder itself is run ``"."``."""
    runs = {}
    for folder, _, files in os.walk(logdir, followlinks=True):
        if any(f.startswith("events.out.tfevents") for f in files):
            runs[os.path.relpath(folder, logdir)] = folder
    return runs


def format_alerts(alerts: list) -> str:
    """Alerts as text, one per line, worst first - for a terminal or a log."""
    if not alerts:
        return "no alerts"
    width = max(len(a.level) for a in alerts)
    return "\n".join(
        f"{a.level.upper():<{width}}  {a.run + '/' if a.run not in ('.', '') else ''}"
        f"{a.tag or '(run)'}  [{a.rule}]  {a.message}" for a in alerts)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Alerts on a TensorBoard log directory.")
    ap.add_argument("logdir")
    ap.add_argument("--serve", action="store_true",
                    help="serve the charts page instead of exiting")
    ap.add_argument("--port", type=int, default=8892)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args(argv)

    board = ScalarBoard()
    info = board.open(args.logdir)
    print(f"{len(info['runs'])} run(s), {len(info['tags'])} series under "
          f"{board.logdir}")
    print(format_alerts(board.check()))
    if args.serve:
        from .server import serve_scalars
        server = serve_scalars(board, host=args.host, port=args.port)
        print(f"charts: http://{args.host}:{server.server_address[1]}/scalars",
              flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            server.shutdown()
    worst = board.check()
    return 2 if any(a.level == "critical" for a in worst) else 0


if __name__ == "__main__":
    raise SystemExit(main())
