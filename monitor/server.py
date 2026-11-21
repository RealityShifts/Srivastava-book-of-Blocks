"""The monitor: a small HTTP server beside a training loop.

The loop owns the model and the batches; the server owns nothing but a
request slot and a result folder. They meet in :meth:`Monitor.step`, which the
loop calls once per iteration:

* nothing requested: one attribute read, and it returns;
* a request active: hooks go on the selected modules, the loop's ``probe``
  runs one extra forward under ``no_grad``, the hooks come off, and that
  iteration's numbers are stored. After the requested number of iterations
  the request closes and is written to ``out_dir`` as JSON.

Everything that touches the model runs in the loop's thread, inside
``step``. The server thread only reads and writes the request slot under a
lock, so no forward pass ever races the training step.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import sys
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from .collector import Collector, summarize_iterations
from .scalars import Alert, ScalarBoard

MAX_ITERATIONS = 1000
#: Magnitudes (largest absolute value in the tensor) at which the page marks
#: it. Too big: ``warn`` in orange, ``danger`` in red. Too small: below
#: ``tiny`` in blue - a collapsed activation, a weight that has shrunk away, a
#: vanishing gradient. All-zero and constant tensors are blue too; non-finite
#: values are always red. Rules of thumb, not laws - override with
#: ``Monitor(thresholds=...)``.
THRESHOLDS = {
    "activation": {"warn": 1e2, "danger": 1e4, "tiny": 1e-6},
    "parameter": {"warn": 1e1, "danger": 1e2, "tiny": 1e-8},
    "gradient": {"warn": 1e0, "danger": 1e2, "tiny": 1e-8},
}
MAX_BODY = 1 << 20
_RESULT = re.compile(r"^r\d{5}$")


def clean(obj: Any) -> Any:
    """``obj`` with every non-finite float replaced by ``None``. Python's
    json writes NaN and Infinity as bare words, which no browser parses - and
    a model with a NaN in it is exactly when the page has to load."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    return obj


CSV_COLUMNS = ("result", "module", "class", "call", "kind", "tensor", "shape",
               "dtype", "numel", "nonfinite", "mean", "median", "mode", "min",
               "max", "std", "lowest", "highest", "time_self_s",
               "time_inclusive_s", "share_of_forward")


def result_csv(result: dict) -> str:
    """One finished request as a flat table, a row per tensor and per timing.

    ``kind`` is ``time``, ``input``, ``output``, ``parameter`` or
    ``gradient``. ``mean`` .. ``std`` are each statistic's mean over the
    request's iterations; ``lowest`` / ``highest`` are the smallest min and
    the largest max any iteration saw. A parameter appears once per picked
    module that contains it.
    """
    header, summary = result["header"], result["summary"]
    forward = header.get("forward_seconds")
    out = io.StringIO()
    writer = csv.DictWriter(out, CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()

    def mid(row: dict, key: str):
        return row[key]["mean"] if row.get(key) else ""

    def tensor(base: dict, kind: str, name: str, row: dict) -> None:
        writer.writerow({
            **base, "kind": kind, "tensor": name,
            "shape": "x".join(map(str, row["shape"])), "dtype": row["dtype"],
            "numel": row["numel"], "nonfinite": row["nonfinite"],
            **{k: mid(row, k) for k in ("mean", "median", "mode", "min",
                                        "max", "std")},
            "lowest": row["min"]["min"] if row.get("min") else "",
            "highest": row["max"]["max"] if row.get("max") else ""})

    params = summary.get("params", {})
    for path, module in summary["modules"].items():
        base = {"result": header["id"], "module": path or "<root>",
                "class": module["cls"]}
        for k, call in enumerate(module["calls"]):
            at = {**base, "call": k}
            if call["time"]:
                inclusive = call["time"]["inclusive"]["mean"]
                writer.writerow({
                    **at, "kind": "time",
                    "time_self_s": call["time"]["self"]["mean"],
                    "time_inclusive_s": inclusive,
                    "share_of_forward": inclusive / forward if forward else ""})
            for row in call["inputs"]:
                tensor(at, "input", row["name"], row)
            for row in call["outputs"]:
                tensor(at, "output", row["name"], row)
        for name in module.get("params", []):
            row = params.get(name)
            if row is None:
                continue
            tensor(base, "parameter", name, row)
            if row.get("grad"):
                tensor(base, "gradient", name, row["grad"])
    return out.getvalue()


def module_tree(model: nn.Module) -> list:
    """Every module of ``model`` as ``{path, name, cls, depth, parent,
    params, own_params}``, root first (path ``""``)."""
    rows = []
    for path, module in model.named_modules():
        own = sum(p.numel() for p in module._parameters.values()
                  if p is not None)
        rows.append({
            "path": path, "name": path.rsplit(".", 1)[-1] or "<root>",
            "cls": type(module).__name__,
            "depth": path.count(".") + 1 if path else 0,
            "parent": path.rsplit(".", 1)[0] if "." in path else
                      (None if not path else ""),
            "params": sum(p.numel() for p in module.parameters()),
            "own_params": own})
    return rows


class Monitor:
    """Serve a page that measures chosen modules of ``model`` on request.

    ``port=0`` picks a free port (see :attr:`port`). ``out_dir`` receives one
    ``r<NNNNN>.json`` per finished request; ``None`` keeps results in memory
    only. ``graph`` is a traced :class:`Blocks.visualized.Graph` of ``model``
    or a function returning one; a function is called once, in the loop's
    thread, the first time the page is opened - without it the page draws the
    module tree with no connections. ``start=False`` builds an inert monitor
    (:meth:`step` does nothing) until :meth:`start` - for the ranks of a
    distributed run that should not serve. ``thresholds`` overrides entries
    of :data:`THRESHOLDS`, the magnitudes the page colours orange and red,
    e.g. ``{"activation": {"warn": 50}}``.

    Scalars and alerts (the ``scalars`` page): ``logdir`` opens a TensorBoard
    log directory at start (one can also be typed into the page);
    ``scalar_limits`` overrides :data:`Blocks.monitor.scalars.DEFAULTS`;
    ``on_alert(alert)`` is called for every new alert - by default it is
    printed to stderr; ``logdir_root`` restricts which directories the page
    may open. :meth:`log` feeds values from the loop without TensorBoard.
    """

    def __init__(self, model: nn.Module, *, port: int = 8891,
                 host: str = "127.0.0.1", out_dir: Optional[str] = None,
                 graph: Any = None, title: Optional[str] = None,
                 thresholds: Optional[dict] = None,
                 logdir: Optional[str] = None,
                 scalar_limits: Optional[dict] = None,
                 on_alert: Optional[Callable[[Alert], None]] = None,
                 logdir_root: Optional[str] = None,
                 start: bool = True) -> None:
        self.model = model
        self.scalars = ScalarBoard(limits=scalar_limits, root=logdir_root,
                                   on_alert=on_alert or print_alert)
        if logdir:
            self.scalars.open(logdir)
        self.thresholds = {k: dict(v) for k, v in THRESHOLDS.items()}
        for kind, limits in (thresholds or {}).items():
            if kind not in self.thresholds:
                raise ValueError(f"thresholds: unknown kind {kind!r}; one of "
                                 f"{sorted(self.thresholds)}")
            unknown = set(limits) - set(self.thresholds[kind])
            if unknown:
                raise ValueError(f"thresholds[{kind!r}]: unknown "
                                 f"{sorted(unknown)}")
            self.thresholds[kind].update(
                {k: float(v) for k, v in limits.items()})
        self.host, self._port = host, port
        self.out_dir = out_dir
        self.title = title or type(model).__name__
        self._graph_source = graph
        self._graph = None if callable(graph) else graph
        self._graph_error: Optional[str] = None
        self._lock = threading.Lock()
        # Read without the lock on every iteration; written under it.
        self._busy = False
        self._want_graph = False
        self._active: Optional[dict] = None
        self._results: dict = {}          # id -> summary header (no payload)
        self._payloads: dict = {}         # id -> full result, when no out_dir
        self._last_step: Optional[int] = None
        self._last_call = None
        self._server: Optional[ThreadingHTTPServer] = None
        self._tree = module_tree(model)
        self._paths = {row["path"] for row in self._tree}
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            for name in sorted(os.listdir(out_dir)):
                rid = name[:-5]
                if name.endswith(".json") and _RESULT.match(rid):
                    try:
                        with open(os.path.join(out_dir, name)) as fh:
                            self._results[rid] = json.load(fh)["header"]
                    except (OSError, ValueError, KeyError):
                        pass
        if start:
            self.start()

    # ---------------------------------------------------------------- serving
    @property
    def port(self) -> int:
        return self._server.server_address[1] if self._server else self._port

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def start(self) -> "Monitor":
        if self._server is None:
            self._server = ThreadingHTTPServer((self.host, self._port),
                                               _handler(self))
            self._server.daemon_threads = True
            threading.Thread(target=self._server.serve_forever,
                             name="blocks-monitor", daemon=True).start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    # ------------------------------------------------------------- scalars
    def log(self, step: int, values: dict) -> None:
        """Give the alert rules numbers straight from the loop:
        ``monitor.log(step, {"loss": loss.item(), "grad_norm": g})``. They
        appear on the scalars page as run ``live``. Not needed when the same
        values are already in an opened TensorBoard log directory."""
        self.scalars.log(step, values)

    def alerts(self) -> list:
        """The current alerts, worst first (re-reading the log directory
        if it is due). Each is a :class:`Blocks.monitor.scalars.Alert`."""
        self.scalars.refresh_if_due()
        return self.scalars.check()

    # ------------------------------------------------------------- the loop
    def step(self, step: int, probe: Callable[[], Any]) -> None:
        """Call once per training iteration, after the training step.

        ``probe`` runs one forward of the model on the current batch, the
        way training does. It is called only while a request is active.
        """
        if not self._busy:
            return
        now = time.perf_counter()
        interval = None if self._last_call is None else now - self._last_call
        self._last_step = step
        if self._want_graph:
            self._trace()
        active = self._active
        if active is not None:
            self._collect(active, step, probe, interval)
        self._last_call = time.perf_counter()
        with self._lock:
            self._busy = self._active is not None or self._want_graph
            if not self._busy:
                self._last_call = None

    def measure(self, probe: Callable[[], Any], paths: Optional[list] = None,
                *, n: int = 1, step: Optional[int] = None, stats: bool = True,
                timing: bool = True, deep: bool = True) -> dict:
        """Measure now, from code, and return the result.

        Runs ``probe`` ``n`` times (once by default) with hooks on ``paths``
        - every module of the model when ``paths`` is ``None`` - without
        waiting for the page or for :meth:`step`. ``probe`` can run anything:
        the current batch, or an input of your own::

            result = monitor.measure(lambda: model(my_input))

        The result is stored like any other, so it also appears on the page
        and can be downloaded. Call it from the thread that owns the model.
        Raises ``RuntimeError`` if a request from the page is still
        collecting, and if ``probe`` fails.
        """
        rid = self.request(sorted(self._paths) if paths is None else paths,
                           n, stats=stats, timing=timing, deep=deep)["id"]
        req = self._active
        base = step if step is not None else (self._last_step or 0)
        for i in range(n):
            if self._active is not req:           # the probe failed
                break
            self._collect(req, base + i, probe, None)
        with self._lock:
            self._busy = self._active is not None or self._want_graph
        result = self.result(rid)
        if result["header"]["error"]:
            raise RuntimeError(f"measure: the probe failed\n"
                               f"{result['header']['error']}")
        return result

    def _trace(self) -> None:
        try:
            self._graph = self._graph_source()
        except Exception:                                   # noqa: BLE001
            self._graph_error = traceback.format_exc(limit=6)
            print("Blocks.monitor: tracing the graph FAILED; the page shows "
                  "the plain module tree\n" + self._graph_error,
                  file=sys.stderr, flush=True)
        self._want_graph = False

    def _collect(self, req: dict, step: int, probe: Callable,
                 interval: Optional[float]) -> None:
        t0 = time.perf_counter()
        try:
            with Collector(self.model, req["paths"], stats=req["stats"],
                           timing=req["timing"],
                           deep=req["deep"]) as collector:
                collector.sync()
                f0 = time.perf_counter()
                with torch.no_grad():
                    probe()
                collector.sync()
                # The whole probe forward, the monitor's own work taken out:
                # the total the per-module times are shares of.
                forward = time.perf_counter() - f0 - collector.overhead
            record = collector.record()
        except Exception:                                   # noqa: BLE001
            # Monitoring must not take the training run down with it, and
            # must not fail quietly either: the request ends, the page shows
            # the traceback, and it goes to stderr.
            req["error"] = traceback.format_exc(limit=8)
            print(f"Blocks.monitor: request {req['id']} FAILED at step "
                  f"{step}\n{req['error']}", file=sys.stderr, flush=True)
            self._finish(req)
            return
        req["iterations"].append(record)
        req["steps"].append(step)
        req["probe_seconds"].append(time.perf_counter() - t0)
        req["forward_seconds"].append(forward)
        # Time between two step() calls with the previous probe taken out:
        # what one training iteration costs without the monitor.
        if interval is not None and len(req["steps"]) > 1:
            req["train_seconds"].append(interval)
        if len(req["iterations"]) >= req["n"]:
            self._finish(req)

    def _finish(self, req: dict) -> None:
        iterations = req.pop("iterations")
        header = {k: req[k] for k in ("id", "paths", "n", "stats", "timing",
                                      "deep", "steps", "error",
                                      "requested_at")}
        header["finished_at"] = time.time()
        header["done"] = len(iterations)
        for key in ("probe_seconds", "train_seconds", "forward_seconds"):
            vals = req[key]
            header[key] = sum(vals) / len(vals) if vals else None
        result = clean({"header": header,
                        "summary": summarize_iterations(iterations),
                        "iterations": iterations})
        header = result["header"]
        if self.out_dir:
            path = os.path.join(self.out_dir, req["id"] + ".json")
            tmp = path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(result, fh, allow_nan=False)
            os.replace(tmp, path)
        else:
            self._payloads[req["id"]] = result
        with self._lock:
            self._results[req["id"]] = header
            self._active = None

    # ------------------------------------------------------------ server side
    def request(self, paths: list, n: int, *, stats: bool = True,
                timing: bool = True, deep: bool = True) -> dict:
        """Queue a request (what ``POST api/request`` does). Raises
        ``ValueError`` on anything invalid. ``deep``: list every parameter
        inside each module, not only the ones it owns directly."""
        if not isinstance(paths, list) or not paths or not all(
                isinstance(p, str) for p in paths):
            raise ValueError("paths: a non-empty list of module paths")
        unknown = sorted(set(paths) - self._paths)
        if unknown:
            raise ValueError(f"unknown module path(s): {unknown[:5]}")
        if not isinstance(n, int) or isinstance(n, bool) or not (
                1 <= n <= MAX_ITERATIONS):
            raise ValueError(f"n: an integer from 1 to {MAX_ITERATIONS}")
        if not (stats or timing):
            raise ValueError("ask for statistics, timing or both")
        with self._lock:
            if self._active is not None:
                raise RuntimeError(
                    f"request {self._active['id']} is still collecting")
            last = max((int(r[1:]) for r in self._results), default=0)
            req = {"id": f"r{last + 1:05d}", "paths": list(dict.fromkeys(paths)),
                   "n": n, "stats": bool(stats), "timing": bool(timing),
                   "deep": bool(deep), "steps": [], "forward_seconds": [], "iterations": [], "probe_seconds": [],
                   "train_seconds": [], "error": None,
                   "requested_at": time.time()}
            self._active = req
            self._last_call = None
            self._busy = True
        return {"id": req["id"]}

    def status(self) -> dict:
        try:
            self.scalars.refresh_if_due()
            found = self.scalars.check()
            alert_counts = {lv: sum(1 for a in found if a.level == lv)
                            for lv in ("critical", "warning", "info")}
        except Exception:                                   # noqa: BLE001
            alert_counts = None
        with self._lock:
            active = self._active
            return {
                "title": self.title,
                "alerts": alert_counts,
                "thresholds": self.thresholds,
                "step": self._last_step,
                "active": None if active is None else {
                    "id": active["id"], "n": active["n"],
                    "done": len(active["steps"])},
                "results": sorted(self._results.values(),
                                  key=lambda h: h["id"], reverse=True),
                "graph": ("traced" if self._graph is not None else
                          "failed" if self._graph_error else
                          "pending" if self._want_graph else
                          "tree" if self._graph_source is None else "untraced"),
                "graph_error": self._graph_error}

    def result(self, rid: str) -> Optional[dict]:
        if not _RESULT.match(rid) or rid not in self._results:
            return None
        if rid in self._payloads:
            return self._payloads[rid]
        with open(os.path.join(self.out_dir, rid + ".json")) as fh:
            return json.load(fh)

    def export(self, rid: str, path: str) -> str:
        """Write result ``rid`` to ``path``: a ``.csv`` path gets the flat
        table (:func:`result_csv`), anything else the full JSON. Returns the
        path. For results kept in memory only (no ``out_dir``) this is how to
        keep one."""
        result = self.result(rid)
        if result is None:
            raise KeyError(f"no result {rid!r}")
        text = (result_csv(result) if path.endswith(".csv") else
                json.dumps(clean(result), allow_nan=False, indent=1))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def page(self) -> str:
        from .page import build_page
        if (self._graph is None and callable(self._graph_source)
                and not self._graph_error):
            with self._lock:               # traced by the loop's next step()
                self._want_graph = True
                self._busy = True
        return build_page(self._graph, self._tree, self.title)


def print_alert(alert: Alert) -> None:
    """The default ``on_alert``: one line on stderr per new alert."""
    where = (alert.run + "/" if alert.run not in (".", "") else "") + (
        alert.tag or "(run)")
    print(f"Blocks.monitor ALERT {alert.level.upper()} {where} "
          f"[{alert.rule}] {alert.message}", file=sys.stderr, flush=True)


class _Base(BaseHTTPRequestHandler):
    """What every handler here shares: quiet logging, sending, and the
    scalars routes (``board`` is set by the subclass)."""
    server_version = "BlocksMonitor/1"
    board: ScalarBoard

    def log_message(self, *args) -> None:           # keep the training log clean
        pass

    def _send(self, code: int, body: bytes, ctype: str,
              filename: Optional[str] = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        if filename:
            self.send_header("Content-Disposition",
                             f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: Any) -> None:
        self._send(code, json.dumps(clean(obj), allow_nan=False).encode(),
                   "application/json")

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_BODY:
            raise ValueError("a JSON body is required")
        body = json.loads(self.rfile.read(length))
        if not isinstance(body, dict):
            raise ValueError("a JSON object is required")
        return body

    def scalars_get(self, path: str) -> bool:
        """Serve a scalars GET route; False when ``path`` is not one."""
        if path in ("/scalars", "/scalars/"):
            from .scalars_page import PAGE
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif path == "/api/scalars":
            try:
                self.board.refresh_if_due()
            except RuntimeError:
                pass                                # reported in the summary
            self._json(200, self.board.summary())
        elif path == "/api/scalars/series":
            query = urllib.parse.parse_qs(self.path.partition("?")[2])
            try:
                data = self.board.series(
                    query.get("run", [""])[0], query.get("tag", [""])[0],
                    max(50, min(20000, int(query.get("max", ["1500"])[0]))))
            except (KeyError, ValueError) as exc:
                self._json(404, {"error": str(exc)})
            else:
                self._json(200, data)
        else:
            return False
        return True

    def scalars_post(self, path: str) -> bool:
        if path != "/api/scalars/open":
            return False
        try:
            self._json(200, self.board.open(str(self._body().get("path", ""))))
        except (ValueError, RuntimeError) as exc:
            self._json(400, {"error": str(exc)})
        return True


def serve_scalars(board: ScalarBoard, *, host: str = "127.0.0.1",
                  port: int = 8892) -> ThreadingHTTPServer:
    """Serve only the scalars page for ``board`` - no model needed. Returns
    the running server (``.shutdown()`` stops it)."""
    class Handler(_Base):
        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            try:
                if path == "/":
                    self.send_response(302)
                    self.send_header("Location", "scalars")
                    self.end_headers()
                elif not self.scalars_get(path):
                    self._json(404, {"error": "not found"})
            except Exception:                               # noqa: BLE001
                self._json(500, {"error": traceback.format_exc(limit=4)})

        def do_POST(self) -> None:
            try:
                if not self.scalars_post(self.path.split("?", 1)[0]):
                    self._json(404, {"error": "not found"})
            except Exception:                               # noqa: BLE001
                self._json(500, {"error": traceback.format_exc(limit=4)})

    Handler.board = board
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="blocks-scalars",
                     daemon=True).start()
    return server


def _handler(monitor: Monitor):
    class Handler(_Base):
        board = monitor.scalars

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            try:
                if self.scalars_get(path):
                    pass
                elif path == "/":
                    self._send(200, monitor.page().encode(),
                               "text/html; charset=utf-8")
                elif path == "/api/status":
                    self._json(200, monitor.status())
                elif path == "/api/tree":
                    self._json(200, monitor._tree)
                elif path.startswith("/api/results/"):
                    # <id> for the page; <id>.json / <id>.csv as downloads.
                    rid, _, ext = path[len("/api/results/"):].partition(".")
                    result = (monitor.result(rid)
                              if ext in ("", "json", "csv") else None)
                    name = re.sub(r"[^\w.-]+", "_", monitor.title)
                    if result is None:
                        self._json(404, {"error": "no such result"})
                    elif ext == "csv":
                        self._send(200, result_csv(result).encode(),
                                   "text/csv; charset=utf-8",
                                   f"{name}_{rid}.csv")
                    elif ext == "json":
                        self._send(
                            200, json.dumps(clean(result), allow_nan=False,
                                            indent=1).encode(),
                            "application/json", f"{name}_{rid}.json")
                    else:
                        self._json(200, result)
                else:
                    self._json(404, {"error": "not found"})
            except Exception:                               # noqa: BLE001
                self._json(500, {"error": traceback.format_exc(limit=4)})

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            try:
                if self.scalars_post(path):
                    return None
            except Exception:                               # noqa: BLE001
                return self._json(500, {"error": traceback.format_exc(limit=4)})
            if path != "/api/request":
                return self._json(404, {"error": "not found"})
            try:
                body = self._body()
                self._json(200, monitor.request(
                    body.get("paths"), body.get("n"),
                    stats=bool(body.get("stats", True)),
                    timing=bool(body.get("timing", True)),
                    deep=bool(body.get("deep", True))))
            except RuntimeError as exc:
                self._json(409, {"error": str(exc)})
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
            except Exception:                               # noqa: BLE001
                self._json(500, {"error": traceback.format_exc(limit=4)})

    return Handler
