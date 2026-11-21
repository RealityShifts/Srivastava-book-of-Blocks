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

import json
import os
import re
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from .collector import Collector, summarize_iterations

MAX_ITERATIONS = 1000
MAX_BODY = 1 << 20
_RESULT = re.compile(r"^r\d{5}$")


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
    distributed run that should not serve.
    """

    def __init__(self, model: nn.Module, *, port: int = 8891,
                 host: str = "127.0.0.1", out_dir: Optional[str] = None,
                 graph: Any = None, title: Optional[str] = None,
                 start: bool = True) -> None:
        self.model = model
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
                           timing=req["timing"]) as collector:
                with torch.no_grad():
                    probe()
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
        # Time between two step() calls with the previous probe taken out:
        # what one training iteration costs without the monitor.
        if interval is not None and len(req["steps"]) > 1:
            req["train_seconds"].append(interval)
        if len(req["iterations"]) >= req["n"]:
            self._finish(req)

    def _finish(self, req: dict) -> None:
        iterations = req.pop("iterations")
        header = {k: req[k] for k in ("id", "paths", "n", "stats", "timing",
                                      "steps", "error", "requested_at")}
        header["finished_at"] = time.time()
        header["done"] = len(iterations)
        for key in ("probe_seconds", "train_seconds"):
            vals = req[key]
            header[key] = sum(vals) / len(vals) if vals else None
        result = {"header": header,
                  "summary": summarize_iterations(iterations),
                  "iterations": iterations}
        if self.out_dir:
            path = os.path.join(self.out_dir, req["id"] + ".json")
            tmp = path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(result, fh)
            os.replace(tmp, path)
        else:
            self._payloads[req["id"]] = result
        with self._lock:
            self._results[req["id"]] = header
            self._active = None

    # ------------------------------------------------------------ server side
    def request(self, paths: list, n: int, *, stats: bool = True,
                timing: bool = True) -> dict:
        """Queue a request (what ``POST api/request`` does). Raises
        ``ValueError`` on anything invalid."""
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
                   "steps": [], "iterations": [], "probe_seconds": [],
                   "train_seconds": [], "error": None,
                   "requested_at": time.time()}
            self._active = req
            self._last_call = None
            self._busy = True
        return {"id": req["id"]}

    def status(self) -> dict:
        with self._lock:
            active = self._active
            return {
                "title": self.title,
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

    def page(self) -> str:
        from .page import build_page
        if (self._graph is None and callable(self._graph_source)
                and not self._graph_error):
            with self._lock:               # traced by the loop's next step()
                self._want_graph = True
                self._busy = True
        return build_page(self._graph, self._tree, self.title)


def _handler(monitor: Monitor):
    class Handler(BaseHTTPRequestHandler):
        server_version = "BlocksMonitor/1"

        def log_message(self, *args) -> None:       # keep the training log clean
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: Any) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            try:
                if path == "/":
                    self._send(200, monitor.page().encode(),
                               "text/html; charset=utf-8")
                elif path == "/api/status":
                    self._json(200, monitor.status())
                elif path == "/api/tree":
                    self._json(200, monitor._tree)
                elif path.startswith("/api/results/"):
                    result = monitor.result(path[len("/api/results/"):])
                    if result is None:
                        self._json(404, {"error": "no such result"})
                    else:
                        self._json(200, result)
                else:
                    self._json(404, {"error": "not found"})
            except Exception:                               # noqa: BLE001
                self._json(500, {"error": traceback.format_exc(limit=4)})

        def do_POST(self) -> None:
            if self.path.split("?", 1)[0] != "/api/request":
                return self._json(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if not 0 < length <= MAX_BODY:
                    raise ValueError("a JSON body is required")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("a JSON object is required")
                self._json(200, monitor.request(
                    body.get("paths"), body.get("n"),
                    stats=bool(body.get("stats", True)),
                    timing=bool(body.get("timing", True))))
            except RuntimeError as exc:
                self._json(409, {"error": str(exc)})
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
            except Exception:                               # noqa: BLE001
                self._json(500, {"error": traceback.format_exc(limit=4)})

    return Handler
