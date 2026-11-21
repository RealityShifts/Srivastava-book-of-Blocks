"""Checks for the monitor, each with a known answer.

    python -m Blocks.monitor._smoke_test

CPU is enough. Exits non-zero on the first failure.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import tempfile
import time
import urllib.error
import urllib.request

import torch
import torch.nn as nn

from ..visualized.tracer_torch import trace_model
from .collector import Collector, summarize_iterations
from .page import build_page
from .server import Monitor, module_tree
from .stats import tensor_stats


def check(ok: bool, what: str) -> None:
    print(("ok    " if ok else "FAIL  ") + what, flush=True)
    if not ok:
        raise SystemExit(1)


def close(a: float, b: float, tol: float = 1e-5) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


class Slow(nn.Module):
    """A module whose forward takes a known time."""

    def __init__(self, seconds: float) -> None:
        super().__init__()
        self.seconds = seconds
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time.sleep(self.seconds)
        return self.lin(x)


class Pair(nn.Module):
    """Takes a tuple and a keyword, returns a dict - the awkward signatures."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((3,), 2.0))

    def forward(self, pair, *, bias):
        a, b = pair
        return {"sum": a + b + bias, "prod": a * b * self.scale[0]}


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner = nn.Sequential(Slow(0.02), nn.ReLU(), Slow(0.03))
        self.shared = nn.Linear(4, 4)
        self.pair = Pair()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time.sleep(0.01)
        h = self.inner(x)
        h = self.shared(self.shared(h))            # one module, two calls
        out = self.pair((h, x), bias=torch.ones(4))
        return out["sum"]


def get(url: str):
    with urllib.request.urlopen(url, timeout=10) as r:
        body = r.read()
        return r.status, (json.loads(body) if "json" in r.headers.get(
            "Content-Type", "") else body.decode())


def post(url: str, obj) -> tuple:
    req = urllib.request.Request(
        url, data=json.dumps(obj).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def test_stats() -> None:
    s = tensor_stats(torch.tensor([1.0, 2.0, 2.0, 3.0, 10.0]), bins=9)
    check(close(s["mean"], 3.6) and s["min"] == 1.0 and s["max"] == 10.0,
          "stats: mean, min, max of [1, 2, 2, 3, 10]")
    check(s["median"] == 2.0, "stats: median of an odd count")
    # 9 bins over [1, 10] are 1 wide; the two 2s share bin [2, 3) -> 2.5.
    check(close(s["mode"], 2.5) and sum(s["hist"]) == 5,
          "stats: mode is the centre of the fullest bin, histogram sums to n")
    s = tensor_stats(torch.tensor([4.0, 1.0, 3.0, 2.0]))
    check(s["median"] == 2.5, "stats: median of an even count is the midpoint")
    s = tensor_stats(torch.full((2, 3), 7.0))
    check(s["mode"] == 7.0 and s["median"] == 7.0 and s["std"] == 0.0
          and s["hist"] == [6], "stats: a constant tensor")
    s = tensor_stats(torch.tensor([1.0, float("nan"), 3.0, float("inf")]))
    check(s["nonfinite"] == 2 and s["mean"] == 2.0,
          "stats: non-finite values counted and left out")
    check(s["head"] == [1.0, None, 3.0, None]
          and "NaN" not in json.dumps(s) and "Infinity" not in json.dumps(s),
          "stats: the value preview is valid JSON (no bare NaN)")
    s = tensor_stats(torch.zeros(0))
    check(s["mean"] is None and s["numel"] == 0, "stats: an empty tensor")
    s = tensor_stats(torch.tensor([True, False, True]))
    check(close(s["mean"], 2 / 3) and s["dtype"] == "bool",
          "stats: a bool tensor")
    big = torch.linspace(0, 1, 50_001)
    s = tensor_stats(big, median_cap=1000)
    check(s["median_sampled"] and abs(s["median"] - 0.5) < 0.06,
          "stats: above the cap the median is sampled and flagged")
    check(s["shape"] == [50_001] and close(s["mean"], 0.5, 1e-4),
          "stats: the mean still uses every value above the cap")


def test_collector() -> None:
    torch.manual_seed(0)
    net = Net()
    x = torch.randn(5, 4)
    paths = ["", "inner", "inner.0", "inner.2", "shared", "pair"]
    hooks_before = sum(len(m._forward_hooks) + len(m._forward_pre_hooks)
                       for m in net.modules())
    with Collector(net, paths) as c:
        with torch.no_grad():
            out = net(x)
    full = c.record()
    rec, table = full["modules"], full["params"]
    hooks_after = sum(len(m._forward_hooks) + len(m._forward_pre_hooks)
                      for m in net.modules())
    check(hooks_before == 0 and hooks_after == 0,
          "collector: no hooks left on the model")

    with torch.no_grad():
        h0 = net.inner[0].lin(x)
    r = rec["inner.0"]["calls"][0]
    check(r["inputs"][0]["shape"] == [5, 4]
          and close(r["inputs"][0]["mean"], float(x.mean()))
          and close(r["outputs"][0]["max"], float(h0.max()))
          and close(r["outputs"][0]["median"], float(h0.flatten().median())
                    if h0.numel() % 2 else r["outputs"][0]["median"]),
          "collector: input/output statistics match a direct computation")
    check(len(rec["shared"]["calls"]) == 2,
          "collector: a module called twice gives two call records")
    names = [t["name"] for t in rec["pair"]["calls"][0]["inputs"]]
    check(names == ["0.0", "0.1", "kw.bias"],
          f"collector: tuple and keyword inputs are named ({names})")
    outs = [t["name"] for t in rec["pair"]["calls"][0]["outputs"]]
    check(outs == ["sum", "prod"], f"collector: dict outputs are named ({outs})")
    check(close(rec[""]["calls"][0]["outputs"][0]["mean"], float(out.mean())),
          "collector: the root module (path '') is measured")

    check(rec["pair"]["params"] == ["pair.scale"]
          and table["pair.scale"]["shape"] == [3]
          and table["pair.scale"]["mean"] == 2.0
          and table["pair.scale"]["grad"] is None
          and table["pair.scale"]["head"] == [2.0, 2.0, 2.0],
          "collector: a parameter's row, its values, no gradient when none exists")
    check(rec["inner"]["own_params"] == 0
          and rec["inner"]["params"] == ["inner.0.lin.weight", "inner.0.lin.bias",
                                         "inner.2.lin.weight", "inner.2.lin.bias"]
          and rec["inner"]["whole"]["params"] == 40,
          "collector: a container owns no parameters and lists every one inside")
    check(len(rec[""]["params"]) == len(list(net.parameters())) == len(table),
          "collector: the root lists all parameters, each measured once")
    with Collector(net, ["inner", "shared"], deep=False) as c:
        with torch.no_grad():
            net(x)
    own = c.record()
    check(own["modules"]["inner"]["params"] == []
          and own["modules"]["shared"]["params"] == ["shared.weight", "shared.bias"],
          "collector: deep=False lists only directly owned parameters")
    net.zero_grad()
    net(x).sum().backward()
    with Collector(net, ["shared"]) as c:
        with torch.no_grad():
            net(x)
    g = c.record()["params"]["shared.weight"]["grad"]
    check(g is not None and close(g["mean"], float(net.shared.weight.grad.mean())),
          "collector: gradient statistics when a gradient is present")

    t = {p: rec[p]["calls"][0]["time"] for p in ("", "inner", "inner.0", "inner.2")}
    check(0.018 < t["inner.0"]["inclusive"] < 0.045
          and 0.028 < t["inner.2"]["inclusive"] < 0.06,
          f"timing: the 20 ms and 30 ms modules read "
          f"{t['inner.0']['inclusive'] * 1e3:.1f} and "
          f"{t['inner.2']['inclusive'] * 1e3:.1f} ms")
    kids = t["inner.0"]["inclusive"] + t["inner.2"]["inclusive"]
    check(close(t["inner"]["self"], t["inner"]["inclusive"] - kids, 1e-9)
          and t["inner"]["self"] < 0.005,
          f"timing: self = inclusive - selected children "
          f"(inner self {t['inner']['self'] * 1e3:.2f} ms)")
    check(0.008 < t[""]["self"] < 0.03,
          f"timing: the root's own 10 ms sleep is its self time "
          f"({t['']['self'] * 1e3:.1f} ms), statistics overhead excluded")

    with Collector(net, ["inner.0"], stats=False) as c:
        with torch.no_grad():
            net(x)
    only = c.record()["modules"]["inner.0"]
    check("params" not in only and only["calls"][0]["inputs"] == []
          and only["calls"][0]["time"] is not None,
          "collector: timing only collects no statistics")

    sick = Net()
    with torch.no_grad():
        sick.shared.bias[0] = float("nan")
        sick.shared.weight[0, 0] = float("inf")
    with Collector(sick, ["", "shared"]) as c:
        with torch.no_grad():
            sick(x)
    bad = c.record()
    w = bad["modules"][""]["whole"]
    check(w["nonfinite"] == 2 and w["mean"] is not None
          and math.isfinite(w["mean"])
          and bad["params"]["shared.bias"]["nonfinite"] == 1,
          "collector: NaN/Inf parameters are counted, the roll-up stays finite")
    from .server import clean
    text = json.dumps(clean({"it": [bad], "s": summarize_iterations([bad])}),
                      allow_nan=False)
    check("NaN" not in text and "Infinity" not in text,
          "server: a model with NaN in it still yields valid JSON")

    summ = summarize_iterations([full, full])
    row = summ["modules"]["inner.0"]["calls"][0]["inputs"][0]
    check(row["mean"]["mean"] == rec["inner.0"]["calls"][0]["inputs"][0]["mean"]
          and summ["modules"]["shared"]["calls"][1]["seen"] == 2,
          "summary: per-call rows across iterations")
    check(summ["params"]["pair.scale"]["mean"]["mean"] == 2.0
          and summ["modules"]["pair"]["params"] == ["pair.scale"],
          "summary: one shared parameter table, modules refer to it by name")


def test_server() -> None:
    torch.manual_seed(0)
    net = Net()
    for m in net.inner:
        if isinstance(m, Slow):
            m.seconds = 0.0
    x = torch.randn(5, 4)
    probes = [0]

    def probe():
        probes[0] += 1
        return net(x)

    with tempfile.TemporaryDirectory() as out:
        mon = Monitor(net, port=0, out_dir=out,
                      graph=lambda: trace_model(net, x))
        base = mon.url
        for step in range(5):
            mon.step(step, probe)
        check(probes[0] == 0, "server: no probe forward while idle")

        code, page = get(base)
        check(code == 200 and "mon-pop" in page and "__EXTRA" not in page,
              "server: the page is served with the monitor controls")
        code, st = get(base + "api/status")
        check(st["graph"] == "pending", "server: opening the page asks for a trace")
        mon.step(5, probe)
        code, st = get(base + "api/status")
        check(st["graph"] == "traced" and probes[0] == 0,
              "server: the graph is traced in the loop's thread, without a probe")
        code, page = get(base)
        check('"edges":[{' in page, "server: the traced graph is on the page")

        check(post(base + "api/request", {"paths": ["nope"], "n": 3})[0] == 400,
              "server: an unknown module is rejected")
        check(post(base + "api/request", {"paths": ["inner"], "n": 0})[0] == 400,
              "server: n = 0 is rejected")
        check(post(base + "api/request", {"paths": [], "n": 3})[0] == 400,
              "server: an empty selection is rejected")
        code, body = post(base + "api/request",
                          {"paths": ["", "inner", "inner.0", "shared"], "n": 4})
        check(code == 200 and body["id"] == "r00001", "server: a request is accepted")
        check(post(base + "api/request", {"paths": ["inner"], "n": 1})[0] == 409,
              "server: a second request while one is collecting is refused")

        for step in range(6, 16):
            mon.step(step, probe)
        check(probes[0] == 4, f"server: exactly n probe forwards ({probes[0]})")
        hooks = sum(len(m._forward_hooks) + len(m._forward_pre_hooks)
                    for m in net.modules())
        check(hooks == 0, "server: no hooks left after the request")
        code, st = get(base + "api/status")
        check(st["active"] is None and st["results"][0]["steps"] == [6, 7, 8, 9],
              "server: the request closed itself after n iterations")
        code, res = get(base + "api/results/r00001")
        check(code == 200 and len(res["iterations"]) == 4
              and set(res["summary"]["modules"]) == {"", "inner", "inner.0", "shared"}
              and len(res["summary"]["modules"]["shared"]["calls"]) == 2
              and "inner.2.lin.weight" in res["summary"]["params"],
              "server: the result has every module and both calls of the shared one")
        with torch.no_grad():
            want = float(net(x).mean())
        got = res["summary"]["modules"][""]["calls"][0]["outputs"][0]["mean"]["mean"]
        check(close(got, want), "server: result numbers match a direct forward")
        fwd = res["header"]["forward_seconds"]
        root = res["summary"]["modules"][""]["calls"][0]["time"]["inclusive"]["mean"]
        check(fwd is not None and root <= fwd * 1.05 and root > fwd * 0.5,
              f"server: total forward time reported ({fwd * 1e3:.2f} ms), "
              f"the root's inclusive time is within it ({root * 1e3:.2f} ms)")
        with urllib.request.urlopen(base + "api/results/r00001.csv") as r:
            disp, text = r.headers.get("Content-Disposition", ""), r.read().decode()
        rows = list(csv.DictReader(io.StringIO(text)))
        kinds = {r["kind"] for r in rows}
        root_out = [r for r in rows if r["module"] == "<root>" and r["kind"] == "output"]
        check("attachment" in disp and disp.endswith('_r00001.csv"')
              and {"time", "input", "output", "parameter"} <= kinds
              and close(float(root_out[0]["mean"]), want)
              and root_out[0]["shape"] == "5x4",
              f"export: the CSV download has {len(rows)} rows and the right numbers")
        with urllib.request.urlopen(base + "api/results/r00001.json") as r:
            dl = json.loads(r.read())
            check("attachment" in r.headers.get("Content-Disposition", "")
                  and dl["header"]["id"] == "r00001" and len(dl["iterations"]) == 4,
                  "export: the JSON download is the full result")
        saved = mon.export("r00001", os.path.join(out, "copy.csv"))
        check(open(saved).read() == text,
              "export: Monitor.export writes the same table to a path")
        try:
            code = get(base + "api/results/r00001.exe")[0]
        except urllib.error.HTTPError as err:
            code = err.code
        check(code == 404, "export: an unknown download format is refused")
        check(get(base + "api/results/r00001")[0] == 200
              and post(base + "api/nothing", {})[0] == 404,
              "server: only the listed endpoints exist")

        def boom():
            raise RuntimeError("probe failed on purpose")
        post(base + "api/request", {"paths": ["inner"], "n": 3})
        mon.step(20, boom)                       # must not raise
        code, st = get(base + "api/status")
        check(st["active"] is None and "on purpose" in st["results"][0]["error"],
              "server: a failing probe ends the request and reports the error")
        mon.stop()

        mon2 = Monitor(net, port=0, out_dir=out)
        code, st = get(mon2.url + "api/status")
        check([r["id"] for r in st["results"]] == ["r00002", "r00001"]
              and st["graph"] == "tree",
              "server: results survive a restart; no graph given -> tree")
        code, page = get(mon2.url)
        check(code == 200 and '"edges":[]' in page,
              "server: without a graph the page draws the module tree")
        mon2.stop()

    solo = Monitor(net, port=0, start=False)
    mine = torch.full((2, 4), 3.0)
    seen = [0]

    def on_mine():
        seen[0] += 1
        return net(mine)
    res = solo.measure(on_mine)
    every = {p for p, _ in net.named_modules()}
    check(seen[0] == 1 and set(res["summary"]["modules"]) == every
          and res["summary"]["modules"][""]["calls"][0]["inputs"][0]["mean"]["mean"] == 3.0
          and res["summary"]["modules"][""]["calls"][0]["inputs"][0]["shape"] == [2, 4]
          and solo.status()["results"][0]["id"] == res["header"]["id"]
          and solo.status()["active"] is None,
          "measure: one pass on a custom input, every module, without a server")
    two = solo.measure(on_mine, ["inner", "shared"], n=2, timing=False)
    check(seen[0] == 3 and two["header"]["done"] == 2
          and set(two["summary"]["modules"]) == {"inner", "shared"},
          "measure: chosen modules, n passes")
    try:
        solo.measure(lambda: 1 / 0, ["inner"])
        raised = False
    except RuntimeError:
        raised = True
    check(raised and solo.status()["active"] is None,
          "measure: a failing probe raises and leaves no request behind")
    hooks = sum(len(m._forward_hooks) + len(m._forward_pre_hooks)
                for m in net.modules())
    check(hooks == 0, "measure: no hooks left")

    custom = Monitor(net, port=0, start=False,
                     thresholds={"activation": {"warn": 5}})
    th = custom.status()["thresholds"]
    check(th["activation"] == {"warn": 5.0, "danger": 1e4, "tiny": 1e-6}
          and th["gradient"]["tiny"] == 1e-8,
          "thresholds: an override changes one limit and keeps the rest")
    try:
        Monitor(net, port=0, start=False, thresholds={"activations": {}})
        bad = False
    except ValueError:
        bad = True
    check(bad, "thresholds: an unknown kind is rejected")
    off = Monitor(net, port=0, start=False)
    off.step(0, probe)
    check(off._server is None and probes[0] == 4,
          "server: start=False is inert")
    tree = module_tree(net)
    check(tree[0]["path"] == "" and tree[0]["parent"] is None
          and {r["path"]: r["parent"] for r in tree}["inner.0.lin"] == "inner.0",
          "tree: paths and parents")
    check("mon-pop" in build_page(None, tree, "Net"), "page: builds from a tree")


def test_scalars() -> None:
    from .scalars import ScalarBoard, format_alerts
    from .scalars import main as scalars_main
    from .server import serve_scalars
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        print(f"SKIPPED  scalars: the TensorBoard checks need the "
              f"'tensorboard' package ({exc})", flush=True)
        return
    import numpy as np
    rng = np.random.default_rng(0)
    n = 400
    with tempfile.TemporaryDirectory() as root:
        logdir = os.path.join(root, "logs")
        w = SummaryWriter(os.path.join(logdir, "runA"))
        for i in range(n):
            t = i / n
            # healthy: a noisy decaying loss, a noisy stationary metric
            w.add_scalar("train/recon_loss", 1.0 * np.exp(-3 * t) + 0.02 * rng.normal(), i)
            w.add_scalar("val/accuracy", 0.8 + 0.01 * rng.normal(), i)
            # faults
            w.add_scalar("train/total_loss", (1 - t) if t < .5 else .5 + 3 * (t - .5), i)
            w.add_scalar("grad_norm/encoder", 1e-2 * 10 ** (-9 * t), i)
            w.add_scalar("act_max/decoder", 2.0 * 10 ** (3 * t), i)
            w.add_scalar("weight_norm/head", 5.0 + 0.01 * rng.normal()
                         + (400.0 if i == 300 else 0.0), i)
            w.add_scalar("train/lpips_loss", float("nan") if i == 250 else 0.3, i)
            w.add_scalar("misc/constant", 7.0, i)
            w.add_scalar("progress/steps_per_sec", 2.0 * 10 ** (3 * t), i)
        w.close()

        fired = []
        board = ScalarBoard(on_alert=fired.append)
        info = board.open(logdir)
        check(info["runs"] == ["runA"] and len(info["tags"]) == 9,
              "scalars: a TensorBoard directory is read, run and tags found")
        alerts = board.check()
        got = {(a.tag, a.rule): a for a in alerts}
        text = format_alerts(alerts)

        def has(tag, rule, level=None):
            a = got.get((tag, rule))
            return a is not None and (level is None or a.level == level)
        check(has("train/lpips_loss", "nonfinite", "critical")
              and got[("train/lpips_loss", "nonfinite")].step == 250,
              "alert: a NaN is critical and reported at its step")
        check(has("act_max/decoder", "growing", "critical"),
              "alert: an activation growing 1000x is critical")
        check(has("grad_norm/encoder", "vanishing"),
              "alert: a gradient norm shrinking away")
        check(has("train/total_loss", "diverging", "warning"),
              "alert: a loss that turned upward")
        check(has("weight_norm/head", "spike", "warning")
              and got[("weight_norm/head", "spike")].step == 300,
              "alert: a one-step spike, at its step")
        check(has("misc/constant", "flat", "info"), "alert: a constant value is info")
        clean_tags = {"train/recon_loss", "val/accuracy"}
        noisy = sorted({a.tag for a in alerts} & clean_tags)
        check(not noisy, f"alert: the healthy curves raise nothing {noisy}")
        check(not any(a.tag.startswith("progress/") for a in alerts),
              "alert: ignored tags are not judged")
        check(_rank_sorted(alerts) and "CRITICAL" in text,
              "alert: listed worst first")
        check(len(fired) == len(alerts) and not board.check() != alerts,
              "notify: on_alert got each alert once")
        board.check()
        check(len(fired) == len(alerts), "notify: nothing is announced twice")

        d = board.series("runA", "weight_norm/head", max_points=60)
        check(len(d["steps"]) <= 60 and max(d["values"]) > 400,
              f"series: downsampled to {len(d['steps'])} points, the spike kept")
        d = board.series("runA", "train/lpips_loss")
        check(d["bad"] == [250] and len(d["values"]) == n - 1,
              "series: the NaN is returned as a marked step, not drawn")

        w = SummaryWriter(os.path.join(logdir, "runA"))
        for i in range(n, n + 30):
            w.add_scalar("val/accuracy", float("inf") if i == n + 5 else 0.8, i)
        w.close()
        board.refresh()
        after = {(a.tag, a.rule) for a in board.check()}
        check(("val/accuracy", "nonfinite") in after
              and board.summary()["tags"][-1]["points"] >= n
              and len(fired) == len(alerts) + 1,
              "refresh: appended points are picked up and the new alert announced")

        board.log(0, {"live_grad": 1.0})
        for i in range(1, 60):
            board.log(i, {"live_grad": float("nan") if i == 40 else 1.0})
        check(("live_grad", "nonfinite") in {(a.tag, a.rule) for a in board.check()
                                             if a.run == "live"},
              "log: values from the loop are judged without TensorBoard")

        fenced = ScalarBoard(root=logdir)
        try:
            fenced.open(root)
            escaped = True
        except ValueError:
            escaped = False
        check(not escaped and fenced.open(os.path.join(logdir, "runA"))["tags"],
              "open: a root restricts which directories can be read")
        try:
            ScalarBoard().open(os.path.join(root, "nope"))
            missing = False
        except ValueError:
            missing = True
        check(missing, "open: a missing directory is an error, not an empty page")

        server = serve_scalars(board, port=0)
        base = f"http://127.0.0.1:{server.server_address[1]}/"
        code, page = get(base + "scalars")
        check(code == 200 and "api/scalars/open" in page, "page: served")
        code, summ = get(base + "api/scalars")
        check(code == 200 and summ["counts"]["critical"] >= 2
              and any(t["level"] == "critical" for t in summ["tags"]),
              "page: the summary lists tags with their alert level")
        code, ser = get(base + "api/scalars/series?run=runA&tag=act_max%2Fdecoder&max=100")
        check(code == 200 and ser["tag"] == "act_max/decoder" and len(ser["steps"]) <= 100,
              "page: a series by run and tag")
        code, opened = post(base + "api/scalars/open", {"path": logdir})
        bad_code, bad = post(base + "api/scalars/open", {"path": "/no/such/dir"})
        check(code == 200 and opened["logdir"] and bad_code == 400 and "error" in bad,
              "page: a directory typed in the page is opened; a wrong one is refused")
        server.shutdown()

        code = scalars_main([logdir])
        check(code == 2, "cli: exits 2 when something critical is found")

    net = Net()
    seen = []
    mon = Monitor(net, port=0, on_alert=seen.append)
    for i in range(60):
        mon.log(i, {"grad_norm": 1e-2 * 10 ** (-9 * i / 60)})
    found = mon.alerts()
    code, st = get(mon.url + "api/status")
    code2, page = get(mon.url + "scalars")
    check(any(a.rule == "vanishing" for a in found) and seen
          and st["alerts"]["warning"] + st["alerts"]["critical"] >= 1
          and code2 == 200,
          "monitor: log() feeds the alerts, the status counts them, the page is served")
    mon.stop()


def _rank_sorted(alerts: list) -> bool:
    order = {"critical": 0, "warning": 1, "info": 2}
    ranks = [order[a.level] for a in alerts]
    return ranks == sorted(ranks)


def main() -> int:
    test_stats()
    test_collector()
    test_server()
    test_scalars()
    print("all monitor checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
