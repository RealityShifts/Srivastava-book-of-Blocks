"""Measure chosen modules of a model while it trains, from a web page.

    from Blocks.monitor import Monitor

    monitor = Monitor(model, port=8891, out_dir="runs/monitor")
    for step, (x, y) in enumerate(loader):
        ...                                   # the training step, untouched
        monitor.step(step, probe=lambda: model(x))

Open ``http://127.0.0.1:8891/``, pick modules on the graph, ask for N
iterations. While a request runs, ``step`` calls ``probe`` once more per
iteration with hooks on the picked modules; otherwise it does nothing.
See ``README.md`` beside this file, and ``python -m Blocks.monitor.demo``.
"""

from .collector import Collector, named_tensors, summarize_iterations
from .scalars import Alert, ScalarBoard, format_alerts
from .server import Monitor, module_tree, result_csv, serve_scalars
from .stats import tensor_stats

__all__ = ["Monitor", "Collector", "tensor_stats", "summarize_iterations",
           "named_tensors", "module_tree", "result_csv", "ScalarBoard",
           "Alert", "format_alerts", "serve_scalars"]
