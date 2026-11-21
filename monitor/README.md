# monitor - measure chosen modules while a model trains

A small web server that sits beside a PyTorch training loop. You open a page,
pick modules on the model's graph, ask for N iterations, and get for each
picked module:

- **inputs and outputs**: mean, median, mode, min, max, shape, dtype and a
  histogram, for every tensor going in and coming out;
- **parameters**: the same for each of the module's own parameters and their
  gradients, plus count, mean, min and max over all its parameters;
- **time**: how long its forward takes, inclusive and self.

Nothing is measured until you ask. While idle, the loop pays one attribute
check per iteration.

![The page: Monitor menu open, six modules picked, timing table on the right](docs/monitor-overview.png)

```
Blocks/monitor/
    server.py     Monitor: the HTTP server and the per-iteration step()
    collector.py  hooks, per-module timing, per-iteration records, summaries
    stats.py      mean / median / mode / min / max / histogram of one tensor
    page.py       the page: Blocks.visualized's graph plus the monitor controls
    demo.py       a small training loop with the monitor attached
    _smoke_test.py
```

PyTorch only. No dependencies beyond `torch` and the standard library.

## Quick start

```python
from Blocks.monitor import Monitor

monitor = Monitor(model, port=8891, out_dir="runs/monitor")

for step, (x, y) in enumerate(loader):
    loss = loss_fn(model(x), y)          # the training forward, untouched
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    monitor.step(step, probe=lambda: model(x))
```

Open `http://127.0.0.1:8891/`.

`probe` is how the monitor gets the current inputs: a function the loop makes
each iteration that runs one forward on this iteration's batch. The server
never sees a batch. `step` calls `probe` only while a request is active.

To try it without a model of your own:

```bash
python -m Blocks.monitor.demo            # prints the address, runs until Ctrl-C
```

## How a request runs

For each of the N iterations after you press **Collect**:

1. the loop's own training step runs as always - no hooks, same compiled code,
   same gradients;
2. `monitor.step` puts hooks on the picked modules, calls `probe()` once under
   `torch.no_grad()`, and takes the hooks off again;
3. that iteration's numbers are stored.

After N iterations the request closes itself, the result is written to
`out_dir` as `r00001.json`, and the page shows it. So the forward runs twice
per iteration during a request and once otherwise.

Everything that touches the model happens in the loop's thread, inside
`step`. The server thread only reads and writes the request slot.

## The page

It is the architecture graph from `Blocks.visualized` - same boxes, expand and
collapse by depth, pan, zoom and details panel - with a **Monitor** menu added.

- **Select: on**, then click boxes to pick modules, at any depth. Or add every
  module at a given depth. Picked modules get a dashed green outline.
- Set the number of iterations, choose statistics and/or timing, **Collect**.
- With nothing selected, the details panel lists the measured modules sorted
  by self time. Click a row to jump to that module.
- Click a module to see its tables. A module called more than once per forward
  (shared weights) has one card per call, each with its own numbers.
- After a timing run each card carries its self time, coloured from green
  (cheap) to red (the most expensive measured).
- The Result list reopens any earlier request.

![One module selected: time, then input, output and parameter tables with histograms](docs/monitor-module.png)

A link can open the page in a given state:
`#panel=1&sel=encoder.0,middle&depth=2&result=r00001&node=encoder.0`.

### The graph

The graph needs one traced forward pass. Give the monitor a traced graph, or a
function that returns one; a function is called once, in the loop's thread,
the first time the page is opened:

```python
from Blocks.visualized.tracer_torch import trace_model

monitor = Monitor(model, port=8891,
                  graph=lambda: trace_model(model, example_input))
```

Without `graph` the page draws the module tree (nesting, classes, parameter
counts) with no connections or shapes. Everything else works the same.

## What the numbers mean

| Number | Definition |
| --- | --- |
| mean, std, min, max | over the finite values; non-finite ones are counted separately |
| median | midpoint of the two middle values. Exact up to 1,000,000 elements; above that, taken on a fixed random sample and marked as such |
| mode | centre of the fullest of 64 equal-width histogram bins between min and max. Floating-point values rarely repeat, so the most frequent exact value would say nothing |
| inclusive time | wall time of the module's forward, device synchronised at entry and exit, with the monitor's own statistics work subtracted |
| self time | inclusive time minus the inclusive time of the *picked* modules called directly inside it. Children you did not pick stay in the parent's self time |

Each table shows the mean over the N iterations; hover a number for its range
across them. The histogram is the last iteration's.

## Limits

- **A request costs a second forward per iteration**, plus memory for one
  forward without gradients. The result reports the probe's time next to the
  training step's.
- **Timings rank modules; they are not training-step times.** They come from
  a forward that is synchronised at every picked module, runs without
  gradients, and excludes the backward pass.
- **Hooks do not fire inside compiled code.** If part of the model is
  `torch.compile`d, call the uncompiled version inside `probe`. A module that
  was not reached reports "not called".
- **Only `forward` is measured.** A module used through another method shows
  in the tree but reports nothing.
- **The probe is a real forward in the model's current mode.** Anything a
  forward changes is changed once more per iteration during a request:
  BatchNorm running statistics, random-number draws (dropout, noise), cached
  attributes. Training is not bit-identical while a request runs.
- **One process.** In a distributed run, start the monitor on one rank
  (`Monitor(..., start=False)` on the others); numbers are that rank's.
- **No login.** The server binds to `127.0.0.1` by default. Reach it through
  an SSH port forward. If you put it behind a public address, anyone with the
  link can make the run do extra forwards.

## API

```python
Monitor(model, *, port=8891, host="127.0.0.1", out_dir=None,
        graph=None, title=None, start=True)
```

| Member | |
| --- | --- |
| `step(step, probe)` | call once per iteration, after the training step |
| `start()` / `stop()` | start or stop serving |
| `url`, `port` | where it is listening (`port=0` picks a free port) |
| `request(paths, n, stats=True, timing=True)` | queue a request from code |
| `status()`, `result(id)` | what the page reads |

A failing `probe` does not stop the loop: the request ends, the traceback is
printed to stderr and shown on the page.

HTTP: `GET /`, `GET /api/status`, `GET /api/tree`, `GET /api/results/<id>`,
`POST /api/request` with `{"paths": [...], "n": 10, "stats": true, "timing": true}`.
The page uses relative URLs, so it works behind a reverse proxy under a prefix.

`Collector` can also be used alone, without a server:

```python
from Blocks.monitor import Collector

with Collector(model, ["encoder", "encoder.0"]) as c:
    with torch.no_grad():
        model(x)
data = c.record()
```

## Checks

```bash
python -m Blocks.monitor._smoke_test
```

46 checks with known answers: statistics on constructed tensors, hooks removed
afterwards, tuple/keyword/dict inputs and outputs, a module called twice,
timing of modules with a known sleep, exactly N probe forwards per request and
none while idle, rejected requests, a failing probe, and results surviving a
restart.
