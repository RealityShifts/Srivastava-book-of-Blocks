"""Render a traced :class:`~Blocks.visualized.tracer.Graph` to interactive HTML.

The output is a single self-contained file - no CDN, no build step - so it
opens straight from disk or inside a notebook iframe. Layout is a longest-path
layering (each node sits below every producer it depends on) with sibling
columns, drawn to SVG and driven by a small amount of vanilla JS for pan,
zoom, collapse and selection.
"""

from __future__ import annotations

import html
import json
import os
from typing import Optional

from .tracer import Graph, INPUT_NODE, IN_BASE


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  --bg:#0f1117; --panel:#171a23; --line:#262b38; --fg:#e6e9ef; --muted:#8b93a7;
  --accent:#6ea8fe; --skip:#f0883e; --edge:#3d4358; --hi:#ffd166; --err:#ff6b6b;
  --out:#5ec9a7;
}
@media (prefers-color-scheme: light) {
  :root { --bg:#f7f8fa; --panel:#fff; --line:#e2e5ec; --fg:#1b1f2a;
          --muted:#5f6880; --edge:#c3c9d6; }
}
* { box-sizing:border-box; }
html,body { margin:0; height:100%; overflow:hidden;
  font:13px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;
  background:var(--bg); color:var(--fg); }
#app { display:flex; height:100vh; }
#stage { flex:1 1 0; min-width:0; position:relative; overflow:hidden;
  cursor:grab; }
#stage.drag { cursor:grabbing; }
svg { width:100%; height:100%; display:block; }
#side { width:310px; flex:none; border-left:1px solid var(--line);
  background:var(--panel); overflow-y:auto; padding:16px; }
#side h1 { font-size:15px; margin:0 0 2px; }
#side .sub { color:var(--muted); font-size:11px; margin-bottom:14px;
  word-break:break-all; }
.stat { display:flex; justify-content:space-between; gap:10px;
  padding:5px 0; border-bottom:1px solid var(--line); font-size:12px; }
.stat span:first-child { color:var(--muted); flex:none; }
.stat span:last-child { font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  text-align:right; word-break:break-word; min-width:0; }
h2 { font-size:11px; text-transform:uppercase; letter-spacing:.08em;
  color:var(--muted); margin:18px 0 6px; }
#toolbar { position:absolute; top:12px; left:12px; display:flex; gap:6px;
  flex-wrap:wrap; align-items:center; z-index:5; }
#lvl { font-size:11px; color:var(--muted); padding-left:4px; }
button { font:inherit; font-size:12px; padding:5px 10px; border-radius:6px;
  border:1px solid var(--line); background:var(--panel); color:var(--fg);
  cursor:pointer; }
button:hover { border-color:var(--accent); }
#legend { position:absolute; bottom:12px; left:12px; font-size:11px;
  color:var(--muted); background:var(--panel); border:1px solid var(--line);
  border-radius:6px; padding:8px 10px; z-index:5; }
#legend div { display:flex; align-items:center; gap:6px; margin:2px 0; }
#legend i { width:16px; height:0; border-top:2px solid var(--edge);
  display:inline-block; }
.node rect { stroke-width:1.5px; }
.node { cursor:pointer; }
.node text { pointer-events:none; }
.nm { font-weight:600; font-size:12px; }
.nc { font-size:10.5px; fill:var(--muted); }
.ns { font-size:10px; font-family:ui-monospace,Menlo,monospace;
  fill:var(--muted); }
/* Only the frame's stroke and label are clickable, so a click on the empty
   space inside a group does not collapse it. */
.flab { font-size:10.5px; font-weight:600; letter-spacing:.02em;
  cursor:pointer; }
.frame rect { pointer-events:stroke; cursor:pointer; }
.node.sel rect, .outnode.sel rect { stroke:var(--hi); stroke-width:2.5px; }
.node.dim, .outnode.dim { opacity:.28; }
.edge { fill:none; stroke:var(--edge); stroke-width:1.6px; }
.edge.skip { stroke:var(--skip); stroke-dasharray:5 4; }
.edge.toout { stroke:var(--out); }
.outnode rect { transition:stroke-width .1s; }
.edge.hot { stroke:var(--hi); stroke-width:2.6px; }
.edge.dim { opacity:.12; }
.badge { font-size:9.5px; fill:var(--bg); font-weight:700; }
#empty { position:absolute; inset:0; display:grid; place-items:center;
  color:var(--muted); }
.err { color:var(--err); border-left:2px solid var(--err); padding-left:8px;
  margin:10px 0; font-size:12px; }
code { font-family:ui-monospace,Menlo,monospace; font-size:11px; }
</style>
</head>
<body>
<div id="app">
  <div id="stage">
    <div id="toolbar">
      <button id="fit">Fit</button>
      <button id="zi">+</button>
      <button id="zo">&minus;</button>
      <button id="col">&minus; depth</button>
      <button id="exp">+ depth</button>
      <span id="lvl"></span>
    </div>
    <svg id="svg"><g id="root"></g></svg>
    <div id="legend">
      <div><i></i> dataflow</div>
      <div><i style="border-color:var(--skip);border-top-style:dashed"></i>
           skip / residual</div>
      <div><i style="border-color:var(--out)"></i> model output</div>
      <div style="margin-top:4px">scroll = zoom &middot; drag = pan &middot;
           click = inspect</div>
    </div>
  </div>
  <div id="side">
    <h1 id="title"></h1>
    <div class="sub" id="subtitle"></div>
    <div id="detail"></div>
  </div>
</div>
<script>
const DATA = __DATA__;

// ---------- palette by module family -------------------------------------
const COLORS = [
  [/conv/i,          '#6ea8fe'],
  [/norm|batchnorm/i,'#8bd450'],
  [/attention|attn/i,'#c792ea'],
  [/linear|dense|mlp|feedforward|ffn/i, '#f0883e'],
  [/embed/i,         '#4dd0e1'],
  [/pool|sample|resize/i, '#f5c542'],
  [/drop/i,          '#9aa3b8'],
];
const colorOf = c => (COLORS.find(([re]) => re.test(c)) || [,'#7c8296'])[1];
const fmt = n => n >= 1e9 ? (n/1e9).toFixed(2)+'B'
              : n >= 1e6 ? (n/1e6).toFixed(2)+'M'
              : n >= 1e3 ? (n/1e3).toFixed(1)+'K' : String(n);
const shp = t => '(' + t.shape.join(', ') + ')';

// ---------- state ---------------------------------------------------------
const byId = new Map(DATA.nodes.map(n => [n.id, n]));
const mergeIds = new Set(
  DATA.nodes.filter(n => n.kind === 'merge').map(n => n.id));
// Execution position. Merge nodes are appended after tracing, so their raw id
// is far higher than their siblings'; ``order`` puts them back in sequence.
const ord = n => (typeof n === 'object' ? n : byId.get(n) || {}).order ?? 0;
const kids = new Map();
DATA.nodes.forEach(n => {
  if (n.parent !== null) {
    if (!kids.has(n.parent)) kids.set(n.parent, []);
    kids.get(n.parent).push(n.id);
  }
});
const hasKids = id => (kids.get(id) || []).length > 0;
// Start with top-level containers collapsed so big models stay readable.
const collapsed = new Set(
  DATA.nodes.filter(n => hasKids(n.id) && n.depth >= 1).map(n => n.id));

// Expand every container whose own depth is below `d`. Deeper containers
// stay collapsed, which keeps at most one level of frames on screen and
// avoids the overlap you get when frames nest several levels deep.
function expandToDepth(d) {
  collapsed.clear();
  DATA.nodes.forEach(n => { if (hasKids(n.id) && n.depth >= d) collapsed.add(n.id); });
}
let selected = null;

const isHidden = id => {          // hidden when any ancestor is collapsed
  let p = byId.get(id).parent;
  while (p !== null && p !== undefined) {
    if (collapsed.has(p)) return true;
    p = byId.get(p).parent;
  }
  return false;
};
// Nearest rendered stand-in for a node (itself, or its collapsed ancestor).
const proxy = id => {
  // Input and output pills are not real nodes, so they have no ancestor to
  // collapse into and stand in for themselves.
  if (id < 0) return id;
  let cur = id, p = byId.get(id) ? byId.get(id).parent : null, top = null;
  while (p !== null && p !== undefined) {
    if (collapsed.has(p)) top = p;
    p = byId.get(p).parent;
  }
  return top !== null ? top : cur;
};

// ---------- layout --------------------------------------------------------
// GAPY must exceed twice the deepest frame padding plus the label strip,
// otherwise a group's frame runs into the frame of the group below it.
const NW = 190, NH = 54, GAPX = 26, GAPY = 92;
// Output pills take ids below every real node id (which start at 0) and below
// __INPUT__, so the three id spaces never collide. Extra input pills (argument
// two onward) sit in the gap between __INPUT__ and OUT_BASE.
const OUT_BASE = -1000;
const IN_BASE = __IN_BASE__;
const isOutput = id => id <= OUT_BASE;
const isInput = id => id === __INPUT__ || (id <= IN_BASE && id > OUT_BASE);
let layout = { nodes: [], edges: [], w: 0, h: 0 };

function relayout() {
  const vis = DATA.nodes.filter(n => !isHidden(n.id));
  const visSet = new Set(vis.map(n => n.id));

  // An expanded container draws no edges of its own - its children do. So a
  // recorded edge touching one is re-pointed at the child that actually
  // produced (last descendant) or consumed (first descendant) the value.
  const expanded = id =>
    id !== __INPUT__ && hasKids(id) && !collapsed.has(id) && visSet.has(id);
  const descOf = id => {
    const out = [];
    const walk = i => (kids.get(i) || []).forEach(c => { out.push(c); walk(c); });
    walk(id);
    return out;
  };
  // Pick by execution order, not by raw id. Merge nodes are appended after
  // tracing, so their ids run far above their siblings' - ``Math.min``/
  // ``Math.max`` over ids therefore picks the wrong end of the subtree and
  // several distinct edges collapse onto one node.
  const pick = (ds, wantLast) => ds.reduce((best, c) =>
    (wantLast ? ord(c) > ord(best) : ord(c) < ord(best)) ? c : best);
  // ``hint`` is the node on the other side of the edge: when a container holds
  // the real endpoint, prefer the descendant that actually carries this edge
  // instead of the subtree's first/last leaf. Without it every input feeding
  // different children of one container lands on the same leaf.
  const resolve = (id, wantLast, hint) => {
    let cur = id;
    while (expanded(cur)) {
      const ds = descOf(cur).filter(c => visSet.has(c) && !expanded(c));
      if (!ds.length) break;
      let pool = ds;
      if (hint !== undefined && hint !== null) {
        // Descendants genuinely linked to the other endpoint by a recorded
        // edge. The far endpoint may itself be a container, so accept an edge
        // landing anywhere in its subtree.
        const far = new Set([hint, ...descOf(hint)]);
        const linked = ds.filter(c => DATA.edges.some(e => wantLast
          ? (e.src === c && far.has(e.dst))
          : (e.dst === c && far.has(e.src))));
        if (linked.length) pool = linked;
      }
      cur = pick(pool, wantLast);
    }
    return cur;
  };

  // Collapse edges onto visible proxies, dropping self-loops.
  const eds = [];
  const seen = new Set();
  for (const e of DATA.edges) {
    // Every input pill is a source, not just the first. Testing only
    // ``__INPUT__`` here dropped the edges of arguments two onward - their ids
    // are not real nodes, so ``visSet`` never contains them and the pills drew
    // with no arrows at all.
    let s = isInput(e.src) ? e.src : proxy(e.src);
    let d = proxy(e.dst);
    if (!isInput(s) && !visSet.has(s)) continue;
    if (!visSet.has(d)) continue;
    // Hint with the raw recorded endpoints, not the already-resolved ones, so
    // the two lookups stay independent of each other's outcome.
    const s0 = s, d0 = d;
    s = isInput(s) ? s : resolve(s, true, d0);    // producer: its last leaf
    d = resolve(d, false, s0);                   // consumer: its first leaf
    if (s === d) continue;
    const k = s + '>' + d;
    if (seen.has(k)) continue;
    seen.add(k);
    eds.push({ src: s, dst: d, skip: e.skip, tensor: e.tensor });
  }

  // Only *leaves* carry dataflow; a visible container is a header that spans
  // its own descendants. Layering containers as if they were peers of their
  // children is what produces diagonal drift, so they are placed afterwards.
  const isLeaf = n => !hasKids(n.id) || collapsed.has(n.id);
  const leaves = vis.filter(isLeaf);
  const leafSet = new Set(leaves.map(n => n.id));

  // Longest-path layering over leaves only.
  const row = new Map(leaves.map(n => [n.id, 0]));
  const preds = new Map();
  eds.forEach(e => {
    if (isInput(e.src) || !leafSet.has(e.src) || !leafSet.has(e.dst)) return;
    if (!preds.has(e.dst)) preds.set(e.dst, []);
    preds.get(e.dst).push(e.src);
  });
  // Sibling leaves inside different containers often have no *direct* edge
  // (the value passes through the containers), which would collapse the whole
  // model into a couple of rows. Chain each leaf to the previously executed
  // one as a weak ordering constraint, except where a real branch exists.
  const seq = leaves.map(n => n.id).sort((a, b) => ord(a) - ord(b));
  const branchTargets = new Set();
  eds.forEach(e => { if (e.skip) branchTargets.add(e.dst); });
  for (let i = 1; i < seq.length; i++) {
    if (branchTargets.has(seq[i])) continue;   // a shortcut starts a new path
    if (mergeIds.has(seq[i])) continue;        // a merge is placed by its edges
    if (!preds.has(seq[i])) preds.set(seq[i], []);
    if (!preds.get(seq[i]).includes(seq[i - 1]))
      preds.get(seq[i]).push(seq[i - 1]);
  }

  for (let it = 0; it < leaves.length + 2; it++) {   // acyclic: settles fast
    let moved = false;
    for (const n of leaves) {
      let want = 0;
      for (const p of (preds.get(n.id) || []))
        want = Math.max(want, (row.get(p) ?? 0) + 1);
      if (want !== row.get(n.id)) { row.set(n.id, want); moved = true; }
    }
    if (!moved) break;
  }

  const rows = new Map();
  leaves.forEach(n => {
    const r = row.get(n.id);
    if (!rows.has(r)) rows.set(r, []);
    rows.get(r).push(n);
  });

  // Container header rows are inserted above the first row they cover, so
  // reserve a slot per depth level that actually appears.
  const rowKeys = [...rows.keys()].sort((a, b) => a - b);

  // Column order matters: siblings of one container must stay contiguous or
  // their frames interleave and overlap. Sort each row by ancestor chain so
  // nodes under the same parent are always adjacent.
  const chain = id => {
    const out = [];
    let c = id;
    while (c !== null && c !== undefined) { out.unshift(c); c = byId.get(c).parent; }
    return out;
  };
  const cmp = (a, b) => {
    const ca = chain(a.id), cb = chain(b.id);
    for (let i = 0; i < Math.max(ca.length, cb.length); i++) {
      const x = ca[i] ?? -1, y2 = cb[i] ?? -1;
      // -1 means "chain ended": the shallower node sorts first.
      if (x !== y2) return (x < 0 ? -1 : ord(x)) - (y2 < 0 ? -1 : ord(y2));
    }
    return 0;
  };

  // Group leaves by the container that owns them, then give each group its
  // own column band *and* a contiguous block of rows. Without the row
  // compaction a group's leaves stagger diagonally, making its frame tall
  // enough to straddle the neighbouring bands.
  // Band by the *innermost* visible container: that is the frame that must
  // stay a tight rectangle. Banding by the outermost one puts every group in
  // the same column, which is what makes inner frames overlap.
  const groupOf = id => {
    let c = byId.get(id).parent;
    while (c !== null && c !== undefined) {
      if (visSet.has(c) && !isLeaf(byId.get(c))) return c;
      c = byId.get(c).parent;
    }
    return id;                      // no visible container: stands alone
  };
  // Group *runs*, not group ids. A container's leaves need not be contiguous
  // in execution order: MotionEncoder's root owns conv1 (first) and the
  // trailing Linears (last), with every res_block in between. Treating that
  // as one block forces the whole tail up beside conv1 and draws a long edge
  // back up the canvas. Splitting on discontinuity keeps each run in place.
  const runOf = new Map();          // leaf id -> run key
  const groupSeq = [];
  {
    const ordered = leaves.slice().sort((a, b) => ord(a) - ord(b));
    let prevGroup = null, runIdx = 0;
    for (const n of ordered) {
      const g = groupOf(n.id);
      if (g !== prevGroup) {        // a new run starts whenever the group flips
        runIdx++;
        prevGroup = g;
        groupSeq.push(`${g}:${runIdx}`);
      }
      runOf.set(n.id, `${g}:${runIdx}`);
    }
  }
  // Which container a run belongs to (for frames and column banding).
  const runGroup = k => +String(k).split(':')[0];

  // Place each group as a block. Groups that are genuinely sequential stack
  // vertically; groups that run *in parallel* (a residual's conv path and its
  // shortcut both read the same value and neither feeds the other) must share
  // rows and sit in adjacent columns, or the picture claims an ordering the
  // model does not have.
  const groupDeps = new Map();      // group -> groups it consumes from
  for (const e of eds) {
    if (e.src === __INPUT__) continue;
    const gs = runOf.get(e.src), gd = runOf.get(e.dst);
    if (gs === undefined || gd === undefined) continue;
    if (gs === gd) continue;
    if (!groupDeps.has(gd)) groupDeps.set(gd, new Set());
    groupDeps.get(gd).add(gs);
  }
  const groupSpan = new Map(groupSeq.map(g => [g,
    new Set(leaves.filter(n => runOf.get(n.id) === g).map(n => row.get(n.id))).size]));

  const groupRow = new Map(groupSeq.map(g => [g, 0]));
  // Chain each group to the previously executed one unless it is a genuine
  // parallel branch - i.e. unless it shares a producer with the group before
  // it and neither feeds the other. Without this, a group whose only inbound
  // edge is `inferred` (MotionEncoder's trailing Linears, each its own
  // single-node group) floats to row 0 and the whole tail renders *above* the
  // res-blocks it comes after, producing a long upward edge.
  const feeds = (a, b) => (groupDeps.get(b) || new Set()).has(a);
  groupSeq.forEach((g, i) => {
    if (i === 0) return;
    const prev = groupSeq[i - 1];
    const mine = groupDeps.get(g) || new Set();
    // Parallel iff g and prev draw from a common producer and are unrelated.
    const prevDeps = groupDeps.get(prev) || new Set();
    const shared = [...mine].some(d => prevDeps.has(d));
    if (shared && !feeds(prev, g) && !feeds(g, prev)) return;
    if (!groupDeps.has(g)) groupDeps.set(g, new Set());
    groupDeps.get(g).add(prev);
  });

  // Then pull each group up to the earliest row its dependencies allow. Two
  // groups fed by the same producer and feeding no one another land on the
  // same row - that is what keeps a residual's conv path and its shortcut
  // side by side instead of stacked.
  for (let it = 0; it < groupSeq.length + 1; it++) {
    let moved = false;
    for (const g of groupSeq) {
      let want = 0;
      for (const dep of (groupDeps.get(g) || []))
        want = Math.max(want, (groupRow.get(dep) ?? 0) + (groupSpan.get(dep) ?? 1));
      if (want !== groupRow.get(g)) { groupRow.set(g, want); moved = true; }
    }
    if (!moved) break;
  }
  // Groups sharing a start row are parallel: give them separate column bands.
  const bandOf = new Map();
  const byStart = new Map();
  for (const g of groupSeq) {
    const r = groupRow.get(g);
    if (!byStart.has(r)) byStart.set(r, []);
    bandOf.set(g, byStart.get(r).length);
    byStart.get(r).push(g);
  }

  const slot = new Map();           // leaf id -> column index
  const gridRow = new Map();        // leaf id -> compacted row index
  let maxCols = 1;
  // Column offset for a band: width of every band to its left, at any row.
  const bandWidth = new Map();
  for (const g of groupSeq) {
    const mine = leaves.filter(n => runOf.get(n.id) === g);
    const perRow = new Map();
    mine.forEach(n => {
      const r = row.get(n.id);
      perRow.set(r, (perRow.get(r) ?? 0) + 1);
    });
    bandWidth.set(g, Math.max(1, ...perRow.values()));
  }
  for (const g of groupSeq) {
    const peers = byStart.get(groupRow.get(g)) || [g];
    const offset = peers.slice(0, bandOf.get(g))
      .reduce((s, p) => s + bandWidth.get(p), 0);
    const mine = leaves.filter(n => runOf.get(n.id) === g).sort(cmp);
    const distinct = [...new Set(mine.map(n => row.get(n.id)))]
      .sort((a, b) => a - b);
    const perRow = new Map();
    mine.forEach(n => {
      const r = row.get(n.id);
      const k = perRow.get(r) ?? 0;
      slot.set(n.id, offset + k);
      perRow.set(r, k + 1);
      gridRow.set(n.id, groupRow.get(g) + distinct.indexOf(r));
    });
    maxCols = Math.max(maxCols, offset + bandWidth.get(g));
  }
  const W2 = maxCols * (NW + GAPX);

  // Renumber the compacted rows so there are no empty bands left behind.
  const usedRows = [...new Set([...gridRow.values()])].sort((a, b) => a - b);
  const rowAt = new Map(usedRows.map((r, i) => [r, i]));

  // Centre each row's occupants across the canvas width.
  const perGrid = new Map();
  leaves.forEach(n => {
    const r = gridRow.get(n.id);
    perGrid.set(r, (perGrid.get(r) ?? 0) + 1);
  });
  // A row holding only merges is drawn as small circles, so it needs far
  // less vertical space than a row of full module cards.
  const rowIsMerge = new Map();
  leaves.forEach(n => {
    const r = gridRow.get(n.id);
    const prev = rowIsMerge.get(r);
    const mine = n.kind === 'merge';
    rowIsMerge.set(r, prev === undefined ? mine : prev && mine);
  });
  const rowTop = new Map();
  let yCursor = 70;
  [...rowAt.keys()].sort((a, b) => rowAt.get(a) - rowAt.get(b)).forEach(r => {
    rowTop.set(r, yCursor);
    yCursor += (rowIsMerge.get(r) ? 40 : NH) + GAPY;
  });

  const pos = new Map();
  for (const n of leaves) {
    const r = gridRow.get(n.id);
    const span = perGrid.get(r) * (NW + GAPX);
    pos.set(n.id, {
      x: (W2 - span) / 2 + slot.get(n.id) * (NW + GAPX) + GAPX / 2,
      y: rowTop.get(r),
    });
  }

  // Expanded containers become background *frames* around their descendants
  // rather than nodes in the flow - a box cannot collide with the children
  // it encloses, and the nesting stays legible at any depth.
  // Nesting depth is expressed by *inset*, measured inward from the deepest
  // level, so the outermost frame is the largest and every frame still fits
  // inside the row gap regardless of how deep the tree goes.
  const containers = vis.filter(n => !isLeaf(n));
  const frames = [];
  const maxDepth = Math.max(0, ...containers.map(c => c.depth));
  const STEP = 7, BASE = 10;
  containers.sort((a, b) => b.depth - a.depth);
  for (const c of containers) {
    const pts = descOf(c.id).map(i => pos.get(i)).filter(Boolean);
    if (!pts.length) continue;
    const pad = BASE + STEP * (maxDepth - c.depth);
    const ids = descOf(c.id).filter(i => pos.get(i));
    const hOf = i => (byId.get(i) && byId.get(i).kind === 'merge' ? 40 : NH);
    const x0 = Math.min(...pts.map(p => p.x)) - pad;
    const x1 = Math.max(...pts.map(p => p.x)) + NW + pad;
    const y0 = Math.min(...pts.map(p => p.y)) - pad - 13;
    const y1 = Math.max(...ids.map(i => pos.get(i).y + hOf(i))) + pad;
    frames.push({ id: c.id, x: x0, y: y0, w: x1 - x0, h: y1 - y0, node: c });
  }

  const frameTop = frames.length ? Math.min(...frames.map(f => f.y)) : 70;
  // One pill per argument the model was called with. The first keeps the id
  // __INPUT__ so the dataflow edges the tracer already emits against it stay
  // valid; the rest use __IN_BASE__ - i, which cannot collide with real node
  // ids or with the output pills.
  const ins = (DATA.input_tensors || []).map((t, i) => ({
    id: i === 0 ? __INPUT__ : IN_BASE - i, tensor: t, index: i,
  }));
  if (!ins.length) ins.push({ id: __INPUT__, tensor: null, index: 0 });
  const inSpan = ins.length * (NW + GAPX);
  const inY = Math.min(frameTop, 70) - 56;
  ins.forEach((n, i) => {
    pos.set(n.id, {
      x: ins.length === 1
        ? (W2 - NW) / 2
        : (W2 - inSpan) / 2 + i * (NW + GAPX) + GAPX / 2,
      y: inY,
    });
  });

  // One output pill per returned array, on a row below everything else and
  // wired to the module that produced it. Outputs use ids __OUT_BASE__ - i so
  // they never collide with real node ids.
  const frameBottom = frames.length
    ? Math.max(...frames.map(f => f.y + f.h))
    : Math.max(...[...pos.values()].map(p => p.y + NH));
  const outs = (DATA.output_sources || []).map((o, i) => {
    // Resolve the producing module down to whatever is currently visible.
    let s = o.src;
    if (s !== null && s !== undefined) {
      s = proxy(s);
      s = visSet.has(s) ? resolve(s, true) : null;
    } else s = null;
    return { id: OUT_BASE - i, tensor: o.tensor, src: o.src, index: i,
             attach: s };
  });
  // Place pills in producer order (left to right) so the connecting curves
  // fan out cleanly instead of crossing each other.
  const ordered = outs.slice().sort((a, b) => {
    const pa = a.attach !== null ? pos.get(a.attach) : null;
    const pb = b.attach !== null ? pos.get(b.attach) : null;
    if (pa && pb && pa.x !== pb.x) return pa.x - pb.x;
    if (pa && pb) return pa.y - pb.y;
    return a.index - b.index;
  });
  const outSpan = Math.max(1, ordered.length) * (NW + GAPX);
  ordered.forEach((o, i) => {
    pos.set(o.id, {
      x: (W2 - outSpan) / 2 + i * (NW + GAPX) + GAPX / 2,
      y: frameBottom + 46,
    });
    if (o.attach !== null)
      eds.push({ src: o.attach, dst: o.id, skip: false, tensor: o.tensor,
                 toOutput: true });
  });

  // Normalize so the whole drawing starts at a small positive margin.
  const dx = 24 - Math.min(...[...pos.values()].map(p => p.x),
                           ...frames.map(f => f.x));
  const dy = 24 - Math.min(...[...pos.values()].map(p => p.y),
                           ...frames.map(f => f.y));
  pos.forEach(p => { p.x += dx; p.y += dy; });
  frames.forEach(f => { f.x += dx; f.y += dy; });

  layout = {
    nodes: leaves, edges: eds, pos, frames, outs, ins,
    containers: new Set(containers.map(c => c.id)),
    w: Math.max(...[...pos.values()].map(p => p.x + NW),
                ...frames.map(f => f.x + f.w)) + 24,
    h: Math.max(...[...pos.values()].map(p => p.y + NH),
                ...frames.map(f => f.y + f.h)) + 24,
  };
  draw();
}

// ---------- draw ----------------------------------------------------------
const SVGNS = 'http://www.w3.org/2000/svg';
const root = document.getElementById('root');
const el = (t, a) => {
  const e = document.createElementNS(SVGNS, t);
  for (const k in a) e.setAttribute(k, a[k]);
  return e;
};

function draw() {
  root.textContent = '';
  const { pos } = layout;

  const gF = el('g'), gE = el('g'), gN = el('g');
  root.append(gF, gE, gN);   // frames sit behind edges and nodes

  for (const f of layout.frames) {
    const col = colorOf(f.node.cls);
    const g = el('g', { class: 'frame' });
    g.dataset.id = f.id;
    g.appendChild(el('rect', { x: f.x, y: f.y, width: f.w, height: f.h,
      rx: 10, fill: col, 'fill-opacity': .05, stroke: col,
      'stroke-opacity': .45, 'stroke-dasharray': '5 4' }));
    const lab = el('text', { x: f.x + 10, y: f.y + 13, class: 'flab',
      fill: col });
    lab.textContent = `${f.node.name}  ·  ${f.node.cls}  ·  ${fmt(f.node.params)}`;
    g.append(lab);
    gF.appendChild(g);
  }

  for (const e of layout.edges) {
    const a = pos.get(e.src), b = pos.get(e.dst);
    if (!a || !b) continue;
    // Merges are drawn as a circle centred in their slot, so edges meet the
    // circle rather than the (invisible) card bounds.
    const srcMerge = mergeIds.has(e.src), dstMerge = mergeIds.has(e.dst);
    const x1 = a.x + NW / 2;
    const y1 = a.y + (isInput(e.src) ? 26 : srcMerge ? 37 : NH);
    const x2 = b.x + NW / 2;
    const y2 = b.y + (dstMerge ? 3 : 0);
    const my = (y1 + y2) / 2;
    let d;
    if (e.skip && Math.abs(x2 - x1) < 4 && y2 - y1 > NH) {
      // A residual that returns to the same column would be hidden under the
      // main chain. Bow it out sideways so the bypass is actually visible.
      const bow = NW * 0.62;
      d = `M${x1},${y1} C${x1 - bow},${y1 + 20} ${x2 - bow},${y2 - 20} `
        + `${x2},${y2}`;
    } else {
      d = `M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}`;
    }
    const p = el('path', {
      d,
      class: 'edge' + (e.skip ? ' skip' : '') + (e.toOutput ? ' toout' : ''),
      'marker-end': e.toOutput ? 'url(#arrowout)' : 'url(#arrow)',
    });
    p.dataset.src = e.src; p.dataset.dst = e.dst;
    gE.appendChild(p);
  }

  // one pill per model input
  for (const n of layout.ins) {
    const ip = pos.get(n.id);
    if (!ip) continue;
    const gi = el('g', { transform: `translate(${ip.x},${ip.y})` });
    gi.appendChild(el('rect', { width: NW, height: 26, rx: 13,
      fill: 'var(--panel)', stroke: 'var(--accent)' }));
    const it = el('text', { x: NW / 2, y: 17, 'text-anchor': 'middle',
      class: 'nc', fill: 'var(--accent)' });
    const label = layout.ins.length > 1 ? `input ${n.index}` : 'input';
    it.textContent = `${label} ${n.tensor ? shp(n.tensor) : ''}`;
    gi.appendChild(it);
    gN.appendChild(gi);
  }

  // one pill per returned array
  for (const o of layout.outs) {
    const p = pos.get(o.id);
    if (!p) continue;
    const g = el('g', { class: 'outnode',
      transform: `translate(${p.x},${p.y})` });
    g.dataset.id = o.id;
    g.appendChild(el('rect', { width: NW, height: 30, rx: 15,
      fill: 'var(--panel)', stroke: 'var(--out)', 'stroke-width': 1.6 }));
    const t = el('text', { x: NW / 2, y: 19, 'text-anchor': 'middle',
      class: 'nc', fill: 'var(--out)' });
    const label = layout.outs.length > 1 ? `output ${o.index}` : 'output';
    t.textContent = `${label} ${shp(o.tensor)}`;
    g.appendChild(t);
    gN.appendChild(g);
  }

  for (const n of layout.nodes) {
    const p = pos.get(n.id);
    const g = el('g', { class: 'node', transform: `translate(${p.x},${p.y})` });
    g.dataset.id = n.id;

    // An array op (a residual ``+``, a concat, an activation) owns no
    // parameters and is not a module, so it is drawn as a small pill on the
    // flow rather than a full card. A one-glyph label keeps the classic
    // circle; a named op like ``leaky_relu`` widens into a rounded pill so
    // the text is not clipped.
    if (n.kind === 'merge') {
      const cx = NW / 2, cy = 20;
      const glyph = [...n.name].length <= 2;
      const fs = glyph ? 18 : 11;
      // SVG has no text metrics before layout; this per-character estimate is
      // close enough for a monospace-ish label and never under-sizes.
      const wRaw = glyph ? 34 : [...n.name].length * fs * 0.62 + 16;
      const w = Math.min(wRaw, NW), h = 34;
      g.appendChild(el('rect', { x: cx - w / 2, y: cy - h / 2, width: w,
        height: h, rx: h / 2, fill: 'var(--panel)',
        stroke: 'var(--skip)', 'stroke-width': 2 }));
      const s = el('text', { x: cx, y: cy + (glyph ? 6 : 4),
        'text-anchor': 'middle', fill: 'var(--skip)', 'font-size': fs,
        'font-weight': 700 });
      s.textContent = n.name;
      g.appendChild(s);
      const sh = el('text', { x: cx, y: cy + 33, 'text-anchor': 'middle',
        class: 'ns' });
      sh.textContent = n.outputs.length ? shp(n.outputs[0]) : '';
      g.appendChild(sh);
      gN.appendChild(g);
      continue;
    }
    const col = n.error ? 'var(--err)' : colorOf(n.cls);
    g.appendChild(el('rect', { width: NW, height: NH, rx: 8,
      fill: 'var(--panel)', stroke: col }));
    g.appendChild(el('rect', { width: 4, height: NH, rx: 2, fill: col }));

    const t1 = el('text', { x: 12, y: 19, class: 'nm', fill: 'var(--fg)' });
    t1.textContent = (n.name.length > 20 ? n.name.slice(0, 19) + '\\u2026' : n.name)
      + (collapsed.has(n.id) ? '  \\u25B8' : '');
    const t2 = el('text', { x: 12, y: 33, class: 'nc' });
    t2.textContent = n.cls;
    const t3 = el('text', { x: 12, y: 47, class: 'ns' });
    t3.textContent = n.outputs.length ? shp(n.outputs[0]) : '';
    g.append(t1, t2, t3);

    if (n.params > 0) {
      const bw = 8 + fmt(n.params).length * 6;
      g.appendChild(el('rect', { x: NW - bw - 8, y: 8, width: bw, height: 15,
        rx: 7, fill: col, opacity: .85 }));
      const tb = el('text', { x: NW - bw / 2 - 8, y: 19,
        'text-anchor': 'middle', class: 'badge' });
      tb.textContent = fmt(n.params);
      g.appendChild(tb);
    }
    gN.appendChild(g);
  }

  const defs = el('defs');
  const mk = el('marker', { id: 'arrow', viewBox: '0 0 10 10', refX: 9,
    refY: 5, markerWidth: 5, markerHeight: 5, orient: 'auto-start-reverse' });
  mk.appendChild(el('path', { d: 'M0,0 L10,5 L0,10 z', fill: 'var(--edge)' }));
  defs.appendChild(mk);
  const mo = el('marker', { id: 'arrowout', viewBox: '0 0 10 10', refX: 9,
    refY: 5, markerWidth: 5, markerHeight: 5, orient: 'auto-start-reverse' });
  mo.appendChild(el('path', { d: 'M0,0 L10,5 L0,10 z', fill: 'var(--out)' }));
  defs.appendChild(mo);
  root.appendChild(defs);

  applySelection();
}

// ---------- interaction ---------------------------------------------------
let tx = 0, ty = 0, k = 1;
const svg = document.getElementById('svg');
const apply = () => root.setAttribute('transform',
  `translate(${tx},${ty}) scale(${k})`);

function fit() {
  const r = svg.getBoundingClientRect();
  // Allow modest upscaling so small graphs fill the canvas, and refuse to
  // shrink past readability - a tall model is meant to be scrolled, not
  // rendered as unreadable confetti.
  k = Math.min(r.width / (layout.w + 80), r.height / (layout.h + 80), 1.9);
  k = Math.max(k, 0.45);
  tx = (r.width - layout.w * k) / 2;
  ty = layout.h * k > r.height ? 16 : (r.height - layout.h * k) / 2;
  apply();
}

let drag = null;
svg.addEventListener('mousedown', e => {
  drag = { x: e.clientX - tx, y: e.clientY - ty };
  document.getElementById('stage').classList.add('drag');
});
addEventListener('mousemove', e => {
  if (!drag) return;
  tx = e.clientX - drag.x; ty = e.clientY - drag.y; apply();
});
addEventListener('mouseup', () => {
  drag = null;
  document.getElementById('stage').classList.remove('drag');
});
svg.addEventListener('wheel', e => {
  e.preventDefault();
  const r = svg.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  const f = e.deltaY < 0 ? 1.12 : 1 / 1.12;
  const nk = Math.min(3, Math.max(0.12, k * f));
  tx = mx - (mx - tx) * (nk / k); ty = my - (my - ty) * (nk / k); k = nk;
  apply();
}, { passive: false });

svg.addEventListener('click', e => {
  // A frame's border/label collapses the group it encloses; a node inside it
  // is matched first, so clicking a child never collapses its parent.
  const g = e.target.closest('.node') || e.target.closest('.outnode')
         || e.target.closest('.frame');
  if (!g) { selected = null; applySelection(); return; }
  const id = +g.dataset.id;
  const isFrame = g.classList.contains('frame');
  if (isFrame || (e.detail === 2 && hasKids(id))) {
    collapsed.has(id) ? collapsed.delete(id) : collapsed.add(id);
    selected = null;
    relayout();
    return;
  }
  selected = id;
  applySelection();
});

function applySelection() {
  const near = new Set();
  if (selected !== null) {
    near.add(selected);
    layout.edges.forEach(e => {
      if (e.src === selected) near.add(e.dst);
      if (e.dst === selected) near.add(e.src);
    });
  }
  root.querySelectorAll('.node, .outnode').forEach(g => {
    const id = +g.dataset.id;
    g.classList.toggle('sel', id === selected);
    g.classList.toggle('dim', selected !== null && !near.has(id));
  });
  root.querySelectorAll('.edge').forEach(p => {
    const s = +p.dataset.src, d = +p.dataset.dst;
    const hot = selected !== null && (s === selected || d === selected);
    p.classList.toggle('hot', hot);
    p.classList.toggle('dim', selected !== null && !hot);
  });
  detail();
}

function detail() {
  const box = document.getElementById('detail');
  if (selected === null) {
    const rows = [
      ['Modules', DATA.nodes.length],
      ['Parameters', fmt(DATA.total_params) + ' (' + DATA.total_params + ')'],
      ['Connections', DATA.edges.length],
      ['Skip edges', DATA.edges.filter(e => e.skip).length],
    ];
    box.innerHTML =
      (DATA.error ? `<div class="err"><b>Forward pass failed</b><br>
         <code>${esc(DATA.error)}</code><br>Graph shows progress up to the
         failure.</div>` : '') +
      '<h2>Model</h2>' + rows.map(r =>
        `<div class="stat"><span>${r[0]}</span><span>${r[1]}</span></div>`).join('') +
      '<h2>Input</h2>' + DATA.input_tensors.map(t =>
        `<div class="stat"><span>${t.dtype}</span><span>${shp(t)}</span></div>`).join('') +
      '<h2>Output</h2>' + DATA.output_tensors.map(t =>
        `<div class="stat"><span>${t.dtype}</span><span>${shp(t)}</span></div>`).join('') +
      '<h2>Tips</h2>' +
      '<div class="stat"><span>Click node</span><span>trace neighbours</span></div>' +
      '<div class="stat"><span>Click group border</span><span>collapse</span></div>' +
      '<div class="stat"><span>&plusmn; depth</span><span>expand a level</span></div>';
    return;
  }
  if (isOutput(selected)) {           // an output pill, not a module
    const o = layout.outs.find(x => x.id === selected);
    const src = o && o.src !== null && o.src !== undefined
      ? byId.get(o.src) : null;
    box.innerHTML = '<h2>Model output</h2>' +
      `<div class="stat"><span>Index</span><span>${o ? o.index : ''}</span></div>` +
      `<div class="stat"><span>Shape</span><span>${o ? shp(o.tensor) : ''}</span></div>` +
      `<div class="stat"><span>dtype</span><span>${o ? o.tensor.dtype : ''}</span></div>` +
      '<h2>Produced by</h2>' +
      (src
        ? `<div class="stat"><span>${esc(src.cls)}</span>` +
          `<span>${esc(src.path)}</span></div>`
        : '<div class="stat"><span>no module</span>' +
          '<span>built by array ops</span></div>');
    return;
  }
  const n = byId.get(selected);
  const rows = [
    ['Class', n.cls], ['Path', n.path || '&lt;root&gt;'],
    ['Depth', n.depth],
    ['Params (total)', fmt(n.params)],
    ['Params (own)', fmt(n.own_params)],
  ];
  if (DATA.total_params > 0)
    rows.push(['Share of model',
      (100 * n.params / DATA.total_params).toFixed(1) + '%']);
  const cfg = Object.entries(n.config);
  box.innerHTML =
    (n.error ? `<div class="err"><code>${esc(n.error)}</code></div>` : '') +
    '<h2>Module</h2>' + rows.map(r =>
      `<div class="stat"><span>${r[0]}</span><span>${r[1]}</span></div>`).join('') +
    '<h2>Inputs</h2>' + (n.inputs.length ? n.inputs.map(t =>
      `<div class="stat"><span>${t.dtype}</span><span>${shp(t)}</span></div>`
      ).join('') : '<div class="stat"><span>none</span><span></span></div>') +
    '<h2>Outputs</h2>' + (n.outputs.length ? n.outputs.map(t =>
      `<div class="stat"><span>${t.dtype}</span><span>${shp(t)}</span></div>`
      ).join('') : '<div class="stat"><span>none</span><span></span></div>') +
    (cfg.length ? '<h2>Config</h2>' + cfg.map(([a, b]) =>
      `<div class="stat"><span>${esc(a)}</span><span>${esc(String(b))}</span></div>`
      ).join('') : '');
}
const esc = s => String(s).replace(/[&<>]/g,
  c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

document.getElementById('fit').onclick = fit;
document.getElementById('zi').onclick = () => { k = Math.min(3, k * 1.2); apply(); };
document.getElementById('zo').onclick = () => { k = Math.max(.12, k / 1.2); apply(); };
const MAXD = Math.max(1, ...DATA.nodes.map(n => n.depth));
let level = 1;
const setLevel = d => {
  level = Math.min(MAXD, Math.max(1, d));
  expandToDepth(level);
  document.getElementById('lvl').textContent = `depth ${level}/${MAXD}`;
  selected = null;
  relayout();
  fit();
};
document.getElementById('exp').onclick = () => setLevel(level + 1);
document.getElementById('col').onclick = () => setLevel(level - 1);

document.getElementById('title').textContent = DATA.model_name;
document.getElementById('subtitle').textContent =
  DATA.nodes.length + ' modules \\u00B7 ' + fmt(DATA.total_params) + ' params';
setLevel(1);
addEventListener('resize', fit);
</script>
</body>
</html>
"""


def render_html(graph: Graph, title: Optional[str] = None) -> str:
    """Return a standalone interactive HTML document for ``graph``."""
    payload = json.dumps(graph.to_dict(), separators=(",", ":"))
    # Guard against a literal "</script>" inside the JSON ending the block.
    payload = payload.replace("</", "<\\/")
    doc = _TEMPLATE.replace("__DATA__", payload)
    doc = doc.replace("__INPUT__", str(INPUT_NODE))
    doc = doc.replace("__IN_BASE__", str(IN_BASE))
    return doc.replace(
        "__TITLE__", html.escape(title or f"{graph.model_name} - visualized"))


def save_html(graph: Graph, path: str, title: Optional[str] = None) -> str:
    """Write the visualization to ``path`` and return the absolute path."""
    doc = render_html(graph, title=title)
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return path


__all__ = ["render_html", "save_html"]
