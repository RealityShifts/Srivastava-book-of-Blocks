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
/* Mermaid's default ("neutral"-ish) look: soft lavender fills, a single
   restrained stroke colour, and mid-grey edges rather than one hue per module
   family. The family colour is kept only as a thin left accent on each card, so
   the graph reads as one calm system instead of a paintbox. */
:root {
  --bg:#0f1117; --panel:#1b1e28; --line:#2b303d; --fg:#e6e9ef; --muted:#9aa3b8;
  --accent:#8494e4; --skip:#e8944a; --edge:#6b7690; --hi:#ffd166; --err:#ff6b6b;
  --out:#5ec9a7;
  --card:#252a38; --cardln:#4a5268; --frame:#1f2431; --frameln:#39415a;
}
@media (prefers-color-scheme: light) {
  :root { --bg:#ffffff; --panel:#fff; --line:#e2e5ec; --fg:#1f2430;
          --muted:#5f6880; --edge:#8a93a6; --accent:#5b6fd6;
          /* Mermaid's signature pale-lavender node fill. */
          --card:#eceffc; --cardln:#9aa5d8; --frame:#f6f7fd;
          --frameln:#c7cde8; }
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
  background:var(--panel); overflow-y:auto; padding:16px;
  transition:width .18s ease, opacity .18s ease; }
/* Collapsed, the panel gives its width back to the canvas. ``visibility``
   rather than ``display`` so the width can animate rather than jump. */
#app.noside #side { width:0; padding-left:0; padding-right:0; opacity:0;
  overflow:hidden; visibility:hidden; border-left-width:0; }
/* The handle rides the panel edge, so it stays clickable either way. */
#sidetog { position:absolute; top:12px; right:12px; z-index:6; }
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
/* Each detail section is a <details>, so a long sidebar can be folded down to
   just the headings the reader cares about. The caret is drawn by hand because
   the native marker sits at a different baseline across browsers. */
.sec { border-bottom:0; }
.sec > summary { font-size:11px; text-transform:uppercase; letter-spacing:.08em;
  color:var(--muted); margin:18px 0 6px; cursor:pointer; list-style:none;
  display:flex; align-items:center; gap:6px; user-select:none; }
.sec > summary::-webkit-details-marker { display:none; }
.sec > summary:hover { color:var(--fg); }
.sec > summary::before { content:''; flex:none; width:0; height:0;
  border-left:4px solid currentColor; border-top:3.5px solid transparent;
  border-bottom:3.5px solid transparent; transition:transform .12s; }
.sec[open] > summary::before { transform:rotate(90deg); }
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
.node rect { stroke-width:1.2px; }
.node { cursor:pointer; }
.node text { pointer-events:none; }
/* The family hue survives as a 3px left rule - enough to group at a glance
   without flooding the canvas with colour. */
.accent { stroke-width:0; }
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
.edge { fill:none; stroke:var(--edge); stroke-width:1.4px;
  stroke-linecap:round; }
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
      <button id="svg-dl" title="Download the current view as SVG">SVG</button>
      <button id="mmd-dl" title="Download Mermaid source for the current view"
        >Mermaid</button>
      <span id="lvl"></span>
    </div>
    <button id="sidetog" title="Show/hide the details panel">Details</button>
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
<script>/*__DAGRE__*/</script>
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
// Is `id` anywhere inside `anc`'s subtree? Used by the Mermaid export to place
// each visible leaf in the frame that encloses it.
const isDesc = (id, anc) => {
  let p = byId.get(id) ? byId.get(id).parent : null;
  while (p !== null && p !== undefined) {
    if (p === anc) return true;
    p = byId.get(p).parent;
  }
  return false;
};
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

  // ---- layout via dagre --------------------------------------------------
  // Ranks, in-rank ordering, crossing reduction and edge routing all come from
  // dagre - the same engine Mermaid uses. The piece that matters for a deep
  // model is its virtual nodes: a long edge occupies real space in every rank it
  // spans, so boxes are placed out of its way instead of underneath it.
  // Containers are declared as compound parents so dagre reserves room for the
  // frames drawn later.
  const G = new dagre.graphlib.Graph({ compound: true });
  // ranksep is the gap between *ranks*, and dagre adds a rank per level a long
  // edge spans, so a residual over three modules multiplies the nominal gap.
  // Keep it tight; GAPY was tuned for the old one-row-per-node layout.
  G.setGraph({ rankdir: 'TB', nodesep: GAPX, ranksep: 34,
               marginx: 24, marginy: 24, ranker: 'network-simplex' });
  G.setDefaultEdgeLabel(() => ({}));

  const hOfNode = n => (n.kind === 'merge' ? 40 : NH);
  for (const n of leaves)
    G.setNode(String(n.id), { width: NW, height: hOfNode(n) });
  // Input pills are real ranked nodes, so their edges are routed like any other
  // rather than being drawn as straight lines over the graph.
  const inIds = (DATA.input_tensors || []).map((t, i) =>
    i === 0 ? __INPUT__ : IN_BASE - i);
  inIds.forEach(id => G.setNode(String(id), { width: NW, height: 26 }));
  // Output pills are ranked too. Placing them by hand afterwards left their
  // edges unrouted, so they cut straight back across the graph.
  const outPre = (DATA.output_sources || []).map((o, i) => {
    let s = o.src;
    if (s !== null && s !== undefined) {
      s = proxy(s);
      s = visSet.has(s) ? resolve(s, true) : null;
    } else s = null;
    return { id: OUT_BASE - i, tensor: o.tensor, index: i, attach: s };
  });
  outPre.forEach(o => G.setNode(String(o.id), { width: NW, height: 30 }));

  // Compound parents. dagre needs every ancestor declared, and a cluster may
  // not be its own parent.
  const contAll = vis.filter(n => !isLeaf(n));
  contAll.forEach(c => G.setNode(String(c.id), {}));
  for (const n of [...leaves, ...contAll]) {
    let p = n.parent;
    while (p !== null && p !== undefined && !contAll.some(c => c.id === p))
      p = byId.get(p) ? byId.get(p).parent : null;
    if (p !== null && p !== undefined && p !== n.id)
      G.setParent(String(n.id), String(p));
  }

  for (const e of eds) {
    if (!G.hasNode(String(e.src)) || !G.hasNode(String(e.dst))) continue;
    // weight keeps the main chain straight and lets skips bend around it.
    G.setEdge(String(e.src), String(e.dst),
              { weight: e.skip ? 1 : 8, minlen: 1 });
  }
  outPre.forEach(o => {
    if (o.attach !== null && G.hasNode(String(o.attach)))
      G.setEdge(String(o.attach), String(o.id), { weight: 8, minlen: 1 });
  });

  dagre.layout(G);

  // dagre reports centres; the drawing code works from top-left corners.
  const pos = new Map();
  for (const n of leaves) {
    const d = G.node(String(n.id));
    if (d) pos.set(n.id, { x: d.x - NW / 2, y: d.y - hOfNode(n) / 2 });
  }
  // Input pills were ranked too; without reading their position back they keep
  // a stale one while their routed edge starts where dagre actually put them,
  // which is what left long wires dangling from the top of the canvas.
  inIds.forEach(id => {
    const d = G.node(String(id));
    if (d) pos.set(id, { x: d.x - NW / 2, y: d.y - 13 });
  });
  // Routed polylines, keyed by edge, for the draw step.
  const routed = new Map();
  for (const e of G.edges()) {
    const pts = (G.edge(e).points || []).map(q => ({ x: q.x, y: q.y }));
    if (pts.length) routed.set(e.v + '>' + e.w, pts);
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
  // Keep the position dagre chose for each pill. Flattening them onto a shared
  // row looks tidier in isolation but discards the ranking that made their
  // edges routable, so a pill feeding three distant consumers ends up dragging
  // three long wires across the graph.
  ins.forEach((n, i) => {
    if (!pos.get(n.id)) pos.set(n.id, { x: i * (NW + GAPX) + 24, y: 0 });
  });
  // Canvas width, now that dagre has decided the extents.
  const W2 = Math.max(NW, ...[...pos.values()].map(p => p.x + NW));

  // One output pill per returned array, positioned by dagre along with the rest
  // so its edge is routed rather than drawn straight back over the graph.
  const outs = outPre.map(o => ({ ...o, src: o.attach }));
  outs.forEach(o => {
    const d = G.node(String(o.id));
    if (d) pos.set(o.id, { x: d.x - NW / 2, y: d.y - 15 });
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
  // Routed polylines live in the same coordinate space, so they shift too.
  routed.forEach(pts => pts.forEach(q => { q.x += dx; q.y += dy; }));

  layout = {
    nodes: leaves, edges: eds, pos, frames, outs, ins, routed,
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
    // Mermaid draws a subgraph as a quiet solid panel with a plain border - no
    // dashes, no per-group hue - so nesting reads as depth rather than as noise.
    g.appendChild(el('rect', { x: f.x, y: f.y, width: f.w, height: f.h,
      rx: 8, fill: 'var(--frame)', 'fill-opacity': .85,
      stroke: 'var(--frameln)' }));
    const lab = el('text', { x: f.x + 10, y: f.y + 13, class: 'flab',
      fill: 'var(--muted)' });
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
    // dagre already routed this edge around whatever lay in its path, so follow
    // its polyline. Drawing our own curve is what put long edges over boxes.
    const pts = (layout.routed || new Map()).get(e.src + '>' + e.dst);
    if (pts && pts.length > 1) {
      // Follow the polyline, rounding each corner with a short quadratic. A
      // per-segment S-curve overshoots the waypoints and reintroduces exactly
      // the crossings dagre routed around, so keep the path on its own line and
      // only soften the joins.
      // Keep every waypoint dagre produced - they are the virtual nodes that
      // steer the edge around boxes. Only the endpoints are nudged onto the
      // node borders, and dropping the interior points (as slicing them off
      // does) collapses the detour back into one long straight line.
      const q = pts.map(p => ({ x: p.x, y: p.y }));
      q[0] = { x: x1, y: y1 };
      q[q.length - 1] = { x: x2, y: y2 };
      const R = 12;
      d = `M${q[0].x},${q[0].y}`;
      for (let i = 1; i < q.length - 1; i++) {
        const p0 = q[i - 1], p1 = q[i], p2 = q[i + 1];
        const len = (u, v) => Math.hypot(v.x - u.x, v.y - u.y) || 1;
        const t1 = Math.min(R, len(p0, p1) / 2) / len(p0, p1);
        const t2 = Math.min(R, len(p1, p2) / 2) / len(p1, p2);
        d += ` L${p1.x + (p0.x - p1.x) * t1},${p1.y + (p0.y - p1.y) * t1}`
          + ` Q${p1.x},${p1.y} `
          + `${p1.x + (p2.x - p1.x) * t2},${p1.y + (p2.y - p1.y) * t2}`;
      }
      d += ` L${q[q.length - 1].x},${q[q.length - 1].y}`;
    } else if (e.skip && Math.abs(x2 - x1) < 4 && y2 - y1 > NH) {
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
        height: h, rx: h / 2, fill: 'var(--card)',
        stroke: 'var(--cardln)', 'stroke-width': 1.2 }));
      const s = el('text', { x: cx, y: cy + (glyph ? 6 : 4),
        'text-anchor': 'middle', fill: 'var(--fg)', 'font-size': fs,
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
    // Mermaid-style card: one shared fill and border for every node, with the
    // module family reduced to a left accent rule. An error still overrides the
    // border outright, since that must not be subtle.
    g.appendChild(el('rect', { width: NW, height: NH, rx: 6,
      fill: 'var(--card)',
      stroke: n.error ? 'var(--err)' : 'var(--cardln)' }));
    g.appendChild(el('rect', { width: 3, height: NH, rx: 1.5, fill: col,
      class: 'accent' }));

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
        rx: 7, fill: col, opacity: .9 }));
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

// A collapsible sidebar section. Open/closed state is keyed by title and kept
// in ``secOpen`` so re-rendering the panel on every selection does not reset
// what the reader has folded away.
const secOpen = new Map();
const sec = (title, body) =>
  `<details class="sec"${secOpen.get(title) === false ? '' : ' open'}>` +
  `<summary>${title}</summary>${body}</details>`;

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
      sec('Model', rows.map(r =>
        `<div class="stat"><span>${r[0]}</span><span>${r[1]}</span></div>`).join('')) +
      sec('Input', DATA.input_tensors.map(t =>
        `<div class="stat"><span>${t.dtype}</span><span>${shp(t)}</span></div>`).join('')) +
      sec('Output', DATA.output_tensors.map(t =>
        `<div class="stat"><span>${t.dtype}</span><span>${shp(t)}</span></div>`).join('')) +
      sec('Tips',
        '<div class="stat"><span>Click node</span><span>trace neighbours</span></div>' +
        '<div class="stat"><span>Click group border</span><span>collapse</span></div>' +
        '<div class="stat"><span>&plusmn; depth</span><span>expand a level</span></div>');
    return;
  }
  if (isOutput(selected)) {           // an output pill, not a module
    const o = layout.outs.find(x => x.id === selected);
    const src = o && o.src !== null && o.src !== undefined
      ? byId.get(o.src) : null;
    box.innerHTML = sec('Model output',
      `<div class="stat"><span>Index</span><span>${o ? o.index : ''}</span></div>` +
      `<div class="stat"><span>Shape</span><span>${o ? shp(o.tensor) : ''}</span></div>` +
      `<div class="stat"><span>dtype</span><span>${o ? o.tensor.dtype : ''}</span></div>`) +
      sec('Produced by',
      (src
        ? `<div class="stat"><span>${esc(src.cls)}</span>` +
          `<span>${esc(src.path)}</span></div>`
        : '<div class="stat"><span>no module</span>' +
          '<span>built by array ops</span></div>'));
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
    sec('Module', rows.map(r =>
      `<div class="stat"><span>${r[0]}</span><span>${r[1]}</span></div>`).join('')) +
    sec('Inputs', n.inputs.length ? n.inputs.map(t =>
      `<div class="stat"><span>${t.dtype}</span><span>${shp(t)}</span></div>`
      ).join('') : '<div class="stat"><span>none</span><span></span></div>') +
    sec('Outputs', n.outputs.length ? n.outputs.map(t =>
      `<div class="stat"><span>${t.dtype}</span><span>${shp(t)}</span></div>`
      ).join('') : '<div class="stat"><span>none</span><span></span></div>') +
    (cfg.length ? sec('Config', cfg.map(([a, b]) =>
      `<div class="stat"><span>${esc(a)}</span><span>${esc(String(b))}</span></div>`
      ).join('')) : '');
}
const esc = s => String(s).replace(/[&<>]/g,
  c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

// ``toggle`` does not bubble, so it is captured rather than listened for on the
// panel itself - the sections are rebuilt on every selection change.
document.getElementById('detail').addEventListener('toggle', e => {
  const d = e.target;
  if (d.classList && d.classList.contains('sec'))
    secOpen.set(d.querySelector('summary').textContent, d.open);
}, true);

// Hiding the panel hands its 310px back to the canvas, so the graph is refit
// once the width transition has finished rather than against a stale size.
document.getElementById('sidetog').onclick = () => {
  const on = document.getElementById('app').classList.toggle('noside');
  document.getElementById('sidetog').setAttribute('aria-expanded', !on);
  setTimeout(fit, 200);
};
// `d` toggles it from the keyboard, ignored while typing in a field.
document.addEventListener('keydown', e => {
  if (e.key === 'd' && !e.metaKey && !e.ctrlKey && !e.altKey &&
      !/^(INPUT|TEXTAREA)$/.test(document.activeElement.tagName))
    document.getElementById('sidetog').click();
});

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

// ---------- export --------------------------------------------------------
const save = (text, name, mime) => {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const a = document.createElement('a');
  a.href = url; a.download = name;
  a.click();
  // Revoking immediately can cancel the download in some browsers.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
const slug = DATA.model_name.replace(/[^\\w.-]+/g, '_');

// Standalone SVG of the current view: the live <svg> carries the pan/zoom
// transform and inherits its colours from the page, so export a detached clone
// sized to the graph with the CSS variables resolved to literals.
document.getElementById('svg-dl').onclick = () => {
  const clone = svg.cloneNode(true);
  const cs = getComputedStyle(document.documentElement);
  // Inline every --var the markup references; a bare `var(...)` would render
  // black once the file is opened outside this page.
  let out = clone.outerHTML.replace(/var\\((--[\\w-]+)\\)/g,
    (_, v) => cs.getPropertyValue(v).trim() || '#888');
  const pad = 20;
  out = out
    .replace(/<svg[^>]*>/, '<svg xmlns="http://www.w3.org/2000/svg" '
      + `width="${layout.w + pad * 2}" height="${layout.h + pad * 2}" `
      + `viewBox="${-pad} ${-pad} ${layout.w + pad * 2} ${layout.h + pad * 2}">`)
    // Drop the interactive pan/zoom so the exported file is framed on content.
    .replace(/<g id="root"[^>]*>/, '<g id="root">');
  const bg = cs.getPropertyValue('--bg').trim() || '#fff';
  // The classes the markup relies on (.edge, .nm, .ns, ...) are defined in the
  // page's <head>, which a detached clone does not carry. Without them every
  // edge falls back to SVG defaults - a 1px black hairline - and the text loses
  // its sizes. Inline the rules the graph actually needs, with the custom
  // properties already resolved.
  const c = v => cs.getPropertyValue(v).trim();
  const css = `<style>
    .edge{fill:none;stroke:${c('--edge')};stroke-width:1.6px;stroke-linecap:round}
    .edge.skip{stroke:${c('--skip')};stroke-dasharray:5 4}
    .edge.toout{stroke:${c('--out')}}
    .node rect,.frame rect,.outnode rect{stroke-width:1.2px}
    .accent{stroke-width:0}
    text{font-family:ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
    .nm{font-weight:600;font-size:12px;fill:${c('--fg')}}
    .nc{font-size:10.5px;fill:${c('--muted')}}
    .ns{font-size:10px;font-family:ui-monospace,Menlo,monospace;
        fill:${c('--muted')}}
    .flab{font-size:10.5px;font-weight:600;fill:${c('--muted')}}
    .badge{font-size:9.5px;fill:${bg};font-weight:700}
  </style>`;
  out = out.replace('<g id="root">',
    `${css}<rect x="${-pad}" y="${-pad}" width="${layout.w + pad * 2}" `
    + `height="${layout.h + pad * 2}" fill="${bg}"/><g id="root">`);
  save(out, slug + '.svg', 'image/svg+xml');
};

// Mermaid source for the current view. Uses the same visible-node set and
// resolved edges the canvas is showing, so collapsing a group collapses it in
// the export too. Containers become `subgraph` blocks, which is Mermaid's
// nearest equivalent to a frame.
document.getElementById('mmd-dl').onclick = () => {
  const mid = id => 'n' + String(id).replace('-', '_');
  const q = s => String(s).replace(/"/g, "'");
  const L = ['flowchart TD'];
  // Mermaid has no styling for our families, so emit classDef + class lines and
  // let the shape carry the rest: rounded box = module, stadium = array op,
  // pill = input/output.
  L.push('  classDef mod fill:#eceffc,stroke:#9aa5d8,color:#1f2430;');
  L.push('  classDef op fill:#fff6e8,stroke:#e8944a,color:#1f2430;');
  L.push('  classDef io fill:#e8f4ee,stroke:#5ec9a7,color:#1f2430;');

  const decl = [];
  for (const n of layout.ins) {
    const t = n.tensor ? ' ' + shp(n.tensor) : '';
    const label = (layout.ins.length > 1 ? 'input ' + n.index : 'input') + t;
    decl.push(`  ${mid(n.id)}(["${q(label)}"]):::io`);
  }
  // Group visible leaves by their frame so each frame becomes a subgraph.
  const inFrame = new Map();
  for (const f of layout.frames)
    for (const n of layout.nodes)
      if (isDesc(n.id, f.id)) inFrame.set(n.id, f.id);
  const loose = layout.nodes.filter(n => !inFrame.has(n.id));
  const nodeLine = n => {
    const sh = n.outputs.length ? '<br/>' + shp(n.outputs[0]) : '';
    if (n.kind === 'merge')
      return `  ${mid(n.id)}(["${q(n.name)}${sh}"]):::op`;
    const ps = n.params > 0 ? '<br/>' + fmt(n.params) + ' params' : '';
    return `  ${mid(n.id)}["${q(n.name)}<br/><i>${q(n.cls)}</i>${sh}${ps}"]:::mod`;
  };
  loose.forEach(n => decl.push(nodeLine(n)));
  for (const f of layout.frames) {
    const mine = layout.nodes.filter(n => inFrame.get(n.id) === f.id);
    if (!mine.length) continue;
    decl.push(`  subgraph ${mid(f.id)}["${q(f.node.name + ' · ' + f.node.cls)}"]`);
    mine.forEach(n => decl.push('  ' + nodeLine(n)));
    decl.push('  end');
  }
  for (const o of layout.outs) {
    const label = (layout.outs.length > 1 ? 'output ' + o.index : 'output')
      + ' ' + shp(o.tensor);
    decl.push(`  ${mid(o.id)}(["${q(label)}"]):::io`);
  }
  L.push(...decl);
  // ``-.->`` marks a skip/branch, matching the dashed orange edge on canvas.
  for (const e of layout.edges) {
    const arrow = e.skip ? '-.->' : '-->';
    const lbl = e.tensor && e.tensor.shape && e.tensor.shape.length
      ? `|"${q(shp(e.tensor))}"|` : '';
    L.push(`  ${mid(e.src)} ${arrow}${lbl} ${mid(e.dst)}`);
  }
  save(L.join('\\n') + '\\n', slug + '.mmd', 'text/plain');
};

document.getElementById('title').textContent = DATA.model_name;
document.getElementById('subtitle').textContent =
  DATA.nodes.length + ' modules \\u00B7 ' + fmt(DATA.total_params) + ' params';
setLevel(1);
addEventListener('resize', fit);
</script>
</body>
</html>
"""


def _dagre_source() -> str:
    """The vendored dagre bundle, inlined so the page stays self-contained.

    Layered graph drawing is not something to hand-roll: the part that keeps a
    deep model readable is inserting a virtual node per rank a long edge spans,
    so boxes get placed *out of the edge's way* rather than under it. dagre is
    what Mermaid itself uses, and vendoring it keeps the output a single file
    with no CDN dependency.
    """
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "dagre.min.js")
    with open(here, encoding="utf-8") as fh:
        return fh.read()


def render_html(graph: Graph, title: Optional[str] = None) -> str:
    """Return a standalone interactive HTML document for ``graph``."""
    payload = json.dumps(graph.to_dict(), separators=(",", ":"))
    # Guard against a literal "</script>" inside the JSON ending the block.
    payload = payload.replace("</", "<\\/")
    doc = _TEMPLATE.replace("__DATA__", payload)
    doc = doc.replace("__INPUT__", str(INPUT_NODE))
    doc = doc.replace("__IN_BASE__", str(IN_BASE))
    # Substituted last: the bundle is ~96KB of minified JS and must not be
    # scanned for the other placeholders.
    doc = doc.replace("/*__DAGRE__*/", _dagre_source())
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
