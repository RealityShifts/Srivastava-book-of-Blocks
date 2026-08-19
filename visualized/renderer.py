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
  --card:#252a38; --cardln:#4a5268; --frame:#1f2431; --frameln:#7b87a8;
  /* Nested frames step towards this tone, one level at a time. */
  --framestep:#aab4d0;
  /* Group titles and their borders carry the structure of the graph, so they
     are held at full strength rather than at the muted tone used for
     secondary text - dimming them read as "disabled" rather than "quiet". */
  --flab:#f2f4f9;
}
@media (prefers-color-scheme: light) {
  :root { --bg:#ffffff; --panel:#fff; --line:#e2e5ec; --fg:#1f2430;
          --muted:#5f6880; --edge:#8a93a6; --accent:#5b6fd6;
          /* Mermaid's signature pale-lavender node fill. */
          --card:#eceffc; --cardln:#9aa5d8; --frame:#f6f7fd;
          --frameln:#7b86bd; --flab:#141824; --framestep:#2b3355; }
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
/* ``#sidetog`` used to float at the top right; it lives in the Show menu now,
   so it must not keep an absolute position - that pulled it out of the popup
   and stacked it on top of the other items. */
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
  flex-wrap:wrap; align-items:center; z-index:30; }
#lvl { font-size:11px; color:var(--muted); padding-left:4px; }
button { font:inherit; font-size:12px; padding:5px 10px; border-radius:6px;
  border:1px solid var(--line); background:var(--panel); color:var(--fg);
  cursor:pointer; }
button:hover { border-color:var(--accent); }
/* Dropdown menus. Each .menu holds a trigger and an absolutely-placed popup
   that only exists while open, so the toolbar stays one row regardless of how
   many commands hang off it. */
.menu { position:relative; }
.mbtn[aria-expanded="true"] { border-color:var(--accent); }
.mpop { display:none; position:absolute; top:calc(100% + 4px); left:0;
  min-width:170px; padding:4px; z-index:20;
  background:var(--panel); border:1px solid var(--line);
  border-radius:8px; box-shadow:0 8px 24px rgba(0,0,0,.32); }
.mpop.open { display:block; }
/* Items fill the popup and read as a list, not a row of chips. */
.mpop button { display:block; width:100%; text-align:left; border:0;
  background:none; padding:6px 8px; border-radius:5px; }
.mpop button:hover { background:var(--accent); color:var(--bg); }
.msep { height:1px; margin:4px 2px; background:var(--line); }
/* A toggle that is currently off loses its check and dims a little. */
.mpop button[aria-pressed="false"], .mpop button[aria-expanded="false"] {
  color:var(--muted); }
#legend { position:absolute; bottom:12px; left:12px; font-size:11px;
  color:var(--muted); background:var(--panel); border:1px solid var(--line);
  border-radius:6px; padding:8px 10px; z-index:5; }
#legend div { display:flex; align-items:center; gap:6px; margin:2px 0; }
#legend i { width:16px; height:0; border-top:2px solid var(--edge);
  display:inline-block; flex:none; }
#legend.off { display:none; }
/* The class key can run long on a deep model; cap it and let it scroll rather
   than letting the legend grow past the top of the canvas. */
#classkey { max-height:34vh; overflow-y:auto; }
.node rect { stroke-width:1.2px; }
.node { cursor:pointer; }
.node text { pointer-events:none; }
/* The class hue is on the card border; the 3px left rule repeats it so the
   grouping still reads at low zoom, where a 1.4px outline thins to nothing. */
.accent { stroke-width:0; }
.nm { font-weight:600; font-size:12px; }
.nc { font-size:10.5px; fill:var(--muted); }
.ns { font-size:10px; font-family:ui-monospace,Menlo,monospace;
  fill:var(--muted); }
/* Only the frame's stroke and label are clickable, so a click on the empty
   space inside a group does not collapse it. */
.flab { font-size:10.5px; font-weight:600; letter-spacing:.02em;
  cursor:pointer; }
/* The frame body is the group's drag surface, so it needs fill events - the
   old ``pointer-events:stroke`` made only the 1.6px border clickable. */
.frame rect { cursor:pointer; }
.fbody { cursor:grab; }
.fbody:active { cursor:grabbing; }
.node.sel rect, .outnode.sel rect, .innode.sel rect {
  stroke:var(--hi); stroke-width:2.5px; }
.node.dim, .outnode.dim, .innode.dim { opacity:.28; }
.innode { cursor:pointer; }
.innode text { pointer-events:none; }
/* Keyboard focus needs a marker distinct from selection: arrowing through the
   graph moves focus, and without this the cursor is invisible on a dimmed node. */
.node.kb rect, .outnode.kb rect, .innode.kb rect {
  stroke:var(--hi); stroke-dasharray:4 3; }
/* Explicit expand control on a container card - replaces the double-click,
   which was undiscoverable and fought with click-to-select. */
.xbtn { cursor:pointer; }
.xbtnt { font-size:12px; font-weight:700; pointer-events:none; }
/* Per-node drag grip: a strip at the card's left edge. Keeping drag on its own
   target means a plain click stays a plain click - selection and the
   incoming/outgoing highlight never have to be disambiguated from a drag. */
.grip { cursor:grab; }
.grip:hover { fill:var(--accent) !important; opacity:.18; }
/* Dragging switched off: grips and group bodies stop advertising a grab, and
   the canvas reads as a plain pan surface. */
#stage.nodrag .grip, #stage.nodrag .fbody { cursor:default; }
#stage.nodrag .grip:hover { fill:transparent !important; }
#stage.nodrag .gripdot { opacity:.35; }
.gripdot { fill:var(--muted); pointer-events:none; }
.node:hover .gripdot, .outnode:hover .gripdot, .innode:hover .gripdot {
  fill:var(--accent); }
/* Floating panels. The header is the drag handle; the body is a solid surface
   so the graph underneath does not show through and confuse the reading. */
/* Expanded-group controls, drawn on the frame's title strip. The handle is a
   distinct target so dragging a group never fights click-to-select. */
.ghandle { fill:var(--accent); opacity:.10; cursor:grab; }
.ghandle:hover { opacity:.24; }
.ghandle:active { cursor:grabbing; }
/* The title rides on the strip and drags with it. */
.flab { cursor:grab; }
.gbtn { font-size:11px; font-weight:700; fill:var(--muted);
  cursor:pointer; user-select:none; }
.gbtn:hover { fill:var(--accent); }
.edge { fill:none; stroke:var(--edge); stroke-width:1.4px;
  stroke-linecap:round; }
.edge.skip { stroke:var(--skip); stroke-dasharray:5 4; }
.edge.toout { stroke:var(--out); }
.outnode rect { transition:stroke-width .1s; }
.edge.hot { stroke:var(--hi); stroke-width:2.6px; }
.edge.dim { opacity:.12; }
/* Shape labels on the wires. ``paint-order:stroke`` lays the halo down before
   the glyphs, so the outline never eats into the letterforms. */
.elab { font-family:ui-monospace,Menlo,monospace; font-size:9.5px;
  fill:var(--muted); pointer-events:none;
  paint-order:stroke; stroke:var(--bg); stroke-width:2.5px;
  stroke-linejoin:round; }
.elab.skip { fill:var(--skip); }
.elabbg { fill:var(--bg); opacity:.72; pointer-events:none; }
.elab.hot { fill:var(--hi); }
.elab.dim, .elabbg.dim { opacity:.1; }
#stage.nolabels .elab, #stage.nolabels .elabbg { display:none; }
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
  <!-- ``nodrag`` matches ``dragEnabled = false``: dragging starts switched off,
       so grips must not advertise a grab until it is turned on. -->
  <div id="stage" class="nodrag">
    <div id="toolbar">
      <!-- Grouped into menus: eleven flat controls read as a wall of buttons,
           and the ones you reach for most (zoom, fit) were lost among the ones
           you touch once. Every id is unchanged, so the handlers below bind
           exactly as before. -->
      <div class="menu">
        <button class="mbtn" data-menu="m-view">View &#9662;</button>
        <div class="mpop" id="m-view">
          <button id="fit">Fit to window</button>
          <button id="zi">Zoom in</button>
          <button id="zo">Zoom out</button>
          <div class="msep"></div>
          <button id="dirtog"
            title="Lay the model out vertically or horizontally">
            &darr; Vertical</button>
        </div>
      </div>
      <div class="menu">
        <button class="mbtn" data-menu="m-depth">Depth &#9662;</button>
        <div class="mpop" id="m-depth">
          <button id="exp">Expand a level</button>
          <button id="col">Collapse a level</button>
          <div class="msep"></div>
          <button id="reset"
            title="Unpin dragged nodes and restore the automatic layout">
            Reset layout</button>
        </div>
      </div>
      <div class="menu">
        <button class="mbtn" data-menu="m-show">Show &#9662;</button>
        <div class="mpop" id="m-show">
          <button id="shapes" title="Show/hide the shape on every connection"
            aria-pressed="true">&#10003; Shapes on edges</button>
          <button id="legtog" title="Show/hide the legend and colour key"
            aria-pressed="true">&#10003; Legend</button>
          <button id="sidetog" title="Show/hide the details panel"
            aria-expanded="true">&#10003; Details panel</button>
          <div class="msep"></div>
          <button id="dragtog" aria-pressed="false"
            title="Allow nodes and groups to be dragged out of the automatic layout">
            &#8199; Dragging</button>
        </div>
      </div>
      <div class="menu">
        <button class="mbtn" data-menu="m-exp">Export &#9662;</button>
        <div class="mpop" id="m-exp">
          <button id="svg-dl" title="Download the current view as SVG">
            SVG image</button>
          <button id="mmd-dl"
            title="Download Mermaid source for the current view">
            Mermaid source</button>
        </div>
      </div>
      <span id="lvl"></span>
    </div>
    <svg id="svg"><g id="root"></g></svg>
    <div id="legend">
      <div><i></i> dataflow</div>
      <div><i style="border-color:var(--skip);border-top-style:dashed"></i>
           skip / residual</div>
      <div><i style="border-color:var(--out)"></i> model output</div>
      <div id="classkey"></div>
      <div style="margin-top:4px">scroll = zoom &middot; drag / arrows = pan
           &middot; click = inspect</div>
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

// ---------- palette by shared weights -------------------------------------
// Colour marks *weight sharing*, not module type. A module's ``path`` is its
// attribute path in the model, so one path appearing on two cards means the
// same object - the same parameters - was called twice. In ``Synthesis`` that
// is ``refernce_modulator.N``, invoked once with the reference style and again
// with the driver style; those two cards are one set of weights and now read as
// one colour.
//
// Colouring by class was the earlier behaviour and answers a different
// question: it groups ``FeatResBlock`` with ``FeatResBlock`` even though each
// carries its own independent parameters. Modules with untied weights are left
// neutral here, so a tinted border means exactly one thing - this block's
// weights appear somewhere else in the graph.
const RAMP = [
  '#6ea8fe', '#e06c9f', '#8bd450', '#c792ea', '#f0883e', '#5ec8c0',
  '#f5c542', '#b58cf0', '#4dd0e1', '#9ccf5a', '#e8944a', '#7fb3f5',
  '#d47ba8', '#4db8a8', '#d4a24c', '#8fa6f0', '#6fc4e8', '#c98ad6',
];
const NEUTRAL = '#7c8296';

// How many times each path was invoked, and how many parameters it owns. A
// path called once is not shared; a path owning no parameters (a bare wrapper,
// a reshape-only block) has no weights to tie, so repeating it means only that
// a structural container ran twice - not weight reuse worth flagging.
const _uses = new Map();
const _pparams = new Map();
for (const n of DATA.nodes) {
  if (n.kind === 'merge') continue;
  _uses.set(n.path, (_uses.get(n.path) || 0) + 1);
  _pparams.set(n.path, n.params);
}
// Deal hues in first-appearance order, which is forward-pass order, so the
// assignment is deterministic rather than dependent on hash luck.
const _hue = new Map();
for (const n of DATA.nodes) {
  if (n.kind === 'merge' || _hue.has(n.path)) continue;
  if ((_uses.get(n.path) || 0) < 2 || !(_pparams.get(n.path) > 0)) continue;
  _hue.set(n.path, RAMP[_hue.size % RAMP.length]);
}
// Keyed on path, so ``colorOf`` takes the node rather than a class string.
const colorOf = n => (n && _hue.get(n.path)) || NEUTRAL;
const isTied = n => !!(n && _hue.has(n.path));

// Colour key: one row per shared-weight group, with how many times it is
// called. Nothing tied means nothing to explain, so the section stays out of
// the way entirely rather than printing an empty heading.
(() => {
  const box = document.getElementById('classkey');
  if (!box) return;
  const byPath = new Map(DATA.nodes.filter(n => n.kind !== 'merge')
    .map(n => [n.path, n]));
  const rows = [..._hue.keys()].map(p => [p, byPath.get(p), _uses.get(p)])
    .sort((a, b) => b[2] - a[2] || a[0].localeCompare(b[0]));
  if (!rows.length) return;
  box.innerHTML = '<div style="margin-top:6px;opacity:.75">shared weights</div>'
    + rows.map(([path, n, k]) =>
      `<div><i style="border-top-width:6px;border-color:${colorOf(n)}"></i>`
      + `${path} \\u00d7${k}</div>`).join('')
    + '<div style="margin-top:2px;opacity:.6">grey = weights not shared</div>';
})();
const fmt = n => n >= 1e9 ? (n/1e9).toFixed(2)+'B'
              : n >= 1e6 ? (n/1e6).toFixed(2)+'M'
              : n >= 1e3 ? (n/1e3).toFixed(1)+'K' : String(n);
// FLOPs span a far wider range than parameter counts, so they get their own
// scale. -1 is the "counter did not run" sentinel and renders as a dash.
const fmtFlops = n => n < 0 ? '\\u2014'
              : n >= 1e12 ? (n/1e12).toFixed(2)+' TFLOP'
              : n >= 1e9  ? (n/1e9).toFixed(2)+' GFLOP'
              : n >= 1e6  ? (n/1e6).toFixed(2)+' MFLOP'
              : n >= 1e3  ? (n/1e3).toFixed(1)+' KFLOP'
              : n + ' FLOP';
const shp = t => '(' + t.shape.join(', ') + ')';
// The parameter name this pill stands for, from ``__call__``'s signature.
// Falls back to positional labelling for a model whose source is unavailable.
const IN_NAMES = DATA.input_names || [];
const inName = n => IN_NAMES[n.index]
  || (DATA.input_tensors.length > 1 ? `input ${n.index}` : 'input');

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

// ---------- expanded groups ----------------------------------------------
// An expanded container stays *in* the graph: its children are laid out by the
// same dagre pass as everything else, wired to the rest of the model by their
// real edges, and wrapped in a frame. What this map adds is per-group state -
// currently the flow direction its contents are arranged in, so one block can
// run left-to-right inside a top-to-bottom model.
const groupDir = new Map();   // id -> 'TB' | 'LR'
// Accumulated drag offset per expanded group. Dragging a group's frame moves
// every member by the same delta, so the block travels as one piece and keeps
// the internal arrangement dagre gave it.
const groupOffset = new Map();   // id -> {dx, dy}

// Has this node been placed by hand - pinned directly, or carried by a dragged
// group? Both the layout (which drops dagre's stale waypoints) and the draw
// step (which switches to a straight line) need the same answer.
const wasMoved = id => {
  if (pinned.has(id)) return true;
  for (const gid of groupOffset.keys())
    if (id === gid || isDesc(id, gid)) return true;
  return false;
};

// Nodes pinned by dragging. dagre still lays out everything else; a pinned node
// is simply written back to its stored point afterwards, so the untouched part
// of the graph keeps arranging itself around what you have placed by hand.
const pinned = new Map();   // id -> {x, y}

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
// Object-drag state (a group or a pinned node). Declared up here because the
// canvas-pan handler below has to test it, and it is installed earlier in the
// file than the drag handlers themselves.
let drag = null, dragMoved = false;
// Whether nodes and groups can be dragged at all. Off by default: reading the
// graph is the common case, and dragging trades away dagre's edge routing for
// the nodes it touches, so it should be something you opt into rather than
// something a stray press can trigger.
let dragEnabled = false;
// Flow direction handed to dagre: 'TB' stacks the model top-to-bottom, 'LR'
// runs it left-to-right. A deep sequential model is far easier to read wide on
// a landscape screen, so this is a per-view choice rather than a fixed one.
let RANKDIR = 'TB';

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
  const sameShape = (a, b) =>
    a && b && a.shape && b.shape && a.shape.length === b.shape.length
    && a.shape.every((v, i) => v === b.shape[i]);
  const resolve = (id, wantLast, hint, tsr) => {
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
        const linkTest = (c, shapeToo) => DATA.edges.some(e => (wantLast
          ? (e.src === c && far.has(e.dst))
          : (e.dst === c && far.has(e.src)))
          && (!shapeToo || sameShape(e.tensor, tsr)));
        // Several edges can join one collapsed container to another (a
        // feature pyramid), and each must land on the descendant that
        // consumes *its* tensor - otherwise all of them collapse onto the
        // first descendant and dedup keeps a single, mislabelled arrow.
        // Match by shape first; fall back to any recorded link.
        let linked = tsr ? ds.filter(c => linkTest(c, true)) : [];
        if (!linked.length) linked = ds.filter(c => linkTest(c, false));
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
    s = isInput(s) ? s : resolve(s, true, d0, e.tensor);  // producer: last leaf
    d = resolve(d, false, s0, e.tensor);                  // consumer: first leaf
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
  // Left-to-right needs a wider rank gap: in TB the gap separates rows of
  // 54px-tall cards, but in LR it separates columns of 190px-wide ones, and 34
  // leaves the edge labels of adjacent ranks overlapping.
  G.setGraph({ rankdir: RANKDIR, nodesep: GAPX,
               ranksep: RANKDIR === 'LR' ? 80 : 34,
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
  //
  // A waypoint list describes the path to where dagre *put* a node. Once an
  // endpoint has been moved by hand - pinned, or carried by a dragged group -
  // those waypoints lead to the old location, so the wire ran out to the
  // vacated spot and only then cut across to the node's new position. Drop the
  // route in that case and let the direct curve be drawn between the live
  // endpoints instead.
  const routed = new Map();
  for (const e of G.edges()) {
    const pts = (G.edge(e).points || []).map(q => ({ x: q.x, y: q.y }));
    if (wasMoved(+e.v) || wasMoved(+e.w)) continue;
    if (pts.length) routed.set(e.v + '>' + e.w, pts);
  }

  // Expanded containers become background *frames* around their descendants
  // rather than nodes in the flow - a box cannot collide with the children
  // it encloses, and the nesting stays legible at any depth.
  // Nesting depth is expressed by *inset*, measured inward from the deepest
  // level, so the outermost frame is the largest and every frame still fits
  // inside the row gap regardless of how deep the tree goes.
  // Per-group flow direction. dagre has no per-cluster ``rankdir``, so a group
  // marked LR inside a TB graph is transposed after the fact: its members are
  // reflected about their own bounding box's diagonal, which turns a column of
  // children into a row while leaving the group where the main layout put it.
  // Positions are what the frame and every edge endpoint read from, so the
  // group stays wired to the rest of the model exactly as before.
  for (const [gid, dir] of groupDir) {
    if (dir === RANKDIR) continue;                 // already that way round
    const members = [];
    for (const n of vis) {
      if (n.id !== gid && isDesc(n.id, gid) && pos.has(n.id))
        members.push(n.id);
    }
    if (members.length < 2) continue;
    const xs = members.map(i => pos.get(i).x), ys = members.map(i => pos.get(i).y);
    const x0 = Math.min(...xs), y0 = Math.min(...ys);
    // Transpose about the group origin, then rescale so cards keep their real
    // width: a plain swap would compress a 190px-wide card into a 54px slot.
    const sx = (NW + GAPX) / (NH + 30), sy = 1 / sx;
    for (const id of members) {
      const p = pos.get(id);
      const dx = p.x - x0, dy = p.y - y0;
      pos.set(id, { x: x0 + dy * sx, y: y0 + dx * sy });
    }
    // The routed polylines inside the group are stale after a transpose; drop
    // them so those edges fall back to direct curves between live endpoints.
    const mset = new Set(members);
    for (const key of [...routed.keys()]) {
      const [a, b] = key.split('>').map(Number);
      if (mset.has(a) || mset.has(b)) routed.delete(key);
    }
  }

  // A dragged *group* moves all its members together, so the block keeps its
  // internal arrangement and stays a single object. Applied before the frames
  // are measured, so each frame is drawn around where its contents ended up.
  for (const [gid, off] of groupOffset) {
    for (const n of vis) {
      if (n.id !== gid && isDesc(n.id, gid) && pos.has(n.id)) {
        const p = pos.get(n.id);
        pos.set(n.id, { x: p.x + off.dx, y: p.y + off.dy });
      }
    }
  }


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

  // A dragged node keeps the point you dropped it at. Applied here - after
  // dagre *and* after the input/output pills have been written back - so a
  // pinned pill is not overwritten by its freshly computed position, which is
  // what stopped output pills from staying where they were dropped.
  for (const [id, p] of pinned) {
    if (pos.has(id)) pos.set(id, { x: p.x, y: p.y });
  }

  // Normalize so the whole drawing starts at a small positive margin. Skipped
  // once anything is pinned: the shift is recomputed on every relayout, and
  // applying it to a hand-placed node would slide it a little further each
  // time something else changed - the node would not stay where it was dropped.
  const held = pinned.size || groupOffset.size;
  const dx = held ? 0 : 24 - Math.min(...[...pos.values()].map(p => p.x),
                           ...frames.map(f => f.x));
  const dy = held ? 0 : 24 - Math.min(...[...pos.values()].map(p => p.y),
                           ...frames.map(f => f.y));
  if (dx || dy) {
    pos.forEach(p => { p.x += dx; p.y += dy; });
    frames.forEach(f => { f.x += dx; f.y += dy; });
    // Routed polylines live in the same coordinate space, so they shift too.
    routed.forEach(pts => pts.forEach(q => { q.x += dx; q.y += dy; }));
  }

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

// Nesting depth as tone. Each level is mixed a fixed step further towards
// ``--framestep`` (lighter in dark mode, darker in light), so a frame inside a
// frame separates from its parent without needing transparency - which is what
// used to bleed the borders through and make them look dim.
const frameFill = d => `color-mix(in srgb, var(--framestep) ` +
  `${Math.min(d, 4) * 9}%, var(--frame))`;

// Rough width of a frame title, used only to decide whether two titles would
// actually overlap horizontally - ~5.6px per char at 10.5px semibold.
const labText = f => `${f.node.name}  ·  ${f.node.cls}  ·  ${fmt(f.node.params)}`;
const labW = f => 10 + labText(f).length * 5.6;

function draw() {
  root.textContent = '';
  const { pos } = layout;

  const gF = el('g'), gFL = el('g'), gE = el('g'), gEL = el('g'), gN = el('g');
  // Frame panels sit behind edges and nodes. Their labels go in a layer of
  // their own: frames are emitted parent-first, so a nested frame's translucent
  // panel would otherwise paint over the enclosing frame's title and wash it
  // out - the titles looked dimmed when they were merely covered.
  // Shape labels ride above every edge (``gEL``) so a crossing wire cannot be
  // drawn over the text, but below nodes, which own the foreground.
  root.append(gF, gFL, gE, gEL, gN);

  // Shallowest first, so a nested frame paints *over* its parent. Now that the
  // body is the drag surface, the deepest frame under the cursor has to be the
  // one that receives the press - otherwise an outer group covers its children
  // and dragging an inner block is impossible.
  const framesByDepth = [...layout.frames].sort(
    (a, b) => (a.node.depth || 0) - (b.node.depth || 0));
  for (const f of framesByDepth) {
    const g = el('g', { class: 'frame' });
    g.dataset.id = f.id;
    // Mermaid draws a subgraph as a quiet solid panel with a plain border - no
    // dashes, no per-group hue - so nesting reads as depth rather than as noise.
    // Each level is stepped a little further from the page background: an
    // opaque fill alone made a nested frame vanish into its parent, and a
    // translucent one washed the borders out. Tone carries the nesting, the
    // stroke keeps every edge crisp.
    let d = 0;
    for (const o of layout.frames) if (o !== f && isDesc(f.id, o.id)) d++;
    // The frame's whole body is the drag surface. A title strip alone was a
    // 20px band that shrank with zoom - accurate to aim at only when zoomed in.
    // The panel already spans the group, so making it the handle costs nothing
    // and turns a sliver into thousands of square pixels. Cards sit above it
    // and take their own clicks, so this only ever catches the empty space
    // between them.
    const body = el('rect', { x: f.x, y: f.y, width: f.w, height: f.h,
      rx: 8, fill: frameFill(d),
      stroke: 'var(--frameln)', 'stroke-width': 1.6, class: 'fbody' });
    body.dataset.drag = f.id;
    g.appendChild(body);
    gF.appendChild(g);
    // The label carries the frame's id too, so clicking a title still collapses
    // the group now that it no longer lives inside the frame's own <g>.
    const lg = el('g', { class: 'frame' });
    lg.dataset.id = f.id;
    // A child frame often starts within a few px of its parent's top edge,
    // which put the two titles on the same baseline. Step this one down past
    // any ancestor title it would otherwise land on top of.
    let ly = f.y + 13;
    for (const o of layout.frames) {
      if (o === f || !isDesc(f.id, o.id)) continue;
      if (Math.abs(ly - (o.y + 13)) < 11 && f.x < o.x + labW(o))
        ly = o.y + 13 + 12;
    }
    const lab = el('text', { x: f.x + 10, y: ly, class: 'flab',
      fill: 'var(--flab)' });
    lab.textContent = labText(f);
    lg.append(lab);

    // The whole title strip is the drag handle. A small dedicated grip was the
    // obvious design but it is a ~22x14px target that shrinks with zoom, so
    // grabbing a group was fiddly; spanning the frame's full width makes it
    // hard to miss. The buttons sit on top and take their clicks first.
    const tw = labW(f);
    const btnW = 44;
    const strip = el('rect', { x: f.x, y: ly - 13, width: f.w, height: 20,
      rx: 5, class: 'ghandle' });
    strip.dataset.drag = f.id;
    // Behind the label, so the text stays legible and still drags.
    lg.insertBefore(strip, lab);
    lab.dataset.drag = f.id;
    // Grip dots directly after the title, marking the strip as grabbable.
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 3; j++) {
        const dot = el('circle', { cx: f.x + 10 + tw + 4 + i * 5,
          cy: ly - 8 + j * 4, r: 1, class: 'gripdot' });
        dot.dataset.drag = f.id;
        lg.appendChild(dot);
      }
    }
    // Buttons are right-aligned in the strip, clear of the label. The root
    // module gets none: collapsing it would hide the entire model behind a
    // single card, which reads as the graph having vanished.
    const bx = f.x + f.w - btnW;
    if (f.node.depth > 0) {
      const db = el('text', { x: bx + 10, y: ly, 'text-anchor': 'middle',
        class: 'gbtn' });
      db.textContent = (groupDir.get(f.id) || RANKDIR) === 'TB'
        ? '\\u2193' : '\\u2192';
      db.dataset.gdir = f.id;
      const cb = el('text', { x: bx + 30, y: ly, 'text-anchor': 'middle',
        class: 'gbtn' });
      cb.textContent = '\\u2212';
      cb.dataset.gclose = f.id;
      lg.append(db, cb);
    }
    gFL.appendChild(lg);
  }

  for (const e of layout.edges) {
    const a = pos.get(e.src), b = pos.get(e.dst);
    if (!a || !b) continue;
    // Merges are drawn as a circle centred in their slot, so edges meet the
    // circle rather than the (invisible) card bounds.
    const srcMerge = mergeIds.has(e.src), dstMerge = mergeIds.has(e.dst);
    // Where an edge meets a card depends on the flow direction: top-to-bottom
    // it leaves the bottom edge and arrives at the top, left-to-right it leaves
    // the right and arrives at the left. Using the TB anchors in LR mode drew
    // every arrow out of the underside of a box and looped it back.
    const lr = RANKDIR === 'LR';
    const srcH = isInput(e.src) ? 26 : srcMerge ? 40 : NH;
    const dstH = dstMerge ? 40 : NH;
    const x1 = lr ? a.x + NW : a.x + NW / 2;
    const y1 = lr ? a.y + srcH / 2
                  : a.y + (isInput(e.src) ? 26 : srcMerge ? 37 : NH);
    const x2 = lr ? b.x : b.x + NW / 2;
    const y2 = lr ? b.y + dstH / 2 : b.y + (dstMerge ? 3 : 0);
    const my = (y1 + y2) / 2, mxx = (x1 + x2) / 2;
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
    } else if (wasMoved(e.src) || wasMoved(e.dst)) {
      // An endpoint placed by hand. The S-curves below leave a node along its
      // rank axis and arrive along the same axis, which reads well for a
      // top-to-bottom flow but bulges absurdly once a node has been dragged
      // sideways - the wire sets off downward, loops around, and comes back.
      // A straight line says exactly what the connection is.
      d = `M${x1},${y1} L${x2},${y2}`;
    } else if (e.skip && !lr && Math.abs(x2 - x1) < 4 && y2 - y1 > NH) {
      // A residual that returns to the same column would be hidden under the
      // main chain. Bow it out sideways so the bypass is actually visible.
      const bow = NW * 0.62;
      d = `M${x1},${y1} C${x1 - bow},${y1 + 20} ${x2 - bow},${y2 - 20} `
        + `${x2},${y2}`;
    } else if (e.skip && lr && Math.abs(y2 - y1) < 4 && x2 - x1 > NW) {
      // The same bypass, rotated: in LR a residual runs along its own row, so
      // bow it vertically instead or it hides under the chain.
      const bow = NH * 1.1;
      d = `M${x1},${y1} C${x1 + 20},${y1 - bow} ${x2 - 20},${y2 - bow} `
        + `${x2},${y2}`;
    } else if (lr) {
      d = `M${x1},${y1} C${mxx},${y1} ${mxx},${y2} ${x2},${y2}`;
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

    // The shape travelling this wire, written on the wire itself. Placed with
    // ``getPointAtLength`` on the path just built rather than at the midpoint
    // of the endpoints: a routed edge detours around intervening boxes, so the
    // straight-line middle can sit far off the visible curve - or on top of a
    // node the edge was routed around.
    if (!e.tensor || !e.tensor.shape) continue;
    let mx = (x1 + x2) / 2, myy = (y1 + y2) / 2;
    try {
      const L = p.getTotalLength();
      if (L > 0) {
        const q = p.getPointAtLength(L * 0.5);
        mx = q.x; myy = q.y;
      }
    } catch (err) { /* not laid out yet: fall back to the chord midpoint */ }

    const txt = shp(e.tensor);
    const lab = el('text', {
      x: mx, y: myy, 'text-anchor': 'middle', 'dominant-baseline': 'middle',
      class: 'elab' + (e.skip ? ' skip' : ''),
    });
    lab.textContent = txt;
    // A halo behind the glyphs keeps them readable where the label lands on
    // the wire itself; ``paint-order`` draws the stroke first so it never
    // thickens the visible letterforms.
    const bg = el('rect', {
      x: mx - (txt.length * 3.15 + 4), y: myy - 8,
      width: txt.length * 6.3 + 8, height: 16, rx: 4,
      class: 'elabbg',
    });
    bg.dataset.src = e.src; bg.dataset.dst = e.dst;
    lab.dataset.src = e.src; lab.dataset.dst = e.dst;
    gEL.appendChild(bg);
    gEL.appendChild(lab);
  }

  // one pill per model input. Carries ``innode`` + ``dataset.id`` so the click
  // handler and keyboard navigation treat it as a real, selectable node - it
  // had neither before, which is why an input could not be clicked.
  for (const n of layout.ins) {
    const ip = pos.get(n.id);
    if (!ip) continue;
    const gi = el('g', { class: 'innode',
      transform: `translate(${ip.x},${ip.y})` });
    gi.dataset.id = n.id;
    gi.appendChild(el('rect', { width: NW, height: 26, rx: 13,
      fill: 'var(--panel)', stroke: 'var(--accent)' }));
    const it = el('text', { x: NW / 2, y: 17, 'text-anchor': 'middle',
      class: 'nc', fill: 'var(--accent)' });
    it.textContent = `${inName(n)} ${n.tensor ? shp(n.tensor) : ''}`;
    gi.appendChild(it);
    gi.appendChild(pillGrip(n.id, 26));
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
    g.appendChild(pillGrip(o.id, 30));
    gN.appendChild(g);
  }

  for (const n of layout.nodes) {
    const p = pos.get(n.id);
    gN.appendChild(drawCard(n, p));
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

// Drag grip for an input/output pill. Pills are rounded and short, so the grip
// sits at the left end inside the curve rather than spanning the full height.
function pillGrip(id, h) {
  const g = el('g');
  const r = el('rect', { x: 2, y: 0, width: 14, height: h,
    class: 'grip', fill: 'transparent' });
  r.dataset.ndrag = id;
  g.appendChild(r);
  for (let i = 0; i < 3; i++) {
    const d = el('circle', { cx: 9, cy: h / 2 - 5 + i * 5, r: 1.1,
      class: 'gripdot' });
    d.dataset.ndrag = id;
    g.appendChild(d);
  }
  return g;
}

// One module card. Shared by the main graph and by panel contents, so a node
// looks and behaves identically wherever it is drawn.
function drawCard(n, p) {
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
      // Array ops (leaky_relu, concat, a residual +) are real steps in the
      // forward pass and move like any other node, so they get a grip too.
      // Theirs rides the left of the pill rather than a card edge.
      const mg = el('rect', { x: cx - w / 2 - 3, y: cy - h / 2,
        width: 12, height: h, class: 'grip', fill: 'transparent' });
      mg.dataset.ndrag = n.id;
      g.appendChild(mg);
      for (let i = 0; i < 3; i++) {
        const dot = el('circle', { cx: cx - w / 2 + 2, cy: cy - 5 + i * 5,
          r: 1.1, class: 'gripdot' });
        dot.dataset.ndrag = n.id;
        g.appendChild(dot);
      }
      return g;
    }
    const tied = isTied(n);
    const col = n.error ? 'var(--err)' : colorOf(n);
    // A tied module wears its group's colour on the border and the left rule;
    // an untied one keeps the quiet default, so colour on the canvas always
    // means shared weights rather than decoration. The 3px bar repeats the hue
    // because at low zoom a 1.4px outline thins to nothing. An error overrides
    // both, since that must not be subtle.
    g.appendChild(el('rect', { width: NW, height: NH, rx: 6,
      fill: 'var(--card)',
      stroke: n.error ? 'var(--err)' : (tied ? col : 'var(--cardln)'),
      'stroke-width': tied && !n.error ? 2 : 1.4 }));
    if (tied || n.error) {
      g.appendChild(el('rect', { width: 3, height: NH, rx: 1.5, fill: col,
        class: 'accent' }));
    }

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

    // Drag grip: a narrow strip down the left edge of the card. Grabbing it
    // moves (and pins) the node; everywhere else on the card stays a click
    // target, so selection and the neighbour highlight are unaffected.
    const grip = el('rect', { x: 0, y: 0, width: 10, height: NH,
      class: 'grip', fill: 'transparent' });
    grip.dataset.ndrag = n.id;
    g.appendChild(grip);
    for (let i = 0; i < 3; i++) {
      const d = el('circle', { cx: 5, cy: NH / 2 - 5 + i * 5, r: 1.1,
        class: 'gripdot' });
      d.dataset.ndrag = n.id;
      g.appendChild(d);
    }

    // A container shows an explicit expand control rather than relying on a
    // double-click: the gesture was undiscoverable, and it collided with
    // click-to-select. The button expands the module in place.
    if (hasKids(n.id) && n.depth > 0) {
      const open = !collapsed.has(n.id);
      const bg = el('rect', { x: NW - 24, y: NH - 22, width: 18, height: 16,
        rx: 4, class: 'xbtn',
        fill: open ? 'var(--accent)' : 'var(--card)', stroke: 'var(--cardln)' });
      bg.dataset.expand = n.id;
      const tx2 = el('text', { x: NW - 15, y: NH - 10, 'text-anchor': 'middle',
        class: 'xbtnt', fill: open ? 'var(--bg)' : 'var(--muted)' });
      tx2.textContent = open ? '\\u2212' : '+';
      tx2.dataset.expand = n.id;
      g.append(bg, tx2);
    }
    return g;
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

// Canvas panning. Distinct from ``drag`` above, which moves objects: a
// pointerdown that grabbed a panel or a node must not also pan the view, so
// this bails whenever an object drag is in progress.
let pan = null;
svg.addEventListener('mousedown', e => {
  if (drag) return;
  // Only the interactive targets block panning. Pressing on the body of a card
  // used to block it too, which made the canvas feel dead wherever the graph
  // was dense; now a card press pans, its grip drags, and its buttons click.
  const ds = e.target.dataset || {};
  if (ds.drag !== undefined || ds.expand !== undefined
      || ds.ndrag !== undefined || ds.pclose !== undefined
      || ds.pdir !== undefined) return;
  pan = { x: e.clientX - tx, y: e.clientY - ty };
  document.getElementById('stage').classList.add('drag');
});
addEventListener('mousemove', e => {
  if (!pan || drag) return;
  tx = e.clientX - pan.x; ty = e.clientY - pan.y; apply();
});
addEventListener('mouseup', () => {
  pan = null;
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

// Expand a container in place. Its children join the main dagre pass and stay
// wired to the rest of the model by their real edges; the frame drawn around
// them is what makes the group readable as one block.
function groupOpen(id) {
  collapsed.delete(id);
  relayout();
}

function groupClose(id) {
  collapsed.add(id);
  groupDir.delete(id);
  groupOffset.delete(id);
  relayout();
}

svg.addEventListener('click', e => {
  // The explicit +/- control on a card expands or collapses it in place.
  // Checked before selection so the button never doubles as a select.
  const xb = e.target.dataset && e.target.dataset.expand;
  if (xb !== undefined) {
    const id = +xb;
    collapsed.has(id) ? groupOpen(id) : groupClose(id);
    return;
  }
  // Per-group direction toggle, drawn on the frame's title bar.
  const gd = e.target.dataset && e.target.dataset.gdir;
  if (gd !== undefined) {
    const id = +gd;
    const cur = groupDir.get(id) || RANKDIR;
    groupDir.set(id, cur === 'TB' ? 'LR' : 'TB');
    relayout();
    return;
  }
  const gc = e.target.dataset && e.target.dataset.gclose;
  if (gc !== undefined) { groupClose(+gc); return; }
  if (dragMoved) { dragMoved = false; return; }   // a drag, not a click

  // A frame's border/label collapses the group it encloses; a node inside it
  // is matched first, so clicking a child never collapses its parent.
  const g = e.target.closest('.node') || e.target.closest('.outnode')
         || e.target.closest('.innode') || e.target.closest('.frame');
  if (!g) { selected = null; applySelection(); return; }
  const id = +g.dataset.id;
  if (g.classList.contains('frame')) {
    // The frame is a drag surface now, so clicking it must not collapse the
    // group - a press that turns into a drag would otherwise fold the block
    // shut the moment you let go. Collapsing is the explicit - button on the
    // title strip. A plain click on empty group space just clears selection.
    selected = null;
    applySelection();
    return;
  }
  selected = id;
  applySelection();
});

// ---------- dragging ------------------------------------------------------
// Two draggable things: a panel (by its header, moving the whole group) and a
// single node in the main graph (which pins it). Both run off one pointer
// handler so a drag can never be interpreted as a click. State lives at the
// top of the file, next to ``layout``.
svg.addEventListener('pointerdown', e => {
  if (e.button !== 0 || !dragEnabled) return;
  // The card's +/- control and the panel header buttons are click targets, not
  // drag handles. Starting a drag here would capture the pointer and the
  // subsequent ``click`` would be retargeted to the SVG, so the button simply
  // never fired - which is exactly what made expand look broken.
  const ds = e.target.dataset || {};
  if (ds.expand !== undefined || ds.pclose !== undefined
      || ds.pdir !== undefined) return;
  const ph = ds.drag;
  if (ph !== undefined) {
    // A group drags by its frame handle and carries every member with it, so
    // an expanded block behaves as one object rather than a pile of cards.
    const off = groupOffset.get(+ph) || { dx: 0, dy: 0 };
    drag = { kind: 'group', id: +ph, ox: off.dx, oy: off.dy,
             sx: e.clientX, sy: e.clientY };
  } else if (ds.ndrag !== undefined) {
    // Nodes move only by their own grip. Dragging from anywhere on the card
    // meant every click had to be disambiguated from a drag, and capturing the
    // pointer to do that swallowed the click - so selecting a node, and the
    // neighbour highlight it drives, stopped working. An explicit grip keeps
    // the two gestures apart with no guessing.
    const id = +ds.ndrag;
    const p = layout.pos.get(id);
    if (!p) return;
    drag = { kind: 'node', id, ox: p.x, oy: p.y,
             sx: e.clientX, sy: e.clientY };
  } else return;
  dragMoved = false;
  svg.setPointerCapture(e.pointerId);
  e.stopPropagation();
});

svg.addEventListener('pointermove', e => {
  if (!drag) return;
  const dx = (e.clientX - drag.sx) / k, dy = (e.clientY - drag.sy) / k;
  if (Math.abs(dx) + Math.abs(dy) > 3) dragMoved = true;
  if (!dragMoved) return;
  if (drag.kind === 'group') {
    groupOffset.set(drag.id, { dx: drag.ox + dx, dy: drag.oy + dy });
    // Shift the members in the live layout and redraw. Calling relayout() here
    // would re-run dagre on every pointer move, which on a graph this size is
    // slow enough that the group visibly lags the cursor.
    const step = { x: dx - (drag.lx || 0), y: dy - (drag.ly || 0) };
    drag.lx = dx; drag.ly = dy;
    for (const n of DATA.nodes) {
      if (n.id !== drag.id && isDesc(n.id, drag.id) && layout.pos.has(n.id)) {
        const p = layout.pos.get(n.id);
        layout.pos.set(n.id, { x: p.x + step.x, y: p.y + step.y });
      }
    }
    for (const f of layout.frames) {
      if (f.id === drag.id || isDesc(f.id, drag.id)) {
        f.x += step.x; f.y += step.y;
      }
    }
    // Discard the routes touching the moving group, so the wires follow the
    // cursor directly instead of first running out to the vacated position.
    if (layout.routed) {
      for (const key of [...layout.routed.keys()]) {
        const [a, b] = key.split('>').map(Number);
        if (a === drag.id || b === drag.id
            || isDesc(a, drag.id) || isDesc(b, drag.id))
          layout.routed.delete(key);
      }
    }
    draw();
  } else {
    pinned.set(drag.id, { x: drag.ox + dx, y: drag.oy + dy });
    layout.pos.set(drag.id, { x: drag.ox + dx, y: drag.oy + dy });
    // Same as for groups: the recorded route ends at the old position, so it
    // has to go the moment the node starts moving.
    if (layout.routed) {
      for (const key of [...layout.routed.keys()]) {
        const [a, b] = key.split('>').map(Number);
        if (a === drag.id || b === drag.id) layout.routed.delete(key);
      }
    }
    draw();
  }
});

svg.addEventListener('pointerup', e => {
  if (!drag) return;
  if (dragMoved) {
    relayout();          // re-route edges around the newly placed node/group
  }
  try { svg.releasePointerCapture(e.pointerId); } catch (err) {}
  drag = null;
});

// ---------- keyboard navigation ------------------------------------------
// Down/Up follow the dataflow - the edges actually drawn - because that is the
// structure the reader is tracing; falling back to nearest-by-geometry only
// when a node has no edge in that direction (an isolated pill, or the ends of
// the graph). Left/Right move between siblings in the same rank, which is what
// the eye does across parallel branches.
const NAV = { ArrowDown: 'down', ArrowUp: 'up',
              ArrowLeft: 'left', ArrowRight: 'right' };

// Every selectable thing on the canvas, with its drawn position.
function navNodes() {
  const out = [];
  for (const [id, p] of layout.pos) {
    if (mergeIds.has(id) && !byId.has(id)) continue;
    out.push({ id, x: p.x, y: p.y });
  }
  return out;
}

function step(from, dir) {
  const all = navNodes();
  const cur = all.find(n => n.id === from);
  if (!cur) return all.length ? all[0].id : null;

  if (dir === 'down' || dir === 'up') {
    // Successors (or predecessors) along real edges, nearest column first so
    // a fan-out lands on the branch sitting straight ahead.
    const linked = layout.edges
      .filter(e => (dir === 'down' ? e.src : e.dst) === from)
      .map(e => all.find(n => n.id === (dir === 'down' ? e.dst : e.src)))
      .filter(Boolean);
    if (linked.length) {
      linked.sort((a, b) => Math.abs(a.x - cur.x) - Math.abs(b.x - cur.x));
      return linked[0].id;
    }
    // No edge that way: fall back to the closest node in that direction.
    const ahead = all.filter(n => dir === 'down' ? n.y > cur.y : n.y < cur.y);
    if (!ahead.length) return null;
    ahead.sort((a, b) => (Math.abs(a.y - cur.y) - Math.abs(b.y - cur.y))
      || (Math.abs(a.x - cur.x) - Math.abs(b.x - cur.x)));
    return ahead[0].id;
  }

  // Sideways: prefer nodes on the same rank, so Left/Right walks the parallel
  // branches rather than drifting diagonally down the graph.
  const band = all.filter(n => n.id !== from && Math.abs(n.y - cur.y) < NH);
  const side = (band.length ? band : all.filter(n => n.id !== from))
    .filter(n => dir === 'right' ? n.x > cur.x : n.x < cur.x);
  if (!side.length) return null;
  side.sort((a, b) => (Math.abs(a.x - cur.x) - Math.abs(b.x - cur.x))
    || (Math.abs(a.y - cur.y) - Math.abs(b.y - cur.y)));
  return side[0].id;
}

// Scroll the view so a keyboard-selected node stays on screen. Only nudges
// when the node is actually outside the viewport, so arrowing around the
// middle of the graph does not make the canvas twitch.
function reveal(id) {
  const p = layout.pos.get(id);
  if (!p) return;
  const r = svg.getBoundingClientRect();
  const sx = p.x * k + tx, sy = p.y * k + ty;
  const m = 60;
  if (sx < m) tx += m - sx;
  if (sx + NW * k > r.width - m) tx -= sx + NW * k - (r.width - m);
  if (sy < m) ty += m - sy;
  if (sy + NH * k > r.height - m) ty -= sy + NH * k - (r.height - m);
  apply();
}

window.addEventListener('keydown', e => {
  // Never hijack typing, and leave modified keys to the browser.
  const t = e.target;
  if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA'
            || t.isContentEditable)) return;
  // Shift is meaningful here (it switches arrows from panning to stepping the
  // selection), so only the browser-owned modifiers bail out.
  if (e.metaKey || e.ctrlKey || e.altKey) return;

  if (NAV[e.key]) {
    e.preventDefault();
    // Bare arrows pan the canvas - on a graph that is tens of thousands of
    // pixels tall, scrolling around is the thing you do constantly, and
    // dragging for it gets tiring. Shift+arrow keeps the structural walk:
    // stepping selection along the dataflow.
    if (!e.shiftKey) {
      // A screen-space step, so the distance felt is the same at any zoom.
      const d = e.repeat ? 90 : 45;
      const dir = NAV[e.key];
      if (dir === 'left')  tx += d;
      if (dir === 'right') tx -= d;
      if (dir === 'up')    ty += d;
      if (dir === 'down')  ty -= d;
      apply();
      return;
    }
    // Nothing selected yet: start at the first input pill, or the first node.
    if (selected === null) {
      const first = (layout.ins[0] && layout.ins[0].id);
      selected = first !== undefined ? first
        : (navNodes()[0] || {}).id ?? null;
    } else {
      const next = step(selected, NAV[e.key]);
      if (next === null || next === undefined) return;
      selected = next;
    }
    applySelection();
    reveal(selected);
    return;
  }
  if (e.key === 'Escape') { selected = null; applySelection(); return; }
  // Enter/Space opens the selected container into a panel - the keyboard
  // equivalent of its +/- button.
  if ((e.key === 'Enter' || e.key === ' ') && selected !== null
      && hasKids(selected)) {
    e.preventDefault();
    collapsed.has(selected) ? groupOpen(selected) : groupClose(selected);
  }
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
  root.querySelectorAll('.node, .outnode, .innode').forEach(g => {
    const id = +g.dataset.id;
    g.classList.toggle('sel', id === selected);
    g.classList.toggle('dim', selected !== null && !near.has(id));
  });
  // Shape labels share their edge's dataset, so the same test lights the wire
  // and the text it carries - a selected module reads its own shapes at full
  // contrast while the rest of the graph recedes.
  root.querySelectorAll('.edge, .elab, .elabbg').forEach(p => {
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
    if (DATA.total_flops >= 0)
      rows.splice(2, 0, ['FLOPs (forward)', fmtFlops(DATA.total_flops)]);
    box.innerHTML =
      (DATA.error ? `<div class="err"><b>Forward pass failed</b><br>
         <code>${esc(DATA.error)}</code><br>Graph shows progress up to the
         failure.</div>` : '') +
      sec('Model', rows.map(r =>
        `<div class="stat"><span>${r[0]}</span><span>${r[1]}</span></div>`).join('')) +
      sec('Input', DATA.input_tensors.map((t, i) =>
        `<div class="stat"><span>${esc(IN_NAMES[i] || t.dtype)}</span>` +
        `<span>${shp(t)}</span></div>`).join('')) +
      sec('Output', DATA.output_tensors.map(t =>
        `<div class="stat"><span>${t.dtype}</span><span>${shp(t)}</span></div>`).join('')) +
      sec('Tips',
        '<div class="stat"><span>Click node</span><span>trace neighbours</span></div>' +
        '<div class="stat"><span>Drag group body</span><span>move whole block</span></div>' +
        '<div class="stat"><span>+ / &minus; on card</span><span>expand / collapse</span></div>' +
        '<div class="stat"><span>&plusmn; depth</span><span>expand a level</span></div>' +
        '<div class="stat"><span>Arrow keys</span><span>pan the view</span></div>' +
        '<div class="stat"><span>Shift + &darr;&uarr;</span><span>step along dataflow</span></div>' +
        '<div class="stat"><span>Shift + &larr;&rarr;</span><span>step across a rank</span></div>' +
        '<div class="stat"><span>Enter / Space</span><span>expand selected</span></div>' +
        '<div class="stat"><span>Esc</span><span>clear selection</span></div>');
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
  if (isInput(selected)) {            // an input pill, not a module
    const inp = layout.ins.find(x => x.id === selected);
    const fed = layout.edges.filter(e => e.src === selected)
      .map(e => byId.get(e.dst)).filter(Boolean);
    box.innerHTML = sec('Model input',
      `<div class="stat"><span>Name</span>` +
      `<span>${inp ? esc(inName(inp)) : ''}</span></div>` +
      `<div class="stat"><span>Shape</span>` +
      `<span>${inp && inp.tensor ? shp(inp.tensor) : ''}</span></div>` +
      `<div class="stat"><span>dtype</span>` +
      `<span>${inp && inp.tensor ? inp.tensor.dtype : ''}</span></div>`) +
      sec('Feeds', fed.length
        ? fed.map(d => `<div class="stat"><span>${esc(d.cls)}</span>` +
            `<span>${esc(d.path)}</span></div>`).join('')
        : '<div class="stat"><span>nothing</span>' +
          '<span>no consumer traced</span></div>');
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
  if (n.flops >= 0) {
    rows.push(['FLOPs (total)', fmtFlops(n.flops)]);
    rows.push(['FLOPs (own)', fmtFlops(n.own_flops)]);
    if (DATA.total_flops > 0)
      rows.push(['Share of FLOPs',
        (100 * n.flops / DATA.total_flops).toFixed(1) + '%']);
  }
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
  mark('sidetog', !on);
  setTimeout(fit, 200);
};
// `d` toggles it from the keyboard, ignored while typing in a field.
document.addEventListener('keydown', e => {
  if (e.key === 'd' && !e.metaKey && !e.ctrlKey && !e.altKey &&
      !/^(INPUT|TEXTAREA)$/.test(document.activeElement.tagName))
    document.getElementById('sidetog').click();
});

// ---------- toolbar menus -------------------------------------------------
// One popup open at a time; a click anywhere else closes it, and Escape backs
// out. Commands inside close the menu too, so the canvas is never left with a
// popup covering it after an action has run.
const closeMenus = () => {
  document.querySelectorAll('.mpop.open').forEach(p => p.classList.remove('open'));
  document.querySelectorAll('.mbtn').forEach(b =>
    b.setAttribute('aria-expanded', 'false'));
};
document.querySelectorAll('.mbtn').forEach(btn => {
  btn.setAttribute('aria-expanded', 'false');
  btn.onclick = ev => {
    ev.stopPropagation();
    const pop = document.getElementById(btn.dataset.menu);
    const wasOpen = pop.classList.contains('open');
    closeMenus();
    if (!wasOpen) {
      pop.classList.add('open');
      btn.setAttribute('aria-expanded', 'true');
    }
  };
});
document.querySelectorAll('.mpop').forEach(pop => {
  pop.addEventListener('click', ev => {
    if (ev.target.tagName === 'BUTTON') closeMenus();
    ev.stopPropagation();
  });
});
document.addEventListener('click', closeMenus);
document.addEventListener('keydown', ev => {
  if (ev.key === 'Escape') closeMenus();
});

// Keep a toggle's label in step with its state, so the tick in the menu is
// the source of truth for whether the thing is on.
// The label is rebuilt from a stored base rather than by stripping a prefix
// off the previous text: the "off" marker is a figure space, which is itself
// whitespace, so a strip-and-prepend left a growing run of blanks each time
// the item was toggled.
const mark = (id, on) => {
  const b = document.getElementById(id);
  if (!b.dataset.base)
    b.dataset.base = b.textContent.replace(/^[\\u2713\\u2007\\s]+/, '');
  b.textContent = (on ? '\\u2713 ' : '\\u2007 ') + b.dataset.base;
};

document.getElementById('shapes').onclick = () => {
  const off = document.getElementById('stage').classList.toggle('nolabels');
  document.getElementById('shapes').setAttribute('aria-pressed', !off);
  mark('shapes', !off);
};
document.getElementById('legtog').onclick = () => {
  const off = document.getElementById('legend').classList.toggle('off');
  document.getElementById('legtog').setAttribute('aria-pressed', !off);
  mark('legtog', !off);
};
document.getElementById('dragtog').onclick = () => {
  dragEnabled = !dragEnabled;
  document.getElementById('dragtog').setAttribute('aria-pressed', dragEnabled);
  mark('dragtog', dragEnabled);
  // Drives the cursor on grips and frame bodies: with dragging off they should
  // not advertise a grab that will not happen.
  document.getElementById('stage').classList.toggle('nodrag', !dragEnabled);
};
document.getElementById('reset').onclick = () => {
  pinned.clear();
  groupDir.clear();
  groupOffset.clear();
  relayout();
  fit();
};
document.getElementById('dirtog').onclick = () => {
  RANKDIR = RANKDIR === 'TB' ? 'LR' : 'TB';
  document.getElementById('dirtog').innerHTML =
    RANKDIR === 'TB' ? '\\u2193 Vertical' : '\\u2192 Horizontal';
  relayout();
  fit();
};
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
    .outnode rect{stroke-width:1.4px}
    .frame rect{stroke-width:1.6px}
    .accent{stroke-width:0}
    text{font-family:ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
    .nm{font-weight:600;font-size:12px;fill:${c('--fg')}}
    .nc{font-size:10.5px;fill:${c('--muted')}}
    .ns{font-size:10px;font-family:ui-monospace,Menlo,monospace;
        fill:${c('--muted')}}
    .flab{font-size:10.5px;font-weight:600;fill:${c('--flab')}}
    .elab{font-size:9.5px;font-family:ui-monospace,Menlo,monospace;
        fill:${c('--muted')};paint-order:stroke;stroke:${bg};
        stroke-width:2.5px;stroke-linejoin:round}
    .elab.skip{fill:${c('--skip')}}
    .elabbg{fill:${bg};opacity:.72}
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
  DATA.nodes.length + ' modules \\u00B7 ' + fmt(DATA.total_params) + ' params'
  + (DATA.total_flops >= 0 ? ' \\u00B7 ' + fmtFlops(DATA.total_flops) : '');
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
