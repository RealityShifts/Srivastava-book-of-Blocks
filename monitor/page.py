"""The monitor's page: the architecture graph plus monitor controls.

The page *is* :func:`Blocks.visualized.renderer.render_html` - same boxes,
same expand/collapse, same details panel - with one stylesheet and one script
added through its ``extra_head`` / ``extra_body`` slots. The script shares the
renderer's top-level names (``DATA``, ``byId``, ``selected``, ``detail``,
``draw``) and wraps ``draw`` and ``detail`` to add selection marks, time
badges and the result tables.

All requests use relative URLs (``api/status``), so the page works unchanged
behind a reverse proxy that mounts it under a prefix.
"""

from __future__ import annotations

from typing import Optional

from ..visualized._core import Graph, Node
from ..visualized.renderer import render_html


def tree_graph(tree: list, title: str) -> Graph:
    """A :class:`Graph` of the bare module tree - nesting, classes and
    parameter counts, no connections or shapes - for a model that has not
    been traced."""
    ids = {row["path"]: i for i, row in enumerate(tree)}
    nodes = [Node(id=i, path=row["path"], name=row["name"], cls=row["cls"],
                  depth=row["depth"],
                  parent=None if row["parent"] is None else ids[row["parent"]],
                  params=row["params"], own_params=row["own_params"],
                  inputs=[], outputs=[], config={}, order=i)
             for i, row in enumerate(tree)]
    return Graph(nodes=nodes, edges=[], model_name=title,
                 total_params=tree[0]["params"] if tree else 0,
                 input_tensors=[], output_tensors=[])


def build_page(graph: Optional[Graph], tree: list, title: str) -> str:
    return render_html(graph if graph is not None else tree_graph(tree, title),
                       title=f"{title} - monitor",
                       extra_head=_CSS, extra_body=_JS)


_CSS = """<style>
#mon { position:relative; }
#mon-pop { display:none; position:absolute; top:calc(100% + 4px); left:0;
  z-index:30; width:290px; padding:10px 12px; background:var(--panel);
  border:1px solid var(--line); border-radius:8px; font-size:12px;
  color:var(--fg); box-shadow:0 8px 24px rgba(0,0,0,.35); }
#mon-pop.open { display:block; }
#mon-pop .row { display:flex; align-items:center; gap:8px; margin:6px 0; }
#mon-pop .row > span { color:var(--muted); flex:1; }
#mon-pop input[type=number] { width:64px; background:var(--card);
  color:var(--fg); border:1px solid var(--cardln); border-radius:5px;
  padding:3px 6px; font:inherit; }
#mon-pop select { flex:1; min-width:0; background:var(--card); color:var(--fg);
  border:1px solid var(--cardln); border-radius:5px; padding:3px 4px;
  font:inherit; }
#mon-pop button { background:var(--card); color:var(--fg);
  border:1px solid var(--cardln); border-radius:5px; padding:4px 9px;
  font:inherit; cursor:pointer; }
#mon-pop button:hover { border-color:var(--accent); }
#mon-pop button.on { background:var(--out); color:var(--bg);
  border-color:var(--out); }
#mon-pop button.go { background:var(--accent); color:var(--bg);
  border-color:var(--accent); font-weight:600; }
#mon-pop button:disabled { opacity:.45; cursor:default; }
#mon-pop a.dl { background:var(--card); color:var(--fg); text-decoration:none;
  border:1px solid var(--cardln); border-radius:5px; padding:4px 9px;
  cursor:pointer; }
#mon-pop a.dl:hover { border-color:var(--accent); }
#mon-pop a.dl.off { opacity:.45; pointer-events:none; }
#mon-pop hr { border:0; border-top:1px solid var(--line); margin:9px 0; }
#mon-status { color:var(--muted); line-height:1.45; }
#mon-status b { color:var(--fg); font-weight:600; }
#mon-status .bad { color:var(--err); }
.node.monsel rect, .frame.monsel > rect { stroke:var(--out) !important;
  stroke-width:2.6px !important; stroke-dasharray:6 3; }
.montime { font-size:10.5px; font-weight:600;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
.mon-t { margin:0 0 10px; }
.mon-t .hd { display:flex; justify-content:space-between; gap:8px;
  font-size:12px; margin-bottom:3px; }
.mon-t .hd b { font-weight:600; color:var(--fg); word-break:break-all; }
.mon-t .hd span { color:var(--muted); flex:none;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:11px; }
.mon-g { display:grid; grid-template-columns:repeat(5,1fr); gap:2px 6px;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:11px; }
.mon-g i { font-style:normal; color:var(--muted); font-size:10px; }
.mon-g span { color:var(--fg); white-space:nowrap; }
.mon-v { margin-top:3px; font-size:10.5px; color:var(--fg); overflow:hidden;
  white-space:nowrap; text-overflow:ellipsis;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
.mon-v i { font-style:normal; color:var(--muted); }
.mon-h { display:block; width:100%; height:34px; margin-top:4px; }
.mon-h rect { fill:var(--accent); }
.mon-h rect.top { fill:var(--hi); }
.mon-h line.mean { stroke:var(--out); stroke-width:1.6;
  vector-effect:non-scaling-stroke; }
.mon-g i.k-mean { color:var(--out); font-weight:700; }
.mon-g i.k-mode { color:var(--hi); font-weight:700; }
.mon-dot.mean { background:var(--out); border-radius:1px; width:3px;
  height:11px; margin:0 8px 0 3px; vertical-align:-1px; }
.mon-r { display:flex; justify-content:space-between; color:var(--muted);
  font-size:10px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
.mon-g span.l1, .mon-tt td.l1, .mon-r .l1 { color:var(--skip); font-weight:600; }
.mon-g span.l2, .mon-tt td.l2, .mon-r .l2 { color:var(--err); font-weight:700; }
.mon-dot { display:inline-block; width:8px; height:8px; border-radius:50%;
  margin-right:6px; vertical-align:baseline; }
.mon-dot.l1 { background:var(--skip); }
.mon-dot.l2 { background:var(--err); }
.mon-dot.l3 { background:#4cc9f0; }
.mon-g span.l3, .mon-tt td.l3, .mon-r .l3 { color:#4cc9f0; font-weight:600; }
.monflag.l3 { fill:#4cc9f0; }
.mon-dot.mode { background:var(--hi); border-radius:2px; }
.mon-dot.bar { background:var(--accent); border-radius:2px; }
.mon-leg { font-size:11px; line-height:1.7; color:var(--muted); }
.mon-leg b { color:var(--fg); font-weight:600; }
.mon-leg code { font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  color:var(--fg); }
.monflag.l1 { fill:var(--skip); }
.monflag.l2 { fill:var(--err); }
.mon-note { color:var(--muted); font-size:11px; line-height:1.4;
  margin:4px 0 8px; }
.mon-err { color:var(--err); font-size:11px; white-space:pre-wrap;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
.mon-tt { width:100%; border-collapse:collapse; font-size:11px;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
.mon-tt td { padding:2px 0; }
.mon-tt td:first-child { color:var(--fg); overflow-wrap:anywhere;
  width:44%; cursor:pointer; }
.mon-tt td:first-child:hover { color:var(--accent); }
.mon-tt td + td { text-align:right; color:var(--muted); padding-left:8px;
  white-space:nowrap; }
</style>
"""


_JS = r"""<script>
// ---------- Blocks.monitor -------------------------------------------------
(() => {
const MON = { sel: new Set(), mode: false, result: null, status: null,
              loaded: null, follow: true };
const isMod = n => n && n.kind === 'module';
// The tracer labels the model itself '<root>'; its module path is ''.
const pathOf = n => n.path === '<root>' ? '' : n.path;
// A module called twice is two cards with one path; the collector numbers
// its calls in the order they start, which is the cards' ``order``.
const callOf = new Map();
{ const seen = new Map();
  DATA.nodes.filter(isMod)
    .sort((a, b) => (a.order ?? a.id) - (b.order ?? b.id))
    .forEach(n => { const k = seen.get(pathOf(n)) || 0;
                    callOf.set(n.id, k); seen.set(pathOf(n), k + 1); }); }
const firstNode = new Map();
DATA.nodes.filter(isMod).forEach(n => {
  if (!firstNode.has(pathOf(n))) firstNode.set(pathOf(n), n.id); });
const label = p => p === '' ? '<root>' : p;
const num = v => v === null || v === undefined ? '—'
  : v === 0 ? '0'
  : (Math.abs(v) >= 1e4 || Math.abs(v) < 1e-3) ? v.toExponential(1)
  : (+v.toPrecision(3)).toString();
const full = v => v === null || v === undefined ? '' : String(v);
// ---------- which values are too big --------------------------------------
// level 0 fine, 1 large (orange), 2 dangerous (red), 3 too small (blue).
// Judged on the extremes across the request's iterations, against the
// server's thresholds. ``RANK`` orders them worst first.
const RANK = { 2: 3, 1: 2, 3: 1, 0: 0 };
const worse = (a, b) => RANK[a] >= RANK[b] ? a : b;
const KINDS = { act: 'activation', param: 'parameter', grad: 'gradient' };
const limits = kind => (MON.status && MON.status.thresholds || {})[KINDS[kind]]
  || {};
const levelOf = (v, kind) => { const T = limits(kind), a = Math.abs(v);
  return v === null || v === undefined ? 0
    : a >= T.danger ? 2 : a >= T.warn ? 1 : 0; };
function flag(r, kind) {
  if (!r) return { level: 0, why: '' };
  if (r.nonfinite) return { level: 2,
    why: r.nonfinite + ' non-finite value' + (r.nonfinite > 1 ? 's' : '') };
  if (!r.min || !r.max) return { level: 0, why: '' };
  const T = limits(kind);
  const big = Math.max(Math.abs(r.min.min), Math.abs(r.max.max));
  if (big >= T.danger) return { level: 2,
    why: `largest magnitude ${num(big)} ≥ ${num(T.danger)}` };
  if (big >= T.warn) return { level: 1,
    why: `largest magnitude ${num(big)} ≥ ${num(T.warn)}` };
  if (big === 0 && r.numel > 1) return { level: 3, why: kind === 'grad'
    ? 'all zero: no gradient reaches this parameter' : 'all zero' };
  if (T.tiny !== undefined && big < T.tiny) return { level: 3,
    why: `largest magnitude ${num(big)} < ${num(T.tiny)}` +
      (kind === 'grad' ? ' (vanishing gradient)' : ' (too small)') };
  if (r.numel > 1 && r.std && r.std.max === 0) return { level: 3,
    why: 'constant: every value is the same' };
  return { level: 0, why: '' };
}
const dot = f => f.level ? `<i class="mon-dot l${f.level}" ` +
  `title="${esc(f.why)}"></i>` : '';
function findFlags(res) {
  const list = [], per = new Map(), owner = new Map();
  // A parameter sits inside every picked module around it: each of those
  // cards gets the mark, but the list names it once, under the innermost.
  const add = (path, call, name, f) => {
    if (!f.level) return;
    per.set(path, worse(per.get(path) || 0, f.level));
    if (call === null) {
      const seen = owner.get(name);
      if (seen) { if (path.length > seen.path.length) seen.path = path; return; }
      const item = { path, call, name, level: f.level, why: f.why, param: true };
      owner.set(name, item);
      list.push(item);
      return;
    }
    list.push({ path, call, name, level: f.level, why: f.why });
  };
  const P = res.summary.params || {};
  Object.entries(res.summary.modules).forEach(([path, m]) => {
    m.calls.forEach((c, k) => {
      c.inputs.forEach(r => add(path, k, 'input ' + r.name, flag(r, 'act')));
      c.outputs.forEach(r => add(path, k, 'output ' + r.name, flag(r, 'act')));
    });
    (m.params || []).forEach(name => { const r = P[name]; if (!r) return;
      add(path, null, name, flag(r, 'param'));
      if (r.grad) add(path, null, name + ' · grad', flag(r.grad, 'grad'));
    });
  });
  list.forEach(f => { if (f.param && f.path && f.name.startsWith(f.path + '.'))
    f.name = f.name.slice(f.path.length + 1); });
  list.sort((a, b) => RANK[b.level] - RANK[a.level]);
  return { list, per };
}
const legend = () => { const T = MON.status && MON.status.thresholds;
  const row = (k, name) => !T ? '' : `<br>${name}: ` +
    `<span style="color:#4cc9f0">&lt; ${num(T[k].tiny)}</span>, ` +
    `<span style="color:var(--skip)">≥ ${num(T[k].warn)}</span>, ` +
    `<span style="color:var(--err)">≥ ${num(T[k].danger)}</span>`;
  return `<div class="mon-leg">` +
    `<b>In each histogram</b><br>` +
    `<i class="mon-dot mean"></i><b>green line</b>: the mean<br>` +
    `<i class="mon-dot mode"></i><b>yellow bar</b>: the mode, the fullest ` +
    `bin<br>` +
    `<i class="mon-dot bar"></i><b>lavender bars</b>: how many values fall ` +
    `in each bin, from min (left) to max (right)<br>` +
    `The <span style="color:var(--out);font-weight:700">mean</span> and ` +
    `<span style="color:var(--hi);font-weight:700">mode</span> labels above ` +
    `the numbers use the same colours. Median, min and max have no mark: ` +
    `min and max are the two ends of the histogram.<br>` +
    `<b>On a number, a tensor or a card</b><br>` +
    `<i class="mon-dot l3"></i><b>blue</b>: too small - all zero, constant, ` +
    `a collapsed activation, a vanishing gradient<br>` +
    `<i class="mon-dot l1"></i><b>orange</b>: large<br>` +
    `<i class="mon-dot l2"></i><b>red</b>: dangerous (an exploding ` +
    `gradient, a runaway activation), or non-finite (NaN / Inf)<br>` +
    `Judged on the largest magnitude over the request's iterations:` +
    row('activation', 'inputs and outputs') + row('parameter', 'parameters') +
    row('gradient', 'gradients') +
    `<br>These limits are rules of thumb, set by <code>Monitor(thresholds=` +
    `…)</code>. A dot on a card marks a module with a flagged tensor.` +
    `</div>`; };
const pct = (t, total) => !total || t === null || t === undefined ? '—'
  : (100 * t / total).toFixed(t / total < 0.1 ? 1 : 0) + '%';
const ms = s => s === null || s === undefined ? '—'
  : s >= 1 ? s.toFixed(2) + ' s'
  : s >= 1e-3 ? (s * 1e3).toFixed(2) + ' ms'
  : (s * 1e6).toFixed(0) + ' µs';

// ---------- panel ----------------------------------------------------------
const wrap = document.createElement('div');
wrap.className = 'menu'; wrap.id = 'mon';
wrap.innerHTML = `
  <button class="mbtn" id="mon-btn" aria-expanded="false">Monitor &#9662;</button>
  <div id="mon-pop">
    <div class="row"><span>Pick modules on the graph</span>
      <button id="mon-mode">Select: off</button></div>
    <div class="row"><span>All modules at depth</span>
      <input id="mon-depth" type="number" min="0" value="1">
      <button id="mon-depth-go">Add</button></div>
    <div class="row"><span>Every module of the model</span>
      <button id="mon-all" title="Pick all modules, at every depth. On a large model this makes the measurement forward and the result large.">Add all</button></div>
    <div class="row"><span id="mon-count">0 selected</span>
      <button id="mon-clear">Clear</button></div>
    <hr>
    <div class="row"><span>Iterations</span>
      <input id="mon-n" type="number" min="1" max="1000" value="10"></div>
    <div class="row"><label><input id="mon-stats" type="checkbox" checked>
      statistics</label>
      <label><input id="mon-time" type="checkbox" checked> timing</label>
      <span></span><button id="mon-go" class="go">Collect</button></div>
    <div class="row"><label title="List every parameter inside each picked module. Off: only the ones the module owns directly (a container owns none).">
      <input id="mon-deep" type="checkbox" checked>
      parameters of everything inside</label></div>
    <hr>
    <div id="mon-status">connecting&hellip;</div>
    <div class="row"><span>Result</span><select id="mon-results"></select></div>
    <div class="row"><span>Download</span>
      <a id="mon-dl-csv" class="dl" title="One row per tensor and per timing: module, kind, shape, mean, median, mode, min, max, time">Table (CSV)</a>
      <a id="mon-dl-json" class="dl" title="Everything: the summary and every iteration's numbers and histograms">Full (JSON)</a></div>
  </div>`;
const bar = document.getElementById('toolbar');
bar.insertBefore(wrap, document.getElementById('lvl'));
const $ = id => document.getElementById(id);
const pop = $('mon-pop');
const openPanel = on => {
  pop.classList.toggle('open', on);
  $('mon-btn').setAttribute('aria-expanded', on ? 'true' : 'false');
};
$('mon-btn').onclick = e => { e.stopPropagation();
  closeMenus(); openPanel(!pop.classList.contains('open')); };
// The panel stays open while modules are picked on the graph, so it does not
// take part in the other menus' close-on-any-click.
pop.addEventListener('click', e => e.stopPropagation());

// ---------- selection ------------------------------------------------------
function marks() {
  root.querySelectorAll('.node, .frame').forEach(g => {
    const n = byId.get(+g.dataset.id);
    g.classList.toggle('monsel', !!n && isMod(n) && MON.sel.has(pathOf(n)));
  });
  $('mon-count').textContent = MON.sel.size + ' selected';
  root.querySelectorAll('.montime, .monflag').forEach(t => t.remove());
  if (MON.flags) root.querySelectorAll('.node').forEach(g => {
    const n = byId.get(+g.dataset.id);
    const lv = n && isMod(n) ? MON.flags.per.get(pathOf(n)) : 0;
    if (!lv) return;
    const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    c.setAttribute('class', 'monflag l' + lv);
    c.setAttribute('cx', 0); c.setAttribute('cy', 0); c.setAttribute('r', 6);
    const t = document.createElementNS('http://www.w3.org/2000/svg', 'title');
    t.textContent = { 2: 'a dangerous or non-finite value in this module',
      1: 'a large value in this module',
      3: 'a too-small, all-zero or constant value in this module' }[lv];
    c.appendChild(t);
    g.appendChild(c);
  });
  const times = MON.times;
  if (!times) return;
  root.querySelectorAll('.node').forEach(g => {
    const n = byId.get(+g.dataset.id);
    if (!n || !isMod(n) || !times.per.has(pathOf(n))) return;
    const calls = times.per.get(pathOf(n));
    const t = calls[callOf.get(n.id)] ?? null;
    if (t === null) return;
    const heat = times.max > 0 ? Math.sqrt(t / times.max) : 0;
    const s = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    s.setAttribute('class', 'montime');
    s.setAttribute('x', NW - 2); s.setAttribute('y', -5);
    s.setAttribute('text-anchor', 'end');
    s.setAttribute('fill',
      `color-mix(in srgb, var(--err) ${Math.round(heat * 100)}%, var(--out))`);
    s.textContent = ms(t);
    g.appendChild(s);
  });
}
const _draw = draw;
draw = function () { _draw.apply(this, arguments); marks(); };

function toggle(path) {
  MON.sel.has(path) ? MON.sel.delete(path) : MON.sel.add(path);
  marks();
}
svg.addEventListener('click', e => {
  if (!MON.mode) return;
  const d = e.target.dataset || {};
  if (d.expand !== undefined || d.gdir !== undefined || d.gclose !== undefined)
    return;                                   // the card's own controls
  const g = e.target.closest('.node') || e.target.closest('.frame')
         || e.target.closest('[data-id]');
  if (!g) return;
  const n = byId.get(+g.dataset.id);
  if (!isMod(n)) return;
  // Not stopped: the page's own click still runs, so picking a module also
  // opens its details.
  toggle(pathOf(n));
}, true);
$('mon-mode').onclick = () => {
  MON.mode = !MON.mode;
  $('mon-mode').textContent = 'Select: ' + (MON.mode ? 'on' : 'off');
  $('mon-mode').classList.toggle('on', MON.mode);
  document.getElementById('stage').style.cursor = MON.mode ? 'crosshair' : '';
};
$('mon-depth-go').onclick = () => {
  const d = +$('mon-depth').value;
  DATA.nodes.filter(n => isMod(n) && n.depth === d)
    .forEach(n => MON.sel.add(pathOf(n)));
  marks();
};
$('mon-all').onclick = () => {
  DATA.nodes.filter(isMod).forEach(n => MON.sel.add(pathOf(n)));
  marks();
};
$('mon-clear').onclick = () => { MON.sel.clear(); marks(); };

// ---------- requests -------------------------------------------------------
async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body) });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}
$('mon-go').onclick = async () => {
  try {
    await api('api/request', { paths: [...MON.sel], n: +$('mon-n').value,
      stats: $('mon-stats').checked, timing: $('mon-time').checked,
      deep: $('mon-deep').checked });
    MON.follow = true;
    MON.note = null;
    poll();
  } catch (err) { MON.note = String(err.message || err); paint(); }
};
$('mon-results').onchange = () => {
  MON.follow = false; load($('mon-results').value); };

async function load(id) {
  if (!id) return;
  try {
    MON.result = await api('api/results/' + id);
    MON.loaded = id;
    const per = new Map();
    let max = 0;
    Object.entries(MON.result.summary.modules).forEach(([path, m]) => {
      const ts = m.calls.map(c => c.time ? c.time.self.mean : null);
      if (ts.some(t => t !== null)) per.set(path, ts);
      ts.forEach(t => { if (t !== null && t > max) max = t; });
    });
    MON.times = per.size ? { per, max } : null;
    MON.flags = findFlags(MON.result);
    marks(); detail();
  } catch (err) { MON.note = String(err.message || err); }
  paint();
}

function paint() {
  const s = MON.status, box = $('mon-status');
  if (!s) { box.innerHTML = '<span class="bad">server not reachable</span>';
            return; }
  const g = { traced: '', tree: 'module tree only (no traced graph given)',
    untraced: 'graph will be traced on the next iteration',
    pending: 'tracing the graph on the next iteration…',
    failed: '<span class="bad">tracing the graph failed</span>' }[s.graph];
  box.innerHTML =
    `step <b>${s.step === null ? '—' : s.step}</b> &middot; ` +
    (s.active ? `collecting <b>${s.active.id}</b> ` +
                `<b>${s.active.done}/${s.active.n}</b>`
              : 'idle (no extra forward)') +
    (g ? `<br>${g}` : '') +
    (MON.note ? `<br><span class="bad">${esc(MON.note)}</span>` : '');
  $('mon-go').disabled = !!s.active || MON.sel.size === 0;
  const sel = $('mon-results'), want = s.results.map(r => r.id).join();
  if (sel.dataset.ids !== want) {
    sel.dataset.ids = want;
    sel.innerHTML = s.results.length ? s.results.map(r =>
      `<option value="${r.id}">${r.id} &middot; ${r.done}/${r.n} it &middot; ` +
      `${r.paths.length} mod${r.error ? ' &middot; FAILED' : ''}</option>`
      ).join('') : '<option value="">none yet</option>';
  }
  if (MON.loaded) sel.value = MON.loaded;
  // Plain links to the server's own files, so a download is the saved result
  // itself and works for a result of any size.
  ['csv', 'json'].forEach(ext => { const a = $('mon-dl-' + ext);
    a.classList.toggle('off', !MON.loaded);
    if (MON.loaded) { a.href = `api/results/${MON.loaded}.${ext}`;
                      a.setAttribute('download', ''); }
    else a.removeAttribute('href'); });
}

let wasGraph = null;
async function poll() {
  try {
    MON.status = await api('api/status');
    const s = MON.status;
    // The page was drawn before the trace existed: redraw it with the graph.
    if (wasGraph !== null && wasGraph !== 'traced' && s.graph === 'traced')
      return location.reload();
    wasGraph = s.graph;
    const newest = s.results.length ? s.results[0].id : null;
    if (MON.follow && newest && newest !== MON.loaded) await load(newest);
  } catch (err) { MON.status = null; }
  paint();
}
setInterval(poll, 1000);
$('mon-count').parentElement.addEventListener('click', paint);
$('mon-all').addEventListener('click', paint);
$('mon-depth-go').addEventListener('click', paint);
svg.addEventListener('click', () => setTimeout(paint, 0));

// ---------- results in the details panel -----------------------------------
const stat5 = (r, kind) => `<div class="mon-g">` +
  ['mean', 'median', 'mode', 'min', 'max'].map(k =>
    `<i class="k-${k}">${k}</i>`).join('') +
  ['mean', 'median', 'mode', 'min', 'max'].map(k => {
    const v = r[k];
    if (!v) return '<span>—</span>';
    // Coloured by the worst this number reached in any iteration.
    const lv = Math.max(levelOf(v.min, kind), levelOf(v.max, kind));
    return `<span${lv ? ` class="l${lv}"` : ''} title="${full(v.mean)}  ` +
      `(across iterations: ${full(v.min)} .. ${full(v.max)})">` +
      `${num(v.mean)}</span>`;
  }).join('') + `</div>`;
function hist(r) {
  if (!r.hist || !r.hist.length || !r.hist_range) return '';
  const n = r.hist.length, top = Math.max(...r.hist), w = 100 / n;
  const at = r.hist.indexOf(top);
  return `<svg class="mon-h" viewBox="0 0 100 34" preserveAspectRatio="none">` +
    r.hist.map((c, i) => { const h = top ? Math.max(c ? .6 : 0, c / top * 34) : 0;
      return `<rect${i === at ? ' class="top"' : ''} x="${(i * w).toFixed(3)}" ` +
        `y="${(34 - h).toFixed(2)}" width="${(w * .86).toFixed(3)}" ` +
        `height="${h.toFixed(2)}"/>`; }).join('') +
    (() => { const m = r.mean ? r.mean.mean : null, [lo, hi] = r.hist_range;
      if (m === null || !(hi > lo)) return '';
      const x = Math.min(99.6, Math.max(0.4, (m - lo) / (hi - lo) * 100));
      return `<line class="mean" x1="${x.toFixed(2)}" x2="${x.toFixed(2)}" ` +
        `y1="0" y2="34"><title>mean ${full(m)}</title></line>`; })() +
    `</svg>` +
    `<div class="mon-r"><span>${num(r.hist_range[0])}</span>` +
    `<span>${r.median_sampled ? 'median on a sample · ' : ''}` +
    `${r.nonfinite ? `<b class="l2">${r.nonfinite} non-finite</b> · ` : ''}` +
    `${n} bins</span><span>${num(r.hist_range[1])}</span></div>`;
}
const values = r => {
  if (!r.head || !r.head.length) return '';
  const all = r.numel <= r.head.length, n = all ? r.head.length : 8;
  return `<div class="mon-v" title="${all ? 'every value' :
    'the first ' + r.head.length + ' of ' + r.numel + ' values, flattened'}` +
    `: ${r.head.map(full).join(', ')}"><i>values</i> ` +
    r.head.slice(0, n).map(num).join(', ') + (all ? '' : ', …') + `</div>`;
};
const tensor = (title, r, kind) => { const f = flag(r, kind);
  return `<div class="mon-t"><div class="hd">` +
  `<b>${dot(f)}${esc(title)}</b><span>(${r.shape.join(', ')}) ${esc(r.dtype)}` +
  `${r.shapes_vary ? ' *' : ''}</span></div>${stat5(r, kind)}${values(r)}` +
  `${hist(r)}` + (f.level ? `<div class="mon-r"><span class="l${f.level}">` +
    `${esc(f.why)}</span></div>` : '') + `</div>`; };
const range = r => r && r.min ? `${num(r.min.mean)} .. ${num(r.max.mean)}`
                              : '—';
const rangeCell = r => { const f = flag(r, 'act');
  return `<td${f.level ? ` class="l${f.level}" title="${esc(f.why)}"` : ''}>` +
    `${range(r)}</td>`; };
const PARAM_ROWS = 40;

function monDetail() {
  const res = MON.result;
  if (!res) return;
  const box = document.getElementById('detail'), h = res.header;
  const head = `steps ${h.steps.length ? h.steps[0] + '–' +
    h.steps[h.steps.length - 1] : '—'} &middot; ${h.done}/${h.n} iterations`;
  let html = '';
  if (selected === null) {
    const rows = [], io = [];
    Object.entries(res.summary.modules).forEach(([path, m]) =>
      m.calls.forEach((c, k) => {
        if (c.time) rows.push([path, k, m.calls.length, c.time]);
        if (c.inputs.length || c.outputs.length)
          io.push([path, k, m.calls.length, c.inputs[0], c.outputs[0]]);
      }));
    rows.sort((a, b) => b[3].self.mean - a[3].self.mean);
    html = sec('Monitor ' + h.id,
      `<div class="mon-note">${head}<br>` +
      `Times are from an uncompiled, synchronised, no-gradient forward.` +
      `</div>` +
      `<div class="stat"><span>Total forward, per iteration</span>` +
      `<span>${ms(h.forward_seconds)}</span></div>` +
      `<div class="stat"><span>Total forward, all ${h.done} iterations</span>` +
      `<span>${ms(h.forward_seconds === null ? null
                  : h.forward_seconds * h.done)}</span></div>` +
      `<div class="stat" title="The forward plus computing the statistics">` +
      `<span>Probe incl. statistics, per iteration</span>` +
      `<span>${ms(h.probe_seconds)}</span></div>` +
      (h.train_seconds !== null
        ? `<div class="stat"><span>Training step, per iteration</span>` +
          `<span>${ms(h.train_seconds)}</span></div>` : '') +
      (h.error ? `<div class="mon-err">${esc(h.error)}</div>` : '') +
      (rows.length ? `<table class="mon-tt" style="margin-top:8px"><tr>` +
        `<td><i>module</i></td><td>self</td><td>inclusive</td>` +
        `<td title="inclusive time as a share of the total forward">%</td></tr>` +
        rows.map(r =>
        `<tr><td data-mon="${esc(r[0])}">${esc(label(r[0]))}` +
        `${r[2] > 1 ? ' #' + r[1] : ''}</td><td>${ms(r[3].self.mean)}</td>` +
        `<td>${ms(r[3].inclusive.mean)}</td>` +
        `<td>${pct(r[3].inclusive.mean, h.forward_seconds)}</td></tr>`).join('') +
        `<tr><td style="cursor:default"><i>picked, sum of self</i></td>` +
        `<td>${ms(rows.reduce((a, r) => a + r[3].self.mean, 0))}</td><td></td>` +
        `<td>${pct(rows.reduce((a, r) => a + r[3].self.mean, 0),
                   h.forward_seconds)}</td></tr></table>` +
        `<div class="mon-note">The sum of self times is the part of the ` +
        `forward spent in the picked modules, counted once; the rest is in ` +
        `modules that were not picked.</div>`
        : '') +
      `<div class="mon-note" style="margin-top:8px">Click a module on the ` +
      `graph, or a name here, for its inputs, outputs, parameters and ` +
      `histograms.</div>`);
    const fl = MON.flags ? MON.flags.list : [];
    html += sec('Monitor: flagged values', (fl.length
      ? `<table class="mon-tt">` + fl.slice(0, 60).map(f =>
          `<tr><td data-mon="${esc(f.path)}"><i class="mon-dot l${f.level}">` +
          `</i>${esc(label(f.path))}${f.call ? ' #' + f.call : ''}</td>` +
          `<td class="l${f.level}" style="white-space:normal">` +
          `${esc(f.name)}<br>${esc(f.why)}</td></tr>`).join('') + `</table>` +
        (fl.length > 60 ? `<div class="mon-note">${fl.length - 60} more ` +
          `not shown.</div>` : '')
      : `<div class="mon-note">Nothing over the limits in the measured ` +
        `modules.</div>`));
    html += sec('Monitor: legend', legend());
    if (io.length) html += sec('Monitor: value ranges',
      `<table class="mon-tt"><tr><td><i>module</i></td>` +
      `<td>input</td><td>output</td></tr>` +
      io.map(r => `<tr><td data-mon="${esc(r[0])}">${esc(label(r[0]))}` +
        `${r[2] > 1 ? ' #' + r[1] : ''}</td>${rangeCell(r[3])}` +
        `${rangeCell(r[4])}</tr>`).join('') + `</table>` +
      `<div class="mon-note">min .. max of the first input and the first ` +
      `output of each module; the module's own page lists every tensor.</div>`);
  } else {
    const n = byId.get(selected);
    const m = isMod(n) ? res.summary.modules[pathOf(n)] : null;
    if (!m) return;
    const k = callOf.get(n.id) || 0, c = m.calls[k];
    let body = `<div class="mon-note">${h.id} &middot; ${head}` +
      (m.calls.length > 1 ? `<br>call ${k + 1} of ${m.calls.length} ` +
        `per forward` : '') + `</div>`;
    if (!c) body += `<div class="mon-note">This module's forward was not ` +
      `called in the probe pass (it may be used through another method, or ` +
      `sit inside compiled code).</div>`;
    if (c && c.time) body += `<div class="stat"><span>Share of the forward</span>` +
      `<span>${pct(c.time.inclusive.mean, h.forward_seconds)} of ` +
      `${ms(h.forward_seconds)}</span></div>` +
      `<div class="stat"><span>Total over ${h.done} iterations</span>` +
      `<span>${ms(c.time.inclusive.mean * c.seen)}</span></div>` +
      (m.calls.length > 1 ? `<div class="stat"><span>All ${m.calls.length} ` +
        `calls, per forward</span><span>${ms(m.calls.reduce((a, x) =>
          a + (x.time ? x.time.inclusive.mean : 0), 0))}</span></div>` : '') +
      `<div class="stat"><span>Time, self</span>` +
      `<span title="${ms(c.time.self.min)} .. ${ms(c.time.self.max)}">` +
      `${ms(c.time.self.mean)}</span></div>` +
      `<div class="stat"><span>Time, inclusive</span>` +
      `<span title="${ms(c.time.inclusive.min)} .. ${ms(c.time.inclusive.max)}">` +
      `${ms(c.time.inclusive.mean)}</span></div>`;
    if (m.whole) body += `<div class="stat"><span>All parameters</span>` +
      `<span>${fmt(m.whole.params)}</span></div>` +
      (m.whole.nonfinite ? `<div class="stat"><span>Non-finite parameters` +
        `</span><span style="color:var(--err);font-weight:700">` +
        `${m.whole.nonfinite}</span></div>` : '') +
      (m.whole.mean ? `<div class="stat"><span>mean / min / max</span><span>` +
        `${num(m.whole.mean.mean)} / ${num(m.whole.min.mean)} / ` +
        `${num(m.whole.max.mean)}</span></div>` : '');
    html = sec('Monitor', body);
    if (c && c.inputs.length) html += sec('Monitor: inputs',
      c.inputs.map(r => tensor('input ' + r.name, r, 'act')).join(''));
    if (c && c.outputs.length) html += sec('Monitor: outputs',
      c.outputs.map(r => tensor('output ' + r.name, r, 'act')).join(''));
    if (m.params) {
      const base = pathOf(n), P = res.summary.params;
      const short = name => base && name.startsWith(base + '.')
        ? name.slice(base.length + 1) : name;
      const shown = m.params.slice(0, PARAM_ROWS);
      html += sec('Monitor: parameters',
        `<div class="mon-note">${m.params.length} tensor` +
        `${m.params.length === 1 ? '' : 's'}` +
        (h.deep ? ` inside this module, ${m.own_params} owned directly`
                : ` owned directly (parameters inside were not requested)`) +
        (m.params.length > shown.length ? `. Showing the first ` +
          `${shown.length}; pick a smaller module for the rest` : '') +
        `.</div>` +
        shown.map(name => { const r = P[name]; return !r ? '' :
          tensor(short(name), r, 'param') +
          (r.grad ? tensor(short(name) + ' · grad', r.grad, 'grad') : ''); }
        ).join(''));
    }
    html += sec('Monitor: legend', legend());
  }
  box.insertAdjacentHTML('afterbegin', html);
}
const _detail = detail;
detail = function () { _detail.apply(this, arguments); monDetail(); };
document.getElementById('detail').addEventListener('click', e => {
  const cell = e.target.closest('[data-mon]');
  if (cell && firstNode.has(cell.dataset.mon))
    gotoNode(firstNode.get(cell.dataset.mon));
});

// ---------- state in the URL ----------------------------------------------
// #panel=1&sel=a,b.c&result=r00001&node=a.b - a link (or a screenshot) can
// open the page in a given state.
const hash = new URLSearchParams(location.hash.slice(1));
if (hash.get('sel') !== null)
  hash.get('sel').split(',').forEach(p => MON.sel.add(p === '~' ? '' : p));
if (hash.get('panel')) openPanel(true);
(async () => {
  await poll();
  if (hash.get('result')) { MON.follow = false; await load(hash.get('result')); }
  if (hash.get('depth')) { expandToDepth(+hash.get('depth')); relayout(); fit(); }
  if (hash.get('node') !== null && firstNode.has(hash.get('node'))) {
    selected = firstNode.get(hash.get('node'));
    applySelection();
    if (hash.get('zoom')) zoomBy(+hash.get('zoom'));
    centerOn(isHidden(selected) ? proxy(selected) : selected);
  }
  marks(); paint();
})();
})();
</script>
"""
