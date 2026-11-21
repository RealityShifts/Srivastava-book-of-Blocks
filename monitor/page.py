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
.mon-h { display:block; width:100%; height:34px; margin-top:4px; }
.mon-h rect { fill:var(--accent); }
.mon-h rect.top { fill:var(--hi); }
.mon-r { display:flex; justify-content:space-between; color:var(--muted);
  font-size:10px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
.mon-note { color:var(--muted); font-size:11px; line-height:1.4;
  margin:4px 0 8px; }
.mon-err { color:var(--err); font-size:11px; white-space:pre-wrap;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
.mon-tt { width:100%; border-collapse:collapse; font-size:11px;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
.mon-tt td { padding:2px 0; }
.mon-tt td:first-child { color:var(--fg); word-break:break-all;
  cursor:pointer; }
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
    <div class="row"><span id="mon-count">0 selected</span>
      <button id="mon-clear">Clear</button></div>
    <hr>
    <div class="row"><span>Iterations</span>
      <input id="mon-n" type="number" min="1" max="1000" value="10"></div>
    <div class="row"><label><input id="mon-stats" type="checkbox" checked>
      statistics</label>
      <label><input id="mon-time" type="checkbox" checked> timing</label>
      <span></span><button id="mon-go" class="go">Collect</button></div>
    <hr>
    <div id="mon-status">connecting&hellip;</div>
    <div class="row"><span>Result</span><select id="mon-results"></select></div>
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
  root.querySelectorAll('.montime').forEach(t => t.remove());
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
  e.stopImmediatePropagation();
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
      stats: $('mon-stats').checked, timing: $('mon-time').checked });
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
    Object.entries(MON.result.summary).forEach(([path, m]) => {
      const ts = m.calls.map(c => c.time ? c.time.self.mean : null);
      if (ts.some(t => t !== null)) per.set(path, ts);
      ts.forEach(t => { if (t !== null && t > max) max = t; });
    });
    MON.times = per.size ? { per, max } : null;
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
svg.addEventListener('click', () => setTimeout(paint, 0));

// ---------- results in the details panel -----------------------------------
const stat5 = r => `<div class="mon-g">` +
  ['mean', 'median', 'mode', 'min', 'max'].map(k => `<i>${k}</i>`).join('') +
  ['mean', 'median', 'mode', 'min', 'max'].map(k => {
    const v = r[k];
    return v ? `<span title="${full(v.mean)}  (across iterations: ` +
               `${full(v.min)} .. ${full(v.max)})">${num(v.mean)}</span>`
             : '<span>—</span>';
  }).join('') + `</div>`;
function hist(r) {
  if (!r.hist || !r.hist.length || !r.hist_range) return '';
  const n = r.hist.length, top = Math.max(...r.hist), w = 100 / n;
  const at = r.hist.indexOf(top);
  return `<svg class="mon-h" viewBox="0 0 100 34" preserveAspectRatio="none">` +
    r.hist.map((c, i) => { const h = top ? Math.max(c ? .6 : 0, c / top * 34) : 0;
      return `<rect${i === at ? ' class="top"' : ''} x="${(i * w).toFixed(3)}" ` +
        `y="${(34 - h).toFixed(2)}" width="${(w * .86).toFixed(3)}" ` +
        `height="${h.toFixed(2)}"/>`; }).join('') + `</svg>` +
    `<div class="mon-r"><span>${num(r.hist_range[0])}</span>` +
    `<span>${r.median_sampled ? 'median on a sample · ' : ''}` +
    `${r.nonfinite ? r.nonfinite + ' non-finite · ' : ''}` +
    `${n} bins</span><span>${num(r.hist_range[1])}</span></div>`;
}
const tensor = (title, r) => `<div class="mon-t"><div class="hd">` +
  `<b>${esc(title)}</b><span>(${r.shape.join(', ')}) ${esc(r.dtype)}` +
  `${r.shapes_vary ? ' *' : ''}</span></div>${stat5(r)}${hist(r)}</div>`;

function monDetail() {
  const res = MON.result;
  if (!res) return;
  const box = document.getElementById('detail'), h = res.header;
  const head = `steps ${h.steps.length ? h.steps[0] + '–' +
    h.steps[h.steps.length - 1] : '—'} &middot; ${h.done}/${h.n} iterations`;
  let html = '';
  if (selected === null) {
    const rows = [];
    Object.entries(res.summary).forEach(([path, m]) => m.calls.forEach((c, k) => {
      if (c.time) rows.push([path, k, m.calls.length, c.time]); }));
    rows.sort((a, b) => b[3].self.mean - a[3].self.mean);
    html = sec('Monitor ' + h.id,
      `<div class="mon-note">${head}<br>` +
      `probe forward ${ms(h.probe_seconds)} per iteration` +
      (h.train_seconds !== null
        ? ` &middot; training step ${ms(h.train_seconds)}` : '') +
      `<br>Times are from an uncompiled, synchronised, no-gradient forward.` +
      `</div>` +
      (h.error ? `<div class="mon-err">${esc(h.error)}</div>` : '') +
      (rows.length ? `<table class="mon-tt"><tr><td><i>module</i></td>` +
        `<td>self</td><td>inclusive</td></tr>` + rows.map(r =>
        `<tr><td data-mon="${esc(r[0])}">${esc(label(r[0]))}` +
        `${r[2] > 1 ? ' #' + r[1] : ''}</td><td>${ms(r[3].self.mean)}</td>` +
        `<td>${ms(r[3].inclusive.mean)}</td></tr>`).join('') + `</table>`
        : `<div class="mon-note">${Object.keys(res.summary).length} modules ` +
          `measured. Click one on the graph.</div>`));
  } else {
    const n = byId.get(selected);
    const m = isMod(n) ? res.summary[pathOf(n)] : null;
    if (!m) return;
    const k = callOf.get(n.id) || 0, c = m.calls[k];
    let body = `<div class="mon-note">${h.id} &middot; ${head}` +
      (m.calls.length > 1 ? `<br>call ${k + 1} of ${m.calls.length} ` +
        `per forward` : '') + `</div>`;
    if (!c) body += `<div class="mon-note">This module's forward was not ` +
      `called in the probe pass (it may be used through another method, or ` +
      `sit inside compiled code).</div>`;
    if (c && c.time) body += `<div class="stat"><span>Time, self</span>` +
      `<span title="${ms(c.time.self.min)} .. ${ms(c.time.self.max)}">` +
      `${ms(c.time.self.mean)}</span></div>` +
      `<div class="stat"><span>Time, inclusive</span>` +
      `<span title="${ms(c.time.inclusive.min)} .. ${ms(c.time.inclusive.max)}">` +
      `${ms(c.time.inclusive.mean)}</span></div>`;
    if (m.whole) body += `<div class="stat"><span>All parameters</span>` +
      `<span>${fmt(m.whole.params)}</span></div>` +
      (m.whole.mean ? `<div class="stat"><span>mean / min / max</span><span>` +
        `${num(m.whole.mean.mean)} / ${num(m.whole.min.mean)} / ` +
        `${num(m.whole.max.mean)}</span></div>` : '');
    html = sec('Monitor', body);
    if (c && c.inputs.length) html += sec('Monitor: inputs',
      c.inputs.map(r => tensor('input ' + r.name, r)).join(''));
    if (c && c.outputs.length) html += sec('Monitor: outputs',
      c.outputs.map(r => tensor('output ' + r.name, r)).join(''));
    if (m.params && m.params.length) html += sec('Monitor: parameters',
      m.params.map(r => tensor(r.name, r) +
        (r.grad ? tensor(r.name + ' · grad', r.grad) : '')).join(''));
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
