"""The scalars page: curves from a TensorBoard log directory, with alerts.

One self-contained document (no CDN, no build step), served at ``scalars``.
Type a log directory, get every scalar as a small chart, filter by name or by
alert level, and see what the rules in :mod:`.scalars` flagged - in a list,
on each chart's title, and as markers on the curve where it happened.

Relative URLs throughout, so it works under a reverse-proxy prefix.
"""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scalars and alerts</title>
<style>
:root { --bg:#0f1117; --panel:#1b1e28; --line:#2b303d; --fg:#e6e9ef;
  --muted:#9aa3b8; --accent:#8494e4; --card:#252a38; --cardln:#4a5268;
  --critical:#ff6b6b; --warning:#e8944a; --info:#4cc9f0; --ok:#5ec9a7; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
  font:13px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
code, .mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
#top { display:flex; gap:8px; align-items:center; padding:10px 14px;
  border-bottom:1px solid var(--line); background:var(--panel);
  position:sticky; top:0; z-index:5; flex-wrap:wrap; }
#top a { color:var(--accent); text-decoration:none; margin-right:6px; }
input[type=text], select { background:var(--card); color:var(--fg);
  border:1px solid var(--cardln); border-radius:6px; padding:5px 8px;
  font:inherit; }
#path { flex:1; min-width:260px; font-family:ui-monospace,Menlo,monospace; }
button { background:var(--card); color:var(--fg); border:1px solid var(--cardln);
  border-radius:6px; padding:5px 11px; font:inherit; cursor:pointer; }
button:hover { border-color:var(--accent); }
button.go { background:var(--accent); color:var(--bg); border-color:var(--accent);
  font-weight:600; }
#msg { color:var(--muted); font-size:12px; }
#msg.bad { color:var(--critical); }
#wrap { display:grid; grid-template-columns:340px 1fr; min-height:calc(100vh - 50px); }
#side { border-right:1px solid var(--line); padding:12px 14px; overflow:auto;
  max-height:calc(100vh - 50px); position:sticky; top:50px; }
#main { padding:12px 14px; }
h2 { font-size:11px; text-transform:uppercase; letter-spacing:.08em;
  color:var(--muted); margin:14px 0 6px; font-weight:600; }
h2:first-child { margin-top:0; }
.chips { display:flex; gap:6px; flex-wrap:wrap; }
.chip { border:1px solid var(--cardln); border-radius:14px; padding:2px 10px;
  cursor:pointer; user-select:none; font-size:12px; }
.chip.off { opacity:.4; }
.chip b { font-weight:700; }
.critical { color:var(--critical); } .warning { color:var(--warning); }
.info { color:var(--info); } .ok { color:var(--ok); }
.dot { display:inline-block; width:8px; height:8px; border-radius:50%;
  margin-right:6px; background:var(--cardln); }
.dot.critical { background:var(--critical); } .dot.warning { background:var(--warning); }
.dot.info { background:var(--info); } .dot.ok { background:var(--ok); }
.alert { padding:7px 0; border-bottom:1px solid var(--line); cursor:pointer; }
.alert:hover .t { color:var(--accent); }
.alert .t { font-family:ui-monospace,Menlo,monospace; font-size:12px;
  overflow-wrap:anywhere; }
.alert .m { color:var(--muted); font-size:12px; margin-top:1px; }
.alert .r { font-size:10px; text-transform:uppercase; letter-spacing:.06em; }
.bar { display:flex; gap:10px; align-items:center; flex-wrap:wrap;
  margin-bottom:12px; }
.bar label { color:var(--muted); display:flex; gap:5px; align-items:center; }
#q { width:260px; font-family:ui-monospace,Menlo,monospace; }
#grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr));
  gap:12px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:8px;
  padding:9px 10px 7px; }
.card.critical { border-color:var(--critical); }
.card.warning { border-color:var(--warning); }
.card.info { border-color:var(--info); }
.card .hd { display:flex; justify-content:space-between; gap:8px;
  align-items:baseline; }
.card .hd .t { font-family:ui-monospace,Menlo,monospace; font-size:12px;
  overflow-wrap:anywhere; }
.card .hd .v { color:var(--muted); font-size:11px; white-space:nowrap;
  font-family:ui-monospace,Menlo,monospace; }
.card svg { display:block; width:100%; height:150px; margin-top:4px;
  cursor:crosshair; }
.card .ft { display:flex; justify-content:space-between; color:var(--muted);
  font-size:10.5px; font-family:ui-monospace,Menlo,monospace; min-height:15px; }
.card .why { font-size:11.5px; margin-top:3px; }
.axis { fill:var(--muted); font-size:9px; font-family:ui-monospace,Menlo,monospace; }
.raw { fill:none; stroke:var(--accent); stroke-width:1; opacity:.35; }
.line { fill:none; stroke:var(--accent); stroke-width:1.5; }
.grid { stroke:var(--line); stroke-width:1; }
.mark { stroke-width:1.5; stroke-dasharray:3 2; }
.mark.critical { stroke:var(--critical); } .mark.warning { stroke:var(--warning); }
.mark.info { stroke:var(--info); }
.badpt { fill:var(--critical); }
.cross { stroke:var(--fg); stroke-width:.7; opacity:.6; }
#empty { color:var(--muted); padding:40px 0; text-align:center; }
.leg { color:var(--muted); font-size:11.5px; line-height:1.7; }
.leg b { color:var(--fg); font-weight:600; }
</style>
</head>
<body>
<div id="top">
  <a href="./" title="The module monitor, when this page is served by one">&larr; modules</a>
  <input id="path" type="text" placeholder="path to a TensorBoard log directory on this machine, e.g. /workspace/run/logs">
  <button id="load" class="go">Load</button>
  <label style="color:var(--muted)"><input id="auto" type="checkbox" checked> refresh every 10 s</label>
  <span id="msg"></span>
</div>
<div id="wrap">
  <div id="side">
    <h2>Alerts</h2>
    <div class="chips" id="chips"></div>
    <div id="alerts" style="margin-top:8px"></div>
    <h2>Legend</h2>
    <div class="leg" id="legend"></div>
  </div>
  <div id="main">
    <div class="bar">
      <input id="q" type="text" placeholder="look up tags: text, or /regex/">
      <select id="run"></select>
      <label><input id="only" type="checkbox"> only with alerts</label>
      <label>smoothing <input id="smooth" type="range" min="0" max="0.99" step="0.01" value="0.6"></label>
      <label><input id="logy" type="checkbox"> log scale</label>
      <span id="count" style="color:var(--muted)"></span>
    </div>
    <div id="grid"></div>
    <div id="empty"></div>
    <div style="text-align:center;margin:14px"><button id="more" style="display:none">Show more</button></div>
  </div>
</div>
<script>
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g,
  c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const num = v => v === null || v === undefined ? '—' : v === 0 ? '0'
  : (Math.abs(v) >= 1e5 || Math.abs(v) < 1e-3) ? v.toExponential(2)
  : (+v.toPrecision(4)).toString();
const RANK = { critical: 0, warning: 1, info: 2, ok: 3 };
const S = { sum: null, levels: { critical: true, warning: true, info: true, ok: true },
            shown: 24, cache: new Map() };

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : { method: 'POST',
    headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}
function say(text, bad) { $('msg').textContent = text; $('msg').className = bad ? 'bad' : ''; }

async function load() {
  const path = $('path').value.trim();
  if (!path) return say('type a log directory', true);
  say('reading…');
  try {
    S.sum = await api('api/scalars/open', { path });
    S.cache.clear();
    try { localStorage.setItem('blocks-monitor-logdir', path); } catch (e) {}
    paint();
  } catch (err) { say(String(err.message || err), true); }
}
async function refresh() {
  try {
    S.sum = await api('api/scalars');
    if (S.sum.logdir && !$('path').value) $('path').value = S.sum.logdir;
    paint();
  } catch (err) { say('server not reachable', true); }
}
$('load').onclick = load;
$('path').addEventListener('keydown', e => { if (e.key === 'Enter') load(); });
setInterval(() => { if ($('auto').checked) refresh(); }, 10000);

const keyOf = t => t.run + '\u0001' + t.tag;
function matcher() {
  const q = $('q').value.trim();
  if (!q) return () => true;
  const m = q.match(/^\/(.*)\/([a-z]*)$/);
  if (m) { try { const re = new RegExp(m[1], m[2] || 'i');
                 return t => re.test(t.tag); } catch (e) { return () => false; } }
  const parts = q.toLowerCase().split(/\s+/);
  return t => parts.every(p => t.tag.toLowerCase().includes(p));
}
function visible() {
  const sum = S.sum, ok = matcher(), run = $('run').value;
  return sum.tags.filter(t => (!run || t.run === run) && ok(t)
      && S.levels[t.level || 'ok'] && (!$('only').checked || t.level))
    .sort((a, b) => RANK[a.level || 'ok'] - RANK[b.level || 'ok']
                    || a.tag.localeCompare(b.tag) || a.run.localeCompare(b.run));
}

function paint() {
  const sum = S.sum;
  if (!sum) return;
  const c = sum.counts, total = sum.tags.length;
  const flagged = sum.tags.filter(t => t.level).length;
  say(sum.error ? sum.error : (sum.logdir ? `${sum.runs.length} run(s), ${total} series` +
    ` · ${sum.logdir}` : (total ? `${total} live series` : 'no log directory open')),
    !!sum.error);
  $('chips').innerHTML = ['critical', 'warning', 'info'].map(lv =>
    `<span class="chip ${lv}${S.levels[lv] ? '' : ' off'}" data-lv="${lv}">` +
    `<b>${c[lv]}</b> ${lv}</span>`).join('') +
    `<span class="chip ok${S.levels.ok ? '' : ' off'}" data-lv="ok">` +
    `<b>${total - flagged}</b> clear</span>`;
  const runs = $('run'), want = sum.runs.join('\u0001');
  if (runs.dataset.v !== want) { runs.dataset.v = want;
    runs.innerHTML = `<option value="">all runs</option>` + sum.runs.map(r =>
      `<option value="${esc(r)}">${esc(r)}</option>`).join('');
    runs.style.display = sum.runs.length > 1 ? '' : 'none'; }
  const many = sum.runs.length > 1;
  const shown = sum.alerts.filter(a => S.levels[a.level]);
  $('alerts').innerHTML = shown.length ? shown.slice(0, 300).map(a =>
    `<div class="alert" data-tag="${esc(a.tag)}" data-run="${esc(a.run)}">` +
    `<div class="r ${a.level}"><i class="dot ${a.level}"></i>${a.level} · ` +
    `${esc(a.rule)}${a.count > 1 ? ' ×' + a.count : ''}</div>` +
    `<div class="t">${many && a.run !== '.' ? esc(a.run) + ' / ' : ''}` +
    `${esc(a.tag || '(whole run)')}</div><div class="m">${esc(a.message)}</div></div>`
    ).join('') : `<div style="color:var(--muted)">${sum.tags.length
      ? 'Nothing flagged at the selected levels.' : 'Load a log directory.'}</div>`;
  const L = sum.limits;
  $('legend').innerHTML =
    `<i class="dot critical"></i><b>critical</b>: NaN / Inf, growth of ` +
    `${num(L.grow_factor_critical)}x or more, a gradient at zero<br>` +
    `<i class="dot warning"></i><b>warning</b>: spikes, growth of ` +
    `${num(L.grow_factor)}x or more, a vanishing gradient, a loss turning upward<br>` +
    `<i class="dot info"></i><b>info</b>: a flat or constant value, a run with ` +
    `no new points<br>` +
    `<i class="dot ok"></i><b>clear</b>: no rule fired<br>` +
    `On a chart: the bright line is smoothed, the faint one is the raw ` +
    `values; a dashed vertical line marks where an alert was seen; red ` +
    `squares at the top are NaN / Inf points.<br>` +
    Object.entries(sum.rules).map(([k, v]) => `<b>${esc(k)}</b>: ${esc(v)}`).join('<br>') +
    `<br>Which tags a rule reads is decided by their names; the patterns ` +
    `and limits are set in code (<code>limits=</code>).`;

  const list = visible();
  $('count').textContent = `${list.length} of ${total} series`;
  $('empty').textContent = list.length ? '' : (total
    ? 'No series match the filters.' : '');
  const part = list.slice(0, S.shown);
  $('more').style.display = list.length > part.length ? '' : 'none';
  $('more').textContent = `Show more (${list.length - part.length} hidden)`;
  const grid = $('grid'), have = new Map([...grid.children].map(n => [n.dataset.k, n]));
  const byTag = new Map();
  sum.alerts.forEach(a => { const k = a.run + '\u0001' + a.tag;
    if (!byTag.has(k)) byTag.set(k, []); byTag.get(k).push(a); });
  grid.innerHTML = '';
  part.forEach(t => {
    const k = keyOf(t);
    const card = have.get(k) || document.createElement('div');
    card.dataset.k = k;
    card.className = 'card ' + (t.level || '');
    const al = byTag.get(k) || [];
    card.innerHTML = `<div class="hd"><span class="t"><i class="dot ${t.level || 'ok'}">` +
      `</i>${many && t.run !== '.' ? esc(t.run) + ' / ' : ''}${esc(t.tag)}</span>` +
      `<span class="v">${num(t.last)} @ ${t.last_step}</span></div>` +
      `<svg viewBox="0 0 340 150" preserveAspectRatio="none"></svg>` +
      `<div class="ft"><span class="rd"></span><span>${t.points} pts</span></div>` +
      al.map(a => `<div class="why ${a.level}">${esc(a.message)}</div>`).join('');
    grid.appendChild(card);
    draw(card, t, al);
  });
}

async function fetchSeries(t) {
  const k = keyOf(t), hit = S.cache.get(k);
  if (hit && hit.points === t.points) return hit;
  const d = await api(`api/scalars/series?run=${encodeURIComponent(t.run)}` +
                      `&tag=${encodeURIComponent(t.tag)}&max=1200`);
  S.cache.set(k, d);
  return d;
}
async function draw(card, t, alerts) {
  let d;
  try { d = await fetchSeries(t); } catch (err) { return; }
  const svg = card.querySelector('svg');
  if (!svg) return;
  const W = 340, H = 150, PL = 44, PR = 6, PT = 8, PB = 16;
  const xs = d.steps, raw = d.values;
  if (!xs.length) { svg.innerHTML = `<text class="axis" x="${W / 2}" y="${H / 2}" ` +
    `text-anchor="middle">no finite points</text>`; return; }
  const useLog = $('logy').checked && raw.every(v => v > 0);
  const f = v => useLog ? Math.log10(v) : v;
  const a = +$('smooth').value;
  let acc = null; const sm = raw.map(v => { const y = f(v);
    acc = acc === null ? y : a * acc + (1 - a) * y; return acc; });
  const ys = raw.map(f);
  let lo = Math.min(...ys), hi = Math.max(...ys);
  if (hi === lo) { hi += Math.abs(hi) * .05 + 1e-9; lo -= Math.abs(lo) * .05 + 1e-9; }
  const x0 = xs[0], x1 = xs[xs.length - 1] === x0 ? x0 + 1 : xs[xs.length - 1];
  const X = s => PL + (s - x0) / (x1 - x0) * (W - PL - PR);
  const Y = v => PT + (hi - v) / (hi - lo) * (H - PT - PB);
  const path = arr => arr.map((v, i) => (i ? 'L' : 'M') + X(xs[i]).toFixed(1) +
    ' ' + Y(v).toFixed(1)).join('');
  const back = v => useLog ? Math.pow(10, v) : v;
  svg.innerHTML =
    [0, .5, 1].map(p => { const v = lo + (hi - lo) * p, y = Y(v);
      return `<line class="grid" x1="${PL}" x2="${W - PR}" y1="${y}" y2="${y}"/>` +
        `<text class="axis" x="${PL - 4}" y="${y + 3}" text-anchor="end">${num(back(v))}</text>`;
    }).join('') +
    `<text class="axis" x="${PL}" y="${H - 3}">${x0}</text>` +
    `<text class="axis" x="${W - PR}" y="${H - 3}" text-anchor="end">${x1}</text>` +
    (useLog ? `<text class="axis" x="${W - PR}" y="${PT + 8}" text-anchor="end">log</text>` : '') +
    `<path class="raw" d="${path(ys)}"/>` + (a > 0 ? `<path class="line" d="${path(sm)}"/>`
      : `<path class="line" d="${path(ys)}"/>`) +
    alerts.filter(al => al.step >= x0 && al.step <= x1 && al.rule !== 'flat').map(al =>
      `<line class="mark ${al.level}" x1="${X(al.step)}" x2="${X(al.step)}" ` +
      `y1="${PT}" y2="${H - PB}"><title>${esc(al.rule + ': ' + al.message)}</title></line>`
      ).join('') +
    d.bad.filter(s => s >= x0 && s <= x1).map(s =>
      `<rect class="badpt" x="${X(s) - 2}" y="${PT - 6}" width="4" height="4">` +
      `<title>non-finite at step ${s}</title></rect>`).join('') +
    `<line class="cross" x1="0" x2="0" y1="${PT}" y2="${H - PB}" style="display:none"/>`;
  const cross = svg.querySelector('.cross'), rd = card.querySelector('.rd');
  svg.onmousemove = e => { const r = svg.getBoundingClientRect();
    const sx = (e.clientX - r.left) / r.width * W;
    const step = x0 + (sx - PL) / (W - PL - PR) * (x1 - x0);
    let i = 0, best = Infinity;
    for (let j = 0; j < xs.length; j++) { const dd = Math.abs(xs[j] - step);
      if (dd < best) { best = dd; i = j; } }
    cross.setAttribute('x1', X(xs[i])); cross.setAttribute('x2', X(xs[i]));
    cross.style.display = '';
    rd.textContent = `step ${xs[i]}: ${num(raw[i])}`; };
  svg.onmouseleave = () => { cross.style.display = 'none'; rd.textContent = ''; };
}

$('chips').onclick = e => { const c = e.target.closest('.chip'); if (!c) return;
  S.levels[c.dataset.lv] = !S.levels[c.dataset.lv]; paint(); };
$('alerts').onclick = e => { const a = e.target.closest('.alert');
  if (!a || !a.dataset.tag) return;
  $('q').value = '/^' + a.dataset.tag.replace(/[.*+?^${}()|[\]\\\/]/g, '\\$&') + '$/';
  if ($('run').style.display !== 'none') $('run').value = a.dataset.run;
  $('only').checked = false; S.shown = 24; paint();
  window.scrollTo({ top: 0, behavior: 'smooth' }); };
['q', 'run', 'only', 'smooth', 'logy'].forEach(id =>
  $(id).addEventListener('input', () => { S.shown = 24; paint(); }));
$('more').onclick = () => { S.shown += 48; paint(); };

// #logdir=/path&q=grad&only=1 opens the page in a given state.
const hash = new URLSearchParams(location.hash.slice(1));
if (hash.get('q')) $('q').value = hash.get('q');
if (hash.get('only')) $('only').checked = true;
if (hash.get('levels')) { const on = hash.get('levels').split(',');
  Object.keys(S.levels).forEach(k => S.levels[k] = on.includes(k)); }
let saved = hash.get('logdir');
try { saved = saved || localStorage.getItem('blocks-monitor-logdir'); } catch (e) {}
(async () => {
  await refresh();
  if (hash.get('logdir')) { $('path').value = hash.get('logdir'); await load(); }
  else if (S.sum && !S.sum.logdir && saved) $('path').value = saved;
})();
</script>
</body>
</html>
"""
