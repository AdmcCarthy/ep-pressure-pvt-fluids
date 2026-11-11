"""3D PVT dashboard (DRAFT): the pressure - volume - temperature surface of a model fluid with the DST tests placed on it.

    (the engine's scripts/build_pvt3d_dashboard.py reads the mid-layer and calls build_html)

Writes reports/dashboards/pvt_3d.html: the same page as the GLR dashboard (map, areas and plays as tick boxes, colour-by, hover linking between the charts, P-T and API
charts) with the GLR chart replaced by an interactive 3D chart: temperature [degC] x molar volume [L/mol, log] x pressure [bar].  The surface is computed with the
Peng-Robinson equation of state of a pure component (eos, config/pvt3d.yaml): compressed liquid, vapour, the two-phase dome with its flat Maxwell
plateau and the two saturated-volume curves (binodal), the critical point and the critical isotherm, and the supercritical region.  A test is placed on the surface of the
selected model fluid at its own temperature and pressure (hollow = estimated temperature): the chart shows which regime a pure fluid with that critical point would be in.
A reservoir fluid is a mixture (its envelope is a loop in the P-T plane and needs the composition): see docs/pvt3d.md.
"""

from __future__ import annotations

import pandas as pd

from ep_pressure_pvt_fluids import eos
from ep_pressure_pvt_fluids.plots.pvt_colours import REGIME_COLOURS
from ep_physics_base.config import load_yaml
from ep_physics_base.figures import hover_panel as hp  # noqa: F401
from ep_pressure_pvt_fluids import pressure as pr  # noqa: F401
from ep_pressure_pvt_fluids.plots import pvt_dashboard as base

PAGE = "pvt_3d.html"


def point_states(tests: pd.DataFrame, fluids: dict[str, eos.Fluid], tolerance: float) -> dict[str, dict[str, list]]:
    """For every model fluid: log10 molar volume [L/mol] and regime of each test (in the order of `tests`) at its temperature and pressure; None where either is missing."""
    out: dict[str, dict[str, list]] = {}
    t = pd.to_numeric(tests["temperature_used_c"], errors="coerce") if "temperature_used_c" in tests else pd.Series(float("nan"), index=tests.index)
    p = pd.to_numeric(tests["reservoir_pressure_bar"], errors="coerce")
    for fid, fluid in fluids.items():
        v, r = [], []
        for tc, pb in zip(t, p):
            lv, regime = eos.state(float(tc), float(pb), fluid, tolerance) if pd.notna(tc) and pd.notna(pb) else (float("nan"), "")
            v.append(None if lv != lv else round(lv, 4))
            r.append(regime or None)
        out[fid] = {"v": v, "r": r}
    return out


def eos_config(tests: pd.DataFrame, cfg3d: dict) -> dict:
    """The page configuration of the 3D chart: the fluids, their surfaces, and the volume and regime of every test on every surface."""
    fluids = eos.fluids_from_config(cfg3d)
    surfaces = {}
    for fid, fluid in fluids.items():
        s = eos.pvt_surface(fluid, cfg3d["grid"])
        s.pop("x")                                              # a row of the mesh is one temperature: the page rebuilds the x coordinate from `t_c`
        surfaces[fid] = s
    return {"eos": {"default": cfg3d["default_fluid"], "fluids": [{"id": f.id, "name": f.name, "role": next(x["role"] for x in cfg3d["fluids"] if x["id"] == f.id), "tc_c": round(f.tc_k - 273.15, 2), "pc_bar": f.pc_bar,
                                                                    "log_vc": round(float(__import__("numpy").log10(f.vc_l_mol)), 4)} for f in fluids.values()],
                    "surfaces": surfaces, "points": point_states(tests, fluids, float(cfg3d["two_phase_tolerance"])), "zones": cfg3d["colours"]},
            "colours_re": REGIME_COLOURS}


def _slice(text: str, start: str, end: str) -> tuple[int, int]:
    a = text.index(start)
    return a, text.index(end, a)


def body3d() -> str:
    """The body of the GLR dashboard with the controls, the main panel, the table and the notes replaced."""
    b = base.BODY
    b = b.replace("<h1>GLR against reservoir pressure: which DST fluids can be one phase?<span", "<h1>Pressure, volume and temperature: where do the DST fluids sit on the surface of a model fluid?<span")
    a, z = _slice(b, '<div class="controls">', '<div class="grid">')
    controls = '''<div class="controls">
  <label>Model fluid (Peng-Robinson, pure component)<select id="f-model"></select></label>
  <label>Fluid class<select id="f-fluid"></select></label>
  <label>Colour by<select id="f-colour">
    <option value="re">Regime on the model fluid</option><option value="st">GLR saturation state</option><option value="fc">Fluid class</option><option value="pl">Play</option><option value="a">Area</option><option value="ac">API class</option><option value="tb">Temperature bracket</option><option value="eb">Envelope basis</option></select></label>
  <div style="display:flex;flex-direction:column;gap:6px">
    <label class="check"><input type="checkbox" id="f-own"> only tests with own API, gas gravity and a gauge or estimated temperature</label>
    <div><button id="b-reset" type="button">Reset</button> <button id="b-csv" type="button">Download filtered tests (CSV)</button></div></div>
  <div style="display:none"><input type="checkbox" id="f-shade" checked><output id="o-api"></output><input type="range" id="s-api" min="10" max="60" step="1"><output id="o-t"></output><input type="range" id="s-t" min="30" max="200" step="5">
    <output id="o-gg"></output><input type="range" id="s-gg" min="0.55" max="1.2" step="0.05"></div>
  <div class="legend3d" id="legend3d"></div>
</div>
'''
    b = b[:a] + controls + b[z:]
    a, z = _slice(b, '<div class="panel"><h2>GLR (log scale', '  <div>\n    <div class="panel"><h2>Pressure against temperature')
    main = '''<div class="panel"><h2>Temperature, molar volume and pressure of the model fluid</h2><p class="sub" id="model-sub"></p>
    <div style="margin:4px 6px"><button type="button" data-view="iso">3D view</button> <button type="button" data-view="pv">P-V view (along temperature)</button> <button type="button" data-view="pt">P-T view (along volume)</button> <button type="button" data-view="vt">V-T view (from above)</button></div>
    <div id="c-main" style="height:860px"></div><p class="sub">Pressure and volume are on log scales so that the dome (tens of bar) and the reservoir data (hundreds of bar) are both visible; volumes above 1,000 L/mol are cut off. Drag to rotate, scroll to zoom, click a legend entry to hide a category. <b>Hollow markers</b>: estimated temperature. <span id="ymax-note"></span></p></div>
'''
    b = b[:a] + main + b[z:]
    a, z = _slice(b, '<div class="panel" style="margin-top:14px"><h2>Tests in the forbidden domain', '<div class="notes">')
    table = '''<div class="panel" style="margin-top:14px"><h2>Regime on the model fluid against the GLR saturation state</h2><p class="sub">Tests per regime of the model fluid (rows) and per saturation state of the GLR chart (columns): where the two views agree and where they do not.</p>
  <div class="tablewrap"><table id="t-two"><thead></thead><tbody></tbody></table></div></div>
'''
    b = b[:a] + table + b[z:]
    a, z = _slice(b, '<div class="notes">', "<footer>")
    notes = '''<div class="notes"><div><h3>How to read</h3>
<p>The surface is the <b>pressure-volume-temperature surface of a pure component</b> from the Peng-Robinson equation of state, so its boundaries are computed, not drawn. Below the <b>critical temperature</b> each isotherm has a <b>compressed-liquid</b> branch (small volume, steep pressure), a flat <b>two-phase plateau</b> at the saturation pressure (liquid and vapour in equilibrium: Maxwell's equal-area construction, equal fugacity) and a <b>vapour</b> branch. The ends of the plateaus trace the <b>dome</b> (the two saturated-volume curves, dark lines); the dome closes at the <b>critical point</b> (red diamond), where liquid and vapour become identical, and the <b>critical isotherm</b> (orange) passes through it. Above the critical temperature there is only one phase: the <b>supercritical</b> region (yellow).
Each test sits on the surface at its own temperature and pressure, at the molar volume the equation gives for the selected model fluid: it shows <i>which regime a pure fluid with that critical point would be in</i> (green liquid, red vapour, yellow supercritical, dark two-phase). A reservoir is an <b>oil</b> reservoir when its temperature is below the critical temperature of its fluid and a <b>gas</b> reservoir when it is above; a light model fluid (methane, ethane) puts every reservoir above its critical temperature, a heavy one (decane) below. The two-phase dome of the 3D chart is the counterpart of the <b>forbidden domain</b> of the GLR chart, the critical point of its <b>critical-fluid band</b>. The map, the pressure-temperature chart and the API chart are those of the GLR dashboard and follow the same selection; hovering a point rings the same test in all charts.</p>
<h3>Sources</h3><p>Peng and Robinson (1976) equation of state with the usual alpha function; critical constants after Poling, Prausnitz and O'Connell; Maxwell construction by equal fugacity; the same DST, pressure and temperature data as the GLR dashboard.</p></div>
<div class="caveat"><h3>Caveats</h3><p>The chart is <b>illustrative</b>: a pure component, not a reservoir fluid. The phase envelope of a mixture is a loop in the pressure-temperature plane (bubble and dew branches meeting at the critical point, with a cricondentherm and cricondenbar) and needs the composition, which Sodir does not give; the critical point of a mixture is not that of any of its components. The surface therefore shows regimes relative to a chosen critical point, not the real phase state of a test. The Peng-Robinson equation is accurate to a few per cent for vapour pressures away from the critical point and less for liquid volumes. Molar volumes of different fluids are not comparable. Estimated temperatures (hollow markers) come from the geothermal gradient of the well. With compositions from PVT reports the surface could be replaced by the mixture envelope.</p></div></div>
'''
    b = b[:a] + notes + b[z:]
    b = b.replace("<footer>Generated by <code>python scripts/build_pvt_dashboard.py</code>", "<footer>Generated by <code>python scripts/build_pvt3d_dashboard.py</code>")
    return b


SCRIPT3D = r"""
// ---- 3D variant: the GLR chart is replaced by the pressure-volume-temperature surface of a model fluid -------------------------------------------------------
const EOS = C.eos;
GROUPS.re = ['Regime on the model fluid', C.colours_re];
C.order.re = ['liquid', 'vapour', 'supercritical', 'two-phase'];
D.forEach(r => { r.re = null; r.lv = null; });
const modelId = () => $('f-model').value;
function applyModel() { const P = EOS.points[modelId()]; D.forEach((r, i) => { r.re = P.r[i]; r.lv = P.v[i]; }); }
const REGIME_NAME = {0: 'liquid', 1: 'two-phase', 2: 'vapour', 3: 'supercritical'};
function surfaceTrace(S) {                         // the mesh: one row per isotherm; vertex colours blend across the region boundaries
  const nr = S.t_c.length, nc = S.y[0].length, x = [], y = [], z = [], col = [];
  const rgb = [hexRgb(EOS.zones.liquid), hexRgb(EOS.zones['two-phase']), hexRgb(EOS.zones.vapour), hexRgb(EOS.zones.supercritical)];
  for (let i = 0; i < nr; i++) for (let j = 0; j < nc; j++) { x.push(S.t_c[i]); y.push(S.y[i][j]); z.push(S.z[i][j]); col.push(rgb[S.region[i][j]]); }
  const I = [], J = [], K = [];
  for (let i = 0; i < nr - 1; i++) for (let j = 0; j < nc - 1; j++) { const a = i * nc + j, b = a + 1, c = a + nc, d = c + 1; I.push(a, b); J.push(b, d); K.push(c, c); }
  return {type: 'mesh3d', x, y, z, i: I, j: J, k: K, vertexcolor: col, opacity: 1, flatshading: false, hoverinfo: 'skip', showlegend: false, name: 'PVT surface',
    lighting: {ambient: 0.88, diffuse: 0.45, specular: 0.05, roughness: 0.9, fresnel: 0.05}};
}
const sym3 = (r, hollow) => { const b = (MT[r.mt] || MT.other).symbol3d; return hollow && ['circle', 'square', 'diamond'].includes(b) ? b + '-open' : b; };          // shape = measurement type (scatter3d has fewer shapes)
function pointTraces3d(S, key, palette, f) {
  const out = [];
  categories(S, key).forEach(cat => {
    const base = palette[cat] || '#8A98A3', all = S.filter(r => fv(r, key) === cat);
    out.push({type: 'scatter3d', mode: 'markers', x: [null], y: [null], z: [null], name: `${labelOf(key, cat)} (${all.length})`, legendgroup: String(cat), marker: {symbol: 'circle', size: 4.5, color: base}, hoverinfo: 'skip'});          // neutral colour key
    [false, true].forEach(hollow => {
      const P = all.filter(r => !!r.ti === hollow);
      if (!P.length) return;
      out.push({type: 'scatter3d', mode: 'markers', x: P.map(r => r.t), y: P.map(r => r.lv), z: P.map(r => r.p), name: `${labelOf(key, cat)} (${all.length})`, legendgroup: String(cat), showlegend: false,
        marker: {symbol: P.map(r => sym3(r, hollow)), size: 4.5, color: base, opacity: 1, line: {width: hollow ? 3 : 0.5, color: hollow ? base : '#FFFFFF'}},
        customdata: P.map(r => [r.w, r.n, null, null, null, null, null, null, null, null, null, null, null, null, null, r.k]),
        text: P.map(r => `<b>${r.w}</b> test ${r.n}<br>${fmt(r.p)} bar, ${fmt(r.t)} °C${r.ti ? ' (estimated)' : ''}<br>molar volume ${Math.pow(10, r.lv).toPrecision(3)} L/mol: <b>${r.re}</b> for ${f.name}<br>${C.areaLabel[r.ar] || r.ar}; ${C.playLabel[r.pl] || r.pl}<br>GLR ${fmt(r.g)} Sm³/Sm³, state: ${r.st}<br>API ${r.api === null ? '–' : r.api.toFixed(1)}, ${r.fc}<br>measurement type: ${mtLabel(r)}`),
        hovertemplate: '%{text}<extra></extra>'});
    });
  });
  return out;
}
mainChart = function (R) {
  const id = modelId(), S = EOS.surfaces[id], f = EOS.fluids.find(x => x.id === id), key = $('f-colour').value, [label, palette] = GROUPS[key];
  applyModel();
  const shown = R.filter(r => r.lv !== null && r.lv !== undefined && r.t !== null && r.t !== undefined && r.p <= YMAX), hidden = R.length - shown.length;
  $('ymax-note').textContent = hidden ? `${hidden} test${hidden === 1 ? '' : 's'} not drawn (above ${YMAX.toFixed(0)} bar or without a temperature).` : '';
  const regime = {liquid: 0, vapour: 0, supercritical: 0, 'two-phase': 0}; shown.forEach(r => { regime[r.re] += 1; });
  $('model-sub').innerHTML = `<b>${f.name}</b> (stands in for: ${f.role}): critical point ${f.tc_c.toFixed(0)} °C and ${f.pc_bar.toFixed(1)} bar. Of ${shown.length} tests drawn, ${regime.liquid} lie in its liquid region, ${regime.vapour} in its vapour region, ${regime.supercritical} above its critical temperature (supercritical) and ${regime['two-phase']} on its saturation line.`;
  const tr = [surfaceTrace(S)];
  const line = (d, color, width, name, dash) => ({type: 'scatter3d', mode: 'lines', x: d.x, y: d.y, z: d.z, line: {color, width, dash: dash || 'solid'}, name, showlegend: false, hoverinfo: 'name'});
  tr.push(line(S.dome_liquid, C.colours.text, 5, 'saturated liquid (dome)'), line(S.dome_vapour, C.colours.text, 5, 'saturated vapour (dome)'), line({x: S.critical_isotherm.x, y: S.critical_isotherm.y, z: S.critical_isotherm.z}, '#E57E00', 5, 'critical isotherm'));
  tr.push({type: 'scatter3d', mode: 'markers+text', x: [S.critical.t_c], y: [S.critical.log_v], z: [S.critical.p_bar], text: ['critical point'], textposition: 'top center', textfont: {size: 11, color: C.colours.text}, name: 'critical point', showlegend: false,
    marker: {symbol: 'diamond', size: 7, color: '#E20337', line: {width: 1, color: '#FFFFFF'}}, hovertemplate: `critical point of ${f.name}<br>${f.tc_c.toFixed(1)} °C, ${f.pc_bar.toFixed(1)} bar<extra></extra>`});
  pointTraces3d(shown, key, palette, f).forEach(x => tr.push(x));
  tr.push({type: 'scatter3d', mode: 'markers', x: [], y: [], z: [], name: 'highlight', showlegend: false, hoverinfo: 'skip', marker: {symbol: 'circle-open', size: 11, color: C.colours.text, line: {width: 4, color: C.colours.text}}});
  const ymin = Math.floor(Math.min(...S.y.map(r => Math.min(...r)))), ymax = Math.min(Math.ceil(Math.max(...S.y.map(r => Math.max(...r)))), 3), ticks = [];
  for (let n = ymin; n <= ymax; n++) ticks.push(n);
  const zlo = Math.pow(10, Math.floor(Math.log10(Math.min(...S.z.map(r => Math.min(...r)))))), zt = [0.01, 0.1, 1, 10, 100, 1000].filter(v => v >= zlo && v <= YMAX);
  const axis = (title, extra) => Object.assign({title: {text: title}, gridcolor: C.colours.grid, backgroundcolor: '#F7FAFC', showbackground: true, zerolinecolor: C.colours.grid}, extra);
  const layout = {font: {family: C.font, color: C.colours.text, size: 12}, paper_bgcolor: '#fff', margin: {l: 0, r: 0, t: 8, b: 0}, uirevision: id,
    legend: {orientation: 'h', y: 0.02, yanchor: 'bottom', font: {size: 11}, title: {text: label + ': '}, entrywidth: 190, entrywidthmode: 'pixels', bgcolor: 'rgba(255,255,255,0.85)'},
    scene: {xaxis: axis('Temperature [°C] (measured tests; hollow = estimated)', {range: [S.t_c[0], S.t_c[S.t_c.length - 1]]}), yaxis: axis('Molar volume [L/mol] (modelled; log scale)', {range: [ymin, ymax], tickvals: ticks, ticktext: ticks.map(n => Math.pow(10, n).toLocaleString('en-GB', {maximumSignificantDigits: 3}))}),
      zaxis: axis('Pressure [bar] (measured tests on a modelled surface; log scale)', {type: 'log', range: [Math.log10(zlo), Math.log10(YMAX)], tickvals: zt, ticktext: zt.map(v => v.toLocaleString('en-GB'))}), camera: {eye: {x: 1.55, y: -1.75, z: 0.95}}, aspectmode: 'manual', aspectratio: {x: 1.25, y: 1.25, z: 0.95}}};
  Plotly.react('c-main', tr, layout, {responsive: true, displaylogo: false});
};
mainPoints = T => T.filter(r => r.lv !== null && r.lv !== undefined && r.t !== null && r.t !== undefined && r.p <= YMAX).map(r => [r.t, r.lv, r.p]);
kpis = function (R) {
  applyModel();
  const f = EOS.fluids.find(x => x.id === modelId()), n = R.length, c = k => R.filter(r => r.re === k).length, wells = new Set(R.map(r => r.id)).size;
  const liquid = c('liquid'), vapour = c('vapour'), sc = c('supercritical'), two = c('two-phase'), est = R.filter(r => r.ti).length;
  $('kpis').innerHTML = [[n, `tests in ${wells} wells`], [liquid, 'in the liquid region'], [vapour, 'in the vapour region'], [sc, 'supercritical (above the critical temperature)'], [two, 'on the saturation line'], [est, 'with an estimated temperature (hollow)'], [R.filter(r => r.dm === 'forbidden').length, 'in the forbidden domain of the GLR chart']]
    .map(t => `<div class="kpi"><b>${t[0]}</b><span>${t[1]}</span></div>`).join('');
  $('headline').innerHTML = n ? `For <b>${f.name}</b> (critical point ${f.tc_c.toFixed(0)} °C, ${f.pc_bar.toFixed(0)} bar) <b>${pct(liquid, n)}</b> of the ${n} tests lie in the liquid region, <b>${pct(vapour, n)}</b> in the vapour region and <b>${pct(sc, n)}</b> above the critical temperature. A reservoir colder than the critical temperature of its fluid is an oil reservoir, a hotter one a gas reservoir: change the model fluid to see where the data cross that boundary.` : 'No tests match the selection: tick at least one area and one play.';
};
table = function (R) {
  const states = C.order.st, rowsOf = ['liquid', 'vapour', 'supercritical', 'two-phase'];
  document.querySelector('#t-two thead').innerHTML = '<tr><th>Regime on the model fluid</th>' + states.map(s => `<th>${s}</th>`).join('') + '<th>all</th></tr>';
  document.querySelector('#t-two tbody').innerHTML = rowsOf.map(g => { const S = R.filter(r => r.re === g); return `<tr><td>${g}</td>${states.map(s => `<td>${S.filter(r => r.st === s).length}</td>`).join('')}<td><b>${S.length}</b></td></tr>`; }).join('');
};
const _reset = reset;
reset = function (redraw = true) { _reset(false); $('f-colour').value = 're'; $('f-model').value = EOS.default; if (redraw) update(); };
$('f-model').innerHTML = EOS.fluids.map(f => `<option value="${f.id}">${f.name} (Tc ${f.tc_c.toFixed(0)} °C): ${f.role}</option>`).join('');
const VIEWS = {iso: {eye: {x: 1.55, y: -1.75, z: 0.95}, up: {x: 0, y: 0, z: 1}}, pv: {eye: {x: 3.2, y: 0.001, z: 0.0}, up: {x: 0, y: 0, z: 1}},      // side views from far away: little perspective distortion
  pt: {eye: {x: 0.0, y: -3.2, z: 0.0}, up: {x: 0, y: 0, z: 1}}, vt: {eye: {x: 0.0, y: 0.001, z: 3.4}, up: {x: 0, y: 1, z: 0}}};
document.querySelectorAll('button[data-view]').forEach(b => b.addEventListener('click', () => Plotly.relayout('c-main', {'scene.camera': VIEWS[b.dataset.view]})));
$('legend3d').innerHTML = [['liquid', EOS.zones.liquid], ['two-phase (dome)', EOS.zones['two-phase']], ['vapour', EOS.zones.vapour], ['supercritical', EOS.zones.supercritical]].map(z => `<span style="display:inline-flex;align-items:center;gap:5px;margin-right:12px"><span style="width:14px;height:14px;background:${z[1]};border:1px solid #B8BEC4"></span>${z[0]}</span>`).join('') + '<span style="color:#5B6B77">dark lines: dome; orange: critical isotherm; red diamond: critical point</span>';
"""

VARIANT = {"name": "3d", "title": "PVT: 3D pressure-volume-temperature surface", "body": None, "script": SCRIPT3D, "config": None}


def build_html(pvt: pd.DataFrame, cfg: dict, context: dict | None = None, pcfg: dict | None = None, cfg3d: dict | None = None) -> str:
    """The 3D page as text (pure)."""
    cfg3d = cfg3d or load_yaml("config/pvt3d.yaml")
    variant = {**VARIANT, "body": body3d(), "config": lambda tests: eos_config(tests, cfg3d)}
    return base.build_html(pvt, cfg, context, pcfg, variant)
