"""PVT dashboard (DRAFT): GLR on a log scale against reservoir pressure for every drill stem test, with a map, a pressure-temperature chart and filters.

    (the engine's scripts/build_pvt_dashboard.py reads the mid-layer and calls build_html)

Reads the mart `well_dst_pvt` (docs/pvt.md) with the well positions (canonical `well_header`), the area of each well (regional marts), the plays and the
outlines (country polygons, area outlines) and writes reports/dashboards/pvt_glr_pressure.html: one self-contained page (data embedded as JSON, Plotly
bundled in reports/dashboards/assets/, nothing is fetched when the page is viewed), EDS light theme.

Page: GLR-pressure chart with the flat-colour zones (forbidden domain, diphasic bands, monophasic domains, critical-fluid band; pressure axis to 1,200 bar,
scale on top); a map of the wells that repositions itself to the selected areas / plays; a pressure-temperature chart with the flat-colour zones normal /
high temperature / high pressure / HPHT / UHPHT / extreme HPHT; an API-temperature chart with the biodegradation brackets; filters (areas and plays as
tick boxes, several at once; fluid class; tests with own fluid properties), a colour-by switch (saturation state, fluid class, API class, temperature
bracket, area, play, envelope basis), live sliders for the reference fluid, a table of forbidden-domain tests and a CSV download of the filtered tests.
Every number on the page is computed in the browser from the embedded tests, so it follows the filters.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from ep_fluids import measurement_types as mtypes
from ep_fluids import pvt as pvt_mod
from ep_fluids.plots.pvt_colours import API_COLOURS, BASIS_COLOURS, BRACKET_COLOURS, FLUID_COLOURS, GREY, PT_COLOURS, PT_LABELS, STATE_COLOURS, YELLOW, ZONE_COLOURS  # noqa: F401
from ep_physics_base.figures import hover_panel as hp
from ep_physics_base.figures import theme
from ep_pressure import pressure as pr

OUT_DIR = "reports/dashboards"
PAGE = "pvt_glr_pressure.html"
FONT = "Segoe UI, Arial, sans-serif"
Y_MAX_BAR = 1200.0                                   # pressure axis of the GLR chart
PT_MAX = (260.0, 1500.0)                             # temperature [degC] and pressure [bar] axes of the pressure-temperature chart
MAP_BOX = (-6.0, 54.0, 36.0, 82.0)                   # the part of the world the map outlines are clipped to (lon0, lat0, lon1, lat1)
PLAY_FALLBACK = [theme.BLUE, theme.GREEN, theme.ORANGE, theme.MOSS_GREEN, theme.RED, theme.NORTH_SEA, YELLOW, "#5B6B77", GREY]
AREA_LABELS = {"southern_north_sea": "Southern North Sea", "central_north_sea": "Central North Sea", "northern_north_sea": "Northern North Sea", "norwegian_sea": "Norwegian Sea", "barents_sea": "Barents Sea", "outside": "Outside the five areas"}
AREA_COLOURS = {"northern_north_sea": theme.BLUE, "central_north_sea": theme.GREEN, "norwegian_sea": theme.ORANGE, "barents_sea": theme.MOSS_GREEN, "southern_north_sea": "#7A8A99", "outside": "#B8BEC4"}
NO_PLAY = "none"

# short field names keep the embedded data small
FIELDS = {"wellbore_id": "id", "well_name": "w", "test_number": "n", "region_id": "ar", "play_id": "pl", "lat": "lat", "lon": "lon", "area": "ma", "mid_md_m": "md", "reservoir_pressure_bar": "p", "temperature_used_c": "t", "temperature_is_estimated": "ti",
          "glr_sm3_sm3": "g", "api": "api", "api_class": "ac", "critical_pressure_bar": "pc", "gas_relative_density": "gg", "fluid_class": "fc", "saturation_state": "st", "saturation_basis": "sb", "envelope_basis": "eb",
          "domain": "dm", "oil_limit_glr": "ol", "gas_limit_glr": "gl", "bubble_point_bar": "pb", "dew_point_bar": "pd", "temperature_bracket": "tb", "biodegradation_assessment": "bio", "flags": "fl",
          "near_critical_zone": "nz", "uplift_area": "up", "measurement_type": "mt", "measurement_confidence": "mc", "sample_link": "sl", "drawdown_ratio": "dd", "test_interval_m": "iv"}

CSS = f"""
:root{{--ns:{theme.NORTH_SEA};--blue:{theme.BLUE};--green:{theme.GREEN};--orange:{theme.ORANGE};--red:{theme.RED};--moss:{theme.MOSS_GREEN};--grid:{theme.GRIDLINE};--surface:{theme.SURFACE}}}
*{{box-sizing:border-box}}
body{{margin:0;background:{theme.CANVAS};color:{theme.TEXT_PRIMARY};font-family:{FONT};font-size:14px}}
header{{background:var(--ns);color:#fff;padding:10px 24px;font-size:13px;letter-spacing:.04em}}
main{{max-width:1800px;margin:0 auto;padding:16px 24px 40px 24px}}
h1{{font-size:26px;line-height:1.2;margin:6px 0 4px 0;color:var(--ns)}}
p.headline{{font-size:16px;margin:4px 0 12px 0;border-left:5px solid var(--blue);padding:6px 14px;background:#EAF4FA}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:10px 0}}
.kpi{{border:1px solid var(--grid);border-top:4px solid var(--ns);padding:8px 12px;background:#fff}}
.kpi b{{display:block;font-size:24px;color:var(--ns)}}.kpi span{{font-size:12px;color:#5B6B77}}
.controls{{display:flex;flex-wrap:wrap;gap:14px 22px;align-items:flex-start;background:var(--surface);padding:10px 14px;border:1px solid var(--grid);margin-bottom:10px}}
.controls label{{display:flex;flex-direction:column;font-size:12px;color:#5B6B77;gap:3px}}
.controls select,.controls input[type=range]{{font-family:inherit;font-size:13px;min-width:150px}}
.controls label.check{{flex-direction:row;align-items:center;gap:6px;color:var(--ns);font-size:13px}}
.controls output{{color:var(--ns);font-weight:600}}
fieldset{{border:1px solid var(--grid);background:#fff;padding:4px 10px 6px 10px;margin:0;min-width:0}}
fieldset legend{{font-size:12px;color:#5B6B77;padding:0 4px}}
fieldset .opts{{display:flex;flex-direction:column;gap:1px;max-height:210px;overflow:auto}}
fieldset label.opt{{flex-direction:row;align-items:center;gap:6px;color:var(--ns);font-size:12.5px}}
fieldset label.opt .n{{color:#5B6B77}}fieldset .sw{{display:inline-block;width:10px;height:10px;border:1px solid #fff;box-shadow:0 0 0 1px var(--grid)}}
fieldset .links{{font-size:11.5px;margin-top:2px}}fieldset .links a{{color:var(--blue);cursor:pointer;margin-right:8px}}
button{{font-family:inherit;font-size:13px;padding:5px 12px;border:1px solid var(--ns);background:#fff;color:var(--ns);cursor:pointer}}
button:hover{{background:var(--ns);color:#fff}}
.grid{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,2.5fr) minmax(0,1.1fr);gap:16px;align-items:start}}
@media(max-width:1300px){{.grid{{grid-template-columns:1fr}}}}
.panel{{border:1px solid var(--grid);padding:6px 8px;background:#fff}}
.panel+.panel{{margin-top:12px}}
.panel h2{{font-size:14px;margin:4px 6px;color:var(--ns)}}
.panel p.sub{{font-size:12px;color:#5B6B77;margin:0 6px 4px 6px}}
table{{border-collapse:collapse;width:100%;font-size:12.5px}}
th{{background:var(--ns);color:#fff;text-align:left;padding:5px 8px;position:sticky;top:0}}
td{{padding:4px 8px;border-bottom:1px solid var(--grid)}}tr:nth-child(even) td{{background:var(--surface)}}
.tablewrap{{max-height:340px;overflow:auto;margin-top:6px}}
.notes{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:18px;font-size:13.5px}}
@media(max-width:1000px){{.notes{{grid-template-columns:1fr}}}}
.notes h3{{font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:var(--moss);margin:0 0 4px 0}}
.caveat{{border-left:5px solid var(--orange);background:#FFF4E5;padding:6px 14px}}
footer{{margin-top:20px;font-size:12px;color:#5B6B77;border-top:1px solid var(--grid);padding-top:8px}}
.glyph{{display:inline-block;width:16px;text-align:center;color:#5B6B77;font-size:13px}}
.draft{{display:inline-block;background:var(--orange);color:#fff;font-size:11px;padding:1px 8px;margin-left:8px;vertical-align:middle;letter-spacing:.06em}}
"""

BODY = """
<header>E&amp;P Geoscience Analytics &middot; Norwegian Continental Shelf &middot; data: Sodir open data (CC BY 4.0) &middot; PVT screening (docs/pvt.md)</header>
<main>
<h1>GLR against reservoir pressure: which DST fluids can be one phase?<span class="draft">DRAFT</span></h1>
<p class="headline" id="headline"></p>
<div class="kpis" id="kpis"></div>
<div class="controls">
  <label>Fluid class<select id="f-fluid"></select></label>
  <label>Colour by<select id="f-colour">
    <option value="st">Saturation state</option><option value="fc">Fluid class</option><option value="pl">Play</option><option value="a">Area</option><option value="ac">API class</option><option value="tb">Temperature bracket</option><option value="eb">Envelope basis</option></select></label>
  <div style="display:flex;flex-direction:column;gap:6px">
    <label class="check"><input type="checkbox" id="f-own"> only tests with own API, gas gravity and a gauge or estimated temperature</label>
    <label class="check"><input type="checkbox" id="f-shade" checked> shade the domains</label>
    <div><button id="b-reset" type="button">Reset</button> <button id="b-csv" type="button">Download filtered tests (CSV)</button></div></div>
  <label>Reference API <output id="o-api"></output><input type="range" id="s-api" min="10" max="60" step="1"></label>
  <label>Reference temperature [&deg;C] <output id="o-t"></output><input type="range" id="s-t" min="30" max="200" step="5"></label>
  <label>Reference gas gravity <output id="o-gg"></output><input type="range" id="s-gg" min="0.55" max="1.2" step="0.05"></label>
</div>
<div class="grid">
  <div class="filters">
    <div class="panel"><h2>Map of the wells</h2><p class="sub" id="map-sub"></p><div id="c-map" style="height:470px"></div></div>
    <div class="panel"><fieldset><legend>Areas (tick one or more)</legend><div class="opts" id="f-areas"></div><div class="links"><a id="a-all">all</a><a id="a-none">none</a></div></fieldset>
      <fieldset style="margin-top:8px"><legend>Plays (tick one or more)</legend><div class="opts" id="f-plays"></div><div class="links"><a id="p-all">all</a><a id="p-none">none</a></div></fieldset>
      <fieldset style="margin-top:8px"><legend>Measurement type (shape of the marker)</legend><div class="opts" id="f-mtypes"></div><div class="links"><a id="m-all">all</a><a id="m-none">none</a></div><label class="opt" style="margin-top:4px"><input type="checkbox" id="f-wire" checked> map: wells with wireline sample records</label></fieldset></div>
  </div>
  <div class="panel"><h2>GLR (log scale, on top) against reservoir pressure (increasing downwards)</h2><p class="sub">Grey: forbidden domain, no single-phase fluid. Beneath it the diphasic bands (a fluid at its saturation pressure: oil with gas cap, gas with oil ring), then the monophasic domains: undersaturated oil, critical fluid (vertical band), undersaturated gas. The colours blend across the zone boundaries; the dashed line is the exact envelope. <b>Hollow markers</b>: the test has no gauge temperature, the temperature is estimated from the geothermal gradient of its well and feeds the envelope.  <b>Shape</b> = measurement type (key beside the map, issues in the table below the chart).<span id="ymax-note"></span></p>
    <div id="c-main" style="height:860px"></div></div>
  <div>
    <div class="panel"><h2>Pressure against temperature</h2><p class="sub">HSE-style classes: HPHT needs both pressure and temperature above the thresholds (690 bar and 149 &deg;C; ultra-HPHT 1,034 bar and 177 &deg;C; extreme 1,379 bar and 204 &deg;C). <b>Hollow markers</b>: estimated temperature (no gauge value); the HPHT class of a test uses gauge temperatures only.</p><div id="c-pt" style="height:400px"></div></div>
    <div class="panel"><h2>API gravity against temperature</h2><p class="sub">Biodegradation window below about 80 &deg;C; hotter reservoirs were pasteurised. Larger markers: uplift area. Hover a point in any chart to highlight the same test (or well) in the others.</p><div id="c-api" style="height:340px"></div></div>
  </div>
</div>
<div class="panel" style="margin-top:14px"><h2>Tests in the forbidden domain (highest pressure first, up to 25)</h2><p class="sub">Pressure below the saturation pressure of the test's GLR: a gas cap, coning, commingled intervals, two-phase inflow, or a wrong pressure or rate.</p>
  <div class="tablewrap"><table id="t-two"><thead><tr><th>Well</th><th>Test</th><th>Area</th><th>Play</th><th>MD [m RT]</th><th>Pressure [bar]</th><th>GLR [Sm&sup3;/Sm&sup3;]</th><th>Oil limit at P</th><th>Gas limit at P</th><th>Critical pressure [bar]</th><th>Basis</th></tr></thead><tbody></tbody></table></div></div>
<div class="panel" style="margin-top:14px"><h2>Measurement types: what each can go wrong with</h2><p class="sub">The shape of a marker is the measurement type, classified from the official Sodir records only (never guessed). Different types have different issues: a DST can show forbidden-domain properties through incomplete clean-up and filtrate contamination or a rapid-drawdown transient; a wireline sample through filtrate contamination. The share of forbidden tests per type follows the selection. Issues are listed with their literature source in docs/measurement_types.md; they are shown, never corrected. Types with no test today are defined for the data to come.</p>
  <div class="tablewrap"><table id="t-types"><thead><tr><th></th><th>Measurement type</th><th>Tests</th><th>Forbidden</th><th>Classifiable from open data</th><th>Typical issues</th></tr></thead><tbody></tbody></table></div></div>
<div class="notes"><div><h3>How to read</h3>
<p>Each point is a drill stem test with a reservoir pressure (bottom-hole) and a producing gas-liquid ratio. The dashed line is the <b>saturation-pressure envelope</b> of the reference fluid set by the sliders: its left branch is the bubble point (the most gas an oil holds in solution at that pressure, Glaso correlation for North Sea oils), its right branch the dew point (the least GLR a gas can have without dropping liquid); they meet in the <b>critical band</b> (vertical yellow) at the critical pressure. The dew-point branch is calibrated to the reference GLR-pressure chart; the reference fluid is 40 API, 55 &deg;C and gas gravity 1.00. Each test is classified with <i>its own</i> API, temperature and gas gravity where it has them (a temperature estimated from the geothermal gradient of the well counts, shown hollow).
<b>Forbidden domain</b> (grey, above the envelope): the pressure is below the saturation pressure of that GLR, so no single-phase fluid exists. <b>Diphasic bands</b> beneath it: the fluid is at its saturation pressure, within the uncertainty of the estimate, so a gas cap (oil) or an oil ring (gas) may be in equilibrium with it. <b>Monophasic domains</b> beneath those: undersaturated oil, critical fluid, undersaturated gas. A state taken from a <b>PVT report</b> (diamond) uses the stated bubble or dew point instead of the envelope.
The <b>map</b> shows the wells of the selected areas and plays and zooms to them; <b>areas</b> are the five Sodir structural-province areas (Southern, Central and Northern North Sea, Norwegian Sea, Barents Sea), <b>plays</b> are placed by the depth of the test inside the play interval of that well (a proxy).</p>
<h3>Sources</h3><p>Sodir drill stem tests (<code>well_dst</code>), pressure screening (<code>well_dst_pressure</code>, docs/pressure.md), PVT classification (<code>well_dst_pvt</code>, docs/pvt.md), well positions (<code>well_header</code>), Natural Earth country outlines. Glaso (1980) and Standing (1947) bubble-point correlations; McCain fluid classes; HSE HPHT thresholds; Wilhelms et al. (Nature, 2001) and Larter et al. for pasteurisation.</p></div>
<div class="caveat"><h3>Caveats</h3><p>The envelopes are literature-based screening assumptions, not field PVT: the bubble-point correlation is good to about &plusmn;7 % (Glaso, on its own data) to &plusmn;15&ndash;20 % (Standing) in GOR, and the dew-point curve was digitised by eye from a photograph of the reference chart, and the reference fluid (40 API, 55 &deg;C, gas gravity 1.00) closes the U at about 378 bar against 387 bar on the chart with the left branch about 30 % low in the mid-range (config/pvt.yaml, DQ-086). Estimated temperatures (hollow markers) come from the seabed temperature and the geothermal gradient of the well (docs/temperature.md, DQ-089, DQ-090) and are less certain than gauge values; bottom-hole temperature corrections that need data we do not have (Horner, Waples, Hermanrud) wait for it.
The critical band (500&ndash;1,200 Sm&sup3;/Sm&sup3;) is configured, not a law. A saturated reading is evidence for a gas cap or oil ring, not proof. The test GLR is a test average, not the initial solution GOR. Tests without oil density, gas density or temperature use the reference fluid (envelope basis). Test temperature is a minimum of the maximum past temperature; in the Barents Sea (uplift) pasteurisation may be under-classified (DQ-087). Forbidden-domain tests are flagged, never corrected. Plays are a depth-based proxy. Only tests with a bottom-hole reservoir pressure and a GLR above zero are shown; tests above the pressure axis limit are counted but not drawn.</p></div></div>
<footer>Generated by <code>python scripts/build_pvt_dashboard.py</code> from the mid-layer; all numbers are computed in the page from the embedded tests. Colours: EDS light theme. DRAFT: layout and filters are open for comment.</footer>
</main>
"""

SCRIPT = r"""
const D = __DATA__, C = __CFG__;
D.forEach((r, i) => { r.k = i; if (!r.mt) r.mt = 'dst'; });               // index of the test: links a point to the same test in the other charts
const $ = id => document.getElementById(id);
const XMIN = 10, XMAX = 1000000, YMAX = C.ymax, TICKS = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000], BAND = C.band, SAT = C.sat;
const GROUPS = {st: ['Saturation state', C.colours.st], fc: ['Fluid class', C.colours.fc], ac: ['API class', C.colours.ac], tb: ['Temperature bracket', C.colours.tb], eb: ['Envelope basis', C.colours.eb], a: ['Area', C.colours.a], pl: ['Play', C.colours.pl]};
const FIELD = {a: 'ar'};
const MT = Object.fromEntries(C.measurement.types.map(t => [t.id, t])), LINE_ONLY = new Set(['x', 'cross']);          // shape = measurement type; a line-only shape cannot be hollow
const symOf = r => (MT[r.mt] || MT.other).symbol, mtLabel = r => (MT[r.mt] || MT.other).label;
const fv = (r, key) => r[FIELD[key] || key];
const labelOf = (key, v) => key === 'a' ? (C.areaLabel[v] || v) : key === 'pl' ? (C.playLabel[v] || v) : v;
const fmt = (v, d = 0) => (v === null || v === undefined || Number.isNaN(v)) ? '–' : Number(v).toLocaleString('en-GB', {maximumFractionDigits: d, minimumFractionDigits: d});
const pct = (n, m) => m ? (100 * n / m).toFixed(0) + ' %' : '–';
const FONT = {family: C.font, color: C.colours.text};

function gasLimit(p) {
  const xs = C.gas.p, ys = C.gas.glr, n = xs.length;
  if (p <= xs[0]) return ys[0];
  if (p >= xs[n - 1]) return ys[n - 1];
  for (let i = 1; i < n; i++) if (p <= xs[i]) { const f = (p - xs[i - 1]) / (xs[i] - xs[i - 1]); return Math.pow(10, Math.log10(ys[i - 1]) + f * (Math.log10(ys[i]) - Math.log10(ys[i - 1]))); }
}
function oilLimit(p, api, t, gg) {
  const tf = t * 1.8 + 32, psia = p * C.psiPerBar;
  if (C.leftBranch === 'glaso') {            // Glaso (1980) Eq. 6a solved for the rising branch (same as pvt.glaso_max_gor)
    const g = C.glaso, disc = Math.max(g.b * g.b - 4 * g.c * (Math.log10(Math.max(psia, 1e-3)) - g.a), 0), pbStar = Math.pow(10, (g.b - Math.sqrt(disc)) / (2 * g.c));
    return gg * Math.pow(pbStar * Math.pow(api, g.api_exponent) / Math.pow(tf, g.temp_exponent), 1 / g.gor_exponent) * C.toSm3;
  }
  const s = C.standing;
  return gg * Math.pow((psia / s.p_divisor + s.p_offset) * Math.pow(10, s.api_coeff * api - s.temp_coeff * tf), s.exponent) * C.toSm3;
}
function bubblePoint(glr, api, t, gg) {      // pressure [bar] at which the bubble-point branch has this GLR (Standing or Glaso inverted, as set in config/pvt.yaml)
  if (C.leftBranch === 'glaso') {
    const g = C.glaso, tfg = t * 1.8 + 32, x = Math.log10(Math.pow(glr / C.toSm3 / gg, g.gor_exponent) * Math.pow(tfg, g.temp_exponent) / Math.pow(api, g.api_exponent));
    return Math.pow(10, g.a + g.b * x - g.c * x * x) / C.psiPerBar;
  }
  const s = C.standing, tf = t * 1.8 + 32, inner = Math.pow(glr / C.toSm3 / gg, 1 / s.exponent) / Math.pow(10, s.api_coeff * api - s.temp_coeff * tf);
  return Math.max((inner - s.p_offset) * s.p_divisor, 0) / C.psiPerBar;
}
function gasDewPressure(glr) {                // pressure [bar] at which the dew-point branch has this GLR (inverse of gasLimit)
  const xs = C.gas.p, ys = C.gas.glr, n = xs.length, lg = Math.log10(glr);
  if (glr >= ys[0]) return xs[0];
  if (glr <= ys[n - 1]) return xs[n - 1];
  for (let i = 1; i < n; i++) if (glr >= ys[i]) { const f = (Math.log10(ys[i - 1]) - lg) / (Math.log10(ys[i - 1]) - Math.log10(ys[i])); return xs[i - 1] + f * (xs[i] - xs[i - 1]); }
}
function optionBoxes(id, cls, items, colourOf) {
  $(id).innerHTML = items.map(it => `<label class="opt"><input type="checkbox" class="${cls}" value="${it.id}" checked><span class="sw" style="background:${colourOf(it)}"></span>${it.label} <span class="n">(${it.n})</span></label>`).join('');
}
function typeBoxes() {
  const n = id => D.filter(r => r.mt === id).length;
  $('f-mtypes').innerHTML = C.measurement.types.filter(t => n(t.id) > 0).map(t => `<label class="opt" title="${t.evidence}"><input type="checkbox" class="f-mtype" value="${t.id}" checked><span class="glyph">${t.glyph}</span>${t.label} <span class="n">(${n(t.id)})</span></label>`).join('');
}
function populate() {
  const fill = (id, values, all) => { $(id).innerHTML = `<option value="all">${all}</option>` + values.map(v => `<option>${v}</option>`).join(''); };
  const count = (field, id) => D.filter(r => r[field] === id).length;
  optionBoxes('f-areas', 'f-area', C.areas.map(a => ({...a, n: count('ar', a.id)})), a => C.colours.a[a.id] || '#B8BEC4');
  optionBoxes('f-plays', 'f-play', C.plays.map(p => ({...p, n: count('pl', p.id)})), p => C.colours.pl[p.id] || '#B8BEC4');
  typeBoxes();
  fill('f-fluid', C.order.fc.filter(v => D.some(r => r.fc === v)), 'All fluid classes');
  reset(false);
}
function setAll(cls, on) { document.querySelectorAll('.' + cls).forEach(e => { e.checked = on; }); update(); }
function reset(redraw = true) {
  $('s-api').value = C.ref.api; $('s-t').value = C.ref.temperature_c; $('s-gg').value = C.ref.gas_gravity;
  $('f-fluid').value = 'all'; $('f-colour').value = 'st'; $('f-own').checked = false; $('f-shade').checked = true;
  document.querySelectorAll('.f-area, .f-play, .f-mtype').forEach(e => { e.checked = true; });
  if ($('f-wire')) $('f-wire').checked = true;
  if (redraw) update();
}
const checkedSet = cls => new Set([...document.querySelectorAll('.' + cls + ':checked')].map(e => e.value));
function rows() {
  const areas = checkedSet('f-area'), plays = checkedSet('f-play'), mtypes = checkedSet('f-mtype'), fluid = $('f-fluid').value, own = $('f-own').checked;
  return D.filter(r => areas.has(r.ar) && plays.has(r.pl) && mtypes.has(r.mt) && (fluid === 'all' || r.fc === fluid) && (!own || r.eb === 'test' || r.eb === 'test (estimated temperature)'));
}
function kpis(R) {
  const n = R.length, c = k => R.filter(r => r.st === k).length;
  const uo = c('undersaturated oil'), ug = c('undersaturated gas'), cf = c('critical fluid'), so = c('saturated oil'), sg = c('saturated gas'), fb = c('forbidden'), rep = R.filter(r => r.sb === 'pvt_report').length;
  const wells = new Set(R.map(r => r.id)).size;
  const tiles = [[n, `tests in ${wells} wells`], [uo, 'monophasic: undersaturated oil'], [ug, 'monophasic: undersaturated gas'], [cf, 'critical fluid'], [so + sg, `diphasic, at saturation pressure (${so} oil with gas cap, ${sg} gas with oil ring)`], [pct(fb, n), `forbidden domain (${fb})`], [R.filter(r => r.ti).length, 'tests with an estimated temperature (hollow)'], [rep, 'states from a PVT report']];
  $('kpis').innerHTML = tiles.map(t => `<div class="kpi"><b>${t[0]}</b><span>${t[1]}</span></div>`).join('');
  const heavy = R.filter(r => r.bio && r.bio.startsWith('likely')).length;
  $('headline').innerHTML = n ? `Of <b>${n}</b> tests with a reservoir pressure and a GLR, <b>${pct(uo + ug + cf, n)}</b> are monophasic (${uo} undersaturated oil, ${ug} undersaturated gas, ${cf} critical fluid), <b>${pct(so + sg, n)}</b> sit at their saturation pressure (${so} oil with gas cap, ${sg} gas with oil ring) and <b>${fb}</b> (${pct(fb, n)}) fall in the forbidden domain, where no single-phase fluid can exist. ${heavy} ${heavy === 1 ? 'test is' : 'tests are'} heavy and cool (likely biodegraded).`
    : 'No tests match the selection: tick at least one area and one play.';
}
function zonePolygons(api, t, gg, pmax) {
  const clip = (v, a, b) => Math.min(Math.max(v, a), b), pg = [];
  for (let p = 0; p <= pmax; p += 5) pg.push(p);
  const oil = pg.map(p => oilLimit(p, api, t, gg)), gas = pg.map(gasLimit);
  const oilEdge = oil.map(v => clip(v, XMIN, BAND[0])), oilUnder = oil.map(v => clip(SAT.under * v, XMIN, BAND[0]));
  const gasEdge = gas.map(v => clip(v, BAND[1], XMAX)), gasRing = gas.map(v => clip(SAT.gas * v, BAND[1], XMAX));
  const pc = Math.max(bubblePoint(BAND[0], api, t, gg), gasDewPressure(BAND[1])), pcd = Math.min(pc, pmax);
  const pf = pg.filter(p => p < pcd).concat([pcd]);
  const of = pf.map(p => clip(oilLimit(p, api, t, gg), XMIN, BAND[0])), gf = pf.map(p => clip(gasLimit(p), BAND[1], XMAX));
  const between = (l, r, y) => ({x: l.concat(r.slice().reverse()), y: y.concat(y.slice().reverse())});
  const zones = {
    'forbidden': between(of, gf, pf),
    'diphasic: oil with gas cap': between(oilUnder, oilEdge, pg), 'diphasic: gas with oil ring': between(gasEdge, gasRing, pg),
    'monophasic: undersaturated oil': between(pg.map(() => XMIN), oilUnder, pg), 'monophasic: undersaturated gas': between(gasRing, pg.map(() => XMAX), pg),
    'monophasic: critical fluid': {x: [BAND[0], BAND[1], BAND[1], BAND[0], BAND[0]], y: [pcd, pcd, pmax, pmax, pcd]}};
  return {zones, outline: {x: of.concat(gf.slice().reverse()), y: pf.concat(pf.slice().reverse())}, pc};
}
function hexRgb(h) { return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]; }
function boxBlur(src, W, H, r) {                 // separable box blur of an RGB float image (edges clamped)
  const tmp = new Float32Array(src.length), out = new Float32Array(src.length), n = 2 * r + 1;
  for (let y = 0; y < H; y++) for (let c = 0; c < 3; c++) {
    let s = 0;
    for (let k = -r; k <= r; k++) s += src[(y * W + Math.min(Math.max(k, 0), W - 1)) * 3 + c];
    for (let x = 0; x < W; x++) {
      tmp[(y * W + x) * 3 + c] = s / n;
      s += src[(y * W + Math.min(x + r + 1, W - 1)) * 3 + c] - src[(y * W + Math.max(x - r, 0)) * 3 + c];
    }
  }
  for (let x = 0; x < W; x++) for (let c = 0; c < 3; c++) {
    let s = 0;
    for (let k = -r; k <= r; k++) s += tmp[(Math.min(Math.max(k, 0), H - 1) * W + x) * 3 + c];
    for (let y = 0; y < H; y++) {
      out[(y * W + x) * 3 + c] = s / n;
      s += tmp[(Math.min(y + r + 1, H - 1) * W + x) * 3 + c] - tmp[(Math.max(y - r, 0) * W + x) * 3 + c];
    }
  }
  return out;
}
function zoneRaster(api, t, gg, pmax) {          // pastel colour of every zone, blurred so the colours adjust into each other (image under the chart)
  const W = 300, H = 260, lx0 = Math.log10(XMIN), lx1 = Math.log10(XMAX);
  const pc = Math.max(bubblePoint(BAND[0], api, t, gg), gasDewPressure(BAND[1])), rgb = {};
  C.zoneOrder.forEach(z => { rgb[z] = hexRgb(C.zones[z]); });
  const img = new Float32Array(W * H * 3);
  for (let j = 0; j < H; j++) {
    const p = (j + 0.5) / H * pmax, oil = oilLimit(p, api, t, gg), gas = gasLimit(p);
    for (let i = 0; i < W; i++) {
      const g = Math.pow(10, lx0 + (i + 0.5) / W * (lx1 - lx0));
      let z;
      if (g < BAND[0]) z = g > oil ? 'forbidden' : g < SAT.under * oil ? 'monophasic: undersaturated oil' : 'diphasic: oil with gas cap';
      else if (g > BAND[1]) z = g < gas ? 'forbidden' : g < SAT.gas * gas ? 'diphasic: gas with oil ring' : 'monophasic: undersaturated gas';
      else z = p < pc ? 'forbidden' : 'monophasic: critical fluid';
      const c = rgb[z], o = (j * W + i) * 3; img[o] = c[0]; img[o + 1] = c[1]; img[o + 2] = c[2];
    }
  }
  let blur = img;
  for (let k = 0; k < 3; k++) blur = boxBlur(blur, W, H, 4);          // three passes approximate a gaussian
  const canvas = document.createElement('canvas'); canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d'), data = ctx.createImageData(W, H);
  for (let k = 0; k < W * H; k++) { data.data[k * 4] = blur[k * 3]; data.data[k * 4 + 1] = blur[k * 3 + 1]; data.data[k * 4 + 2] = blur[k * 3 + 2]; data.data[k * 4 + 3] = 255; }
  ctx.putImageData(data, 0, 0);
  return canvas.toDataURL('image/png');
}
const highlightTrace = () => ({x: [], y: [], mode: 'markers', type: 'scatter', name: 'highlight', showlegend: false, hoverinfo: 'skip', marker: {symbol: 'circle', size: 21, color: 'rgba(255,255,255,0)', line: {width: 3, color: C.colours.text}}});
function categories(R, key) {
  const present = new Set(R.map(r => fv(r, key)));
  const order = (C.order[key] || []).filter(c => present.has(c));
  return [...order, ...[...present].filter(c => !order.includes(c)).sort()];
}
function pointTraces(S, key, palette, extra, hover, sizeOf) {
  const out = [];
  categories(S, key).forEach(cat => {
    const base = palette[cat] || '#8A98A3', all = S.filter(r => fv(r, key) === cat);
    if (!(extra.more && extra.more.showlegend === false)) out.push(Object.assign({x: [null], y: [null], mode: 'markers', type: 'scatter', name: `${labelOf(key, cat)} (${all.length})`, legendgroup: String(cat), marker: {symbol: 'circle', size: 8, color: base}, hoverinfo: 'skip'}, extra.more || {}));          // neutral colour key: the SHAPE belongs to the measurement type
    [false, true].forEach(hollow => {                         // solid markers: gauge temperature; hollow markers: estimated temperature (one trace each, one legend entry per category)
      const P = all.filter(r => !!r.ti === hollow);
      if (!P.length) return;
      out.push(Object.assign({x: P.map(r => extra.x(r)), y: P.map(r => extra.y(r)), mode: 'markers', type: 'scatter', name: `${labelOf(key, cat)} (${all.length})`, legendgroup: String(cat), showlegend: false,
        marker: {color: P.map(r => hollow && !LINE_ONLY.has(symOf(r)) ? 'rgba(255,255,255,0)' : base), size: P.map(sizeOf || (r => r.mt === 'pvt_report' ? 11 : 7.5)), symbol: P.map(symOf), opacity: 0.95,
          line: {width: P.map(r => hollow ? 2 : (r.mt === 'pvt_report' ? 1.5 : 0.7)), color: P.map(r => hollow ? base : (r.mt === 'pvt_report' ? C.colours.text : '#FFFFFF'))}},
        customdata: P.map(r => [r.w, r.n, r.md, r.t, r.api, r.fc, r.st, r.sb, r.eb, r.fl || '', r.g / C.toSm3, r.pb, r.pd, C.areaLabel[r.ar] || r.ar, C.playLabel[r.pl] || r.pl, r.k, r.ti ? ' (estimated from the well gradient)' : '', mtLabel(r), r.mc || '–', r.p, r.g]), hovertemplate: hover}, extra.more || {}));
    });
  });
  return out;
}
const HOVER = '<b>%{customdata[0]}</b> test %{customdata[1]}<br>%{customdata[19]:.0f} bar, %{customdata[3]:.0f} °C, GLR %{customdata[20]:,.0f}<extra></extra>';      // two lines next to the point; the rest is in the detail panel (src/utils/hover_panel.py)
const NUM = (v, d) => (v === null || v === undefined || isNaN(v)) ? '–' : Number(v).toLocaleString('en-GB', {minimumFractionDigits: d || 0, maximumFractionDigits: d || 0});
window.HP_DETAIL = (tr, p) => { const cd = p.customdata || (tr.customdata && tr.customdata[p.pointNumber]); if (!cd || cd.length < 21) return null;
  return `<b>${cd[0]}</b> test ${cd[1]}<br>${cd[13]}; ${cd[14]}<br>MD ${NUM(cd[2])} m RT, ${NUM(cd[19])} bar, ${NUM(cd[3])} °C${cd[16]}<br>GLR ${NUM(cd[20])} Sm³/Sm³ (${NUM(cd[10])} scf/stb)<br>API ${NUM(cd[4], 1)}, ${cd[5]}<br>state: ${cd[6]} (${cd[7]})<br>envelope basis: ${cd[8]}<br>Pb ${NUM(cd[11])} bar, Pdew ${NUM(cd[12])} bar<br>measurement type: ${cd[17]} (${cd[18]} confidence)<br>${cd[9]}`; };
function mainChart(R) {
  const key = $('f-colour').value, [label, palette] = GROUPS[key];
  const api = +$('s-api').value, t = +$('s-t').value, gg = +$('s-gg').value;
  $('o-api').textContent = api; $('o-t').textContent = t; $('o-gg').textContent = gg.toFixed(2);
  const pmax = YMAX, Z = zonePolygons(api, t, gg, pmax), tr = [];
  const shown = R.filter(r => r.p <= pmax), hidden = R.length - shown.length;
  $('ymax-note').textContent = hidden ? `${hidden} test${hidden === 1 ? '' : 's'} above ${pmax.toFixed(0)} bar not drawn.` : '';
  const shade = $('f-shade').checked;
  if (shade) {
    C.zoneOrder.forEach(name => {
      tr.push({x: Z.zones[name].x, y: Z.zones[name].y, mode: 'lines', line: {width: 0}, fill: 'toself', fillcolor: 'rgba(255,255,255,0)', name: name, showlegend: false, hoveron: 'fills', hoverinfo: 'name'});
      tr.push({x: [null], y: [null], mode: 'markers', marker: {symbol: 'square', size: 11, color: C.zones[name], line: {width: 1, color: '#B8BEC4'}}, name: name, legend: 'legend', hoverinfo: 'skip'});
    });
    tr.push({x: Z.outline.x, y: Z.outline.y, mode: 'lines', legend: 'legend', name: `envelope: ${api} API, ${t} °C, γg ${gg.toFixed(2)}, Pc ${Z.pc.toFixed(0)} bar`, line: {color: C.colours.text, width: 1.6, dash: 'dash'}, hoverinfo: 'skip'});
  }
  pointTraces(shown, key, palette, {x: r => r.g, y: r => r.p, more: {legend: 'legend2'}}, HOVER).forEach(x => tr.push(x));
  if (shown.some(r => r.ti)) tr.push({x: [null], y: [null], mode: 'markers', legend: 'legend2', name: `hollow: estimated temperature (${shown.filter(r => r.ti).length})`, marker: {symbol: 'circle', size: 8, color: 'rgba(255,255,255,0)', line: {width: 2, color: C.colours.text}}, hoverinfo: 'skip'});
  tr.push(highlightTrace());
  C.measurement.types.filter(t => shown.some(r => r.mt === t.id)).forEach(t => tr.push({x: [null], y: [null], mode: 'markers', legend: 'legend3', name: `${t.label} (${shown.filter(r => r.mt === t.id).length})`, marker: {symbol: t.symbol, size: 8, color: '#8A98A3', line: {width: 1, color: '#5B6B77'}}, hoverinfo: 'skip'}));
  const lab = (x, y, text, extra = {}) => Object.assign({x: Math.log10(x), y, xref: 'x', yref: 'y', text, showarrow: false, font: {color: C.colours.text, size: 12}}, extra);
  const layout = {font: {family: C.font, color: C.colours.text, size: 12}, paper_bgcolor: '#fff', plot_bgcolor: '#fff', margin: {l: 70, r: 20, t: 100, b: 215}, hovermode: 'closest',
    xaxis: {type: 'log', side: 'top', range: [Math.log10(XMIN), Math.log10(XMAX)], title: {text: 'GLR [Sm³/Sm³] (log scale)  ·  1 Sm³/Sm³ = 5.6 scf/stb', standoff: 12}, gridcolor: C.colours.grid, layer: 'above traces', ticks: 'outside', zeroline: false, showline: true, linecolor: C.colours.text,
      tickvals: TICKS, ticktext: TICKS.map(v => v.toLocaleString('en-GB'))},
    yaxis: {range: [pmax, 0], title: {text: 'Reservoir pressure [bar], increasing downwards'}, gridcolor: C.colours.grid, layer: 'above traces', ticks: 'outside', zeroline: false, showline: true, linecolor: C.colours.text},
    legend: {orientation: 'h', y: -0.03, yanchor: 'top', font: {size: 11}, entrywidth: 215, entrywidthmode: 'pixels', title: {text: 'Zones: '}},
    legend2: {orientation: 'h', y: -0.17, yanchor: 'top', font: {size: 11}, entrywidth: 215, entrywidthmode: 'pixels', title: {text: label + ': '}},
    legend3: {orientation: 'h', y: -0.26, yanchor: 'top', font: {size: 11}, entrywidth: 260, entrywidthmode: 'pixels', title: {text: 'Measurement type (shape): '}},
    images: shade ? [{source: zoneRaster(api, t, gg, pmax), xref: 'paper', yref: 'paper', x: 0, y: 1, sizex: 1, sizey: 1, xanchor: 'left', yanchor: 'top', sizing: 'stretch', layer: 'below', opacity: 1}] : [],
    annotations: shade ? [lab(1000, 38, '<b>FORBIDDEN DOMAIN</b>'), lab(25, pmax * 0.94, 'undersaturated oil'), lab(100000, pmax * 0.94, 'undersaturated gas'), lab(Math.sqrt(BAND[0] * BAND[1]), Math.min(pmax * 0.96, Z.pc + 0.5 * (pmax - Z.pc)), 'critical fluid', {textangle: -90})] : []};
  Plotly.react('c-main', tr, layout, {responsive: true, displaylogo: false});
}
function mapChart(R) {
  const key = $('f-colour').value, palette = GROUPS[key][1], byWell = new Map();
  R.forEach(r => {
    if (r.lat === null || r.lat === undefined || r.lon === null || r.lon === undefined) return;
    let w = byWell.get(r.id);
    if (!w) { w = {id: r.id, name: r.w, lat: r.lat, lon: r.lon, n: 0, cats: {}, mts: {}, areaId: r.ar}; byWell.set(r.id, w); }
    const c = fv(r, key); w.n++; w.cats[c] = (w.cats[c] || 0) + 1; w.mts[r.mt] = (w.mts[r.mt] || 0) + 1;
  });
  const W = [...byWell.values()];
  W.forEach(w => { w.cat = Object.keys(w.cats).sort((a, b) => w.cats[b] - w.cats[a])[0]; w.mt = Object.keys(w.mts).sort((a, b) => w.mts[b] - w.mts[a])[0]; });
  const noPos = new Set(R.filter(r => r.lat === null || r.lat === undefined).map(r => r.id)).size;
  $('map-sub').textContent = `${W.length} wells with tests${noPos ? ` (${noPos} more without coordinates)` : ''}; the map follows the area and play selection. Marker size: number of tests; colour: most frequent ${GROUPS[key][0].toLowerCase()}; shape: most frequent measurement type; grey hexagons: wells with wireline formation-tester sample records (no GLR, so not tests).`;
  let box = [0, 56, 32, 75];
  if (W.length) {
    const lons = W.map(w => w.lon), lats = W.map(w => w.lat);
    let lo0 = Math.min(...lons), lo1 = Math.max(...lons), la0 = Math.min(...lats), la1 = Math.max(...lats);
    const dx = Math.max(lo1 - lo0, 2.5), dy = Math.max(la1 - la0, 1.2), cx = (lo0 + lo1) / 2, cy = (la0 + la1) / 2;
    box = [cx - dx * 0.62, cy - dy * 0.62, cx + dx * 0.62, cy + dy * 0.62];
  }
  const midLat = (box[1] + box[3]) / 2, selected = checkedSet('f-area');
  const tr = [{x: C.outlines.countries.x, y: C.outlines.countries.y, mode: 'lines', fill: 'toself', fillcolor: '#EEF1F3', line: {color: '#B8BEC4', width: 0.6}, hoverinfo: 'skip', showlegend: false}];
  C.outlines.areas.forEach(a => tr.push({x: a.x, y: a.y, mode: 'lines', line: {color: selected.has(a.id) ? C.colours.text : '#B8BEC4', width: selected.has(a.id) ? 1.6 : 0.8, dash: 'dot'}, hoverinfo: 'skip', showlegend: false}));
  const wire = ($('f-wire') && $('f-wire').checked) ? C.wireline.filter(w => selected.has(w.ar)) : [];
  if (wire.length) tr.push({x: wire.map(w => w.lon), y: wire.map(w => w.lat), mode: 'markers', type: 'scatter', showlegend: false, text: wire.map(w => `<b>${w.w}</b><br>${w.n} wireline sample record${w.n === 1 ? '' : 's'} (MDT / RFT / FMT): no GLR in the open data`), hovertemplate: '%{text}<extra></extra>',
    marker: {symbol: MT.wireline_tester.symbol, size: 7, color: 'rgba(255,255,255,0)', line: {width: 1, color: '#8A98A3'}}});
  tr.push({x: C.outlines.areas.map(a => a.cx), y: C.outlines.areas.map(a => a.cy), mode: 'text', text: C.outlines.areas.map(a => a.label), textfont: {size: 9, color: '#5B6B77'}, hoverinfo: 'skip', showlegend: false});
  categories(W, 'cat').forEach(cat => {
    const P = W.filter(w => w.cat === cat);
    tr.push({x: P.map(w => w.lon), y: P.map(w => w.lat), mode: 'markers', type: 'scatter', showlegend: false, text: P.map(w => `<b>${w.name}</b><br>${w.n} test${w.n === 1 ? '' : 's'}; ${labelOf(key, cat)}`),
      marker: {color: palette[cat] || '#8A98A3', symbol: P.map(w => (MT[w.mt] || MT.other).symbol), size: P.map(w => Math.min(6 + 2 * Math.sqrt(w.n), 16)), opacity: 0.9, line: {width: 0.7, color: '#FFFFFF'}}, customdata: P.map(w => [w.id]), hovertemplate: '%{text}<extra></extra>'});
  });
  tr.push(highlightTrace());
  Plotly.react('c-map', tr, {font: FONT, paper_bgcolor: '#fff', plot_bgcolor: '#F7FAFC', margin: {l: 48, r: 10, t: 8, b: 36}, hovermode: 'closest',
    xaxis: {range: [box[0], box[2]], ticksuffix: '°E', gridcolor: C.colours.grid, zeroline: false, showline: true, linecolor: C.colours.text},
    yaxis: {range: [box[1], box[3]], ticksuffix: '°N', gridcolor: C.colours.grid, zeroline: false, showline: true, linecolor: C.colours.text, scaleanchor: 'x', scaleratio: 1 / Math.cos(midLat * Math.PI / 180)}}, {responsive: true, displaylogo: false});
}
function ptChart(R) {
  const key = $('f-colour').value, palette = GROUPS[key][1], S = R.filter(r => r.t !== null && r.t !== undefined && r.p !== null);
  const tr = C.pt.zones.map(z => ({x: z.x, y: z.y, mode: 'lines', line: {width: 0}, fill: 'toself', fillcolor: z.colour, opacity: 1, name: z.label, hoveron: 'fills', hoverinfo: 'name', showlegend: false}));
  pointTraces(S, key, palette, {x: r => r.t, y: r => r.p, more: {showlegend: false}}, HOVER, r => r.mt === 'pvt_report' ? 10 : 6.5).forEach(x => tr.push(x));
  tr.push(highlightTrace());
  Plotly.react('c-pt', tr, {font: {family: C.font, color: C.colours.text, size: 11}, paper_bgcolor: '#fff', plot_bgcolor: '#fff', margin: {l: 66, r: 12, t: 8, b: 56}, hovermode: 'closest',
    xaxis: {range: [0, C.pt.tmax], automargin: true, title: {text: `Test temperature [°C] (n=${S.length}; ${S.filter(r => r.ti).length} hollow: estimated)`}, gridcolor: C.colours.grid, layer: 'above traces', zeroline: false, showline: true, linecolor: C.colours.text},
    yaxis: {range: [0, C.pt.pmax], automargin: true, title: {text: 'Reservoir pressure [bar]'}, gridcolor: C.colours.grid, layer: 'above traces', zeroline: false, showline: true, linecolor: C.colours.text},
    annotations: C.pt.zones.map(z => ({x: z.lx, y: z.ly, xref: 'x', yref: 'y', text: z.label, showarrow: false, font: {size: 10, color: C.colours.text}}))}, {responsive: true, displaylogo: false});
}
function apiChart(R) {
  const S = R.filter(r => r.api !== null && r.t !== null), key = $('f-colour').value, palette = GROUPS[key][1];
  const shapes = C.brackets.map(b => ({type: 'rect', xref: 'x', yref: 'paper', x0: b[0], x1: b[1], y0: 0, y1: 1, fillcolor: b[2], opacity: 0.09, line: {width: 0}, layer: 'below'}));
  [C.bio.heavy, C.bio.light].forEach(v => shapes.push({type: 'line', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: v, y1: v, line: {color: C.colours.text, width: 1, dash: 'dash'}}));
  const tr = [{x: S.map(r => r.t), y: S.map(r => r.api), mode: 'markers', type: 'scatter', showlegend: false, text: S.map(r => `${r.w} test ${r.n}<br>${mtLabel(r)}<br>${r.bio || ''}`),
    marker: {color: S.map(r => r.ti && !LINE_ONLY.has(symOf(r)) ? 'rgba(255,255,255,0)' : (palette[fv(r, key)] || '#8A98A3')), symbol: S.map(symOf), size: S.map(r => r.up ? 10 : 7), opacity: 0.85, line: {width: S.map(r => r.ti ? 2 : 0), color: S.map(r => palette[fv(r, key)] || '#8A98A3')}}, customdata: S.map(r => [r.w, r.n, null, null, null, null, null, null, null, null, null, null, null, null, null, r.k]), hovertemplate: '%{text}<br>%{x:.0f} °C, %{y:.1f} API<extra></extra>'}, highlightTrace()];
  Plotly.react('c-api', tr, {font: {family: C.font, color: C.colours.text, size: 11}, paper_bgcolor: '#fff', plot_bgcolor: '#fff', margin: {l: 55, r: 12, t: 26, b: 48}, shapes,
    xaxis: {range: [0, 200], title: {text: 'Test temperature [°C]  (shape: type; hollow: estimated)'}, gridcolor: C.colours.grid}, yaxis: {range: [5, 62], title: {text: 'Stock-tank gravity [°API]'}, gridcolor: C.colours.grid},
    annotations: C.brackets.map(b => ({x: (b[0] + Math.min(b[1], 200)) / 2, y: 1, xref: 'x', yref: 'paper', text: b[3], showarrow: false, font: {size: 10, color: b[2]}, yanchor: 'bottom'}))}, {responsive: true, displaylogo: false});
}
function table(R) {
  const T = R.filter(r => r.dm === 'forbidden').sort((a, b) => b.p - a.p).slice(0, 25);
  document.querySelector('#t-two tbody').innerHTML = T.length ? T.map(r => `<tr><td>${r.w}</td><td>${r.n}</td><td>${C.areaLabel[r.ar] || r.ar}</td><td>${C.playLabel[r.pl] || r.pl}</td><td>${fmt(r.md)}</td><td>${fmt(r.p)}</td><td>${fmt(r.g)}</td><td>${fmt(r.ol)}</td><td>${fmt(r.gl)}</td><td>${fmt(r.pc)}</td><td>${r.sb === 'pvt_report' ? 'PVT report' : 'envelope estimate'}</td></tr>`).join('')
    : '<tr><td colspan="11">No forbidden-domain tests in the selection.</td></tr>';
}
function typesTable(R) {                         // every configured type with its tests, forbidden share and issues (the taxonomy is shown even where no test exists yet)
  document.querySelector('#t-types tbody').innerHTML = C.measurement.types.map(t => {
    const S = R.filter(r => r.mt === t.id), fb = S.filter(r => r.dm === 'forbidden').length;
    const issues = t.issues.map(i => `<span title="${C.measurement.issues[i].effect} (data: ${C.measurement.issues[i].measurable})">${C.measurement.issues[i].label}</span>`).join('; ');
    return `<tr><td class="glyph">${t.glyph}</td><td title="${t.evidence}">${t.label}<br><span class="n">${C.measurement.groups[t.group]}</span></td><td>${S.length}</td><td>${S.length ? pct(fb, S.length) + ' (' + fb + ')' : '–'}</td><td>${t.classifiable}</td><td>${issues || '–'}</td></tr>`;
  }).join('');
}
function csv() {
  const R = rows(), cols = [['w', 'well'], ['n', 'test'], ['ar', 'area'], ['pl', 'play'], ['md', 'mid_md_m'], ['p', 'reservoir_pressure_bar'], ['t', 'temperature_used_c'], ['ti', 'temperature_is_estimated'], ['g', 'glr_sm3_sm3'], ['api', 'api'], ['fc', 'fluid_class'], ['st', 'saturation_state'], ['sb', 'saturation_basis'], ['dm', 'domain'], ['mt', 'measurement_type'], ['mc', 'measurement_confidence'], ['dd', 'drawdown_ratio'], ['fl', 'flags']];
  const q = v => v === null || v === undefined ? '' : /[",\n]/.test(String(v)) ? '"' + String(v).replace(/"/g, '""') + '"' : v;
  const text = [cols.map(c => c[1]).join(','), ...R.map(r => cols.map(c => q(r[c[0]])).join(','))].join('\n');
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], {type: 'text/csv'})); a.download = 'pvt_filtered_tests.csv'; a.click();
}
let CUR = [], bound = false;
let mainPoints = T => T.filter(r => r.p <= YMAX).map(r => [r.g, r.p]);        // where a test sits in the main chart (a variant may change it)
function highlight(ks) {                         // ring the same test(s) in every chart (the map shows wells: one ring per well)
  const T = ks.map(k => D[k]), seen = new Set();
  const sets = {'c-main': mainPoints(T), 'c-pt': T.filter(r => r.t !== null && r.t !== undefined).map(r => [r.t, r.p]), 'c-api': T.filter(r => r.api !== null && r.t !== null && r.api !== undefined).map(r => [r.t, r.api]),
    'c-map': T.filter(r => r.lat !== null && r.lat !== undefined && !seen.has(r.id) && seen.add(r.id)).map(r => [r.lon, r.lat])};
  Object.entries(sets).forEach(([id, pts]) => {
    const gd = $(id); if (!gd || !gd.data) return;
    const upd = {x: [pts.map(q => q[0])], y: [pts.map(q => q[1])]};
    if (id === 'c-main' && C.variant === '3d') upd.z = [pts.map(q => q[2])];
    Plotly.restyle(gd, upd, [gd.data.length - 1]);
  });
}
function bindHover() {
  ['c-main', 'c-pt', 'c-api', 'c-map'].forEach(id => {
    const gd = $(id);
    gd.on('plotly_hover', ev => {
      const pt = ev.points && ev.points[0];
      const cd = pt && (pt.customdata || (pt.data && pt.data.customdata && pt.data.customdata[pt.pointNumber]));      // 3D traces give the data trace and the point number
      if (!cd) return;
      highlight(id === 'c-map' ? CUR.filter(r => r.id === cd[0]).map(r => r.k) : (cd[15] === undefined ? [] : [cd[15]]));
    });
    gd.on('plotly_unhover', () => highlight([]));
  });
  bound = true;
}
function update() { const R = rows(); CUR = R; kpis(R); mainChart(R); mapChart(R); ptChart(R); apiChart(R); table(R); typesTable(R); if (!bound) bindHover(); }
__VARIANT_JS__
populate();
document.addEventListener('input', e => { if (e.target.closest('.controls, .filters')) update(); });
$('b-reset').addEventListener('click', () => reset()); $('b-csv').addEventListener('click', csv);
$('a-all').addEventListener('click', () => setAll('f-area', true)); $('a-none').addEventListener('click', () => setAll('f-area', false));
$('p-all').addEventListener('click', () => setAll('f-play', true)); $('p-none').addEventListener('click', () => setAll('f-play', false));
$('m-all').addEventListener('click', () => setAll('f-mtype', true)); $('m-none').addEventListener('click', () => setAll('f-mtype', false));
update();
"""


# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# data preparation (pure)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
def prepare_rows(pvt: pd.DataFrame, header: pd.DataFrame | None = None, well_area: pd.DataFrame | None = None) -> pd.DataFrame:
    """The tests with the well position (`lat`, `lon`, from the well header), the area of the well (`region_id`, 'outside' when in none) and the play ('none' when none)."""
    t = pvt.copy()
    if header is not None and not header.empty and {"wellbore_id", "lat_etrs89", "lon_etrs89"} <= set(header.columns):
        pos = header.drop_duplicates("wellbore_id")[["wellbore_id", "lat_etrs89", "lon_etrs89"]].rename(columns={"lat_etrs89": "lat", "lon_etrs89": "lon"})
        t = t.merge(pos, on="wellbore_id", how="left")
    if well_area is not None and not well_area.empty:
        t = t.merge(well_area.drop_duplicates("wellbore_id")[["wellbore_id", "region_id"]], on="wellbore_id", how="left")
    t["region_id"] = t["region_id"].fillna("outside") if "region_id" in t else "outside"
    t["play_id"] = t["play_id"].astype("string").fillna(NO_PLAY) if "play_id" in t else NO_PLAY
    return t


def _clean(frame: pd.DataFrame) -> list[dict]:
    """The tests as short-keyed records; NaN becomes null (valid JSON) and numbers are rounded to what a person can read."""
    f = frame[[c for c in FIELDS if c in frame.columns]].rename(columns=FIELDS).copy()
    for c, digits in (("md", 1), ("p", 1), ("t", 1), ("ol", 1), ("gl", 1), ("pb", 1), ("pd", 1), ("pc", 0), ("g", 2), ("api", 2), ("gg", 2), ("lat", 4), ("lon", 4), ("dd", 3), ("iv", 0)):
        if c in f:
            f[c] = pd.to_numeric(f[c], errors="coerce").astype("float64").round(digits)
    for c in ("nz", "up", "ti"):
        if c in f:
            f[c] = f[c].fillna(False).astype(bool)
    for c in ("id",):
        if c in f:
            f[c] = f[c].astype("Int64").astype(object)
    f = f.astype(object).where(f.notna(), None)
    return f.to_dict("records")


def _rings(geometry: Any, digits: int = 3) -> tuple[list[float | None], list[float | None]]:
    """x (lon) and y (lat) of the outer rings of a (Multi)Polygon as one list with None between the rings (a Plotly line / fill trace)."""
    parts = list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]
    x: list[float | None] = []
    y: list[float | None] = []
    for part in parts:
        if getattr(part, "is_empty", True) or part.geom_type != "Polygon":
            continue
        coords = list(part.exterior.coords)
        x += [round(c[0], digits) for c in coords] + [None]
        y += [round(c[1], digits) for c in coords] + [None]
    return x[:-1], y[:-1]


def page_config(cfg: dict, pcfg: dict, areas: list[dict], plays: list[dict], outlines: dict) -> dict:
    psi_per_bar, to_sm3 = pvt_mod._factors()
    brackets = cfg["biodegradation"]["temperature_brackets_c"]
    edges = [0.0] + [float(b["below"]) for b in brackets[:-1]] + [200.0]
    bracket_rows = [[edges[i], edges[i + 1], BRACKET_COLOURS.get(b["name"], GREY), b["name"]] for i, b in enumerate(brackets)]
    tmax, pmax = PT_MAX
    zones = pr.hpht_zone_polygons(pcfg, tmax, pmax)
    (p1, p2, p3), (t1, t2, t3) = pcfg["hpht"]["pressure_bar"], pcfg["hpht"]["temperature_c"]
    names = pcfg["hpht"]["names"]
    anchor = {"normal": (t1 / 2, p1 / 2), "high temperature": ((t1 + tmax) / 2, p1 / 2), "high pressure": (t1 / 2, (p1 + pmax) / 2), names[1]: ((t1 + t2) / 2, (p1 + p2) / 2), names[2]: ((t2 + t3) / 2, (p2 + p3) / 2), names[3]: ((t3 + tmax) / 2, (p3 + pmax) / 2)}
    pt_zones = [{"name": n, "label": PT_LABELS.get(n, n), "colour": PT_COLOURS.get(n, GREY), "x": x, "y": y, "lx": anchor[n][0], "ly": anchor[n][1]} for n, (x, y) in zones.items()]
    return {"font": FONT, "psiPerBar": psi_per_bar, "toSm3": to_sm3, "standing": cfg["standing"], "glaso": cfg["glaso"], "leftBranch": pvt_mod.left_branch_name(cfg), "ref": cfg["reference"], "gas": {"p": cfg["gas_envelope"]["pressure_bar"], "glr": cfg["gas_envelope"]["min_glr"]},
            "bio": {"heavy": cfg["biodegradation"]["heavy_api_below"], "light": cfg["biodegradation"]["light_api_above"]}, "brackets": bracket_rows,
            "band": cfg["critical_band"]["glr_sm3_sm3"], "sat": {"under": cfg["saturation"]["undersaturated_below"], "gas": cfg["saturation"]["saturated_gas_below"]},
            "zones": ZONE_COLOURS, "zoneOrder": pvt_mod.ZONES, "ymax": Y_MAX_BAR, "pt": {"zones": pt_zones, "tmax": tmax, "pmax": pmax},
            "areas": areas, "plays": plays, "areaLabel": {a["id"]: a["label"] for a in areas}, "playLabel": {p["id"]: p["label"] for p in plays}, "outlines": outlines,
            "colours": {"st": STATE_COLOURS, "fc": FLUID_COLOURS, "ac": API_COLOURS, "tb": BRACKET_COLOURS, "eb": BASIS_COLOURS, "a": {a["id"]: a["colour"] for a in areas}, "pl": {p["id"]: p["colour"] for p in plays},
                        "text": theme.TEXT_PRIMARY, "grid": theme.GRIDLINE},
            "order": {"st": list(STATE_COLOURS), "fc": list(FLUID_COLOURS), "ac": list(API_COLOURS), "tb": list(BRACKET_COLOURS), "eb": list(BASIS_COLOURS), "a": [a["id"] for a in areas], "pl": [p["id"] for p in plays]}}


def measurement_config(mcfg: dict) -> dict:
    """The measurement types (symbol, glyph, evidence, issues) and the issue catalogue for the page (config/measurement_types.yaml, docs/measurement_types.md)."""
    types = [{"id": k, "group": v["group"], "label": v["label"], "symbol": v["symbol"], "symbol3d": v["symbol_3d"], "glyph": v["glyph"], "classifiable": v["classifiable"], "evidence": v["evidence"], "issues": v["issues"]} for k, v in mcfg["types"].items()]
    return {"types": types, "groups": mcfg["groups"], "issues": {k: {"label": v["label"], "effect": v["effect"], "measurable": v["measurable"]} for k, v in mcfg["issues"].items()}}


def default_context(tests: pd.DataFrame) -> dict:
    """Areas, plays and (empty) outlines derived from the tests themselves, for when no mid-layer context is available (tests, previews)."""
    area_ids = list(dict.fromkeys(list(AREA_LABELS) + sorted(set(tests.get("region_id", pd.Series(dtype=str)).dropna()) - set(AREA_LABELS))))
    play_ids = sorted(set(tests.get("play_id", pd.Series(dtype=str)).dropna()) - {NO_PLAY})
    areas = [{"id": a, "label": AREA_LABELS.get(a, a), "colour": AREA_COLOURS.get(a, GREY)} for a in area_ids]
    plays = [{"id": p, "label": p, "colour": PLAY_FALLBACK[i % len(PLAY_FALLBACK)]} for i, p in enumerate(play_ids)] + [{"id": NO_PLAY, "label": "No play (outside play intervals)", "colour": GREY}]
    return {"areas": areas, "plays": plays, "outlines": {"countries": {"x": [], "y": []}, "areas": []}, "wireline": []}


def build_html(pvt: pd.DataFrame, cfg: dict, context: dict | None = None, pcfg: dict | None = None, variant: dict | None = None) -> str:
    """The whole page as text (pure: the prepared tests, the PVT and pressure configuration and the map / area / play context in, HTML out).
    `variant` (the 3D PVT page): {name, title, body, script, config(tests) -> dict} replaces the body, adds JavaScript and page configuration."""
    tests = pvt[pvt["glr_sm3_sm3"].notna() & pvt["reservoir_pressure_bar"].notna()] if not pvt.empty else pvt
    if tests.empty:
        body = '<header>E&amp;P Geoscience Analytics</header><main><h1>GLR against reservoir pressure</h1><p class="headline">No DST tests with a reservoir pressure and a GLR are in the mid-layer yet: build the pipeline and rebuild this page.</p></main>'
        return f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>PVT dashboard</title><style>{CSS}</style></head><body>{body}</body></html>'
    if "region_id" not in tests.columns or "play_id" not in tests.columns:
        tests = prepare_rows(tests)                                  # no area / play known: 'outside' and 'none', so the filters still work
    context = context or default_context(tests)
    config = page_config(cfg, pcfg or pr.load_config(), context["areas"], context["plays"], context["outlines"])
    tests = tests.reset_index(drop=True)
    config["measurement"] = measurement_config(mtypes.load_config())
    config["wireline"] = context.get("wireline", [])                 # wells with wireline formation-tester sample records (map context: no GLR, so not tests)
    config["variant"] = variant["name"] if variant else "glr"
    if variant:
        config.update(variant["config"](tests))
    data = json.dumps(_clean(tests), ensure_ascii=False, allow_nan=False)
    script = SCRIPT.replace("__VARIANT_JS__", variant["script"] if variant else "").replace("__DATA__", data).replace("__CFG__", json.dumps(config, ensure_ascii=False, allow_nan=False))
    title = variant["title"] if variant else "PVT: GLR and reservoir pressure"
    return (f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>'
            f'<style>{CSS}</style><script src="assets/plotly.min.js"></script>{hp.SCRIPT_TAG}</head><body>{variant["body"] if variant else BODY}<script>{script}</script></body></html>')
