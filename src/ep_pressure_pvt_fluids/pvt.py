"""PVT fluid classification of drill stem tests: API class, fluid class, the zones of the GLR-pressure plane (forbidden domain, diphasic bands, monophasic
domains with the critical-fluid band), saturation state and the biodegradation / pasteurisation temperature brackets.

Marts (derived, schema 1.0)
    well_dst_pvt       one row per DST with a reservoir pressure and a producing GOR / GLR
    play_pvt_summary   tests per play: fluid classes, saturation states, domains, median API, biodegradation assessment

Rules (AGENTS.md sections 8, 16, 19)
    * Thresholds, envelopes and temperature brackets are configuration (`config/pvt.yaml`); the literature values are ASSUMPTIONS and are labelled so.
    * The left branch of the saturation-pressure U is the maximum solution GOR at the test pressure (bubble point; Glaso by default, or Standing: `left_branch` in config/pvt.yaml), the right branch a configured
      minimum-GLR curve (dew point), the critical band a configured GLR band.  A test above the U (pressure below the saturation pressure of its GLR) is in
      the `forbidden` domain (no single-phase fluid): flagged, never dropped or corrected.  A test at the saturation pressure (within the uncertainty band)
      is `diphasic`: oil with gas cap (saturated oil) or gas with oil ring (saturated gas).
    * A test without oil density / temperature / gas density is evaluated against the reference fluid (`envelope_basis = reference`): indicative only.
    * The temperature of a test is the gauge temperature, else (when `use_estimated_temperature`) the screening estimate from the geothermal gradient of its well
      (`temperature_est_c`, docs/temperature.md): `temperature_used_c`, flagged `temperature_is_estimated` (drawn hollow); a test with an estimated temperature has
      `envelope_basis = 'test (estimated temperature)'`.
    * The present-day temperature is a MINIMUM of the maximum past temperature that decides pasteurisation (uplift), so a degradable bracket is only
      "possible" and an uplift area is flagged.
    * Pure functions: DataFrames in, DataFrames out, plus data-quality events.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from ep_pressure_pvt_fluids import measurement_types as mt
from ep_physics_base.events import dq_event
from ep_physics_base.config import load_yaml

logger = logging.getLogger(__name__)

CONFIG_FILE = "config/pvt.yaml"
UNIT_FILE = "config/unit_factors.yaml"
SCHEMA_VERSIONS = {"well_dst_pvt": "1.0", "play_pvt_summary": "1.0"}
STATE_ORDER = ["undersaturated oil", "saturated oil", "critical fluid", "saturated gas", "undersaturated gas", "forbidden"]
# domain of each saturation state: a saturated fluid is at its saturation pressure, in equilibrium with a second phase where one exists (diphasic band)
DOMAIN_OF_STATE = {"undersaturated oil": "monophasic liquid", "undersaturated gas": "monophasic gas", "critical fluid": "critical fluid", "saturated oil": "diphasic", "saturated gas": "diphasic", "forbidden": "forbidden"}
ZONES = ["forbidden", "diphasic: oil with gas cap", "diphasic: gas with oil ring", "monophasic: undersaturated oil", "monophasic: critical fluid", "monophasic: undersaturated gas"]


def load_config() -> dict[str, Any]:
    return load_yaml(CONFIG_FILE)


def _factors() -> tuple[float, float]:
    """(psi per bar, Sm3/Sm3 per scf/stb) from unit_factors.yaml."""
    f = load_yaml(UNIT_FILE)
    return float(f["constants"]["psi_per_bar"]), float(f["text_units"]["gor"]["units"]["scf/stb"])


def _band(values: pd.Series, bands: list[dict[str, Any]]) -> pd.Series:
    """Name of the first band whose `below` bound the value is under (NA where the value is missing)."""
    out = pd.Series(pd.NA, index=values.index, dtype="string")
    v = pd.to_numeric(values, errors="coerce")
    assigned = pd.Series(False, index=values.index)
    for b in bands:
        hit = (v < float(b["below"])) & ~assigned & v.notna()
        out[hit] = b["name"]
        assigned |= hit
    return out


def api_from_density(density_g_cm3: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    """Degrees API from the stock-tank density; outside the plausible range the value is NaN (kept in the source)."""
    d = pd.to_numeric(density_g_cm3, errors="coerce").astype("float64")
    api = 141.5 / (d / float(cfg["api"]["water_density_g_cm3"])) - 131.5
    low, high = cfg["api"]["plausible"]
    return api.where((api >= low) & (api <= high))


def standing_max_gor(pressure_bar: pd.Series, temperature_c: pd.Series, api: pd.Series, gas_gravity: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    """Maximum solution GOR [Sm3/Sm3] of a liquid at this pressure = the bubble-point envelope (Standing)."""
    psi_per_bar, to_sm3 = _factors()
    s = cfg["standing"]
    p_psia = pd.to_numeric(pressure_bar, errors="coerce").astype("float64") * psi_per_bar
    t_f = pd.to_numeric(temperature_c, errors="coerce").astype("float64") * 1.8 + 32.0
    gg = pd.to_numeric(gas_gravity, errors="coerce").astype("float64")
    inner = (p_psia / s["p_divisor"] + s["p_offset"]) * 10.0 ** (s["api_coeff"] * api - s["temp_coeff"] * t_f)
    return gg * inner ** s["exponent"] * to_sm3


def standing_bubble_point(gor_sm3_sm3: pd.Series, temperature_c: pd.Series, api: pd.Series, gas_gravity: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    """Bubble-point pressure [bar] that would hold this GOR in solution (Standing inverted; the producing GOR of a single-phase oil test is Rs at Pb)."""
    psi_per_bar, to_sm3 = _factors()
    s = cfg["standing"]
    rs = pd.to_numeric(gor_sm3_sm3, errors="coerce").astype("float64") / to_sm3
    t_f = pd.to_numeric(temperature_c, errors="coerce").astype("float64") * 1.8 + 32.0
    gg = pd.to_numeric(gas_gravity, errors="coerce").astype("float64")
    inner = (rs / gg) ** (1.0 / s["exponent"]) / 10.0 ** (s["api_coeff"] * api - s["temp_coeff"] * t_f)
    return ((inner - s["p_offset"]) * s["p_divisor"]).clip(lower=0) / psi_per_bar


def _glaso_terms(temperature_c: pd.Series, api: pd.Series, gas_gravity: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    t_f = pd.to_numeric(temperature_c, errors="coerce").astype("float64") * 1.8 + 32.0
    return t_f, pd.to_numeric(api, errors="coerce").astype("float64"), pd.to_numeric(gas_gravity, errors="coerce").astype("float64")


def glaso_bubble_point(gor_sm3_sm3: pd.Series, temperature_c: pd.Series, api: pd.Series, gas_gravity: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    """Bubble-point pressure [bar] from Glaso (1980, SPE 8016, Eq. 6; North Sea oils, flash-separator GOR and total surface-gas gravity):
    Pb* = (R / gg)^0.816 T^0.172 / API^0.989 (R scf/STB, T degF) and log10 Pb[psia] = 1.7669 + 1.7447 log10 Pb* - 0.30218 (log10 Pb*)^2.  No correction for CO2, N2 or H2S
    (no composition) and the black-oil temperature exponent is used (the paper gives 0.130 for volatile oils, Eq. 20; not applied: a volatile oil cannot be told from the data)."""
    psi_per_bar, to_sm3 = _factors()
    g = cfg["glaso"]
    t_f, a_, gg = _glaso_terms(temperature_c, api, gas_gravity)
    rs = pd.to_numeric(gor_sm3_sm3, errors="coerce").astype("float64") / to_sm3
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.log10((rs / gg) ** g["gor_exponent"] * t_f ** g["temp_exponent"] / a_ ** g["api_exponent"])
        return 10.0 ** (g["a"] + g["b"] * x - g["c"] * x**2) / psi_per_bar


def glaso_max_gor(pressure_bar: pd.Series, temperature_c: pd.Series, api: pd.Series, gas_gravity: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    """Maximum solution GOR [Sm3/Sm3] at this pressure from Glaso (the quadratic of Eq. 6a solved for the rising branch; above the vertex of the parabola, about 19,000 psia,
    the limit is held at the vertex)."""
    psi_per_bar, to_sm3 = _factors()
    g = cfg["glaso"]
    t_f, a_, gg = _glaso_terms(temperature_c, api, gas_gravity)
    log_p = np.log10(pd.to_numeric(pressure_bar, errors="coerce").astype("float64").clip(lower=1e-3) * psi_per_bar)
    with np.errstate(invalid="ignore", divide="ignore"):
        disc = (g["b"] ** 2 - 4.0 * g["c"] * (log_p - g["a"])).clip(lower=0.0)
        pb_star = 10.0 ** ((g["b"] - np.sqrt(disc)) / (2.0 * g["c"]))
        return gg * (pb_star * a_ ** g["api_exponent"] / t_f ** g["temp_exponent"]) ** (1.0 / g["gor_exponent"]) * to_sm3


def left_branch_name(cfg: dict[str, Any]) -> str:
    name = str(cfg.get("left_branch", {}).get("correlation", "standing")).lower()
    if name not in ("standing", "glaso"):
        raise ValueError(f"config/pvt.yaml left_branch.correlation must be standing or glaso, not {name!r}")
    return name


def oil_limit_glr(pressure_bar: pd.Series, temperature_c: pd.Series, api: pd.Series, gas_gravity: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    """The bubble-point (left) branch of the envelope as a maximum solution GOR, with the correlation chosen in config/pvt.yaml (`left_branch.correlation`)."""
    return (glaso_max_gor if left_branch_name(cfg) == "glaso" else standing_max_gor)(pressure_bar, temperature_c, api, gas_gravity, cfg)


def bubble_point_estimate(gor_sm3_sm3: pd.Series, temperature_c: pd.Series, api: pd.Series, gas_gravity: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    """Bubble-point pressure [bar] that would hold this GOR in solution, with the chosen correlation."""
    return (glaso_bubble_point if left_branch_name(cfg) == "glaso" else standing_bubble_point)(gor_sm3_sm3, temperature_c, api, gas_gravity, cfg)


def gas_min_glr(pressure_bar: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    """Minimum GLR [Sm3/Sm3] of a single-phase gas at this pressure (configured dew-point envelope, interpolated in log GLR, held flat outside the table)."""
    e = cfg["gas_envelope"]
    p = pd.to_numeric(pressure_bar, errors="coerce").astype("float64")
    return pd.Series(10.0 ** np.interp(p, e["pressure_bar"], np.log10(e["min_glr"])), index=p.index).where(p.notna())


def gas_dew_pressure(glr: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    """Pressure [bar] at which the dew-point envelope has this GLR (the inverse of `gas_min_glr`, held at the table ends)."""
    e = cfg["gas_envelope"]
    g = pd.to_numeric(glr, errors="coerce").astype("float64")
    xs, ps = np.log10(np.asarray(e["min_glr"], float))[::-1], np.asarray(e["pressure_bar"], float)[::-1]       # min GLR falls with pressure: reversed it rises
    return pd.Series(np.interp(np.log10(g.clip(lower=1e-9)), xs, ps), index=g.index).where(g.notna())


def critical_pressure(temperature_c: pd.Series, api: pd.Series, gas_gravity: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    """Pressure [bar] where the saturation-pressure U closes: the higher of the pressures at which the bubble-point branch reaches the lower edge and the
    dew-point branch reaches the upper edge of the critical band.  Above it the critical band is single phase."""
    low, high = cfg["critical_band"]["glr_sm3_sm3"]
    left = bubble_point_estimate(pd.Series(low, index=api.index), temperature_c, api, gas_gravity, cfg)
    right = gas_dew_pressure(pd.Series(high, index=api.index), cfg)
    return pd.concat([left, right], axis=1).max(axis=1, skipna=False)


def classify_domain(glr: pd.Series, pressure: pd.Series, oil_limit: pd.Series, gas_limit: pd.Series, p_crit: pd.Series, cfg: dict[str, Any]) -> pd.DataFrame:
    """Zone of the GLR-pressure plane and the saturation state of each test (zones: see the header of config/pvt.yaml).
        left of the critical band : forbidden when GLR is above the bubble-point limit at that pressure, else a liquid (undersaturated oil, or oil with gas cap near the limit)
        right of the critical band: forbidden when GLR is below the dew-point limit, else a gas (undersaturated gas, or gas with oil ring near the limit)
        inside the critical band  : critical fluid above the critical pressure, forbidden below it"""
    sat = cfg["saturation"]
    low, high = cfg["critical_band"]["glr_sm3_sm3"]
    g = pd.to_numeric(glr, errors="coerce").astype("float64")
    p = pd.to_numeric(pressure, errors="coerce").astype("float64")
    known = (g.notna() & p.notna() & oil_limit.notna() & gas_limit.notna() & p_crit.notna()).to_numpy(dtype=bool)
    gv, pv, ol, gl, pc = (x.to_numpy(dtype="float64") for x in (g, p, oil_limit, gas_limit, p_crit))
    with np.errstate(invalid="ignore", divide="ignore"):
        in_band, left, right = known & (gv >= low) & (gv <= high), known & (gv < low), known & (gv > high)
        critical = in_band & (pv >= pc)
        forbidden = (in_band & (pv < pc)) | (left & (gv > ol)) | (right & (gv < gl))
        liquid, gas = left & ~forbidden, right & ~forbidden
        state = pd.Series(pd.NA, index=g.index, dtype="string")
        state[liquid & (gv / ol < sat["undersaturated_below"])] = "undersaturated oil"
        state[liquid & (gv / ol >= sat["undersaturated_below"])] = "saturated oil"
        state[gas & (gv / gl < sat["saturated_gas_below"])] = "saturated gas"
        state[gas & (gv / gl >= sat["saturated_gas_below"])] = "undersaturated gas"
    state[critical], state[forbidden] = "critical fluid", "forbidden"
    return pd.DataFrame({"domain": state.map(DOMAIN_OF_STATE).astype("string"), "saturation_state": state})


FORBIDDEN_RESOLUTIONS = ["within typical GOR error", "within worst-case GOR error", "beyond GOR error"]


def forbidden_glr_factor(glr: pd.Series, oil_limit: pd.Series, gas_limit: pd.Series, domain: pd.Series, band: tuple[float, float]) -> pd.Series:
    """For a forbidden-domain test: the factor (>= 1) by which its GLR would have to change to reach the nearest ALLOWED GLR at its pressure.  Allowed = a liquid at or below
    the oil limit (and left of the critical band `band`) or a gas at or above the gas limit (and right of the band), so the factor is
    min(GLR / min(oil limit, band low), max(gas limit, band high) / GLR); inside the band below the critical pressure both sides count.  Missing for every other test.
    The envelope is an estimate, so this is a distance to the ENVELOPE, not to the truth."""
    g, ol, gl = (pd.to_numeric(x, errors="coerce").astype("float64") for x in (glr, oil_limit, gas_limit))
    low, high = band
    with np.errstate(invalid="ignore", divide="ignore"):
        factor = pd.concat([g / ol.clip(upper=low), gl.clip(lower=high) / g], axis=1).min(axis=1, skipna=False)
    return factor.where(domain.eq("forbidden").fillna(False).astype(bool) & (factor > 0))


def classify_forbidden_resolution(factor: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    """Could the producing-GOR error alone explain a forbidden test?  A factor within 1 + typical error (Moffatt & Williams 1998: 5 %), within 1 + worst-case error (25 %), or beyond.
    Evidence, not proof: a pressure error, a commingled interval or a wrong envelope give the same picture."""
    u = cfg["gor_uncertainty"]
    f = pd.to_numeric(factor, errors="coerce").astype("float64")
    out = pd.Series(pd.NA, index=f.index, dtype="string")
    out[f.notna() & (f <= 1.0 + float(u["typical"]))] = FORBIDDEN_RESOLUTIONS[0]
    out[f.notna() & (f > 1.0 + float(u["typical"])) & (f <= 1.0 + float(u["worst_case"]))] = FORBIDDEN_RESOLUTIONS[1]
    out[f.notna() & (f > 1.0 + float(u["worst_case"]))] = FORBIDDEN_RESOLUTIONS[2]
    return out


def zone_polygons(pressure_grid: np.ndarray, temperature_c: float, api: float, gas_gravity: float, cfg: dict[str, Any], xmin: float = 10.0, xmax: float = 1.0e6) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Closed polygons (GLR, pressure) of the six zones for ONE reference fluid, each to be filled with one flat colour (keys: `ZONES`).  The same geometry as
    `classify_domain` (the dashboard repeats it in JavaScript so the reference fluid can be changed with sliders)."""
    p = np.asarray(pressure_grid, float)
    low, high = cfg["critical_band"]["glr_sm3_sm3"]
    sat = cfg["saturation"]

    def limits(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = pd.Series(grid)
        oil = oil_limit_glr(s, pd.Series(temperature_c, index=s.index), pd.Series(api, index=s.index), pd.Series(gas_gravity, index=s.index), cfg).to_numpy()
        return oil, gas_min_glr(s, cfg).to_numpy()

    oil, gas = limits(p)
    p_crit = float(critical_pressure(pd.Series([temperature_c]), pd.Series([api]), pd.Series([gas_gravity]), cfg).iloc[0])

    def between(left: np.ndarray, right: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.r_[left, right[::-1]], np.r_[grid, grid[::-1]]

    oil_edge, oil_under = np.clip(oil, xmin, low), np.clip(sat["undersaturated_below"] * oil, xmin, low)          # the left branch stops at the critical band
    gas_edge, gas_ring = np.clip(gas, high, xmax), np.clip(sat["saturated_gas_below"] * gas, high, xmax)         # the right branch stops at the critical band
    pf = np.r_[p[p < p_crit], p_crit]                                                                              # the forbidden domain closes at the critical pressure
    of, gf = limits(pf)
    out = {"forbidden": between(np.clip(of, xmin, low), np.clip(gf, high, xmax), pf),
           "diphasic: oil with gas cap": between(oil_under, oil_edge, p), "diphasic: gas with oil ring": between(gas_edge, gas_ring, p),
           "monophasic: undersaturated oil": between(np.full_like(p, xmin), oil_under, p), "monophasic: undersaturated gas": between(gas_ring, np.full_like(p, xmax), p)}
    top, bottom = p_crit, max(float(p.max()), p_crit)
    out["monophasic: critical fluid"] = (np.array([low, high, high, low, low]), np.array([top, top, bottom, bottom, top]))
    return out


def zone_raster(cfg: dict[str, Any], api: float, temperature_c: float, gas_gravity: float, pressure_max_bar: float, colours: dict[str, str], zone_of_state: dict[str, str],
                width: int = 300, height: int = 260, xmin: float = 10.0, xmax: float = 1.0e6, sigma: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """The GLR-pressure plane as an image for ONE reference fluid: the pastel colour of the zone of every pixel centre (x: log GLR from `xmin` to `xmax`, y: pressure from 0
    down to `pressure_max_bar`), blurred with a Gaussian so the colours run into each other across the zone boundaries.  Returns (RGB image 0-1, zone name per pixel before the
    blur): the classification is `classify_domain`, so the picture and the data agree."""
    from scipy.ndimage import gaussian_filter

    lg = np.linspace(np.log10(xmin), np.log10(xmax), width + 1)
    glr = 10.0 ** ((lg[:-1] + lg[1:]) / 2.0)
    press = (np.arange(height) + 0.5) / height * pressure_max_bar
    g, p = np.meshgrid(glr, press)
    n = g.size
    const = lambda v: pd.Series(np.full(n, float(v)))
    p_flat = pd.Series(p.ravel())
    oil = oil_limit_glr(p_flat, const(temperature_c), const(api), const(gas_gravity), cfg)
    gas = gas_min_glr(p_flat, cfg)
    p_crit = critical_pressure(const(temperature_c), const(api), const(gas_gravity), cfg)
    state = classify_domain(pd.Series(g.ravel()), p_flat, oil, gas, p_crit, cfg)["saturation_state"]
    zone = state.map(zone_of_state).to_numpy().reshape(height, width)
    rgb = np.zeros((height, width, 3))
    for name, hex_colour in colours.items():
        rgb[zone == name] = [int(hex_colour[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
    return np.clip(gaussian_filter(rgb, sigma=(sigma, sigma, 0), mode="nearest"), 0.0, 1.0), zone


def classify_biodegradation(temperature_c: pd.Series, api: pd.Series, area: pd.Series, cfg: dict[str, Any]) -> pd.DataFrame:
    """Temperature bracket and a screening assessment of biodegradation / pasteurisation from API and the present-day temperature."""
    b = cfg["biodegradation"]
    bracket = _band(temperature_c, b["temperature_brackets_c"])
    heavy = (api < b["heavy_api_below"]).fillna(False).to_numpy(dtype=bool)
    light = (api > b["light_api_above"]).fillna(False).to_numpy(dtype=bool)
    deg, trans, past = (bracket == "degradable").fillna(False).to_numpy(dtype=bool), (bracket == "transition").fillna(False).to_numpy(dtype=bool), (bracket == "pasteurised").fillna(False).to_numpy(dtype=bool)
    up = area.astype("string").str.upper().isin([a.upper() for a in b["uplift_areas"]]).fillna(False).to_numpy(dtype=bool)
    text = pd.Series(pd.NA, index=api.index, dtype="object")
    text[bracket.notna() & api.notna()] = "intermediate API: no clear signal"
    text[deg & heavy], text[trans & heavy] = "likely biodegraded (heavy and cool)", "possibly biodegraded (heavy, 60-80 degC)"
    text[past & heavy] = "heavy but hot: not present-day biodegradation (palaeo-biodegradation before burial, or low-maturity heavy oil)"
    text[(deg | trans) & light] = "light oil in the degradable window: pasteurised in the past (deeper burial then uplift) or charged after sterilisation"
    text[past & light] = "light and hot: consistent with pasteurisation"
    text[bracket.isna() | api.isna()] = pd.NA
    return pd.DataFrame({"temperature_bracket": bracket, "biodegradation_assessment": text.astype("string"), "uplift_area": up & bracket.notna().to_numpy(dtype=bool)})


REPORT_COLUMNS = ["sample_type", "bubble_point_bar", "dew_point_bar", "rs_sm3_sm3", "cgr_sm3_msm3", "api", "gas_gravity", "reservoir_temperature_c", "fluid_type", "source"]


def match_pvt_reports(tests: pd.DataFrame, reports: pd.DataFrame | None, cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[dict]]:
    """Manual PVT-report rows matched to tests: same wellbore, sample MD within `report.match_tolerance_m` of the test mid depth (closest wins).
    Returns one row per matched test (index = the test's index) with the report columns prefixed `report_`; unmatched reports are logged."""
    ctx = ("manual", "pvt_report")
    empty = pd.DataFrame(columns=["report_" + c for c in REPORT_COLUMNS])
    if reports is None or reports.empty or not {"wellbore_id", "md_m"} <= set(reports.columns):
        return empty, []
    tol = float(cfg["report"]["match_tolerance_m"])
    r = reports.copy()
    r["wellbore_id"] = r["wellbore_id"].astype("Int64")
    for c in REPORT_COLUMNS:
        if c not in r:
            r[c] = np.nan
    r = r.reset_index(names="report_row")
    m = tests[["wellbore_id", "mid_md_m"]].astype({"wellbore_id": "Int64"}).reset_index(names="test_row").merge(r[["report_row", "wellbore_id", "md_m"] + REPORT_COLUMNS], on="wellbore_id")
    m["gap_m"] = (m["md_m"].astype("float64") - m["mid_md_m"].astype("float64")).abs()
    m = m[m["gap_m"] <= tol].sort_values("gap_m").drop_duplicates("test_row")
    events = [dq_event(*ctx, "pvt_report_matched", int(m["report_row"].nunique()), "info", detail=f"{m['report_row'].nunique()} of {len(r)} PVT report rows matched to {len(m)} tests (sample MD within {tol:g} m of the test mid depth)")]
    lost = r[~r["report_row"].isin(m["report_row"])]
    if len(lost):
        events.append(dq_event(*ctx, "pvt_report_unmatched", len(lost), "warning", lost["wellbore_id"].astype(str).head(10).tolist(), f"no DST with a reservoir pressure within {tol:g} m: the report is not used"))
    out = m.set_index("test_row")[REPORT_COLUMNS].rename(columns=lambda c: "report_" + c)
    return out, events


def apply_pvt_report_values(t: pd.DataFrame, matched: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[dict]]:
    """Add the report columns to the tests.  Official DST values win: report API, gas gravity and temperature only fill gaps (a disagreement over
    `report.conflict_ratio` is logged); the stated bubble / dew point and fluid type have no official counterpart and are used by the caller."""
    ctx = ("manual", "pvt_report")
    t = t.join(matched, how="left")
    for c in ("report_" + x for x in REPORT_COLUMNS):
        if c not in t:
            t[c] = np.nan
    for c in ("report_bubble_point_bar", "report_dew_point_bar", "report_rs_sm3_sm3", "report_cgr_sm3_msm3", "report_api", "report_gas_gravity", "report_reservoir_temperature_c"):
        t[c] = pd.to_numeric(t[c], errors="coerce").astype("float64")
    events: list[dict] = []
    ratio = float(cfg["report"]["conflict_ratio"])
    t["api_source"] = np.where(t["api"].notna(), "official_dst", None)
    for official, stated, label in (("api", "report_api", "API"), ("gas_relative_density", "report_gas_gravity", "gas gravity"), ("temperature_c", "report_reservoir_temperature_c", "temperature")):
        both = t[official].notna() & t[stated].notna()
        far = both & ((t[official] - t[stated]).abs() / t[stated].abs() > ratio)
        if far.any():
            events.append(dq_event(*ctx, "pvt_report_conflict", int(far.sum()), "info", t.loc[far, "well_name"].dropna().unique()[:10], f"stated {label} differs from the DST by more than {ratio:.0%}: the official DST value is kept"))
        fill = t[official].isna() & t[stated].notna()
        t.loc[fill, official] = t.loc[fill, stated]
        if official == "api":
            t.loc[fill, "api_source"] = "pvt_report"
    return t, events


def apply_stated_saturation(t: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """A stated bubble point (liquid tests) or dew point (gas tests) decides the saturation state, replacing the envelope estimate (kept in `envelope_state`).
    Above the stated pressure the fluid is undersaturated, within `report.stated_tolerance` of it saturated (diphasic band: oil with gas cap / gas with oil ring), and
    below it in the forbidden domain (reservoir pressure under the saturation pressure of the sampled fluid: free gas or liquid drop-out, a mixture)."""
    tol = float(cfg["report"]["stated_tolerance"])
    p = t["reservoir_pressure_bar"].astype("float64")
    liquid = t["glr_sm3_sm3"] < float(cfg["liquid_below_gor_sm3_sm3"])
    use_pb = t["report_bubble_point_bar"].notna() & (liquid | t["report_dew_point_bar"].isna())                               # a liquid test uses the bubble point, a gas test the dew point
    use_pd = ~use_pb & t["report_dew_point_bar"].notna()
    sat_p = t["report_bubble_point_bar"].where(use_pb, t["report_dew_point_bar"].where(use_pd))
    is_liquid_sat = use_pb.to_numpy(dtype=bool)
    ratio = p / sat_p
    above, near, below = (ratio > 1 + tol).fillna(False).to_numpy(dtype=bool), ((ratio - 1).abs() <= tol).fillna(False).to_numpy(dtype=bool), (ratio < 1 - tol).fillna(False).to_numpy(dtype=bool)
    t["envelope_state"] = t["saturation_state"]
    t["saturation_basis"] = np.where(sat_p.notna(), "pvt_report", "envelope_estimate")
    for mask, liquid_flag, state in ((above, True, "undersaturated oil"), (near, True, "saturated oil"), (below, True, "forbidden"),
                                     (above, False, "undersaturated gas"), (near, False, "saturated gas"), (below, False, "forbidden")):
        hit = mask & (is_liquid_sat == liquid_flag) & sat_p.notna().to_numpy(dtype=bool)
        t.loc[hit, "saturation_state"], t.loc[hit, "domain"] = state, DOMAIN_OF_STATE[state]
    stated_class = t["report_fluid_type"].astype("string")
    t["fluid_class_source"] = np.where(stated_class.notna(), "pvt_report", np.where(t["fluid_class"].notna(), "producing_gor", None))
    t["fluid_class"] = stated_class.where(stated_class.notna(), t["fluid_class"])
    return t


def build_well_dst_pvt(pressure: pd.DataFrame, dst: pd.DataFrame, cfg: dict[str, Any], reports: pd.DataFrame | None = None, well_play: pd.DataFrame | None = None,
                       use_estimated_temperature: bool = False, oil_samples: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """One row per DST with a producing GOR: API, fluid class, GLR-pressure domain, saturation state, near-critical flag and biodegradation assessment.
    `reports` = accepted rows of the manual `pvt_report` sheet (stated bubble / dew point, API, gas gravity, temperature, fluid type)."""
    ctx = ("all", "well_dst_pvt")
    need = {"wellbore_id", "test_number", "reservoir_pressure_bar"}
    if pressure.empty or dst.empty or not need <= set(pressure.columns):
        return pd.DataFrame(), [dq_event(*ctx, "pvt_inputs_missing", 0, "info", detail="no well_dst_pressure or well_dst")]
    keep = [c for c in ("wellbore_id", "test_number", "gor_sm3_sm3", "oil_density_g_cm3", "gas_relative_density") if c in dst.columns]
    fluid = dst[keep].drop_duplicates(["wellbore_id", "test_number"])
    t = pressure.merge(fluid, on=["wellbore_id", "test_number"], how="left")
    for c in ("gor_sm3_sm3", "oil_density_g_cm3", "gas_relative_density"):
        t[c] = pd.to_numeric(t[c], errors="coerce").astype("float64") if c in t else np.nan
    t["glr_sm3_sm3"] = t["gor_sm3_sm3"].where(t["gor_sm3_sm3"] > 0)
    t["api"] = api_from_density(t["oil_density_g_cm3"], cfg)
    t["temperature_c"] = pd.to_numeric(t["temperature_c"], errors="coerce").astype("float64")
    t["play_id"] = place_tests_in_plays(t, well_play)
    matched, events = match_pvt_reports(t, reports, cfg)
    t, more = apply_pvt_report_values(t, matched, cfg)
    events += more
    t["api_class"] = _band(t["api"], cfg["api"]["classes"])
    t["fluid_class"] = _band(t["glr_sm3_sm3"], cfg["fluid_class_by_gor_sm3_sm3"])
    for c in ("temperature_est_c", "temperature_est_gradient_c_km"):
        t[c] = pd.to_numeric(t[c], errors="coerce").astype("float64") if c in t else np.nan
    if "temperature_est_basis" not in t:
        t["temperature_est_basis"] = pd.Series(pd.NA, index=t.index, dtype="object")
    estimate = t["temperature_est_c"] if use_estimated_temperature else pd.Series(np.nan, index=t.index)
    t["temperature_used_c"] = t["temperature_c"].where(t["temperature_c"].notna(), estimate)         # gauge (or report) temperature first, then the labelled estimate
    t["temperature_is_estimated"] = t["temperature_c"].isna() & t["temperature_used_c"].notna()
    ref = cfg["reference"]
    api_e, temp_e, gg_e = t["api"].fillna(ref["api"]), t["temperature_used_c"].fillna(ref["temperature_c"]), t["gas_relative_density"].fillna(ref["gas_gravity"])
    own = t["api"].notna() & t["temperature_used_c"].notna() & t["gas_relative_density"].notna()
    t["envelope_basis"] = np.where(own & ~t["temperature_is_estimated"], "test", np.where(own, "test (estimated temperature)",
                                   np.where(t["api"].isna() & t["temperature_used_c"].isna() & t["gas_relative_density"].isna(), "reference", "partly reference")))
    t["oil_limit_glr"] = oil_limit_glr(t["reservoir_pressure_bar"], temp_e, api_e, gg_e, cfg)
    t["gas_limit_glr"] = gas_min_glr(t["reservoir_pressure_bar"], cfg)
    t["critical_pressure_bar"] = critical_pressure(temp_e, api_e, gg_e, cfg)
    t = pd.concat([t, classify_domain(t["glr_sm3_sm3"], t["reservoir_pressure_bar"], t["oil_limit_glr"], t["gas_limit_glr"], t["critical_pressure_bar"], cfg)], axis=1)
    liquid_test = t["glr_sm3_sm3"] < float(cfg["liquid_below_gor_sm3_sm3"])
    t["bubble_point_est_bar"] = bubble_point_estimate(t["glr_sm3_sm3"], temp_e, api_e, gg_e, cfg).where(liquid_test)
    t = apply_stated_saturation(t, cfg)
    envelope_forbidden = t["domain"].where(t["saturation_basis"].eq("envelope_estimate"))                  # a state decided by a stated bubble / dew point has no GLR distance to the estimate
    t["forbidden_glr_factor"] = forbidden_glr_factor(t["glr_sm3_sm3"], t["oil_limit_glr"], t["gas_limit_glr"], envelope_forbidden, tuple(cfg["critical_band"]["glr_sm3_sm3"]))
    t["forbidden_resolution"] = classify_forbidden_resolution(t["forbidden_glr_factor"], cfg)
    t["bubble_point_bar"] = t["report_bubble_point_bar"].where(t["report_bubble_point_bar"].notna(), t["bubble_point_est_bar"])
    t["bubble_point_source"] = np.where(t["report_bubble_point_bar"].notna(), "pvt_report", np.where(t["bubble_point_est_bar"].notna(), f"{left_branch_name(cfg)}_estimate", None))
    t["dew_point_bar"] = t["report_dew_point_bar"]
    t["pressure_to_bubble_point"] = (t["reservoir_pressure_bar"] / t["bubble_point_bar"]).where(t["bubble_point_bar"] > 0)
    changed = t["saturation_basis"].eq("pvt_report") & t["envelope_state"].notna() & t["saturation_state"].ne(t["envelope_state"])
    if changed.any():
        events.append(dq_event(*ctx, "pvt_report_changes_state", int(changed.sum()), "info", t.loc[changed, "well_name"].dropna().unique()[:10],
                               "the stated bubble / dew point gives another saturation state than the envelope estimate: a check of the envelopes (config/pvt.yaml)"))
    t["near_critical_zone"] = t["glr_sm3_sm3"].between(*cfg["critical_band"]["glr_sm3_sm3"])         # GLR inside the critical band (at any pressure)
    t = pd.concat([t, classify_biodegradation(t["temperature_used_c"], t["api"], t["area"], cfg)], axis=1)
    measured, more = mt.classify_measurements(t, oil_samples)                       # measurement type of each test: from the official oil-sample records, never guessed
    events += more
    t = pd.concat([t, measured], axis=1)
    flags = pd.Series("", index=t.index, dtype="object")
    for mask, text in ((t["domain"].eq("forbidden"), "forbidden_domain_no_single_phase_fluid"), (t["forbidden_resolution"].eq(FORBIDDEN_RESOLUTIONS[2]), "forbidden_beyond_gor_error"), (t["domain"].eq("diphasic"), "diphasic_at_saturation_pressure"), (t["envelope_basis"].isin(["partly reference", "reference"]), "envelope_uses_reference_fluid"), (t["temperature_is_estimated"], "temperature_estimated"),
                       (t["oil_density_g_cm3"].notna() & t["api"].isna(), "api_outside_plausible_range"), (t["uplift_area"], "uplift_area_past_temperature_higher")):
        flags = flags.where(~mask.fillna(False).astype(bool), flags + text + ";")
    t["flags"] = flags.str.rstrip(";").replace("", pd.NA)
    columns = ["wellbore_id", "well_name", "test_number", "area", "play_id", "mid_md_m", "tvdss_m", "reservoir_pressure_bar", "temperature_c", "temperature_used_c", "temperature_is_estimated", "temperature_est_c", "temperature_est_basis", "temperature_est_gradient_c_km", "glr_sm3_sm3", "oil_density_g_cm3", "api", "api_class",
               "gas_relative_density", "fluid_class", "fluid_class_source", "api_source", "envelope_basis", "oil_limit_glr", "gas_limit_glr", "critical_pressure_bar", "domain", "saturation_state", "saturation_basis", "envelope_state", "forbidden_glr_factor", "forbidden_resolution",
               "bubble_point_est_bar", "bubble_point_bar", "bubble_point_source", "dew_point_bar", "report_rs_sm3_sm3", "report_cgr_sm3_msm3", "report_sample_type", "report_source", "pressure_to_bubble_point",
               "near_critical_zone", "temperature_bracket", "biodegradation_assessment", "uplift_area", "measurement_type", "measurement_group", "measurement_confidence", "measurement_evidence", "sample_link", "sample_fluid_type", "sample_types_in_well",
               "wireline_samples_in_well", "test_interval_m", "drawdown_ratio", "drawdown_class", "flags"]
    out = t[t["glr_sm3_sm3"].notna() & t["reservoir_pressure_bar"].notna()][[c for c in columns if c in t.columns]].reset_index(drop=True)
    events.append(dq_event(*ctx, "well_dst_pvt_rows", len(out), "info", detail=f"states: {out['saturation_state'].value_counts().to_dict()}; fluid classes: {out['fluid_class'].value_counts().to_dict()}; {int(out['saturation_basis'].eq('pvt_report').sum())} states from a PVT report"))
    n_forbidden = int(out["domain"].eq("forbidden").sum())
    if n_forbidden:
        events.append(dq_event(*ctx, "pvt_forbidden_tests", n_forbidden, "info", out.loc[out["domain"].eq("forbidden"), "well_name"].dropna().unique()[:10],
                               "pressure below the saturation pressure of this GLR: not a single-phase fluid (gas cap, coning, commingled intervals, two-phase inflow, or a pressure / rate error); envelopes are ASSUMPTIONS (config/pvt.yaml)"))
        shares = out["forbidden_resolution"].value_counts().to_dict()
        events.append(dq_event(*ctx, "pvt_forbidden_gor_error", int(sum(shares.values())), "info", [],
                               f"forbidden tests (envelope estimate) against the GOR error of the literature (+-{cfg['gor_uncertainty']['typical']:.0%} typical, +-{cfg['gor_uncertainty']['worst_case']:.0%} worst case; knowledge entry {cfg['gor_uncertainty']['knowledge_entry']}): {shares}"))
    return out, events


def place_tests_in_plays(tests: pd.DataFrame, well_play: pd.DataFrame | None) -> pd.Series:
    """play_id of each test: the play whose interval in that wellbore contains the test mid depth (MD; the shortest interval wins), NA when none does.
    The play of a test is a depth-based proxy: the interval is the one the stratigraphy of that well gives for the play."""
    out = pd.Series(pd.NA, index=tests.index, dtype="string")
    if tests.empty or well_play is None or well_play.empty or not {"wellbore_id", "mid_md_m"} <= set(tests.columns):
        return out
    t = tests[["wellbore_id", "mid_md_m"]].dropna().reset_index(names="test_row")
    m = t.merge(well_play[["wellbore_id", "play_id", "top_md_m", "base_md_m"]], on="wellbore_id")
    m = m[(m["mid_md_m"] >= m["top_md_m"]) & (m["mid_md_m"] <= m["base_md_m"])].copy()
    m["span"] = m["base_md_m"] - m["top_md_m"]
    m = m.sort_values("span").drop_duplicates("test_row")
    out.loc[m["test_row"].to_numpy()] = m["play_id"].astype("string").to_numpy()
    return out


def build_play_pvt_summary(pvt: pd.DataFrame, well_play: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Fluid classes, saturation states and biodegradation assessments per play (`play_id` of the test, see `place_tests_in_plays`; computed here from `well_play` when the mart has none)."""
    ctx = ("all", "play_pvt_summary")
    if pvt.empty:
        return pd.DataFrame(), [dq_event(*ctx, "play_pvt_inputs_missing", 0, "info", detail="no PVT tests")]
    t = pvt.copy()
    if ("play_id" not in t.columns or t["play_id"].isna().all()) and well_play is not None:
        t["play_id"] = place_tests_in_plays(t, well_play)
    m = t[t["play_id"].notna()]
    if m.empty:
        return pd.DataFrame(), [dq_event(*ctx, "play_pvt_no_match", 0, "info", detail="no test depth falls inside a play interval")]
    rows = []
    for play, g in m.groupby("play_id"):
        row = {"play_id": play, "n_tests": len(g), "n_wells": int(g["wellbore_id"].nunique()), "n_with_api": int(g["api"].notna().sum()),
               "median_api": float(g["api"].median()) if g["api"].notna().sum() >= 5 else np.nan,              # fewer than 5 values: blank, never a number from two tests
               "median_glr_sm3_sm3": float(g["glr_sm3_sm3"].median()) if len(g) >= 5 else np.nan,
               "n_likely_biodegraded": int(g["biodegradation_assessment"].astype("string").str.startswith("likely", na=False).sum()),
               "n_pasteurised_light": int(g["biodegradation_assessment"].astype("string").str.contains("pasteurised", na=False).sum()), "n_near_critical_zone": int(g["near_critical_zone"].fillna(False).sum())}
        for name in STATE_ORDER:
            row["n_" + name.replace(" ", "_").replace("-", "_")] = int((g["saturation_state"] == name).sum())
        rows.append(row)
    out = pd.DataFrame(rows)
    return out, [dq_event(*ctx, "play_pvt_rows", len(out), "info", detail=f"{int(out['n_tests'].sum())} tests placed in {len(out)} plays")]
