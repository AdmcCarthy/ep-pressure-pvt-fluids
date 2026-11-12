"""Pressure regimes, HPHT classes, hydrostatic gradients and fracture gradients from drill stem tests, leak-off tests and the wellbore header.

Marts (all derived, schema 1.0)
    well_dst_pressure      one row per DST (plus optional manual formation-pressure points): which pressure is a reservoir pressure, at what depth, its
                           ratio to the hydrostatic pressure, pressure regime, temperature, HPHT class
    well_pressure_summary  one row per wellbore: regime, HPHT class, pressure-depth gradient, geothermal gradient, leak-off / fracture gradient and the
                           mud window, with what is missing to improve it
    play_pressure_summary  tests, regimes and HPHT classes per play

Rules (AGENTS.md sections 6, 16)
    * Pressures arrive in MPa and are stored in bar (`well_dst`); a running check (`check_pressure_units`) makes a wrong unit visible.
    * Of the three DST pressures only the BOTTOM-HOLE pressure is a reservoir pressure.  Shut-in and flowing pressures are mostly wellhead values (the
      shut-in is about 0.36 x bottom hole for oil tests, 0.77 x for gas tests): each shut-in is classified `bottom_hole` or `wellhead` from the test itself.
    * Depth is MD [m RT] unless a survey gives TVD.  TVD is never fabricated: the depth used carries `depth_basis` and `depth_confidence`, and where the
      depth is MD of a deviated or unknown well the overpressure is a MINIMUM (`overpressure_is_minimum`).
    * A well with no information stays in the mart with empty values and `missing` listing what would improve it (seabed depth, survey, ...): the set-up
      is ready for enrichment (`tvd_lookup`, manual `well_depth` and `formation_pressure` sheets, docs/enrichment.md) without code changes.
    * Pure functions: DataFrames in, DataFrames out, plus data-quality events.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from ep_physics_base.events import dq_event
from ep_physics_base.config import load_yaml

logger = logging.getLogger(__name__)

CONFIG_FILE = "config/pressure.yaml"
SCHEMA_VERSIONS = {"well_contact_tvdss": "1.0", "well_dst_pressure": "1.1", "well_pressure_summary": "1.0", "play_pressure_summary": "1.0"}
HPHT_ORDER = ["standard", "HPHT", "ultra-HPHT", "extreme-HPHT"]


def load_config() -> dict[str, Any]:
    return load_yaml(CONFIG_FILE)


# --------------------------------------------------------------------------- #
# Depth reference
# --------------------------------------------------------------------------- #
def tvd_from_lookup(wellbore_ids: pd.Series, md: pd.Series, lookup: pd.DataFrame) -> pd.Series:
    """TVD [m RT] at the given MD from MANUAL points (wellbore_id, md_m, tvd_m): linear interpolation inside the points of a well, NaN outside them or with fewer than two points.  Never extrapolated.
    Survey depths are not converted here: the host fills the canonical TVD columns from the selected survey (minimum-curvature arc) and this module reads them (docs/md_to_tvd.md)."""
    out = pd.Series(np.nan, index=md.index, dtype="float64")
    if lookup is None or lookup.empty or not {"wellbore_id", "md_m", "tvd_m"} <= set(lookup.columns):
        return out
    points = lookup.dropna(subset=["wellbore_id", "md_m", "tvd_m"]).sort_values("md_m")
    for well, group in points.groupby("wellbore_id"):
        rows = (wellbore_ids == well).to_numpy()
        if not rows.any():
            continue
        x, y = group["md_m"].astype(float).to_numpy(), group["tvd_m"].astype(float).to_numpy()
        values = md[rows].astype(float).to_numpy()
        result = np.full(len(values), np.nan)
        if len(np.unique(x)) >= 2:
            inside = (values >= x.min()) & (values <= x.max())
            result[inside] = np.interp(values[inside], x, y)
        for xi, yi in zip(x, y):                           # a point given at (or within 1 m of) the depth of a test is used as it is
            result[np.abs(values - xi) <= 1.0] = yi
        out[rows] = result
    return out


def well_depth_reference(well_header: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Per wellbore: RT elevation, water depth, TD MD / TVD, TVD/MD ratio at TD and the depth basis it allows."""
    columns = ["wellbore_id", "rt_elev_m", "water_depth_m", "td_md_m", "td_tvd_m", "main_area"]
    if well_header.empty:
        return pd.DataFrame(columns=columns + ["tvd_md_ratio", "rt_assumed", "base_depth_basis"])
    ref = well_header.drop_duplicates("wellbore_id").reindex(columns=columns).copy()
    for column in columns[1:5]:
        ref[column] = pd.to_numeric(ref[column], errors="coerce").astype("float64")
    ref["tvd_md_ratio"] = ref["td_tvd_m"] / ref["td_md_m"]
    ref["rt_assumed"] = ref["rt_elev_m"].isna()
    ref["rt_elev_m"] = ref["rt_elev_m"].fillna(float(cfg["depth"]["assumed_rt_elev_m"]))
    near = ref["tvd_md_ratio"] >= float(cfg["depth"]["near_vertical_tvd_md_ratio"])
    ref["base_depth_basis"] = np.where(near, "md_near_vertical", np.where(ref["tvd_md_ratio"].notna(), "md_deviated", "md_no_tvd"))
    return ref


# --------------------------------------------------------------------------- #
# Classification helpers
# --------------------------------------------------------------------------- #
def classify_regime(ratio: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    """Pressure regime from the ratio to the hydrostatic pressure (first regime whose `below` bound is above the ratio)."""
    out = pd.Series(pd.NA, index=ratio.index, dtype="string")
    previous = -np.inf
    for regime in cfg["regimes"]:
        out[(ratio >= previous) & (ratio < float(regime["below"]))] = regime["name"]
        previous = float(regime["below"])
    return out


def _rank(values: pd.Series, thresholds: list[float]) -> pd.Series:
    """Number of thresholds the value reaches (NaN stays NaN)."""
    ranks = sum((values >= t).astype("float64") for t in thresholds)
    return ranks.where(values.notna())


def classify_hpht(pressure_bar: pd.Series, temperature_c: pd.Series, cfg: dict[str, Any]) -> pd.DataFrame:
    """Pressure class, temperature class and the combined HPHT class (both must exceed the class thresholds), plus the pattern when only one does."""
    h = cfg["hpht"]
    names = h["names"]
    p_rank, t_rank = _rank(pressure_bar, h["pressure_bar"]), _rank(temperature_c, h["temperature_c"])
    both = pd.concat([p_rank, t_rank], axis=1).min(axis=1, skipna=False)
    out = pd.DataFrame(index=pressure_bar.index)
    out["pressure_class"] = p_rank.map(lambda r: pd.NA if pd.isna(r) else names[int(r)]).astype("string")
    out["temperature_class"] = t_rank.map(lambda r: pd.NA if pd.isna(r) else names[int(r)]).astype("string")
    out["hpht_class"] = both.map(lambda r: pd.NA if pd.isna(r) else names[int(r)]).astype("string")
    pattern = pd.Series(pd.NA, index=out.index, dtype="string")
    pattern[p_rank.notna() & t_rank.notna()] = "standard"
    pattern[(p_rank >= 1) & (t_rank >= 1)] = "high pressure and high temperature"
    pattern[(p_rank >= 1) & (t_rank < 1)] = "high pressure only"
    pattern[(p_rank < 1) & (t_rank >= 1)] = "high temperature only"
    pattern[p_rank.notna() & t_rank.isna() & (p_rank >= 1)] = "high pressure (temperature unknown)"
    pattern[p_rank.isna() & t_rank.notna() & (t_rank >= 1)] = "high temperature (pressure unknown)"
    out["hp_ht_pattern"] = pattern
    return out


def hpht_zone_polygons(cfg: dict[str, Any], t_max: float = 260.0, p_max: float = 1500.0) -> dict[str, tuple[list[float], list[float]]]:
    """Closed polygons (temperature [degC], pressure [bar]) of the pressure-temperature zones, each to be filled with one flat colour: `normal` (neither
    threshold reached), `high temperature` (only temperature), `high pressure` (only pressure) and the combined classes of `classify_hpht` (`HPHT`,
    `ultra-HPHT`, `extreme-HPHT`: both reach the class thresholds, the class is the LOWER of the two ranks, so each class is an L-shaped region)."""
    h = cfg["hpht"]
    (p1, p2, p3), (t1, t2, t3) = h["pressure_bar"], h["temperature_c"]
    names = h["names"]
    box = lambda t0, t_1, p0, p_1: [(t0, p0), (t_1, p0), (t_1, p_1), (t0, p_1)]

    def level(t_lo: float, p_lo: float, t_hi: float | None, p_hi: float | None) -> list[tuple[float, float]]:
        """Both at or above (t_lo, p_lo) but not both at or above (t_hi, p_hi)."""
        if t_hi is None or p_hi is None:
            return box(t_lo, t_max, p_lo, p_max)
        return [(t_lo, p_lo), (t_max, p_lo), (t_max, p_hi), (t_hi, p_hi), (t_hi, p_max), (t_lo, p_max)]

    zones = {"normal": box(0.0, t1, 0.0, p1), "high temperature": box(t1, t_max, 0.0, p1), "high pressure": box(0.0, t1, p1, p_max),
             names[1]: level(t1, p1, t2, p2), names[2]: level(t2, p2, t3, p3), names[3]: level(t3, p3, None, None)}
    return {name: ([x for x, _ in pts] + [pts[0][0]], [y for _, y in pts] + [pts[0][1]]) for name, pts in zones.items()}


def classify_shut_in(shut_in: pd.Series, bhp: pd.Series, gradient: pd.Series, cfg: dict[str, Any]) -> pd.DataFrame:
    """Is the shut-in pressure of a test a bottom-hole value or a wellhead value?  From the test itself: ratio to the bottom-hole pressure when both exist,
    otherwise from the pressure gradient (probable)."""
    s = cfg["shut_in"]
    basis = pd.Series(pd.NA, index=shut_in.index, dtype="string")
    evidence = pd.Series(pd.NA, index=shut_in.index, dtype="string")
    ratio = shut_in / bhp
    both = shut_in.notna() & bhp.notna()
    basis[both & (ratio >= float(s["bottom_hole_min_ratio"]))] = "bottom_hole"
    basis[both & (ratio < float(s["bottom_hole_min_ratio"]))] = "wellhead"
    evidence[both] = "ratio to bottom-hole pressure"
    only = shut_in.notna() & bhp.isna()
    basis[only & (gradient >= float(s["probable_bottom_hole_gradient_bar_per_m"]))] = "bottom_hole_probable"
    basis[only & (gradient < float(s["probable_bottom_hole_gradient_bar_per_m"]))] = "wellhead_probable"
    evidence[only] = "pressure gradient (no bottom-hole pressure in the test)"
    return pd.DataFrame({"shut_in_basis": basis, "shut_in_basis_evidence": evidence, "shut_in_to_bhp_ratio": ratio.where(both)})


# --------------------------------------------------------------------------- #
# Tests: reservoir pressure per DST
# --------------------------------------------------------------------------- #
def _test_key(value: Any) -> Any:
    """Test number as text: 1.0 -> "1", 2.1 -> "2.1" (a number typed in a spreadsheet matches the Sodir label)."""
    if value is None or pd.isna(value):
        return pd.NA
    text = str(value).strip()
    try:
        number = float(text)
        return str(int(number)) if number == int(number) else f"{number:g}"
    except ValueError:
        return text


def build_well_dst_pressure(dst: pd.DataFrame, header: pd.DataFrame, cfg: dict[str, Any], tvd_lookup: pd.DataFrame | None = None,
                            extra_points: pd.DataFrame | None = None, detail: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """One row per drill stem test (and per manual formation-pressure point) with the reservoir pressure, depth basis, overpressure, regime and HPHT class."""
    ctx = ("all", "well_dst_pressure")
    columns = ["wellbore_id", "well_name", "test_number", "pressure_source", "top_md_m", "base_md_m", "mid_md_m", "depth_used_m", "depth_basis", "depth_confidence", "depth_sigma_m", "tvdss_m",
               "reservoir_pressure_bar", "reservoir_pressure_basis", "pressure_confidence", "shut_in_bar", "shut_in_basis", "shut_in_basis_evidence", "shut_in_to_bhp_ratio",
               "flow_pressure_bar", "bottom_hole_pressure_bar", "hydrostatic_bar", "hydrostatic_sigma_bar", "overpressure_bar", "overpressure_sigma_bar", "pressure_ratio", "pressure_ratio_sigma", "pore_emw_sg", "pore_emw_sigma_sg", "sigma_basis", "pressure_regime", "overpressure_is_minimum",
               "temperature_c", "temperature_source", "geothermal_gradient_c_km", "pressure_class", "temperature_class", "hpht_class", "hp_ht_pattern", "fluid_hint", "rt_assumed", "pressure_screen", "rejected_pressure_bar", "area", "flow_basis", "siwhp_bar", "sibhp_bar", "fwhp_bar", "fbhp_bar", "bhp_kind", "static_bhp_confirmed",
               "pressure_definition_source", "underpressure_suspect_outside_area", "underpressure_unconfirmed_static", "flags"]
    if dst.empty:
        return pd.DataFrame(columns=columns), [dq_event(*ctx, "dst_inputs_missing", 0, "warning", detail="well_dst is empty")]
    t = dst.copy()
    t["pressure_source"] = "official_dst"
    t["shut_in_bar"], t["flow_pressure_bar"] = t["shut_in_pressure_bar"], t["flow_pressure_bar"]
    if extra_points is not None and not extra_points.empty:
        extra = extra_points.rename(columns={"md_m": "mid_md_m", "pressure_bar": "bottom_hole_pressure_bar"}).copy()
        extra["pressure_source"] = "manual_formation_pressure"
        extra["top_md_m"] = extra["base_md_m"] = extra["mid_md_m"]
        t = pd.concat([t, extra.reindex(columns=t.columns.union(extra.columns, sort=False))], ignore_index=True)
    t["wellbore_id"] = t["wellbore_id"].astype("Int64")
    t["mid_md_m"] = ((t["top_md_m"].astype("float64") + t["base_md_m"].astype("float64")) / 2).where(t["top_md_m"].notna() & t["base_md_m"].notna(), pd.to_numeric(t.get("mid_md_m"), errors="coerce")).astype("float64")
    ref = well_depth_reference(header, cfg)
    t = t.merge(ref, on="wellbore_id", how="left")
    t["area"] = t["main_area"].astype("string")
    manual_cols = ["siwhp_bar", "sibhp_bar", "fwhp_bar", "fbhp_bar", "bhp_kind"]
    for column in manual_cols:
        t[column + "_manual"] = np.nan if column != "bhp_kind" else pd.NA
    if detail is not None and not detail.empty:                       # wellhead / bottom-hole and static / flowing values given by a person (sheet dst_pressure_detail)
        d = detail.copy()
        d["test_number"] = d["test_number"].map(_test_key)
        d = d.drop_duplicates(["wellbore_id", "test_number"]).set_index(["wellbore_id", "test_number"])
        keys = pd.MultiIndex.from_arrays([t["wellbore_id"].astype("Int64"), t["test_number"].map(_test_key)])
        official = t["pressure_source"].eq("official_dst").to_numpy()
        for column in manual_cols:
            if column in d.columns:
                got = pd.Series(d[column].reindex(keys).to_numpy(), index=t.index)
                t[column + "_manual"] = got.where(official, t[column + "_manual"])
    # The TVD of a test is read from the canonical table (`tvd_mid_m`, filled from the selected survey by the shared MD-to-TVD step, docs/md_to_tvd.md, with its `tvd_source` and `tvd_sigma_m`); this
    # function converts no survey depth itself.  `tvd_lookup` holds MANUAL points only (no angles, linear), used for rows the canonical table could not place.
    canonical_tvd = pd.to_numeric(t["tvd_mid_m"], errors="coerce").astype("float64") if "tvd_mid_m" in t else pd.Series(np.nan, index=t.index, dtype="float64")
    if "tvd_source" in t:
        canonical_tvd = canonical_tvd.where(t["tvd_source"].astype("string").isin(["survey", "reported"]))
    manual_tvd = tvd_from_lookup(t["wellbore_id"], t["mid_md_m"], tvd_lookup if tvd_lookup is not None else pd.DataFrame())
    from_canonical, from_manual = canonical_tvd.notna(), canonical_tvd.isna() & manual_tvd.notna()
    tvd = canonical_tvd.where(from_canonical, manual_tvd)
    t["depth_used_m"] = tvd.where(tvd.notna(), t["mid_md_m"].astype("float64"))
    t["depth_basis"] = np.where(from_canonical, "tvd_survey", np.where(from_manual, "tvd_manual", t["base_depth_basis"].fillna("md_no_tvd")))
    t["depth_confidence"] = t["depth_basis"].map({"tvd_survey": "high", "tvd_manual": "high", "md_near_vertical": "medium", "md_deviated": "low", "md_no_tvd": "low"})
    t["depth_sigma_m"] = pd.to_numeric(t["tvd_sigma_m"], errors="coerce").astype("float64").where(from_canonical) if "tvd_sigma_m" in t else np.nan          # the survey's own vertical uncertainty at the test
    t["tvdss_m"] = t["depth_used_m"] - t["rt_elev_m"]
    t["bhp"] = pd.to_numeric(t["bottom_hole_pressure_bar"], errors="coerce").astype("float64")
    shut = pd.to_numeric(t["shut_in_bar"], errors="coerce").astype("float64")
    t = pd.concat([t, classify_shut_in(shut, t["bhp"], shut / t["tvdss_m"], cfg)], axis=1)
    flow = pd.to_numeric(t["flow_pressure_bar"], errors="coerce").astype("float64")
    flow_class = classify_shut_in(flow, t["bhp"], flow / t["tvdss_m"], cfg)
    t["flow_basis"] = flow_class["shut_in_basis"]
    # wellhead (SIWHP, FWHP) and bottom-hole (SIBHP, FBHP) values: inferred from each test (ratio to the bottom-hole pressure, else gradient); a person's value wins
    inferred = {"siwhp_bar": shut.where(t["shut_in_basis"].isin(["wellhead", "wellhead_probable"])), "sibhp_bar": shut.where(t["shut_in_basis"].isin(["bottom_hole", "bottom_hole_probable"])),
                "fwhp_bar": flow.where(t["flow_basis"].isin(["wellhead", "wellhead_probable"])), "fbhp_bar": flow.where(t["flow_basis"].isin(["bottom_hole", "bottom_hole_probable"]))}
    for column, values in inferred.items():
        t[column] = pd.to_numeric(t[column + "_manual"], errors="coerce").astype("float64").combine_first(values)
    has_manual = pd.concat([pd.to_numeric(t[c + "_manual"], errors="coerce").notna() for c in manual_cols[:4]], axis=1).any(axis=1) | t["bhp_kind_manual"].notna()
    t["pressure_definition_source"] = np.where(has_manual, "manual", np.where(t[["siwhp_bar", "sibhp_bar", "fwhp_bar", "fbhp_bar"]].notna().any(axis=1), "inferred from ratio / gradient", None))
    t["bhp_kind"] = t["bhp_kind_manual"].astype("string").fillna(pd.Series(np.where(t["bhp"].notna(), "unspecified (static or flowing)", None), index=t.index).astype("string"))
    t["static_bhp_confirmed"] = t["bhp_kind"].eq("static").fillna(False)
    bhp_usable = t["bhp"].notna() & ~t["bhp_kind"].eq("flowing").fillna(False)          # a bottom-hole pressure that a person says is a FLOWING value is not a reservoir pressure
    reservoir = t["bhp"].where(bhp_usable)
    basis = pd.Series(np.where(bhp_usable, "bottom_hole pressure", None), index=t.index, dtype="object")
    confidence = pd.Series(np.where(bhp_usable, "high", None), index=t.index, dtype="object")
    static_shut = pd.to_numeric(t["sibhp_bar_manual"], errors="coerce").astype("float64")
    use_manual = static_shut.notna()
    reservoir[use_manual], basis[use_manual], confidence[use_manual] = static_shut[use_manual], "static shut-in bottom-hole pressure (manual)", "high"
    manual = t["pressure_source"].eq("manual_formation_pressure")
    confidence[manual & t["bhp"].notna()] = "medium"
    use_shut = reservoir.isna() & t["shut_in_basis"].isin(["bottom_hole", "bottom_hole_probable"]) & shut.notna()
    reservoir[use_shut] = shut[use_shut]
    basis[use_shut] = "shut-in (bottom-hole value)"
    confidence[use_shut] = np.where(t.loc[use_shut, "shut_in_basis"].eq("bottom_hole"), "medium", "low")
    g = float(cfg["hydrostatic"]["gradient_bar_per_m"])
    sg = float(cfg["hydrostatic"]["bar_per_m_per_sg"])
    t["hydrostatic_bar"] = g * t["tvdss_m"]
    t["hydrostatic_sigma_bar"] = g * t["depth_sigma_m"]                      # the hydrostatic pressure moves by the gradient times the vertical uncertainty of the depth (empty where the survey states none)
    # screen: a static reservoir pressure far below hydrostatic is a wellhead or flowing value, one far above is a typing / unit error (kept, flagged, not used)
    screen_cfg = cfg["screen"]
    candidate_ratio = reservoir / t["hydrostatic_bar"]
    low, high = candidate_ratio < float(screen_cfg["min_ratio"]), candidate_ratio > float(screen_cfg["max_ratio"])
    bar_ratio = candidate_ratio / 10.0
    possible_bar = high & bar_ratio.between(float(screen_cfg["min_ratio"]), float(screen_cfg["max_ratio"]))
    t["pressure_screen"] = np.select([possible_bar, low, high], ["rejected: possibly delivered in bar (not MPa)", "rejected: far below hydrostatic (wellhead or flowing value?)", "rejected: far above any plausible regime"], default=np.where(reservoir.notna(), "ok", None))
    t["rejected_pressure_bar"] = reservoir.where(low | high)
    reject = low | high
    reservoir, basis, confidence = reservoir.where(~reject), basis.where(~reject), confidence.where(~reject)
    t["reservoir_pressure_bar"], t["reservoir_pressure_basis"], t["pressure_confidence"] = reservoir, basis, confidence
    t["overpressure_bar"] = t["reservoir_pressure_bar"] - t["hydrostatic_bar"]
    t["pressure_ratio"] = t["reservoir_pressure_bar"] / t["hydrostatic_bar"]
    t["pore_emw_sg"] = t["reservoir_pressure_bar"] / (sg * t["depth_used_m"])        # on the same RT basis as a leak-off test (pressure / (g x TVD below RT))
    # The uncertainty that follows from the depth only (the gauge accuracy and drift are not stated, so these are a LOWER BOUND of the whole): the overpressure moves with the hydrostatic pressure, the ratio and
    # the equivalent density by the relative depth uncertainty.  Empty where no survey states one.
    t["overpressure_sigma_bar"] = t["hydrostatic_sigma_bar"]
    t["pressure_ratio_sigma"] = t["pressure_ratio"] * t["hydrostatic_sigma_bar"] / t["hydrostatic_bar"]
    t["pore_emw_sigma_sg"] = t["pore_emw_sg"] * t["depth_sigma_m"] / t["depth_used_m"]
    t["sigma_basis"] = pd.Series(np.where(t["depth_sigma_m"].notna(), "survey vertical uncertainty of the depth only; gauge accuracy not stated", "not stated"), index=t.index).astype("string")
    t["pressure_regime"] = classify_regime(t["pressure_ratio"], cfg)
    u = cfg["underpressure"]
    in_up = t["area"].str.upper().isin([a.upper() for a in u["areas"]]).fillna(False).to_numpy(dtype=bool)
    sub = t["pressure_regime"].eq("subnormal").fillna(False).to_numpy(dtype=bool)
    t.loc[sub & in_up, "pressure_regime"] = u["regime_name"]                      # real underpressure (uplift and erosion): a valid regime in the Barents Sea
    suspect = (t["pressure_ratio"] < float(u["suspect_below_outside"])).fillna(False).to_numpy(dtype=bool) & ~in_up
    t.loc[suspect & t["pressure_confidence"].isin(["high", "medium"]), "pressure_confidence"] = "low"       # outside the Barents Sea a low value is a screening suspect
    t["underpressure_suspect_outside_area"] = suspect
    t["underpressure_unconfirmed_static"] = (t["pressure_regime"] == u["regime_name"]) & ~t["static_bhp_confirmed"]
    t["overpressure_is_minimum"] = t["depth_basis"].isin(["md_deviated", "md_no_tvd"]) & t["reservoir_pressure_bar"].notna()
    dst_temp = pd.to_numeric(t.get("downhole_temperature_c"), errors="coerce").astype("float64") if "downhole_temperature_c" in t else pd.Series(np.nan, index=t.index)
    t["temperature_c"], t["temperature_source"] = dst_temp, np.where(dst_temp.notna(), "official_dst", None)
    below_seabed = t["depth_used_m"] - t["rt_elev_m"] - t["water_depth_m"]
    t["geothermal_gradient_c_km"] = ((t["temperature_c"] - float(cfg["hydrostatic"]["seabed_temperature_c"])) / below_seabed * 1000).where(below_seabed > 100)
    t = pd.concat([t, classify_hpht(t["reservoir_pressure_bar"], t["temperature_c"], cfg)], axis=1)
    oil, gas = t.get("oil_rate_sm3d"), t.get("gas_rate_sm3d")
    hint = pd.Series(pd.NA, index=t.index, dtype="string")
    if oil is not None and gas is not None:
        gor = pd.to_numeric(t.get("gor_sm3_sm3"), errors="coerce")
        hint[gor >= 1000] = "gas or condensate (GOR >= 1000)"
        hint[gor < 1000] = "oil (GOR < 1000)"
    if "fluid" in t:                                                                    # a fluid given for a manual formation-pressure point (RFT / MDT: water, oil, gas) wins over the GOR hint
        manual_fluid = t["fluid"].astype("string").str.lower()
        manual_fluid = manual_fluid.where(manual_fluid.isin(["water", "oil", "gas", "condensate"]))
        hint = hint.where(manual_fluid.isna(), manual_fluid + " (manual)")
    t["fluid_hint"] = hint
    flags = pd.Series("", index=t.index, dtype="object")
    for mask, text in ((t["rt_assumed"].fillna(False), "rt_assumed"), (t["overpressure_is_minimum"], "overpressure_minimum"), (t["pressure_ratio"] > 2.0, "ratio_above_2"),
                       (t["shut_in_to_bhp_ratio"] > 1.08, "shut_in_above_bhp"), (t["reservoir_pressure_bar"].notna() & t["tvdss_m"].isna(), "no_depth"), (t["pressure_screen"].astype("string").str.startswith("rejected"), "pressure_rejected"),
                       (t["underpressure_suspect_outside_area"], "low_pressure_outside_underpressure_area"), (t["underpressure_unconfirmed_static"], "underpressure_static_unconfirmed")):
        flags = flags.where(~mask.fillna(False).astype(bool), flags + text + ";")
    t["flags"] = flags.str.rstrip(";").replace("", pd.NA)
    out = t[columns].sort_values(["wellbore_id", "mid_md_m"], ignore_index=True)
    events = [dq_event(*ctx, "well_dst_pressure_rows", len(out), "info",
                       detail=f"{int(out['reservoir_pressure_bar'].notna().sum())} tests with a reservoir pressure; shut-in: {out['shut_in_basis'].value_counts().to_dict()}; regimes: {out['pressure_regime'].value_counts().to_dict()}")]
    return out, events


# --------------------------------------------------------------------------- #
# Buoyancy: the pressure of a test inside a hydrocarbon column against the aquifer pressure
# --------------------------------------------------------------------------- #
BUOYANCY_COLUMNS = ["buoyancy_fluid", "buoyancy_contacts", "contact_tvdss_m", "contact_basis", "column_below_test_m", "hc_gradient_bar_per_m", "buoyancy_excess_bar", "buoyancy_excess_low_bar",
                    "buoyancy_excess_high_bar", "buoyancy_is_minimum", "aquifer_pressure_bar", "aquifer_ratio", "aquifer_regime", "overpressure_beyond_column_bar", "buoyancy_status", "hc_gradient_basis", "water_gradient_bar_per_m"]


def well_contacts_tvdss(facts: pd.DataFrame, header: pd.DataFrame, lookup: pd.DataFrame | None, cfg: dict[str, Any]) -> pd.DataFrame:
    """Fluid contacts stated in the well-history text (fact_type `contact`: OWC, GOC, GWC, HWC and the down-to / up-to depths ODT, GDT, WUT) as depths [m TVDSS] with their basis.

    A stated TVDSS / MSL depth is used as it is, a stated TVD is read as TVD below the rotary table (ASSUMPTION: the reference of the history text is RT), an MD is converted with a survey
    or manual point where there is one and otherwise only for a near-vertical well (never fabricated).  Contacts are text evidence, `confidence` as extracted."""
    columns = ["wellbore_id", "label", "contact_class", "tvdss_m", "basis", "confidence", "evidence"]
    if facts is None or facts.empty or "fact_type" not in facts:
        return pd.DataFrame(columns=columns)
    labels = cfg["buoyancy"]["contact_labels"]
    cls = {lab: name for name, labs in labels.items() for lab in labs}
    c = facts[facts["fact_type"] == "contact"].copy()
    c["contact_class"] = c["label"].astype("string").str.upper().map(cls)
    c = c.dropna(subset=["contact_class"])
    if c.empty:
        return pd.DataFrame(columns=columns)
    ref = well_depth_reference(header, cfg)[["wellbore_id", "rt_elev_m", "base_depth_basis"]]
    c = c.merge(ref, on="wellbore_id", how="left")
    value = pd.to_numeric(c["value"], errors="coerce")
    unit = c["unit"].astype("string").str.upper().fillna("")
    md = pd.to_numeric(c["depth_top_m"], errors="coerce")
    refc = c["depth_reference"].astype("string").str.upper().fillna("")
    tvd_lookup = tvd_from_lookup(c["wellbore_id"], md, lookup if lookup is not None else pd.DataFrame())               # manual points only
    if "depth_top_tvd_m" in c:                                                                                         # the canonical TVD of the depth (shared MD-to-TVD step) outranks a manual point
        canonical = pd.to_numeric(c["depth_top_tvd_m"], errors="coerce").astype("float64")
        if "depth_tvd_source" in c:
            canonical = canonical.where(c["depth_tvd_source"].astype("string").eq("survey"))
        tvd_lookup = canonical.where(canonical.notna(), tvd_lookup)
    tvdss = pd.Series(np.nan, index=c.index, dtype="float64")
    basis = pd.Series("", index=c.index, dtype="object")
    for mask, vals, text in (
            (value.notna() & unit.isin(["M TVDSS", "M MSL"]), value, "stated TVDSS"),
            (value.notna() & unit.eq("M TVD") & c["rt_elev_m"].notna(), value - c["rt_elev_m"], "stated TVD (read as below RT)"),
            (md.notna() & refc.eq("MSL"), md, "stated depth below MSL"),
            (md.notna() & refc.str.startswith("TVD") & c["rt_elev_m"].notna(), md - c["rt_elev_m"], "stated TVD (read as below RT)"),
            (md.notna() & tvd_lookup.notna() & c["rt_elev_m"].notna(), tvd_lookup - c["rt_elev_m"], "MD converted with the survey / a manual point"),
            (md.notna() & c["base_depth_basis"].eq("md_near_vertical") & c["rt_elev_m"].notna(), md - c["rt_elev_m"], "MD of a near-vertical well")):
        free = tvdss.isna() & mask
        tvdss[free], basis[free] = vals[free], text
    out = pd.DataFrame({"wellbore_id": c["wellbore_id"], "label": c["label"].astype("string").str.upper(), "contact_class": c["contact_class"], "tvdss_m": tvdss, "basis": basis,
                        "confidence": c["confidence"], "evidence": c["evidence"]}).dropna(subset=["tvdss_m"])
    return out.drop_duplicates(["wellbore_id", "label", "tvdss_m"]).reset_index(drop=True)


def _hc_gradients(fluid: str, info: dict[str, Any], cfg: dict[str, Any]) -> tuple[float, float, float, str]:
    """(central, low, high, basis) in-situ gradient [bar/m] of the hydrocarbon leg of a test, from its fluid (PVT class, stock-tank density, GLR, gas gravity, pressure and temperature), not from
    the area (`fluid_gradients.py`); the configured band is the fallback where the test has none of these."""
    from ep_pressure_pvt_fluids import fluid_gradients as fg

    return fg.hydrocarbon_gradients(fluid, info, cfg)


def water_gradient_band(cfg: dict[str, Any]) -> tuple[float, float, float, str]:
    """(central, low, high, basis) of the water gradient: the hydrostatic gradient with the fresh-to-brine band (a salinity moves it; `fluid_gradients.water_gradient`)."""
    from ep_pressure_pvt_fluids import fluid_gradients as fg

    return fg.water_gradient(None, cfg)


def add_buoyancy_correction(pressure: pd.DataFrame, contacts: pd.DataFrame, oil_density: pd.Series | None, cfg: dict[str, Any], pvt: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """For a test in a gas or oil leg: the pressure the same column puts on the aquifer at its contact, and the part of the excess over hydrostatic that is the column itself.

    Hydraulically connected fluids follow their own gradients (Zhang, `zh-hydraulic-connection`): from a test at depth z down to the contact z_c the pressure rises by the hydrocarbon gradient, the water
    outside the column by the water gradient, so  p_contact = p_test + g_hc (z_c - z)  and  buoyancy excess = (g_w - g_hc)(z_c - z).  A gas leg over an oil leg is two segments (GOC, then the oil contact).
    The excess over hydrostatic that remains once the column is taken out is the real overpressure of the aquifer (`overpressure_beyond_column_bar`).  Contacts are text evidence (see
    `well_contacts_tvdss`); a test with no contact keeps its raw regime and is flagged, never corrected by a guess.  A down-to depth (ODT, GDT) is a minimum column: the excess is a minimum."""
    ctx = ("all", "well_dst_pressure")
    out = pressure.copy()
    for column in BUOYANCY_COLUMNS:
        out[column] = pd.NA
    for column in ("contact_tvdss_m", "column_below_test_m", "hc_gradient_bar_per_m", "buoyancy_excess_bar", "buoyancy_excess_low_bar", "buoyancy_excess_high_bar", "aquifer_pressure_bar", "aquifer_ratio",
                   "overpressure_beyond_column_bar"):
        out[column] = np.nan
    out["buoyancy_is_minimum"] = False
    b = cfg["buoyancy"]
    gw, gw_low, gw_high, _ = water_gradient_band(cfg)
    pvt_rows = {(int(k[0]), str(k[1])): v for k, v in pvt.set_index(["wellbore_id", "test_number"]).to_dict("index").items()} if pvt is not None and not pvt.empty else {}
    by_well = {k: g.sort_values("tvdss_m") for k, g in contacts.groupby("wellbore_id")} if contacts is not None and not contacts.empty else {}
    status = pd.Series("no reservoir pressure", index=out.index, dtype="object")
    for i, r in out.iterrows():
        p, z = r["reservoir_pressure_bar"], r["tvdss_m"]
        if pd.isna(p) or pd.isna(z):
            continue
        hint = str(r["fluid_hint"]) if pd.notna(r["fluid_hint"]) else ""
        if hint.startswith("water"):
            status[i] = "water leg (no column effect)"
            continue
        fluid = "oil" if hint.startswith("oil") else "gas" if hint.startswith(("gas", "condensate")) else None
        if fluid is None:
            status[i] = "fluid of the test unknown"
            continue
        cons = by_well.get(r["wellbore_id"])
        near = cons[(cons["tvdss_m"] > z) & (cons["tvdss_m"] <= z + float(b["max_column_m"]))] if cons is not None else pd.DataFrame()
        if near.empty:
            if cons is None:
                status[i] = "no contact on record"
            elif not (cons["tvdss_m"] > z).any():
                status[i] = "test below the stated contact (no column below)"
            else:
                status[i] = "nearest contact below is further than the column window"
            continue
        pv = pvt_rows.get((int(r["wellbore_id"]), str(r.get("test_number"))), {})
        info = {"pressure_bar": float(p), "temperature_c": pv.get("temperature_used_c", r.get("temperature_c")), "gas_gravity": pv.get("gas_relative_density"),
                "sto_density_g_cm3": pv.get("oil_density_g_cm3") if pd.notna(pv.get("oil_density_g_cm3", np.nan)) else (oil_density.get(r["wellbore_id"]) if oil_density is not None else None),
                "glr": pv.get("glr_sm3_sm3"), "fluid_class": pv.get("fluid_class")}
        bases = []
        segs, names, zc, minimum = [], [], z, False
        cur = fluid
        for c in near.itertuples():
            if c.contact_class == "gas_oil":
                if cur != "gas":
                    continue
                segs.append(("gas", c.tvdss_m - zc)), names.append(c.label)
                zc, cur = c.tvdss_m, "oil"
                continue
            if c.contact_class == "down_to":
                if (c.label == "ODT" and cur == "oil") or (c.label == "GDT" and cur == "gas"):
                    segs.append((cur, c.tvdss_m - zc)), names.append(c.label + " (bound)")
                    zc, minimum = c.tvdss_m, True
                break
            if c.contact_class in ("oil_water", "gas_water"):
                segs.append((cur, c.tvdss_m - zc)), names.append(c.label)
                zc, minimum = c.tvdss_m, False
                break
        if not segs or zc <= z:
            status[i] = "no usable contact below the test"
            continue
        if names and names[-1].endswith("(bound)"):
            minimum = True
        elif cur == "oil" and fluid == "gas" and not any(n in ("OWC", "HWC", "GWC") for n in names):
            minimum = True                                                                      # gas cap down to the GOC only: the oil leg below is not counted
        ex = [0.0, 0.0, 0.0]
        pc = float(p)
        for f, dz in segs:
            central, low, high, basis = _hc_gradients(f, info, cfg)
            bases.append(f"{f}: {basis}")
            pc += central * dz
            ex[0] += (gw - central) * dz
            ex[1] += (gw_low - high) * dz                                                       # the heavier the hydrocarbon and the lighter the water, the smaller the excess
            ex[2] += (gw_high - low) * dz
        total_dz = zc - z
        out.loc[i, ["buoyancy_fluid", "buoyancy_contacts", "contact_basis"]] = [fluid, "+".join(names), "; ".join(sorted(set(near["basis"])))]
        out.loc[i, "contact_tvdss_m"], out.loc[i, "column_below_test_m"] = zc, total_dz
        out.loc[i, "hc_gradient_bar_per_m"] = (pc - p) / total_dz
        out.loc[i, "hc_gradient_basis"], out.loc[i, "water_gradient_bar_per_m"] = "; ".join(bases), gw
        out.loc[i, ["buoyancy_excess_bar", "buoyancy_excess_low_bar", "buoyancy_excess_high_bar"]] = ex
        out.loc[i, "buoyancy_is_minimum"] = bool(minimum)
        out.loc[i, "aquifer_pressure_bar"] = pc
        out.loc[i, "aquifer_ratio"] = pc / (gw * zc)
        out.loc[i, "overpressure_beyond_column_bar"] = float(r["overpressure_bar"]) - ex[0]
        status[i] = "corrected (down-to depth only: minimum)" if minimum else "corrected"
    out["buoyancy_status"] = status
    regime = classify_regime(pd.to_numeric(out["aquifer_ratio"], errors="coerce"), cfg)
    u = cfg["underpressure"]
    in_up = out["area"].astype("string").str.upper().isin([a.upper() for a in u["areas"]]).fillna(False).to_numpy(dtype=bool)
    regime[regime.eq("subnormal").fillna(False).to_numpy(dtype=bool) & in_up] = u["regime_name"]
    out["aquifer_regime"] = regime
    unknown = out["buoyancy_status"].eq("no contact on record") & (pd.to_numeric(out["pressure_ratio"], errors="coerce") >= float(b["unknown_contact_ratio"]))
    out.loc[unknown, "flags"] = (out.loc[unknown, "flags"].fillna("").astype(str) + ";buoyancy_unknown_contact").str.strip(";")
    done = out["buoyancy_status"].astype("string").str.startswith("corrected")
    events = [dq_event(*ctx, "buoyancy_correction", int(done.sum()), "info", detail=f"{int(done.sum())} of {int(out['reservoir_pressure_bar'].notna().sum())} reservoir pressures in a gas or oil leg have a contact below the test; "
                       f"median buoyancy excess {out.loc[done, 'buoyancy_excess_bar'].astype(float).median():.1f} bar; regime changes: "
                       f"{int((out.loc[done, 'pressure_regime'] != out.loc[done, 'aquifer_regime']).sum())}; hydrocarbon tests above ratio {b['unknown_contact_ratio']} with no contact: {int(unknown.sum())}")]
    return out, events


def check_pressure_units(pressure: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[dict]]:
    """Running unit check of the converted pressures: the median bottom-hole pressure / depth gradient [bar/m] must be near hydrostatic."""
    ctx = ("all", "well_dst_pressure")
    c = cfg["unit_check"]
    use = pressure[pressure["pressure_source"].eq("official_dst") & pressure["bottom_hole_pressure_bar"].notna() & pressure["depth_used_m"].gt(0)] if not pressure.empty else pressure
    n = len(use)
    median = float((use["bottom_hole_pressure_bar"] / use["depth_used_m"]).median()) if n else np.nan
    ok = bool(c["bhp_gradient_bar_per_m"][0] <= median <= c["bhp_gradient_bar_per_m"][1]) if n else None
    table = pd.DataFrame([{"check": "median bottom-hole pressure / depth", "n_tests": n, "median_bar_per_m": median, "expected_min": c["bhp_gradient_bar_per_m"][0],
                           "expected_max": c["bhp_gradient_bar_per_m"][1], "unit_ok": ok,
                           "meaning": "hydrostatic is about 0.10 bar/m: a value near 0.01 would mean the MPa values were not converted, near 1.0 that bar were converted again"}])
    events = []
    if n >= int(c["min_tests"]) and not ok:
        events.append(dq_event(*ctx, "pressure_unit_suspect", n, "warning", detail=f"median bottom-hole pressure gradient {median:.3f} bar/m is outside {c['bhp_gradient_bar_per_m']}: check the MPa to bar conversion"))
    else:
        events.append(dq_event(*ctx, "pressure_unit_check", n, "info", detail=f"median bottom-hole pressure gradient {median:.3f} bar/m (expected {c['bhp_gradient_bar_per_m']}): units consistent with MPa -> bar"))
    return table, events


# --------------------------------------------------------------------------- #
# Wells: gradients, fracture gradient, mud window
# --------------------------------------------------------------------------- #
def _gradient_fit(group: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    g = cfg["gradient"]
    use = group.dropna(subset=["reservoir_pressure_bar", "tvdss_m"])
    use = use[use["pressure_confidence"].isin(["high", "medium"])]
    if len(use) < int(g["min_points"]) or (use["tvdss_m"].max() - use["tvdss_m"].min()) < float(g["min_depth_range_m"]):
        return pd.Series({"gradient_bar_per_m": np.nan, "gradient_r2": np.nan, "gradient_points": len(use)})
    slope, intercept = np.polyfit(use["tvdss_m"], use["reservoir_pressure_bar"], 1)
    fitted = slope * use["tvdss_m"] + intercept
    ss_tot = float(((use["reservoir_pressure_bar"] - use["reservoir_pressure_bar"].mean()) ** 2).sum())
    r2 = 1 - float(((use["reservoir_pressure_bar"] - fitted) ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan
    return pd.Series({"gradient_bar_per_m": float(slope), "gradient_r2": r2, "gradient_points": len(use)})


def build_well_fracture(casing_lot: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[dict]]:
    """Per wellbore: the deepest leak-off (LOT) and integrity (FIT) test with a usable equivalent mud density, as a fracture-gradient estimate."""
    ctx = ("all", "well_pressure_summary")
    columns = ["wellbore_id", "n_lot_fit", "lot_shoe_md_m", "lot_emw_sg", "lot_test_type", "fracture_gradient_bar_per_m", "fracture_gradient_kind", "max_lot_emw_sg"]
    if casing_lot.empty:
        return pd.DataFrame(columns=columns), [dq_event(*ctx, "casing_lot_missing", 0, "info", detail="well_casing_lot is empty: no fracture gradient")]
    low, high = cfg["fracture"]["emw_plausible_sg"]
    t = casing_lot.copy()
    t["emw"] = pd.to_numeric(t["lot_emw_sg"], errors="coerce").astype("float64")
    implausible = t["emw"].notna() & ~t["emw"].between(low, high)
    use = t[t["emw"].between(low, high)].copy()
    use["shoe"] = pd.to_numeric(use["casing_depth_md_m"], errors="coerce").astype("float64")
    use = use.sort_values(["wellbore_id", "shoe"])
    deepest = use.groupby("wellbore_id").tail(1).set_index("wellbore_id")
    out = pd.DataFrame({"n_lot_fit": use.groupby("wellbore_id").size(), "lot_shoe_md_m": deepest["shoe"], "lot_emw_sg": deepest["emw"], "lot_test_type": deepest["test_type"],
                        "max_lot_emw_sg": use.groupby("wellbore_id")["emw"].max()}).reset_index()
    out["fracture_gradient_bar_per_m"] = out["lot_emw_sg"] * float(cfg["hydrostatic"]["bar_per_m_per_sg"])
    kind = out["lot_test_type"].astype("string").str.upper().str.strip()
    out["fracture_gradient_kind"] = kind.map({"FIT": "integrity_minimum (FIT)", "LOT": "leak_off (LOT)"}).fillna("unspecified test").astype("string")
    out["wellbore_id"] = out["wellbore_id"].astype("Int64")
    events = [dq_event(*ctx, "fracture_gradient_wells", len(out), "info", detail=f"{len(out)} wellbores with a leak-off or integrity test; {int(implausible.sum())} values outside {low}-{high} sg kept out")]
    if implausible.any():
        events.append(dq_event(*ctx, "lot_emw_implausible", int(implausible.sum()), "warning", t.loc[implausible, "well_name"].dropna().unique()[:20],
                               f"equivalent mud density outside {low}-{high} sg: kept in the source, not used"))
    return out[columns], events


def build_well_pressure_summary(pressure: pd.DataFrame, fracture: pd.DataFrame, header: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[dict]]:
    """One row per wellbore with a test or a leak-off test: pressure regime and HPHT class (the most extreme test), pressure-depth gradient, geothermal gradient,
    fracture gradient and the mud window, and what is missing to improve the result."""
    ctx = ("all", "well_pressure_summary")
    ids = pd.concat([pressure["wellbore_id"] if not pressure.empty else pd.Series(dtype="Int64"), fracture["wellbore_id"] if not fracture.empty else pd.Series(dtype="Int64")]).dropna().unique()
    if len(ids) == 0:
        return pd.DataFrame(), [dq_event(*ctx, "pressure_summary_inputs_missing", 0, "info", detail="no tests and no leak-off tests")]
    base = pd.DataFrame({"wellbore_id": pd.array(ids, dtype="Int64")})
    ref = well_depth_reference(header, cfg)
    names = header.drop_duplicates("wellbore_id")[["wellbore_id", "well_name", "bottom_hole_temp_c"]] if not header.empty else pd.DataFrame(columns=["wellbore_id", "well_name", "bottom_hole_temp_c"])
    out = base.merge(names, on="wellbore_id", how="left").merge(ref[["wellbore_id", "base_depth_basis", "water_depth_m", "td_tvd_m"]], on="wellbore_id", how="left")
    if not pressure.empty:
        have = pressure[pressure["reservoir_pressure_bar"].notna()]
        top = have.sort_values("pressure_ratio", ascending=False).groupby("wellbore_id").head(1).set_index("wellbore_id")
        deepest = have.sort_values("depth_used_m").groupby("wellbore_id").tail(1).set_index("wellbore_id")
        agg = pd.DataFrame({"n_tests": pressure.groupby("wellbore_id").size(), "n_tests_with_pressure": have.groupby("wellbore_id").size(), "n_tests_with_temperature": pressure[pressure["temperature_c"].notna()].groupby("wellbore_id").size(),
                            "max_reservoir_pressure_bar": have.groupby("wellbore_id")["reservoir_pressure_bar"].max(), "max_temperature_c": pressure.groupby("wellbore_id")["temperature_c"].max(),
                            "max_pressure_ratio": top["pressure_ratio"], "min_pressure_ratio": have.groupby("wellbore_id")["pressure_ratio"].min(), "any_underpressure": have["pressure_regime"].eq("underpressured").groupby(have["wellbore_id"]).any(),
                            "area": pressure.groupby("wellbore_id")["area"].first(), "max_overpressure_bar": top["overpressure_bar"], "regime": top["pressure_regime"], "depth_at_max_ratio_m": top["depth_used_m"],
                            "depth_basis": top["depth_basis"], "overpressure_is_minimum": top["overpressure_is_minimum"], "pressure_confidence": top["pressure_confidence"],
                            "deepest_test_depth_m": deepest["depth_used_m"], "deepest_pore_emw_sg": deepest["pore_emw_sg"], "geothermal_gradient_c_km": pressure.groupby("wellbore_id")["geothermal_gradient_c_km"].median()})
        out = out.merge(agg.reset_index(), on="wellbore_id", how="left")
        fits = have.groupby("wellbore_id").apply(lambda g: _gradient_fit(g, cfg), include_groups=False).reset_index()
        out = out.merge(fits, on="wellbore_id", how="left")
    for column in ("n_tests", "n_tests_with_pressure", "n_tests_with_temperature", "max_reservoir_pressure_bar", "max_temperature_c", "max_pressure_ratio", "min_pressure_ratio", "max_overpressure_bar", "depth_at_max_ratio_m",
                   "deepest_test_depth_m", "deepest_pore_emw_sg", "geothermal_gradient_c_km", "gradient_bar_per_m", "gradient_r2", "gradient_points"):
        if column not in out:
            out[column] = np.nan
    for column in ("regime", "depth_basis", "pressure_confidence"):
        if column not in out:
            out[column] = pd.NA
    if "overpressure_is_minimum" not in out:
        out["overpressure_is_minimum"] = pd.NA
    previous = -np.inf
    out["gradient_fluid_hint"] = pd.Series(pd.NA, index=out.index, dtype="string")
    good = (out["gradient_r2"] >= float(cfg["gradient"]["min_r2_for_hint"])) & (out["gradient_bar_per_m"] > 0)          # a poor fit says nothing about the fluid
    for band in cfg["gradient"]["fluid_bands_bar_per_m"]:
        out.loc[good & (out["gradient_bar_per_m"] >= previous) & (out["gradient_bar_per_m"] < float(band["below"])), "gradient_fluid_hint"] = band["name"]
        previous = float(band["below"])
    out["bht_c"] = pd.to_numeric(out["bottom_hole_temp_c"], errors="coerce").astype("float64")       # header value: not corrected to formation temperature (a minimum)
    out["temperature_for_class_c"] = out["max_temperature_c"].fillna(out["bht_c"])
    out["temperature_for_class_source"] = np.where(out["max_temperature_c"].notna(), "official_dst", np.where(out["bht_c"].notna(), "header_bht_uncorrected_minimum", None))
    out = pd.concat([out, classify_hpht(out["max_reservoir_pressure_bar"], out["temperature_for_class_c"], cfg)], axis=1)
    if not fracture.empty:
        out = out.merge(fracture, on="wellbore_id", how="left")
    else:
        for column in ("n_lot_fit", "lot_shoe_md_m", "lot_emw_sg", "lot_test_type", "fracture_gradient_bar_per_m", "fracture_gradient_kind", "max_lot_emw_sg"):
            out[column] = np.nan
    out["mud_window_sg"] = out["lot_emw_sg"] - out["deepest_pore_emw_sg"]
    out["mud_window_is_minimum"] = out["mud_window_sg"].notna()          # the shoe is shallower than the deepest test and fracture gradients rise with depth: the real window is wider
    out["narrow_window"] = out["mud_window_sg"] < float(cfg["fracture"]["narrow_window_sg"])
    missing = pd.Series("", index=out.index, dtype="object")
    for mask, text in ((out["td_tvd_m"].isna(), "td_tvd_m"), (out["base_depth_basis"].isin(["md_deviated", "md_no_tvd"]), "survey (tvd)"), (out["water_depth_m"].isna(), "water_depth_m"),
                       (out["n_tests_with_pressure"].fillna(0).eq(0), "bottom-hole or formation pressure"), (out["max_temperature_c"].isna(), "downhole temperature"), (out["lot_emw_sg"].isna(), "leak-off test")):
        missing = missing.where(~mask.fillna(False).astype(bool), missing + text + "; ")
    out["missing"] = missing.str.rstrip("; ").replace("", pd.NA)
    events = [dq_event(*ctx, "well_pressure_summary_rows", len(out), "info", detail=f"{int(out['regime'].notna().sum())} wells with a regime, {int((out['hpht_class'].isin(HPHT_ORDER[1:])).sum())} HPHT or higher, {int(out['lot_emw_sg'].notna().sum())} with a fracture gradient, {int(out['gradient_bar_per_m'].notna().sum())} with a pressure-depth gradient")]
    return out.sort_values("wellbore_id", ignore_index=True), events


def build_play_pressure_summary(pressure: pd.DataFrame, well_play: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[dict]]:
    """Tests, regimes and HPHT classes per play: a test belongs to the play whose interval in that well contains the test's mid depth (shortest interval wins)."""
    ctx = ("all", "play_pressure_summary")
    if pressure.empty or well_play.empty:
        return pd.DataFrame(), [dq_event(*ctx, "play_pressure_inputs_missing", 0, "info", detail="no pressure tests or no well_play_summary")]
    t = pressure[pressure["mid_md_m"].notna()][["wellbore_id", "mid_md_m", "reservoir_pressure_bar", "pressure_ratio", "pressure_regime", "temperature_c", "hpht_class", "pore_emw_sg", "overpressure_is_minimum"]].reset_index(names="test_row")
    m = t.merge(well_play[["wellbore_id", "play_id", "top_md_m", "base_md_m"]], on="wellbore_id")
    m = m[(m["mid_md_m"] >= m["top_md_m"]) & (m["mid_md_m"] <= m["base_md_m"])].copy()
    m["span"] = m["base_md_m"] - m["top_md_m"]
    m = m.sort_values("span").drop_duplicates("test_row")
    if m.empty:
        return pd.DataFrame(), [dq_event(*ctx, "play_pressure_no_match", 0, "info", detail="no test depth falls inside a play interval")]
    order = ["underpressured", "subnormal", "normal", "mildly overpressured", "overpressured", "highly overpressured"]
    rows = []
    for play, g in m.groupby("play_id"):
        p = g["pressure_ratio"].dropna()
        row = {"play_id": play, "n_tests": len(g), "n_with_pressure": int(g["reservoir_pressure_bar"].notna().sum()), "n_wells": int(g["wellbore_id"].nunique()),
               "median_pressure_ratio": float(p.median()) if len(p) else np.nan, "p90_pressure_ratio": float(p.quantile(0.9)) if len(p) else np.nan,
               "max_reservoir_pressure_bar": float(g["reservoir_pressure_bar"].max()) if g["reservoir_pressure_bar"].notna().any() else np.nan,
               "max_temperature_c": float(g["temperature_c"].max()) if g["temperature_c"].notna().any() else np.nan, "n_hpht": int(g["hpht_class"].isin(HPHT_ORDER[1:]).sum())}
        for name in order:
            row["n_" + name.replace(" ", "_")] = int((g["pressure_regime"] == name).sum())
        rows.append(row)
    out = pd.DataFrame(rows)
    return out, [dq_event(*ctx, "play_pressure_rows", len(out), "info", detail=f"{int(out['n_tests'].sum())} tests placed in {len(out)} plays")]


def add_overburden_share(pressure: pd.DataFrame, well_header: pd.DataFrame, pp_cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[dict]]:
    """Add the reservoir pressure as a share of the MODELLED overburden stress at the depth of each test (Zhang, pp. 3-4: pore pressure runs up to about 95 % of the overburden stress).

    Columns added: `overburden_bar`, `overburden_share`, `overburden_share_is_minimum` (the depth is a measured depth of a deviated or unsurveyed well, so the true vertical depth is smaller, the modelled
    overburden at that depth is too large and the share is a MINIMUM) and `share_above_limit` (above `overburden_share.upper_limit`: a finding about the pressure, the depth or the modelled
    overburden, never corrected).  The overburden is a compaction trend until density logs arrive.  Existing columns are not changed.
    """
    from ep_pressure_pvt_fluids import pore_pressure as pp

    ctx = ("all", "well_dst_pressure")
    if pressure.empty or "tvdss_m" not in pressure:
        return pressure, []
    wd = well_header.drop_duplicates("wellbore_id").set_index("wellbore_id")["water_depth_m"] if not well_header.empty else pd.Series(dtype="float64")
    water = pd.to_numeric(pressure["wellbore_id"].map(wd), errors="coerce").to_numpy(dtype="float64")
    share = pp.overburden_share(pd.to_numeric(pressure["reservoir_pressure_bar"], errors="coerce"), pd.to_numeric(pressure["tvdss_m"], errors="coerce"), water, pp_cfg)
    out = pressure.copy()
    out["overburden_bar"] = share["overburden_bar"].to_numpy()
    out["overburden_share"] = share["overburden_share"].to_numpy()
    out["overburden_share_is_minimum"] = pressure["depth_basis"].astype("string").isin(["md_deviated", "md_no_tvd"]).to_numpy() & out["overburden_share"].notna().to_numpy()
    out["share_above_limit"] = share["above_upper_limit"].to_numpy()
    n_over = int(out["share_above_limit"].sum())
    events = [dq_event(*ctx, "overburden_share", int(out["overburden_share"].notna().sum()), "info",
                       detail=f"{int(out['overburden_share'].notna().sum())} tests with a share of the modelled overburden; {n_over} above the upper limit (a finding about the pressure, the depth or the modelled overburden)")]
    if n_over:
        events.append(dq_event(*ctx, "overburden_share_above_limit", n_over, "warning", out.loc[out["share_above_limit"], "well_name"].dropna().unique()[:10],
                               "share of the modelled overburden above the physical limit: kept, not corrected"))
    return out, events
