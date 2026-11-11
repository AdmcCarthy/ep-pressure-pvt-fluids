"""Pressure-depth gradients by fluid, the free water level where a hydrocarbon gradient meets the water gradient, and the column a seal can hold.  Pure functions.

Why (Adam, 2026-11-04): pressures measured in a hydrocarbon leg follow the gradient of that fluid, those in the water leg follow the water gradient, and the depth where the two lines cross is
the FREE WATER LEVEL (FWL), the pressure-defined level at which the hydrocarbon and water pressures are equal.  It is the most robust way to place a contact because it needs no log or
sample reading of saturation.  The observed oil-water contact lies ABOVE the FWL by the height the entry pressure of the rock holds up (the transition zone): `contact_above_fwl_m`.
A seal holds a hydrocarbon column of at most  entry pressure / (water gradient - hydrocarbon gradient): `max_column_height_m` (seal capacity).

Depth is TVD below sea level [m]; pressure is bar; gradients are bar/m (a gradient of 0.0981 bar/m is 1.0 g/cm3).  The inputs are formation pressure points with a fluid label (RFT / MDT / XPT
points of the `formation_pressure` enrichment sheet, `docs/enrichment.md`); Sodir has no such table, so the results exist where points are supplied.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ep_physics_base.events import dq_event

BAR_PER_M_PER_SG = 0.0980665
SCHEMA_VERSIONS = {"well_free_water_level": "1.0"}
ORDER = ["gas", "oil", "water"]                      # lightest first: a gradient must increase along this list


def fit_gradient(depth_m: pd.Series | np.ndarray, pressure_bar: pd.Series | np.ndarray) -> dict[str, float]:
    """Least-squares line  p = intercept + gradient x z  with its standard errors; fewer than two points or no depth range gives NaN."""
    z, p = np.asarray(depth_m, dtype=float), np.asarray(pressure_bar, dtype=float)
    ok = np.isfinite(z) & np.isfinite(p)
    z, p = z[ok], p[ok]
    n = len(z)
    nan = {"n": n, "gradient": np.nan, "intercept": np.nan, "se_gradient": np.nan, "se_intercept": np.nan, "density_g_cm3": np.nan, "r2": np.nan, "depth_range_m": float(np.ptp(z)) if n else np.nan}
    if n < 2 or np.ptp(z) <= 0:
        return nan
    slope, intercept = np.polyfit(z, p, 1)
    resid = p - (intercept + slope * z)
    if n > 2:
        s2 = float(resid @ resid) / (n - 2)
        sxx = float(((z - z.mean()) ** 2).sum())
        se_g, se_i = np.sqrt(s2 / sxx), np.sqrt(s2 * (1 / n + z.mean() ** 2 / sxx))
        ss_tot = float(((p - p.mean()) ** 2).sum())
        r2 = 1 - float(resid @ resid) / ss_tot if ss_tot > 0 else np.nan
    else:
        se_g = se_i = r2 = np.nan
    return {"n": n, "gradient": float(slope), "intercept": float(intercept), "se_gradient": float(se_g), "se_intercept": float(se_i), "density_g_cm3": float(slope) / BAR_PER_M_PER_SG, "r2": r2, "depth_range_m": float(np.ptp(z))}


def intersect(light: dict[str, float], heavy: dict[str, float]) -> float:
    """Depth where the light-fluid line meets the heavy-fluid line (NaN when the gradients do not allow it: the light fluid must have the LOWER gradient)."""
    dg = light["gradient"] - heavy["gradient"]
    if not np.isfinite(dg) or dg >= 0:
        return np.nan
    return (heavy["intercept"] - light["intercept"]) / dg


def free_water_level(points: pd.DataFrame, cfg: dict[str, Any], n_boot: int = 500, seed: int = 20261105) -> pd.DataFrame:
    """Gas-oil and oil-water (or gas-water) levels from the crossing of fluid gradients, one row per pair with data.

    `points`: columns `tvdss_m`, `pressure_bar`, `fluid` (gas / oil / water).  A fluid with one point only is allowed for WATER when `fwl.assumed_water_gradient_bar_per_m` is set: the line then runs
    through that point with the assumed gradient (`method` says so).  The 5-95 % range is a bootstrap over the points of each fluid (fixed seed).  A crossing outside `fwl.plausible_range_m` of the
    tested hydrocarbon depths is reported with status `outside the tested range` (an extrapolation)."""
    fc = cfg["fwl"]
    rng = np.random.default_rng(seed)
    by = {f: points[points["fluid"] == f] for f in ORDER}
    rows = []
    for light, heavy, name in (("gas", "oil", "gas-oil level"), ("oil", "water", "free water level (oil-water)"), ("gas", "water", "free water level (gas-water)")):
        if name.endswith("(gas-water)") and len(by["oil"]) >= 1:
            continue                                                         # a leg with oil between gas and water is read as two pairs
        a, b = by[light], by[heavy]
        method = "two fitted gradients"
        fl = fit_gradient(a["tvdss_m"], a["pressure_bar"])
        fh = fit_gradient(b["tvdss_m"], b["pressure_bar"])
        if heavy == "water" and not np.isfinite(fh["gradient"]) and len(b) >= 1 and fc.get("assumed_water_gradient_bar_per_m"):
            g = float(fc["assumed_water_gradient_bar_per_m"])
            fh = {"n": len(b), "gradient": g, "intercept": float((b["pressure_bar"] - g * b["tvdss_m"]).mean()), "density_g_cm3": g / BAR_PER_M_PER_SG, "r2": np.nan, "depth_range_m": 0.0}
            method = "hydrocarbon gradient fitted, water line through the point(s) with the assumed water gradient"
        if not (np.isfinite(fl["gradient"]) and np.isfinite(fh["gradient"])):
            continue
        z = intersect(fl, fh)
        status = "ok"
        if not np.isfinite(z):
            status = "gradients do not cross (the light fluid is not lighter)"
        boots = []
        if np.isfinite(z):
            for _ in range(n_boot):
                sa, sb = a.sample(len(a), replace=True, random_state=int(rng.integers(1 << 31))), b.sample(len(b), replace=True, random_state=int(rng.integers(1 << 31)))
                gl, gh = fit_gradient(sa["tvdss_m"], sa["pressure_bar"]), fit_gradient(sb["tvdss_m"], sb["pressure_bar"])
                if method != "two fitted gradients" and not np.isfinite(gh["gradient"]):
                    g = float(fc["assumed_water_gradient_bar_per_m"])
                    gh = {"gradient": g, "intercept": float((sb["pressure_bar"] - g * sb["tvdss_m"]).mean())}
                zb = intersect(gl, gh) if np.isfinite(gl["gradient"]) and np.isfinite(gh.get("gradient", np.nan)) else np.nan
                if np.isfinite(zb):
                    boots.append(zb)
            lo = float(fc["plausible_range_m"])
            if not (a["tvdss_m"].min() - lo <= z <= max(a["tvdss_m"].max(), b["tvdss_m"].min()) + lo):
                status = "outside the tested range"
            if fl["gradient"] < float(fc["min_gradient_bar_per_m"][light]) or fl["gradient"] > float(fc["max_gradient_bar_per_m"][light]):
                status = "hydrocarbon gradient outside the plausible band for " + light
        rows.append({"level": name, "depth_tvdss_m": z, "depth_p5_m": float(np.percentile(boots, 5)) if boots else np.nan, "depth_p95_m": float(np.percentile(boots, 95)) if boots else np.nan,
                     "light_fluid": light, "light_gradient_bar_per_m": fl["gradient"], "light_density_g_cm3": fl["density_g_cm3"], "light_points": fl["n"],
                     "heavy_fluid": heavy, "heavy_gradient_bar_per_m": fh["gradient"], "heavy_density_g_cm3": fh["density_g_cm3"], "heavy_points": fh["n"], "method": method, "status": status})
    return pd.DataFrame(rows)


def contact_above_fwl_m(entry_pressure_bar: float, water_gradient: float, hc_gradient: float) -> float:
    """Height of the 100 % water level (the contact seen on logs) above the FWL: the entry pressure of the reservoir rock divided by the gradient contrast."""
    return entry_pressure_bar / (water_gradient - hc_gradient)


def max_column_height_m(seal_entry_pressure_bar: float, water_gradient: float, hc_gradient: float) -> float:
    """Seal capacity as a hydrocarbon column: the column at which the buoyancy pressure at the top equals the entry pressure of the seal (static, no leakage by faults or fractures)."""
    return seal_entry_pressure_bar / (water_gradient - hc_gradient)


def build_free_water_levels(points: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[dict]]:
    """Per wellbore with labelled pressure points (`wellbore_id`, `tvdss_m`, `pressure_bar`, `fluid`): the levels from `free_water_level`."""
    ctx = ("all", "well_free_water_level")
    columns = ["wellbore_id", "level", "depth_tvdss_m", "depth_p5_m", "depth_p95_m", "light_fluid", "light_gradient_bar_per_m", "light_density_g_cm3", "light_points", "heavy_fluid",
               "heavy_gradient_bar_per_m", "heavy_density_g_cm3", "heavy_points", "method", "status"]
    if points is None or points.empty or not {"wellbore_id", "tvdss_m", "pressure_bar", "fluid"} <= set(points.columns):
        return pd.DataFrame(columns=columns), [dq_event(*ctx, "fwl_no_points", 0, "info", detail="no formation-pressure points with a fluid label (sheet formation_pressure, column fluid): no free water level can be derived")]
    p = points.dropna(subset=["tvdss_m", "pressure_bar", "fluid"])
    p = p[p["fluid"].isin(ORDER)]
    parts = []
    for well, g in p.groupby("wellbore_id"):
        r = free_water_level(g, cfg)
        if len(r):
            parts.append(r.assign(wellbore_id=well))
    out = pd.concat(parts, ignore_index=True)[columns] if parts else pd.DataFrame(columns=columns)
    return out, [dq_event(*ctx, "fwl_levels", len(out), "info", detail=f"{len(out)} levels from {p['wellbore_id'].nunique()} wellbores with labelled pressure points; {int((out['status'] == 'ok').sum()) if len(out) else 0} ok")]
