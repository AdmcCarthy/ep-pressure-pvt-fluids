"""Compositional grading of a hydrocarbon column with a cubic equation of state, and the envelope of a column's pressure profile.  Pure functions.

The traditional column is LINEAR: the pressure rises from the apex with the density at the apex times g.  In a real column the heavy components settle, so the fluid below the apex is heavier and, for a
volatile oil or a gas condensate, closer to or further from its saturation pressure.  Isothermal gravity-chemical equilibrium (Gibbs; Schulte 1980; Montel and Gouel 1985, via Hoier and Whitson, not read):

    ln f_i(x(z), p(z), T) = ln f_i(x_0, p_0, T) + M_i g (z - z_0) / (R T)               for every component i (z positive downwards), with  sum x_i = 1

so that composition and pressure at every depth follow from the apex values; dp/dz = rho g then holds by the Gibbs-Duhem relation.  `grade_column` solves it step by step with a cubic EOS
(`eos_mixture`).  Temperature may be uniform (`isothermal`) or rise with depth (`geothermal`): the geothermal option evaluates the same condition at the local temperature and has NO thermal diffusion
term, so it is an approximation that brackets the non-isothermal case (a temperature gradient reduces, can enhance, balance or remove the gravity grading: Hoier and Whitson slides, not read).
A fluid close to its critical point is the hard case: the solution may not exist or be very sensitive to the equation; a failed step is reported (`converged = False`), never smoothed over.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ep_pressure_pvt_fluids import eos_mixture as em

G = em.G


def _t_k(t0_c: float, dz_m: float, mode: str, gradient_c_km: float) -> float:
    return t0_c + 273.15 + (gradient_c_km * dz_m / 1000.0 if mode == "geothermal" else 0.0)


def _solve_hydrostatic(m: em.Mixture, ln_target: np.ndarray, x_prev: np.ndarray, p_prev: float, rho_prev: float, dz: float, t_k: float, tol: float, max_iter: int) -> tuple[np.ndarray, float, bool]:
    """Non-isothermal step without a thermal-diffusion term: the pressure follows from dp = rho g dz (trapezoid), the composition from the fugacity condition at that pressure and the LOCAL temperature
    (normalised).  The equilibrium condition is then only approximate (Gibbs-Duhem is not satisfied when T varies), which is why the result is reported as an approximation."""
    x = x_prev.copy()
    p = p_prev + rho_prev * G * dz / em.BAR
    for _ in range(max_iter):
        lnphi, _z = em.ln_phi(m, x, t_k, p, "stable")
        xu = np.exp(ln_target - lnphi) / p
        xn = xu / xu.sum()
        rho_new = em.density_kg_m3(m, xn, t_k, p, "stable")
        p_new = p_prev + 0.5 * (rho_prev + rho_new) * G * dz / em.BAR
        if np.max(np.abs(xn - x)) < tol and abs(p_new - p) < 1e-9:
            return xn, p_new, True
        x, p = xn, p_new
    return x, p, np.max(np.abs(xn - x)) < 1e-6


def _solve_step(m: em.Mixture, ln_f0: np.ndarray, x_prev: np.ndarray, p_prev: float, rho_prev: float, dz_total: float, dz: float, t_k: float, tol: float, max_iter: int, hydrostatic: bool = False) -> tuple[np.ndarray, float, bool]:
    """Composition and pressure at the new depth: for a trial pressure the composition follows from the fugacity condition, the pressure is the one that makes the mole fractions sum to 1 (secant)."""
    mi = m.mw_g_mol / 1000.0
    ln_target = ln_f0 + mi * G * dz_total / (em.R * t_k)                  # depth is positive downwards: the heavier the component, the more its fugacity rises with depth
    x = x_prev.copy()
    if hydrostatic:
        return _solve_hydrostatic(m, ln_target, x_prev, p_prev, rho_prev, dz, t_k, tol, max_iter)
    state = {"x": x}

    def residual(p: float) -> float:
        xx = state["x"]
        s = 1.0
        for _ in range(60):
            lnphi, _z = em.ln_phi(m, xx, t_k, p, "stable")
            xu = np.exp(ln_target - lnphi) / p
            s = float(xu.sum())
            xn = xu / s
            if np.max(np.abs(xn - xx)) < tol:
                xx = xn
                break
            xx = xn
        state["x"] = xx
        return s - 1.0

    p0 = p_prev + rho_prev * G * dz / em.BAR
    p1 = p0 * 1.002
    r0 = residual(p0)
    r1 = residual(p1)
    for _ in range(max_iter):
        if abs(r1) < 1e-10:
            return state["x"], p1, True
        if r1 == r0:
            break
        p2 = p1 - r1 * (p1 - p0) / (r1 - r0)
        p0, r0 = p1, r1
        p1 = float(np.clip(p2, 0.5 * p_prev, 1.5 * p_prev + 50.0))
        r1 = residual(p1)
    return state["x"], p1, abs(r1) < 1e-7


def grade_column(m: em.Mixture, x0: np.ndarray, t0_c: float, p0_bar: float, offsets_m: np.ndarray, mode: str = "isothermal", gradient_c_km: float = 30.0, step_m: float = 5.0, tol: float = 1e-9,
                 max_iter: int = 40, saturation: bool = True, along_profile: bool = False) -> pd.DataFrame:
    """Profile at the given offsets [m, positive downwards, 0 = the reference point]: composition, pressure, temperature, density, saturation pressure, and the linear-gradient pressure for comparison.

    The offsets are visited in steps of at most `step_m` away from the reference in both directions.  Columns: dz_m, t_c, p_bar, rho_kg_m3, mw_g_mol, psat_bar, psat_kind, p_minus_psat_bar, p_linear_bar,
    delta_p_bar (EOS minus linear), converged and one x_<component> per component.

    The saturation pressure is computed at the reference point only (`along_profile = False`): for a composition close to the critical point the pressure-scan routine of `eos_mixture` can return a
    spurious root (a bubble point with the wrong branch), so a profile of saturation pressures is not reported as a result unless a caller asks for it and checks it.  A contact (the depth where the
    pressure reaches the saturation pressure) needs a stability-based saturation routine (Michelsen) that is not implemented."""
    x0 = np.asarray(x0, dtype=float) / np.sum(x0)
    t_ref = t0_c + 273.15
    lnphi0, _ = em.ln_phi(m, x0, t_ref, p0_bar, "stable")
    ln_f0 = np.log(x0) + lnphi0 + np.log(p0_bar)
    rho0 = em.density_kg_m3(m, x0, t_ref, p0_bar, "stable")
    kind, psat0 = None, float("nan")
    if saturation:
        for k in ("bubble", "dew"):
            ps = em.saturation_pressure(m, x0, t_ref, k)
            if np.isfinite(ps) and (not np.isfinite(psat0) or abs(ps - p0_bar) < abs(psat0 - p0_bar)):
                kind, psat0 = k, ps
    rows = {0.0: (x0, p0_bar, t_ref, rho0, True)}
    for sign in (1.0, -1.0):
        targets = sorted({float(o) for o in offsets_m if o * sign > 0}, key=abs)
        x, p, rho, dz_prev, ok_all = x0, p0_bar, rho0, 0.0, True
        for target in targets:
            n = max(1, int(np.ceil(abs(target - dz_prev) / step_m)))
            for i in range(1, n + 1):
                dz_total = dz_prev + (target - dz_prev) * i / n
                dz = (target - dz_prev) / n
                t_k = _t_k(t0_c, dz_total, mode, gradient_c_km)
                if not ok_all:
                    break
                x, p, ok = _solve_step(m, ln_f0, x, p, rho, dz_total, dz, t_k, tol, max_iter, hydrostatic=(mode == "geothermal"))
                ok_all = ok_all and ok
                rho = em.density_kg_m3(m, x, t_k, p, "stable") if ok else float("nan")
            rows[target] = (x, p, _t_k(t0_c, target, mode, gradient_c_km), rho, ok_all)
            dz_prev = target
            if not ok_all:
                break
    out = []
    last_psat = psat0
    for dz in sorted(rows):
        x, p, t_k, rho, ok = rows[dz]
        psat = float("nan")
        if saturation and kind is not None and ok and np.isfinite(last_psat) and (dz == 0.0 or along_profile):
            try:
                psat = em.saturation_pressure(m, x, t_k, kind, n_scan=45)                           # a full pressure scan: a narrow window near the critical point returns spurious roots
            except (ValueError, FloatingPointError):
                psat = float("nan")
            if np.isfinite(psat):
                last_psat = psat
        lin = p0_bar + rho0 * G * dz / em.BAR
        rec = {"dz_m": dz, "t_c": t_k - 273.15, "p_bar": p, "rho_kg_m3": rho, "mw_g_mol": m.mw(x) if ok else float("nan"), "psat_bar": psat, "psat_kind": kind,
               "p_minus_psat_bar": p - psat if np.isfinite(psat) else float("nan"), "p_linear_bar": lin, "delta_p_bar": p - lin, "converged": ok}
        rec.update({f"x_{c}": float(v) for c, v in zip(m.ids, x)})
        out.append(rec)
    return pd.DataFrame(out)


def contact_offset(profile: pd.DataFrame) -> float:
    """Offset [m] where the pressure reaches the saturation pressure (a gas-oil contact or the dew / bubble boundary), by linear interpolation; NaN if the profile does not cross it."""
    d = profile.dropna(subset=["p_minus_psat_bar"]).sort_values("dz_m")
    s = np.sign(d["p_minus_psat_bar"].to_numpy())
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    if len(idx) == 0:
        return float("nan")
    i = idx[0]
    a, b = d.iloc[i], d.iloc[i + 1]
    return float(a["dz_m"] + (0 - a["p_minus_psat_bar"]) * (b["dz_m"] - a["dz_m"]) / (b["p_minus_psat_bar"] - a["p_minus_psat_bar"]))


def perturb_composition(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Log-normal relative perturbation of every mole fraction but the lightest, renormalised."""
    y = np.asarray(x, dtype=float).copy()
    y[1:] = y[1:] * np.exp(rng.normal(0.0, sigma, len(y) - 1))
    return y / y.sum()


def column_envelope(cfg: dict[str, Any], fluid_key: str, height_m: float, n_points: int = 13, draws: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The range of a column of `height_m` below the apex over the ensemble of `cfg['ensemble']`: equation (PR, SRK), volume translation, temperature mode and composition draws.

    Returns (envelope per depth: p_min / p_p05 / p_median / p_p95 / p_max, the linear-gradient pressure of the base case, the range of the saturation pressure and of the density;  members: one row per
    ensemble member with its maximum deviation from the linear gradient and its contact offset)."""
    ens = cfg["ensemble"]
    f = cfg["fluids"][fluid_key]
    x0 = np.asarray(f["composition"], dtype=float)
    rng = np.random.default_rng(int(ens["seed"]))
    n_draw = int(draws if draws is not None else ens["n_composition_draws"])
    comps = [x0] + [perturb_composition(x0, float(ens["composition_sigma"]), rng) for _ in range(n_draw)]
    offsets = np.linspace(0.0, height_m, n_points)
    step = float(cfg["grid"]["step_m"])
    grad = float(cfg["temperature"]["geothermal_gradient_c_km"])
    base = em.mixture_from_config(cfg, "PR", True)
    profiles, members = [], []
    for eos in ens["eos"]:
        for shift in ens["volume_shift"]:
            m = base.with_options(eos, bool(shift))
            for mode in ens["temperature_modes"]:
                for ci, x in enumerate(comps):
                    pr = grade_column(m, x, float(f["t_c"]), float(f["p_ref_bar"]), offsets, mode, grad, step, saturation=(ci == 0))
                    p0 = float(f["p_ref_bar"])
                    psat0_member = float(pr["psat_bar"].iloc[0]) if ci == 0 and len(pr) else float("nan")
                    ok = bool(pr["converged"].all())
                    members.append({"eos": eos, "volume_shift": bool(shift), "temperature_mode": mode, "composition_draw": ci, "converged": ok, "max_abs_delta_p_bar": float(pr["delta_p_bar"].abs().max()) if ok else np.nan,
                                    "delta_p_base_bar": float(pr["delta_p_bar"].iloc[-1]) if ok else np.nan, "saturation_margin_apex_bar": float(p0 - psat0_member) if ci == 0 and np.isfinite(psat0_member) else np.nan})
                    if ok:
                        profiles.append(pr.assign(eos=eos, volume_shift=bool(shift), temperature_mode=mode, composition_draw=ci))
    allp = pd.concat(profiles, ignore_index=True)
    g = allp.groupby("dz_m")
    env = pd.DataFrame({"p_min_bar": g["p_bar"].min(), "p_p05_bar": g["p_bar"].quantile(0.05), "p_median_bar": g["p_bar"].median(), "p_p95_bar": g["p_bar"].quantile(0.95), "p_max_bar": g["p_bar"].max(),
                        "rho_min": g["rho_kg_m3"].min(), "rho_max": g["rho_kg_m3"].max(), "delta_p_min_bar": g["delta_p_bar"].min(), "delta_p_max_bar": g["delta_p_bar"].max()}).reset_index()
    base_profile = allp[(allp["eos"] == "PR") & allp["volume_shift"] & (allp["temperature_mode"] == "isothermal") & (allp["composition_draw"] == 0)].sort_values("dz_m")
    env = env.merge(base_profile[["dz_m", "p_bar", "p_linear_bar", "psat_bar"]].rename(columns={"p_bar": "p_base_bar"}), on="dz_m", how="left")
    return env, pd.DataFrame(members)
