"""Peng-Robinson equation of state of a pure component: saturation pressure, volumes, regime of a (T, P) point and the pressure-volume-temperature surface.

The physical boundaries of the 3D PVT chart (`scripts/build_pvt3d_dashboard.py`, docs/pvt3d.md) are computed here, not drawn:
    * pressure P(V, T) from the cubic equation of state: Z^3 - (1 - B) Z^2 + (A - 3 B^2 - 2 B) Z - (A B - B^2 - B^3) = 0, A = a P / (R T)^2, B = b P / (R T),
      a(T) = 0.45724 R^2 Tc^2 / Pc * [1 + kappa (1 - sqrt(T / Tc))]^2, b = 0.07780 R Tc / Pc, kappa = 0.37464 + 1.54226 w - 0.26992 w^2
      (above w = 0.491 the Robinson-Peng polynomial 0.379642 + 1.48503 w - 0.164423 w^2 + 0.016666 w^3, as in common implementations);
    * below the critical temperature an isotherm of the equation has a van der Waals loop: it is replaced by the horizontal tie line at the SATURATION pressure,
      where liquid and vapour have equal fugacity (equivalent to Maxwell's equal-area rule); its ends are the saturated liquid and vapour volumes, whose curves
      form the two-phase dome (the binodal);
    * the dome closes at the critical point (Tc, Pc, Vc = Zc R Tc / Pc, Zc = 0.3074 for this equation): there the first and second pressure derivatives with
      respect to volume vanish; above Tc there is one phase (supercritical) on every isotherm.
Pure functions, no side effects.  A pure component is NOT a reservoir mixture: the phase envelope of a mixture is a loop in the P-T plane (cricondentherm,
cricondenbar) and needs the composition, which Sodir does not give.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.optimize import brentq

R = 8.314462618                      # J / (mol K)
SQRT2 = float(np.sqrt(2.0))
BAR = 1.0e5                          # Pa per bar
CRITICAL_Z = 0.3074                  # compressibility factor at the critical point of the Peng-Robinson equation


@dataclass(frozen=True)
class Fluid:
    id: str
    name: str
    tc_k: float
    pc_bar: float
    omega: float

    @property
    def vc_l_mol(self) -> float:
        return CRITICAL_Z * R * self.tc_k / (self.pc_bar * BAR) * 1000.0


def fluids_from_config(cfg: dict[str, Any]) -> dict[str, Fluid]:
    return {f["id"]: Fluid(f["id"], f["name"], float(f["tc_k"]), float(f["pc_bar"]), float(f["omega"])) for f in cfg["fluids"]}


def _kappa(omega: float) -> float:
    if omega > 0.491:
        return 0.379642 + 1.48503 * omega - 0.164423 * omega ** 2 + 0.016666 * omega ** 3
    return 0.37464 + 1.54226 * omega - 0.26992 * omega ** 2


def ab(t_k: float, p_bar: float, fluid: Fluid) -> tuple[float, float]:
    """Dimensionless A = a P / (R T)^2 and B = b P / (R T) of the equation at T [K] and P [bar]."""
    pc = fluid.pc_bar * BAR
    a = 0.45724 * R ** 2 * fluid.tc_k ** 2 / pc * (1.0 + _kappa(fluid.omega) * (1.0 - np.sqrt(t_k / fluid.tc_k))) ** 2
    b = 0.07780 * R * fluid.tc_k / pc
    p = p_bar * BAR
    return a * p / (R * t_k) ** 2, b * p / (R * t_k)


def z_roots(a_: float, b_: float) -> np.ndarray:
    """Real compressibility factors above B (physical roots), ascending: one (single phase) or three (liquid, unstable, vapour)."""
    roots = np.roots([1.0, -(1.0 - b_), a_ - 3.0 * b_ ** 2 - 2.0 * b_, -(a_ * b_ - b_ ** 2 - b_ ** 3)])
    real = np.sort(roots[np.abs(roots.imag) < 1e-9 * np.maximum(1.0, np.abs(roots.real))].real)
    return real[real > b_]


def ln_phi(z: float, a_: float, b_: float) -> float:
    """Natural log of the fugacity coefficient of a pure component."""
    return z - 1.0 - np.log(z - b_) - a_ / (2.0 * SQRT2 * b_) * np.log((z + (1.0 + SQRT2) * b_) / (z + (1.0 - SQRT2) * b_))


def _fugacity_gap(t_k: float, p_bar: float, fluid: Fluid) -> float:
    """ln phi(liquid) - ln phi(vapour) where the equation has a liquid and a vapour root (NaN elsewhere): zero at the saturation pressure."""
    a_, b_ = ab(t_k, p_bar, fluid)
    z = z_roots(a_, b_)
    if len(z) < 2 or z[-1] <= z[0] * (1.0 + 1e-12):
        return float("nan")
    return float(ln_phi(z[0], a_, b_) - ln_phi(z[-1], a_, b_))


def _three_root_window(t_k: float, fluid: Fluid, n_scan: int = 3000) -> tuple[float, float] | None:
    """Pressure range [bar] in which the cubic has three real roots (a liquid, an unstable and a vapour root): where its discriminant is positive.  The scan is
    geometric over nine decades plus two linear ones close to the critical pressure, where the window narrows to nothing at the critical temperature."""
    p = fluid.pc_bar * np.unique(np.r_[np.geomspace(1e-9, 1.0, n_scan), np.linspace(0.8, 1.0, n_scan), np.linspace(0.995, 1.0, 20000)])
    a_, b_ = ab(t_k, p, fluid)
    c2, c1, c0 = -(1.0 - b_), a_ - 3.0 * b_ ** 2 - 2.0 * b_, -(a_ * b_ - b_ ** 2 - b_ ** 3)
    disc = 18.0 * c2 * c1 * c0 - 4.0 * c2 ** 3 * c0 + c2 ** 2 * c1 ** 2 - 4.0 * c1 ** 3 - 27.0 * c0 ** 2
    inside = np.flatnonzero(disc > 0.0)
    return (float(p[inside[0]]), float(p[inside[-1]])) if len(inside) else None


@lru_cache(maxsize=8192)
def saturation_pressure(t_k: float, fluid: Fluid) -> float:
    """Saturation (vapour) pressure [bar] below the critical temperature by equal fugacity of liquid and vapour; the critical pressure at the critical
    temperature; NaN above it.  The fugacity gap changes sign once inside the pressure window with three roots: refined by bracketing."""
    if t_k > fluid.tc_k:
        return float("nan")
    if t_k >= fluid.tc_k * (1.0 - 1e-9):
        return fluid.pc_bar
    window = _three_root_window(t_k, fluid)
    if window is None:
        return float("nan")
    lo, hi = window[0] * (1.0 + 1e-6), window[1] * (1.0 - 1e-6)
    f_lo, f_hi = _fugacity_gap(t_k, lo, fluid), _fugacity_gap(t_k, hi, fluid)
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0.0:
        return float("nan")
    return float(brentq(lambda pp: _fugacity_gap(t_k, pp, fluid), lo, hi, xtol=1e-12, rtol=1e-10))


def volume_l_mol(t_k: float, p_bar: float, fluid: Fluid, phase: str) -> float:
    """Molar volume [L/mol]: `liquid` the smallest root, `vapour` the largest (one root: that one)."""
    a_, b_ = ab(t_k, p_bar, fluid)
    z = z_roots(a_, b_)
    zz = z[0] if phase == "liquid" else z[-1]
    return float(zz * R * t_k / (p_bar * BAR) * 1000.0)


def state(t_c: float, p_bar: float, fluid: Fluid, tolerance: float = 0.002) -> tuple[float, str]:
    """(log10 of the molar volume [L/mol], regime) of the fluid at T [degC] and P [bar]: `liquid`, `vapour`, `two-phase` (on the saturation line) or `supercritical`."""
    t_k = t_c + 273.15
    if not (np.isfinite(t_c) and np.isfinite(p_bar)) or p_bar <= 0:
        return float("nan"), ""
    if t_k >= fluid.tc_k:
        return float(np.log10(volume_l_mol(t_k, p_bar, fluid, "vapour"))), "supercritical"
    ps = saturation_pressure(t_k, fluid)
    if not np.isfinite(ps):
        return float("nan"), ""
    if abs(p_bar / ps - 1.0) <= tolerance:
        return float(np.log10(np.sqrt(volume_l_mol(t_k, ps, fluid, "liquid") * volume_l_mol(t_k, ps, fluid, "vapour")))), "two-phase"
    phase = "liquid" if p_bar > ps else "vapour"
    return float(np.log10(volume_l_mol(t_k, p_bar, fluid, phase))), phase


def temperature_range_k(fluid: Fluid, grid: dict[str, Any]) -> tuple[float, float]:
    return min(0.6 * fluid.tc_k, float(grid["t_low_k"])), min(max(1.3 * fluid.tc_k, float(grid["t_high_k"])), float(grid.get("t_cap_k", 1.0e9)))


def pvt_surface(fluid: Fluid, grid: dict[str, Any]) -> dict[str, Any]:
    """The surface as one regular mesh: row i is the isotherm at T_i, the columns follow the isotherm from high pressure (liquid) through the two-phase plateau to low
    pressure (vapour).  Returns lists: `t_c` (row temperatures, degC), `x` / `y` / `z` (T [degC], log10 V [L/mol], P [bar], 2D), `region` (0 liquid, 1 two-phase, 2 vapour,
    3 supercritical, 2D), plus the dome curves (`dome_liquid`, `dome_vapour` as (T, log V, P) lists), the critical point and the critical isotherm."""
    n_t, n_l, n_p, n_v = (int(grid[k]) for k in ("n_temperature", "n_liquid", "n_plateau", "n_vapour"))
    p_max, floor = float(grid["p_max_bar"]), float(grid["vapour_floor"])
    t_lo, t_hi = temperature_range_k(fluid, grid)
    t_all = np.unique(np.r_[np.linspace(t_lo, t_hi, n_t - 1), fluid.tc_k])                 # the critical isotherm is one of the rows
    xs, ys, zs, reg = [], [], [], []
    for t_k in t_all:
        if t_k < fluid.tc_k:
            ps = saturation_pressure(t_k, fluid)
            p_liq = np.geomspace(p_max, ps, n_l)
            p_vap = np.geomspace(ps, ps * floor, n_v)
            v_l, v_v = volume_l_mol(t_k, ps, fluid, "liquid"), volume_l_mol(t_k, ps, fluid, "vapour")
            v_plateau = np.geomspace(v_l, v_v, n_p)
            y = np.r_[[np.log10(volume_l_mol(t_k, p, fluid, "liquid")) for p in p_liq], np.log10(v_plateau), [np.log10(volume_l_mol(t_k, p, fluid, "vapour")) for p in p_vap]]
            p = np.r_[p_liq, np.full(n_p, ps), p_vap]
            r = np.r_[np.zeros(n_l), np.ones(n_p), np.full(n_v, 2.0)]
        else:
            n = n_l + n_p + n_v
            p = np.geomspace(p_max, fluid.pc_bar * floor, n)
            y = np.array([np.log10(volume_l_mol(t_k, pp, fluid, "vapour")) for pp in p])
            r = np.full(n, 3.0)
        xs.append(np.full(len(p), t_k - 273.15))
        ys.append(y)
        zs.append(p)
        reg.append(r)
    below = t_all[t_all < fluid.tc_k]
    dome_t = np.r_[np.linspace(below.min(), fluid.tc_k * (1.0 - 2e-4), 60)]
    ps_line = np.array([saturation_pressure(t, fluid) for t in dome_t])
    liquid = [np.log10(volume_l_mol(t, p, fluid, "liquid")) for t, p in zip(dome_t, ps_line)]
    vapour = [np.log10(volume_l_mol(t, p, fluid, "vapour")) for t, p in zip(dome_t, ps_line)]
    crit = {"t_c": fluid.tc_k - 273.15, "log_v": float(np.log10(fluid.vc_l_mol)), "p_bar": fluid.pc_bar}
    idx = int(np.argmin(np.abs(t_all - fluid.tc_k)))
    rounder = lambda arr, d: np.round(np.asarray(arr, float), d).tolist()
    return {"t_c": rounder(t_all - 273.15, 3), "x": [rounder(r_, 3) for r_ in xs], "y": [rounder(r_, 4) for r_ in ys], "z": [rounder(r_, 3) for r_ in zs], "region": [rounder(r_, 0) for r_ in reg],
            "dome_liquid": {"x": rounder(dome_t - 273.15, 3), "y": rounder(liquid, 4), "z": rounder(ps_line, 3)}, "dome_vapour": {"x": rounder(dome_t - 273.15, 3), "y": rounder(vapour, 4), "z": rounder(ps_line, 3)},
            "critical": crit, "critical_isotherm": {"x": rounder(xs[idx], 3), "y": rounder(ys[idx], 4), "z": rounder(zs[idx], 3)}}
