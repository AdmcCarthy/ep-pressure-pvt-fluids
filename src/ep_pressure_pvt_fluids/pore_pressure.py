"""Pore pressure, overburden, fracture pressure and the drilling window.  Pure functions (docs/pore_pressure.md).

Definitions used (each with its source in `knowledge/`):
    OBG   overburden stress / gradient: the weight of the water column and the sediments above a depth (sigma_v);
    PP    pore pressure: the fluid pressure in the pores (hydrostatic when connected to the surface, overpressured when the fluid cannot drain);
    SHmin minimum horizontal stress: the lowest of the three principal stresses in the normal-faulting regime; the fracture closure pressure (FCP, the pressure where losses stop) equals it;
    FG    fracture gradient: the pressure to open and extend a fracture: a lower bound (the minimum stress, FCP) and an upper bound (breakdown); a leak-off test gives a value between them;
    SFG   shear failure gradient: the lowest mud pressure that keeps the borehole wall from shear failure (collapse); needs rock strength, the stresses and the well path;
    drilling window  the mud pressure range that is safe: from max(PP, SFG) up to the fracture gradient (FCP / SHmin as the conservative limit).
Nothing is invented: SFG without rock strength is NaN and the window is then reported with its lower limit as a minimum (PP only).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ep_physics_base import units as pu


_DEFAULT_OVERLAY: Any = None


def register_overlay(overlay: Any) -> None:
    """Set the function that `load_config()` applies when the caller passes none (None removes it).  A host application uses this to let a knowledge pack replace values
    everywhere in the library, including the library's own calls, while the library never imports a knowledge store."""
    global _DEFAULT_OVERLAY
    _DEFAULT_OVERLAY = overlay


def load_config(overlay: Any = None) -> dict[str, Any]:
    """config/pore_pressure.yaml (the packaged defaults, or a project file of that name).  `overlay` is an optional function that takes the configuration and returns it with values
    replaced, for example the values of a knowledge pack (default: the one set with `register_overlay`): the library never imports a knowledge store."""
    from ep_physics_base.config import load_yaml

    cfg = load_yaml("config/pore_pressure.yaml")
    overlay = overlay or _DEFAULT_OVERLAY
    return overlay(cfg) if overlay else cfg


def bulk_density_g_cm3(depth_below_seabed_m, cfg: dict[str, Any]):
    """Bulk density of the exponential compaction trend at a depth below the seabed."""
    o = cfg["overburden"]
    phi = float(o["surface_porosity"]) * np.exp(-np.asarray(depth_below_seabed_m, dtype=float) / float(o["compaction_length_m"]))
    return float(o["matrix_density_g_cm3"]) * (1.0 - phi) + float(o["fluid_density_g_cm3"]) * phi


def overburden_pressure_bar(z_msl_m, water_depth_m: float, cfg: dict[str, Any], n_step: int = 400):
    """Overburden stress [bar] at depths `z_msl_m` below sea level: seawater column to the seabed plus the integral of rho_b g over the sediments (trapezoid on a fine grid).  Above the seabed only water."""
    z = np.atleast_1d(np.asarray(z_msl_m, dtype=float))
    o = cfg["overburden"]
    wd = float(water_depth_m)
    out = np.empty_like(z)
    for i, zi in enumerate(z):
        if zi <= wd:
            out[i] = float(o["seawater_gradient_bar_per_m"]) * zi
            continue
        s = np.linspace(0.0, zi - wd, n_step)
        rho = bulk_density_g_cm3(s, cfg)
        out[i] = float(o["seawater_gradient_bar_per_m"]) * wd + float(np.trapezoid(rho * pu.BAR_PER_M_PER_SG, s))
    return out


def lithostatic_line_bar(z_msl_m, water_depth_m: float, cfg: dict[str, Any]):
    return overburden_pressure_bar(z_msl_m, water_depth_m, cfg)


def leakage_line_bar(obg_bar, cfg: dict[str, Any]):
    """Indicative upper limit of pore pressure on the Norwegian shelf: the lithostatic pressure minus about 100 bar (Sodir overpressure slides; water depth 250-400 m)."""
    return np.asarray(obg_bar, dtype=float) - float(cfg["leakage_line"]["below_lithostatic_bar"])


def shmin_lower_bound_bar(obg_bar, pp_bar, nu: float):
    """Minimum horizontal stress, the lower bound of the fracture pressure: nu / (1 - nu) (sigma_v - p) + p."""
    obg, pp = np.asarray(obg_bar, dtype=float), np.asarray(pp_bar, dtype=float)
    return nu / (1.0 - nu) * (obg - pp) + pp


def fracture_upper_bound_bar(shmin_bar, pp_bar):
    """Upper bound of the fracture pressure (breakdown, neglecting tensile strength and temperature): 2 sigma_min - p."""
    return 2.0 * np.asarray(shmin_bar, dtype=float) - np.asarray(pp_bar, dtype=float)


def fracture_band_bar(obg_bar, pp_bar, cfg: dict[str, Any]) -> pd.DataFrame:
    """Fracture pressure band for the Poisson band: lower bound (SHmin) for the lowest and highest nu, the upper bound and the mean (`most likely`) for the central nu."""
    lo_nu, mid_nu, hi_nu = (float(x) for x in cfg["poisson_ratio"]["band"])
    s_lo, s_mid, s_hi = (shmin_lower_bound_bar(obg_bar, pp_bar, n) for n in (lo_nu, mid_nu, hi_nu))
    up = fracture_upper_bound_bar(s_mid, pp_bar)
    return pd.DataFrame({"shmin_low_bar": s_lo, "shmin_mid_bar": s_mid, "shmin_high_bar": s_hi, "fg_upper_bar": up, "fg_most_likely_bar": 0.5 * (s_mid + up)})


def drilling_window(pp_sg, sfg_sg, fg_sg, margin_sg: float = 0.05) -> pd.DataFrame:
    """The safe mud-weight window [sg] between the lower limit max(pore pressure, shear failure gradient) and the fracture gradient.

    A missing shear failure gradient (NaN) leaves the lower limit at the pore pressure and says so (`lower_is_minimum`); a missing fracture gradient or pore pressure leaves the window undefined."""
    pp, sfg, fg = (np.atleast_1d(np.asarray(x, dtype=float)) for x in (pp_sg, sfg_sg, fg_sg))
    pp, sfg, fg = np.broadcast_arrays(pp, sfg, fg)
    lower = np.where(np.isnan(sfg), pp, np.fmax(pp, sfg))
    width = fg - lower
    status = np.select([np.isnan(pp) | np.isnan(fg), width <= 0, width < margin_sg], ["undefined (pore pressure or fracture gradient missing)", "closed: the fracture gradient is at or below the lower limit", "narrow"], "open")
    return pd.DataFrame({"lower_sg": lower, "upper_sg": fg, "width_sg": width, "lower_is_minimum": np.isnan(sfg) & ~np.isnan(pp), "status": status})


def overbalance(mud_weight_sg, pp_sg):
    """Mud weight minus pore pressure [sg]: positive = overbalanced (the mud holds the formation), negative = underbalanced."""
    return np.asarray(mud_weight_sg, dtype=float) - np.asarray(pp_sg, dtype=float)