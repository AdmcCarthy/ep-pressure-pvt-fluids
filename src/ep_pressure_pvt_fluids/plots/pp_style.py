"""The pore pressure line style guide (config/pore_pressure_style.yaml): one place, used by every pore pressure figure.

    line_style("fg_max")   -> {"color": "#0084C4", "ls": "--", "lw": 1.5, "label": "FG max"}
    plot_curve(ax, "obg", x, z)        draws a curve in its style; returns the line
    legend_handles(keys)               legend entries in the same styles
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from matplotlib.lines import Line2D

from ep_physics_base.config import load_yaml


@lru_cache(maxsize=1)
def _cfg() -> dict[str, Any]:
    return load_yaml("config/pore_pressure_style.yaml")


def line_style(key: str) -> dict[str, Any]:
    try:
        return dict(_cfg()["lines"][key])
    except KeyError as exc:
        raise KeyError(f"no pore pressure line style {key!r}; known: {sorted(_cfg()['lines'])}") from exc


def plot_curve(ax, key: str, x, z, label: str | bool = True, **kw):
    """Draw one quantity against depth in its style (x = the quantity, z = depth)."""
    s = line_style(key)
    lab = s["label"] if label is True else (label or None)
    return ax.plot(x, z, color=s["color"], ls=s["ls"], lw=s["lw"], label=lab, **kw)[0]


def legend_handles(keys: list[str]) -> list[Line2D]:
    return [Line2D([0], [0], color=line_style(k)["color"], ls=line_style(k)["ls"], lw=line_style(k)["lw"], label=line_style(k)["label"]) for k in keys]


def fracture_curves(obg_bar, pp_bar, nu_band) -> dict[str, Any]:
    """Pressure [bar] of the Sh and FG family for the Poisson band (low, central, high), from the Zhang bounds (src/processing/pore_pressure.py)."""
    import numpy as np

    from ep_pressure_pvt_fluids import pore_pressure as pp

    obg, p = np.asarray(obg_bar, dtype=float), np.asarray(pp_bar, dtype=float)
    lo, mid, hi = (float(x) for x in nu_band)
    sh = {n: pp.shmin_lower_bound_bar(obg, p, n) for n in (lo, mid, hi)}
    fgm = {n: 0.5 * (sh[n] + pp.fracture_upper_bound_bar(sh[n], p)) for n in sh}
    return {"sh": sh[mid], "sh_min": sh[lo], "fg": fgm[mid], "fg_min": fgm[lo], "fg_max": fgm[hi]}