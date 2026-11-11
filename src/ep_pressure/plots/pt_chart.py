"""The pressure-temperature chart of the drill stem tests with the HPHT classes (normal, high temperature, high pressure, HPHT, ultra-HPHT, extreme HPHT).

It needs the pressure library (the class limits and their polygons) and the PVT colours and test markers of ep-fluids, so it lives here and not in ep-fluids.
A hollow marker is a test whose temperature is estimated; the HPHT class itself uses gauge temperatures only (docs/temperature.md in the engine)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ep_fluids.plots import pvt_colours as pc
from ep_fluids.plots.pvt_plots import estimated_mask, scatter_tests, types_of
from ep_physics_base.figures import theme
from ep_pressure import pressure as pr


def plot_pt_chart(ax: Any, pv: pd.DataFrame, pcfg: dict[str, Any], t_max: float = 260.0, p_max: float = 1500.0, legend: bool = False) -> None:
    """Pressure against temperature with the flat-colour zones normal pressure / high temperature / high pressure / HPHT / UHPHT / extreme HPHT; tests coloured by saturation
    state, hollow where the temperature is estimated (the HPHT class itself uses gauge temperatures only)."""
    from matplotlib.patches import Polygon

    zones = pr.hpht_zone_polygons(pcfg, t_max, p_max)
    for name, (x, y) in zones.items():
        ax.add_patch(Polygon(np.c_[x, y], closed=True, facecolor=pc.PT_COLOURS[name], edgecolor="none", zorder=0))
        ax.text(np.mean(x[:-1]) if name in ("normal", "high temperature", "high pressure") else {"HPHT": 165, "ultra-HPHT": 190, "extreme-HPHT": 232}[name],
                {"normal": 340, "high temperature": 340, "high pressure": 1100, "HPHT": 860, "ultra-HPHT": 1210, "extreme-HPHT": 1440}[name], pc.PT_LABELS[name], ha="center", fontsize=8.5, color=theme.TEXT_PRIMARY, zorder=1)
    t = pv[pv["temperature_used_c"].notna() & pv["reservoir_pressure_bar"].notna()]
    for state, colour in pc.STATE_COLOURS.items():
        s = t[t["saturation_state"] == state]
        if len(s):
            scatter_tests(ax, s["temperature_used_c"].astype(float), s["reservoir_pressure_bar"].astype(float), colour, estimated_mask(s), size=16, label=f"{state} ({len(s)})", types=types_of(s))
    n_est = int(estimated_mask(t).sum())
    ax.set_xlim(0, t_max)
    ax.set_ylim(0, p_max)
    ax.set_xlabel(f"Test temperature [degC] (n={len(t)}; {n_est} hollow: estimated)")
    ax.set_ylabel("Reservoir pressure [bar]")
    if legend:
        ax.legend(fontsize=8.5, loc="lower right", frameon=False)
