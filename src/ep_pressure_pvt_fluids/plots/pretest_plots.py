"""Plots of formation pretests on the pressure-depth figures: one marker per test, its shape and fill say how far the draw-down can be trusted (mobility category), its ring says a possible
supercharge, the reference lines are the hydrostatic gradient and, where the report states it, the mud density at the test.  Categories, limits and colours come from `config/pretest_categories.yaml`
(assumptions until a source states them).  Every test is a SINGLE-TEST estimate: nothing is averaged or joined into a curve.  Depth is TVD below sea level (stated or from the survey); a test without TVD is not drawn.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ep_physics_base import units as pu
from ep_physics_base.figures import theme
from ep_physics_base.config import load_yaml

EDGE = theme.NORTH_SEA


def load_categories() -> dict[str, Any]:
    return load_yaml("config/pretest_categories.yaml")


ENGINEER_TEXT = {"excluded": "excluded (lost seal, tight or dry)", "positive": "engineer: good test", "negative": "engineer: problem reported", "none_stated": "no engineer's remark"}


def engineer_key(row_pol: str, excluded: bool) -> str:
    """Which colour class a test takes: excluded, the engineer's positive or negative remark, or none stated (a mixed remark counts as a problem reported)."""
    return "excluded" if excluded else {"positive": "positive", "negative": "negative", "mixed": "negative"}.get(row_pol, "none_stated")


def _marker(category_id: str, cc: dict[str, Any]) -> tuple[str, str]:
    allcats = list(cc["mobility_categories"]) + [cc["no_mobility"]]
    c = next((x for x in allcats if x["id"] == category_id), cc["no_mobility"])
    return c["marker"], c["label"]


def quantity(pre: pd.DataFrame, what: str, rt_m: float, gw: float) -> np.ndarray:
    """The x value of each test: `bar`, `sg` (equivalent density, rotary-table basis, depth = TVDSS + RT) or `overpressure` (bar over the water gradient)."""
    p, z = pre["pressure_bar"].to_numpy(float), pre["tvdss_used_m"].to_numpy(float)
    if what == "bar":
        return p
    if what == "sg":
        return np.asarray(pu.emw_sg(p, z, rt_m, "RT"), dtype=float)
    return np.asarray(pu.overpressure_bar(p, z, gw), dtype=float)


def draw_pretests(ax, pre: pd.DataFrame, what: str, rt_m: float, gw: float, cc: dict[str, Any], show_numbers: bool = False, count_frame: pd.DataFrame | None = None) -> list[str]:
    """Draw the tests of one wellbore on `ax`: SHAPE = the mobility category (how demanding the measurement is), COLOUR = what the engineer's remark says (good test, problem reported, no remark) and a grey cross for a hard
    exclusion.  Nothing is ranked: a good test at low mobility is a green square.  The legend entry of each combination carries its count, so the symbols are traceable.  A red ring marks a possible supercharge or a
    remark that says the pressure did not stabilise.  Returns the legend labels drawn."""
    d = pre[pre["tvdss_used_m"].notna() & pre["pressure_bar"].notna()].copy()
    d["_ek"] = [engineer_key(p, bool(h)) for p, h in zip(d["remark_polarity"], d["hard_exclusion"].ne(""))]
    drawn: list[str] = []
    cf = (count_frame if count_frame is not None else pre)                                  # counts in the legend: of the whole set when several wellbores are drawn on one axes
    cf = cf[cf["tvdss_used_m"].notna() & cf["pressure_bar"].notna()]
    counts = pd.Series([f"{c}|{engineer_key(p, bool(h))}" for c, p, h in zip(cf["mobility_category"], cf["remark_polarity"], cf["hard_exclusion"].ne(""))]).value_counts()
    for (cid, ek), g in d.groupby(["mobility_category", "_ek"]):
        marker, mlabel = _marker(cid, cc)
        colour = cc["colours"][ek]
        label = f"{mlabel}; {ENGINEER_TEXT[ek]} (n={int(counts.get(f'{cid}|{ek}', len(g)))})"
        x, z = quantity(g, what, rt_m, gw), g["tvdss_used_m"].to_numpy(float)
        if ek == "excluded":
            ax.scatter(x, z, s=50, marker="x", color=colour, linewidth=1.6, zorder=6, label=label)
        else:
            ax.scatter(x, z, s=46, marker=marker, facecolor=colour, edgecolor="#22303A", linewidth=0.7, alpha=0.9, zorder=6, label=label)
        drawn.append(label)
    ring = d["possible_supercharge"].fillna(False).astype(bool) | d["remark_flags"].fillna("").str.contains("supercharge|not_stabilised")
    sc = d[ring]
    if len(sc):
        ax.scatter(quantity(sc, what, rt_m, gw), sc["tvdss_used_m"], s=170, facecolor="none", edgecolor="#B3261E", linewidth=1.4, zorder=5, label=f"possible supercharge or unstabilised: trend screen or the engineer's remark (n={len(sc)})")
        drawn.append("ring")
    return drawn


def pretest_figure(pre: pd.DataFrame, well_name: str, rt_m: float, gw: float, cc: dict[str, Any], plt):
    """Three panels (bar, equivalent density on the rotary-table basis, overpressure) of the tests of one wellbore with the hydrostatic reference; the mud density stated by the report is drawn as a line when it exists."""
    d = pre[pre["tvdss_used_m"].notna()]
    zmin, zmax = float(d["tvdss_used_m"].min()) - 60, float(d["tvdss_used_m"].max()) + 60
    zz = np.linspace(zmin, zmax, 30)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, what, xl in zip(axes, ("bar", "sg", "overpressure"), ("Pressure [bar] (measured, single test)", "Equivalent density, RT basis [sg] (derived)", "Overpressure over the water gradient [bar] (derived)")):
        hyd = {"bar": gw * zz, "sg": np.asarray(pu.emw_sg(gw * zz, zz, rt_m, "RT"), dtype=float), "overpressure": np.zeros_like(zz)}[what]
        ax.plot(hyd, zz, color="#7EC8F0", lw=2.0, label="hydrostatic (seawater gradient, assumed)")
        draw_pretests(ax, d, what, rt_m, gw, cc, show_numbers=False)
        core = d                                                                             # no measured point is dropped from a figure (decision of 2026-11-08), also those below the water gradient
        if len(core):
            x = np.append(quantity(core, what, rt_m, gw), hyd[[0, -1]])
            pad = max(0.06 * float(x.max() - x.min()), 0.5 if what != "sg" else 0.02)
            ax.set_xlim(float(x.min()) - pad, float(x.max()) + pad)
        ax.set_xlabel(xl, fontsize=9); ax.grid(color=theme.GRIDLINE, lw=0.5)
    off = int(d["below_water_gradient"].fillna(False).astype(bool).sum())
    axes[0].set_ylim(zmax, zmin); axes[0].set_ylabel("TVD below sea level [m MSL] (derived)", fontsize=9)
    fig.suptitle(f"{well_name}: {len(d)} formation pretests, each one test, all drawn" + (f" ({off} below the water gradient: not a formation pressure, kept)" if off else ""), fontsize=10, color=EDGE, x=0.01, y=0.99, ha="left")
    return fig, axes


def combination_table(pre: pd.DataFrame) -> pd.DataFrame:
    """The caption of the figures: every combination of mobility category and engineer's remark present, how many tests, the provisional description of the combination and how many need a person's judgement."""
    g = pre.groupby("combination_id").agg(tests=("test_no", "size"), description=("combination_note", "first"), excluded=("attention", lambda s: int((s == "excluded").sum())), needs_judgement=("attention", lambda s: int((s == "needs judgement").sum())), no_flag=("attention", lambda s: int((s == "no flag").sum())))
    return g.reset_index().sort_values("tests", ascending=False)


def category_table(pre: pd.DataFrame) -> pd.DataFrame:
    """Tests per well and mobility category, with the number below 1 mD/cP and the numbers by attention."""
    g = pre.groupby(["well_folder", "mobility_category"]).size().unstack(fill_value=0)
    g["tests"] = g.sum(axis=1)
    extra = pre.groupby("well_folder").agg(below_1_md=("low_mobility", "sum"), excluded=("attention", lambda s: int((s == "excluded").sum())), needs_judgement=("attention", lambda s: int((s == "needs judgement").sum())), no_flag=("attention", lambda s: int((s == "no flag").sum())))
    return g.join(extra).reset_index()


def dedupe_labels(ax) -> None:
    """Keep one legend entry per label when several wellbores were drawn on the same axes."""
    seen: set[str] = set()
    for h in ax.collections + ax.lines:
        lab = h.get_label()
        if lab in seen and not lab.startswith("_"):
            h.set_label("_nolegend_")
        seen.add(lab)
