"""Figures of the permanent-gauge shut-in pressures (notebooks 22 and 24).  A shut-in value is the LAST daily average of a run of fully shut-in days: shape and fill say how far it can be read as a shut-in pressure
(filled circle = flat at the end, open circle = still rising, a lower bound; grey cross = pressure falls during the shut-in, not a build-up).  Nothing is extrapolated.  Pressures at the gauge are measured; pressures at the datum are a
BAND from an assumed fluid gradient (derived).  Wells are told apart by colour and named in the legend, with the count of episodes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ep_physics_base.figures import theme

WELL_COLOURS = ["#0B6FA4", "#2E7D32", "#E08A00", "#6B3FA0", "#B3261E", "#4B8F8C"]


def _status(ep: pd.DataFrame) -> pd.Series:
    iv = ep["intervention_logged"].fillna("").ne("") if "intervention_logged" in ep else pd.Series(False, index=ep.index)
    return pd.Series(np.where(iv, "intervention", np.where(ep["falls_during_shutin"], "falls", np.where(ep["near_stabilised"], "flat", "rising"))), index=ep.index)


def shutin_figure(ep: pd.DataFrame, sc: dict, plt, fstate: pd.DataFrame | None = None):
    """Top: the last shut-in pressure of every episode at the gauge (measured) against time; bottom: the same pressure carried to the datum as a band (derived), with the initial pressure of the plan for development.
    Wells without a gauge depth appear in the top panel only."""
    d = ep.copy(); d["_s"] = _status(d)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8.2), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    names = list(d.groupby("wellbore_name")["start"].min().sort_values().index)
    col = {n: WELL_COLOURS[i % len(WELL_COLOURS)] for i, n in enumerate(names)}
    for n in names:
        g = d[d["wellbore_name"] == n]
        for st, marker, filled in (("flat", "o", True), ("rising", "o", False), ("falls", "x", False), ("intervention", "D", False)):
            h = g[g["_s"] == st]
            if not len(h):
                continue
            kw = dict(s=44, marker=marker, zorder=5)
            if marker == "D":
                a1.scatter(h["end"], h["last_bar"], facecolor="#DDE6EC", edgecolor="#22303A", linewidth=1.2, **{**kw, "s": 36})
            elif marker == "x":
                a1.scatter(h["end"], h["last_bar"], color="#6B7782", linewidth=1.6, **kw)
            else:
                a1.scatter(h["end"], h["last_bar"], facecolor=col[n] if filled else "white", edgecolor=col[n], linewidth=1.4, **kw)
        ok = g[~g["_s"].isin(["falls", "intervention"]) & g["at_datum_low_bar"].notna()]
        if len(ok):
            a2.vlines(ok["end"], ok["at_datum_low_bar"], ok["at_datum_high_bar"], color=col[n], lw=2.4, alpha=0.8, zorder=4)
            a2.scatter(ok["end"], (ok["at_datum_low_bar"] + ok["at_datum_high_bar"]) / 2, s=14, color=col[n], zorder=5)
    a2.axhline(float(sc["initial_pressure_bar"]), color="#22303A", lw=1.2, ls="--")
    a1.set_ylabel("Shut-in gauge pressure, last day [bar] (measured)", fontsize=9)
    a2.set_ylabel(f"Pressure at {sc['datum_tvdss_m']:.0f} m TVDSS [bar] (derived band)", fontsize=9)
    a2.set_xlabel("Date [calendar] (measured)", fontsize=9)
    for a in (a1, a2):
        a.grid(color=theme.GRIDLINE, lw=0.5)
    cnt = d.groupby("wellbore_name").size()
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", markerfacecolor=col[n], markeredgecolor=col[n], label=f"{n} (n={int(cnt[n])} shut-ins)") for n in names]
    handles += [Line2D([], [], color="#22303A", lw=1.2, ls="--", label=f"initial pressure of the plan for development, {sc['initial_pressure_bar']:.0f} bar at {sc['datum_tvdss_m']:.0f} m TVDSS (published, 2005)"),
                Line2D([], [], marker="o", ls="", markerfacecolor="#22303A", markeredgecolor="#22303A", label=f"filled: flat at the end ({int((d['_s'] == 'flat').sum())})"),
                Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="#22303A", label=f"open: still rising, a lower bound ({int((d['_s'] == 'rising').sum())})"),
                Line2D([], [], marker="x", ls="", color="#6B7782", label=f"cross: pressure falls during the shut-in, not a build-up ({int((d['_s'] == 'falls').sum())})"),
                Line2D([], [], marker="D", ls="", markerfacecolor="#DDE6EC", markeredgecolor="#22303A", label=f"diamond: a well intervention was logged (kill fluid or transient possible) ({int((d['_s'] == 'intervention').sum())})")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.5, frameon=False)
    fig.tight_layout(); fig.subplots_adjust(bottom=0.24)
    return fig, (a1, a2)


def episode_table(ep: pd.DataFrame) -> pd.DataFrame:
    """One row per gauged well: shut-ins, how many flat at the end, the range of the last pressure, and the datum band of the flat ones."""
    d = ep.copy(); d["_s"] = _status(d)
    g = d.groupby("wellbore_name").agg(shut_ins=("start", "size"), flat=("_s", lambda s: int((s == "flat").sum())), rising=("_s", lambda s: int((s == "rising").sum())), falling=("_s", lambda s: int((s == "falls").sum())), intervention=("_s", lambda s: int((s == "intervention").sum())),
                                       gauge_md_m=("gauge_md_m", "first"), gauge_tvdss_m=("gauge_tvdss_m", "first"), first_end=("end", "min"), last_end=("end", "max"),
                                       last_bar_min=("last_bar", "min"), last_bar_max=("last_bar", "max"))
    ok = d[~d["_s"].isin(["falls", "intervention"]) & d["at_datum_low_bar"].notna()]
    b = ok.groupby("wellbore_name").agg(datum_low_min=("at_datum_low_bar", "min"), datum_high_max=("at_datum_high_bar", "max"))
    return g.join(b).reset_index().round(1)
