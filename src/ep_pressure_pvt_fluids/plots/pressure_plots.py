"""Pressure-depth figures with contacts (docs/pressure.md section 10; knowledge `sw05-gradients-and-contacts`).  One axis per well.

Depth is TVD below sea level [m MSL] (down), pressure [bar].  Drawn per well:
    * the hydrostatic line (1.025 sg) with the fresh-to-brine band of the water gradient shaded;
    * every DST reservoir pressure, filled with the Geostandard colour of its fluid (oil green, gas red), the test number beside it;
    * the contacts stated in the well history (OWC, GOC, GWC, HWC as solid lines, ODT / GDT as dotted lines: the contact lies at or below them);
    * for a test with a contact below it: the hydrocarbon-gradient segment down to the contact and the AQUIFER-EQUIVALENT point (diamond) at the contact depth, with the water line through it:
      the hydrocarbon line and the water line cross at the contact, the free-water-level construction;
    * the leak-off / integrity tests of the casing shoes (equivalent mud density x depth) as triangles, where the shoe depth is a TVD or a near-vertical MD.
A depth that is an MD proxy for a deviated well is not drawn as TVD (the test is left out and counted in the title note).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ep_pressure_pvt_fluids import pressure as pr
from ep_physics_base import units as pu
from ep_pressure_pvt_fluids.plots import fluid_class_colours as fcc
from ep_pressure_pvt_fluids.plots import shoe_tests as st
from ep_physics_base.figures import theme
from ep_physics_base.figures.hc_colours import hc_colour, hc_line_colour

CONTACT_STYLE = {"OWC": ("-", "#009900"), "HWC": ("-", "#009900"), "GWC": ("-", "#B00000"), "GOC": ("-", "#993300"), "ODT": (":", "#009900"), "GDT": (":", "#B00000"), "WUT": (":", theme.BLUE)}


def lot_points(well_id: int, lot: pd.DataFrame, header: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Casing-shoe tests of a well as (tvdss_m, pressure_bar, test_type): equivalent mud density x depth below RT, for a near-vertical well only (never a fabricated TVD)."""
    g = lot[(lot["wellbore_id"] == well_id) & lot["lot_emw_sg"].notna() & lot["casing_depth_md_m"].notna()]
    ref = pr.well_depth_reference(header[header["wellbore_id"] == well_id], cfg)
    if g.empty or ref.empty or ref.iloc[0]["base_depth_basis"] != "md_near_vertical":
        return pd.DataFrame(columns=["tvdss_m", "pressure_bar", "test_type"])
    rt = float(ref.iloc[0]["rt_elev_m"])
    sg = float(cfg["hydrostatic"]["bar_per_m_per_sg"])
    d = pd.to_numeric(g["casing_depth_md_m"], errors="coerce")
    return pd.DataFrame({"tvdss_m": d - rt, "pressure_bar": pd.to_numeric(g["lot_emw_sg"], errors="coerce") * sg * d, "test_type": st.norm_type(g["test_type"]).to_numpy()})


WATER_BOX, RT_BOX = "#DCEEFB", "#FBEFCB"                                  # pastel boxes: water column (MSL to seabed) and the air gap (RT above MSL)


def casing_points(well_id: int, lot: pd.DataFrame, header: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Casing shoes of a well as (tvdss_m, casing_type, size_in): measured depth below RT converted to TVD MSL for a near-vertical well only (never a fabricated TVD)."""
    out = pd.DataFrame(columns=["tvdss_m", "casing_type", "size_in"])
    if lot is None or not len(lot):
        return out
    g = lot[(lot["wellbore_id"] == well_id) & lot["casing_depth_md_m"].notna()]
    ref = pr.well_depth_reference(header[header["wellbore_id"] == well_id], cfg)
    if g.empty or ref.empty or ref.iloc[0]["base_depth_basis"] != "md_near_vertical":
        return out
    rt = float(ref.iloc[0]["rt_elev_m"])
    return pd.DataFrame({"tvdss_m": pd.to_numeric(g["casing_depth_md_m"], errors="coerce").to_numpy() - rt, "casing_type": (g["casing_type"].astype("string") if "casing_type" in g else pd.Series(pd.NA, index=g.index, dtype="string")).to_numpy(),
                         "size_in": (pd.to_numeric(g["casing_diameter_mm"], errors="coerce") if "casing_diameter_mm" in g else pd.Series(np.nan, index=g.index)).to_numpy() / 25.4}).dropna(subset=["tvdss_m"]).sort_values("tvdss_m").reset_index(drop=True)


def inch_text(size_in: float) -> str:
    """Nominal pipe size as the industry writes it: 13 3/8 in, 9 5/8 in (nearest eighth); 'casing' when the size is missing."""
    if size_in is None or pd.isna(size_in):
        return "casing"
    eighths = int(round(float(size_in) * 8))
    whole, frac = divmod(eighths, 8)
    return f"{whole} in" if frac == 0 else f"{whole} {frac}/8 in".replace("2/8", "1/4").replace("4/8", "1/2").replace("6/8", "3/4")


def draw_casing_points(ax, cp: pd.DataFrame, label: bool = True, fontsize: float = 8.5) -> int:
    """A short tick at the left edge for every casing shoe, labelled with the pipe size (inches) where the labels do not collide; returns the number drawn."""
    if cp is None or cp.empty:
        return 0
    y0, y1 = sorted(ax.get_ylim())
    vis = cp[(cp["tvdss_m"] >= y0) & (cp["tvdss_m"] <= y1)]
    last = -1e9
    for r in vis.itertuples():
        ax.plot([0, 0.035], [r.tvdss_m, r.tvdss_m], transform=ax.get_yaxis_transform(), color="#243746", lw=2.0, solid_capstyle="butt", zorder=8, clip_on=False)
        if label and abs(r.tvdss_m - last) > 0.035 * (y1 - y0):
            ax.text(0.04, r.tvdss_m, f"{inch_text(r.size_in)} shoe", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=fontsize, color="#243746", bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.8), zorder=8)
            last = r.tvdss_m
    return int(len(vis))


def draw_datum_boxes(ax, water_depth_m: float | None, rt_m: float | None) -> None:
    """Pastel boxes behind the data: the water column (MSL = 0 down to the seabed) and the air gap between MSL and the rotary table (RT above MSL).  Drawn only where visible."""
    y0, y1 = sorted(ax.get_ylim())
    span = y1 - y0
    for lo, hi, col, text in ((0.0, water_depth_m, WATER_BOX, "water column: MSL to seabed, {:.0f} m"), (-rt_m if rt_m else None, 0.0, RT_BOX, "MSL to rotary table (RT), {:.0f} m")):
        if lo is None or hi is None or hi <= lo:
            continue
        a, b = max(lo, y0), min(hi, y1)
        if b <= a:
            continue
        ax.axhspan(a, b, color=col, alpha=0.9, lw=0, zorder=0)
        if text.startswith("water"):
            if (b - a) > 0.03 * span:
                ax.text(0.995, (a + b) / 2, text.format(hi - lo), transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8.5, color="#243746")
        else:
            ax.text(0.995, a, text.format(hi - lo), transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8.5, color="#243746")


def plot_well(ax, well_id: int, pressure: pd.DataFrame, contacts: pd.DataFrame, lot: pd.DataFrame, header: pd.DataFrame, cfg: dict[str, Any], z_pad: float = 150.0) -> dict[str, Any]:
    """Draw one well on `ax`; returns what was drawn (counts) for the key finding."""
    t = pressure[(pressure["wellbore_id"] == well_id) & pressure["reservoir_pressure_bar"].notna() & pressure["tvdss_m"].notna()]
    t = t[~t["depth_basis"].isin(["md_deviated", "md_no_tvd"])]
    c = contacts[contacts["wellbore_id"] == well_id] if contacts is not None and len(contacts) else pd.DataFrame(columns=["label", "tvdss_m"])
    gw, gw_lo, gw_hi, _ = pr.water_gradient_band(cfg)
    zs = list(t["tvdss_m"]) + list(c["tvdss_m"]) + list(t["contact_tvdss_m"].dropna())
    zmin, zmax = (min(zs) - z_pad, max(zs) + z_pad) if zs else (1000.0, 3000.0)
    zz = np.array([zmin, zmax])
    ax.fill_betweenx(zz, gw_lo * zz, gw_hi * zz, color="#7EC8F0", alpha=0.18, lw=0, label="hydrostatic band (fresh water to brine)")
    ax.plot(gw * zz, zz, color="#7EC8F0", lw=2.0, label=f"hydrostatic ({gw / 0.0980665:.3f} sg)")
    ax.plot(gw_hi * zz, zz, color="#7EC8F0", lw=1.5, ls="--", label="hydrostatic, strong brine"); ax.plot(gw_lo * zz, zz, color="#7EC8F0", lw=1.6, ls=":", label="hydrostatic, fresh water")
    for r in c.itertuples():
        ls, col = CONTACT_STYLE.get(r.label, ("-", "#555555"))
        ax.axhline(r.tvdss_m, color=col, ls=ls, lw=1.0, alpha=0.9)
        ax.text(0.995, r.tvdss_m, f"{r.label} {r.tvdss_m:.0f} m", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8.5, color=col)
    drawn = {"tests": int(len(t)), "corrected": 0, "contacts": int(len(c)), "lot": 0}
    for r in t.itertuples():
        fluid = "gas" if str(r.fluid_hint).startswith(("gas", "condensate")) else "oil" if str(r.fluid_hint).startswith("oil") else None
        face = hc_colour(fluid) if fluid else "#B8BEC4"
        ax.scatter([r.reservoir_pressure_bar], [r.tvdss_m], s=46, color=face, edgecolor=theme.NORTH_SEA, linewidth=0.8, zorder=5)
        ax.annotate(f"{r.test_number}", (r.reservoir_pressure_bar, r.tvdss_m), xytext=(5, -3), textcoords="offset points", fontsize=8.5, color=theme.NORTH_SEA)
        if pd.notna(r.aquifer_pressure_bar) and str(r.buoyancy_status).startswith("corrected"):
            drawn["corrected"] += 1
            line = hc_line_colour(r.buoyancy_fluid) if r.buoyancy_fluid in ("oil", "gas") else theme.NORTH_SEA
            ax.plot([r.reservoir_pressure_bar, r.aquifer_pressure_bar], [r.tvdss_m, r.contact_tvdss_m], color=line, lw=1.6, zorder=4)
            ax.scatter([r.aquifer_pressure_bar], [r.contact_tvdss_m], marker="D", s=34, color="white", edgecolor=line, linewidth=1.2, zorder=6)
            zw = np.array([zmin, zmax])
            ax.plot(r.aquifer_pressure_bar + gw * (zw - r.contact_tvdss_m), zw, color="#7EC8F0", lw=0.9, ls="--", alpha=0.9, zorder=2)
    lp = lot_points(well_id, lot, header, cfg) if lot is not None and len(lot) else pd.DataFrame()
    if len(lp):
        lp = lp[(lp["tvdss_m"] >= zmin) & (lp["tvdss_m"] <= zmax)]
        drawn["lot"] = int(len(lp))
        st.scatter_tests(ax, lp["pressure_bar"], lp["tvdss_m"], lp["test_type"], s=34, label=False)
    ax.set_ylim(zmax, zmin)
    ref_ = pr.well_depth_reference(header[header["wellbore_id"] == well_id], cfg)
    draw_datum_boxes(ax, float(ref_.iloc[0]["water_depth_m"]) if len(ref_) and pd.notna(ref_.iloc[0]["water_depth_m"]) else None, float(ref_.iloc[0]["rt_elev_m"]) if len(ref_) else None)
    drawn["casing_shoes"] = draw_casing_points(ax, casing_points(well_id, lot, header, cfg))
    ax.set_xlabel("Pressure [bar]")
    ax.set_ylabel("TVD [m MSL]")
    ax.grid(color=theme.GRIDLINE, lw=0.5)
    return drawn


def contact_regimes(t: pd.DataFrame, tol_bar: float = 2.0) -> dict:
    """Classify every contact-corrected test: 'single' (one equivalent for its contact: solid line from the measurement to the contact, apex above unknown),
    'consistent' (several points whose aquifer-equivalent pressures agree within `tol_bar`: one gradient; only the shallowest apex is unknown) or
    'multi' (they do not line up: probably more than one pressure regime, so every gradient is dashed; the deepest start is ASSUMED to connect down to the contact).
    Returns {row index: (mode, is_deepest)}.  The tolerance is an assumption (bar)."""
    out: dict = {}
    c = t[t["aquifer_pressure_bar"].notna() & t["buoyancy_status"].astype(str).str.startswith("corrected")]
    for _, g in c.groupby([c["wellbore_id"], c["contact_tvdss_m"].round(0)]):
        deepest = g["tvdss_m"].idxmax()
        mode = "single" if len(g) == 1 else ("consistent" if float(g["aquifer_pressure_bar"].max() - g["aquifer_pressure_bar"].min()) <= tol_bar else "multi")
        for i in g.index:
            out[i] = (mode, i == deepest)
    return out


def pressure_depth_figure(wells: list[tuple[int, str]], pressure: pd.DataFrame, contacts: pd.DataFrame, lot: pd.DataFrame, header: pd.DataFrame, cfg: dict[str, Any], ncols: int = 3):
    """Small multiples, one pressure-depth axis per (wellbore_id, well_name); a shared legend for what the symbols mean."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n = len(wells)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 5.2 * nrows), squeeze=False)
    summary = {}
    for ax, (wid, name) in zip(axes.ravel(), wells):
        summary[name] = plot_well(ax, wid, pressure, contacts, lot, header, cfg)
        d = summary[name]
        ax.set_title(f"{name}: {d['tests']} tests, {d['corrected']} with a contact below", fontsize=9, color=theme.NORTH_SEA)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=hc_colour("oil"), markeredgecolor=theme.NORTH_SEA, label="oil test"),
               Line2D([0], [0], marker="o", color="w", markerfacecolor=hc_colour("gas"), markeredgecolor=theme.NORTH_SEA, label="gas / condensate test"),
               Line2D([0], [0], marker="D", color="w", markerfacecolor="white", markeredgecolor=theme.NORTH_SEA, label="aquifer-equivalent pressure at the contact"),
               Line2D([0], [0], color=theme.NORTH_SEA, lw=1.6, label="hydrocarbon gradient, test to contact"),
               Line2D([0], [0], color="#7EC8F0", lw=0.9, ls="--", label="water line through the aquifer-equivalent point"),
               Line2D([0], [0], color="#7EC8F0", lw=2.0, label="hydrostatic (band: fresh to brine, dashed strong brine, dotted fresh water)"),
               Line2D([0], [0], color="#009900", lw=1.0, label="OWC / HWC (dotted: ODT, oil down to)"), Line2D([0], [0], color="#993300", lw=1.0, label="GOC"), Line2D([0], [0], color="#B00000", lw=1.0, label="GWC (dotted: GDT)"),
               *st.legend_handles()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig, summary

# --------------------------------------------------------------------------------------------------------------------------------
# Three views of the same data: pressure [bar], equivalent density [sg] and overpressure [bar] (hydrostatic = the vertical zero line)
# --------------------------------------------------------------------------------------------------------------------------------
PVT_TO_PROXY = {"black oil": "black_oil", "volatile oil": "volatile_oil", "near-critical": "near_critical", "gas condensate": "rich_gas_condensate", "wet gas": "rich_gas_condensate", "dry gas": "lean_gas"}


def proxy_fluid_key(fluid_class: str | None, fluid: str, p_bar: float, t_c: float | None, cfg: dict[str, Any]) -> str:
    """The example fluid of `config/column_models.yaml` that stands in for a test: its PVT class (else oil -> black oil, gas -> rich gas condensate), the HPHT variant above the HPHT thresholds."""
    key = PVT_TO_PROXY.get(str(fluid_class), "black_oil" if fluid == "oil" else "rich_gas_condensate")
    h = cfg["hpht"]
    if key in ("volatile_oil", "rich_gas_condensate") and p_bar >= float(h["pressure_bar"][0]) and t_c is not None and pd.notna(t_c) and float(t_c) >= float(h["temperature_c"][0]):
        key = "hpht_volatile_oil" if key == "volatile_oil" else "hpht_gas_condensate"
    return key


def envelope_deviation(env: pd.DataFrame | None, key: str, heights_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(minimum, maximum) deviation [bar] from the linear gradient of the modelled envelope of proxy fluid `key` at the given heights below the apex (interpolated)."""
    if env is None or env.empty or key not in set(env["fluid_key"]):
        return np.zeros_like(heights_m, dtype=float), np.zeros_like(heights_m, dtype=float)
    e = env[env["fluid_key"] == key].sort_values("dz_m")
    return np.interp(heights_m, e["dz_m"], e["delta_p_min_bar"]), np.interp(heights_m, e["dz_m"], e["delta_p_max_bar"])


def mud_points(well_id: int, mud: pd.DataFrame, header: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Mud weight [sg, surface density] against TVD below sea level, for a near-vertical well only (an MD is not turned into a TVD for a deviated well)."""
    g = mud[(mud["wellbore_id"] == well_id) & mud["mud_weight_sg"].notna()] if mud is not None and len(mud) else pd.DataFrame()
    ref = pr.well_depth_reference(header[header["wellbore_id"] == well_id], cfg)
    if len(g) == 0 or ref.empty or ref.iloc[0]["base_depth_basis"] != "md_near_vertical":
        return pd.DataFrame(columns=["tvdss_m", "mud_weight_sg", "mud_type"])
    g = g[(g["mud_weight_sg"] > 0.9) & (g["mud_weight_sg"] < 2.8) & (g["md_m"] > 0)]
    return pd.DataFrame({"tvdss_m": g["md_m"].astype(float) - float(ref.iloc[0]["rt_elev_m"]), "mud_weight_sg": g["mud_weight_sg"].astype(float), "mud_type": g["mud_type"]}).sort_values("tvdss_m")


def plot_triplet(axes, well_id: int, well_name: str, pressure: pd.DataFrame, contacts: pd.DataFrame, lot: pd.DataFrame, mud: pd.DataFrame | None, header: pd.DataFrame, cfg: dict[str, Any],
                 env: pd.DataFrame | None = None, classes: dict | None = None, datum: str = "RT", z_pad: float = 120.0) -> dict[str, Any]:
    """Draw one well in three panels: (a) pressure [bar] with the fan of lines of constant equivalent density, (b) equivalent density [sg] on the `datum` basis (RT: the mud and leak-off basis; MSL: the
    water-gradient basis), (c) overpressure [bar] with the hydrostatic gradient as the vertical zero line.  Annotated with the gradients; a modelled envelope (proxy fluid of the PVT class, no
    composition) is drawn around every hydrocarbon leg and as an error bar on the aquifer-equivalent point."""
    ref = pr.well_depth_reference(header[header["wellbore_id"] == well_id], cfg)
    rt = float(ref.iloc[0]["rt_elev_m"]) if len(ref) else float(cfg["depth"]["assumed_rt_elev_m"])
    gw, gw_lo, gw_hi, _ = pr.water_gradient_band(cfg)
    t = pressure[(pressure["wellbore_id"] == well_id) & pressure["reservoir_pressure_bar"].notna() & pressure["tvdss_m"].notna()]
    t = t[~t["depth_basis"].isin(["md_deviated", "md_no_tvd"])]
    c = contacts[contacts["wellbore_id"] == well_id] if contacts is not None and len(contacts) else pd.DataFrame(columns=["label", "tvdss_m"])
    lp = lot_points(well_id, lot, header, cfg) if lot is not None and len(lot) else pd.DataFrame(columns=["tvdss_m", "pressure_bar", "test_type"])
    mp = mud_points(well_id, mud, header, cfg) if mud is not None else pd.DataFrame(columns=["tvdss_m", "mud_weight_sg"])
    zs = list(t["tvdss_m"]) + list(c["tvdss_m"]) + list(t["contact_tvdss_m"].dropna())
    zmin, zmax = (min(zs) - z_pad, max(zs) + z_pad) if zs else (1000.0, 3000.0)
    zz = np.linspace(zmin, zmax, 60)
    ax_a, ax_b, ax_c = axes

    def xa(p, z):
        return np.asarray(p, dtype=float)

    def xb(p, z):
        return pu.emw_sg(p, z, rt, datum)

    def xc(p, z):
        return pu.overpressure_bar(p, z, gw)

    fans = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0)
    for sg in fans:                                                                          # the fan of lines of constant equivalent density (RT basis): pressure and sg read off the same plot
        pl = pu.iso_sg_line(sg, zz, rt)
        ax_a.plot(pl, zz, color="#D5DBDF", lw=0.7, zorder=1)
        ax_a.text(pl[-1], zz[-1], f"{sg:g} sg", fontsize=8.5, color="#5F6B76", ha="center", va="bottom", clip_on=True, bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))
    for ax, fx in ((ax_a, xa), (ax_b, xb), (ax_c, xc)):
        ax.fill_betweenx(zz, fx(gw_lo * zz, zz), fx(gw_hi * zz, zz), color="#7EC8F0", alpha=0.18, lw=0)
        ax.plot(fx(gw * zz, zz), zz, color="#7EC8F0", lw=2.0, zorder=2)                              # hydrostatic (config/pore_pressure_style.yaml)
        ax.plot(fx(gw_hi * zz, zz), zz, color="#7EC8F0", lw=1.3, ls="--", zorder=2); ax.plot(fx(gw_lo * zz, zz), zz, color="#7EC8F0", lw=1.4, ls=":", zorder=2)
        for r in c.itertuples():
            ls, col = CONTACT_STYLE.get(r.label, ("-", "#555555"))
            ax.axhline(r.tvdss_m, color=col, ls=ls, lw=0.9, alpha=0.9)
    for r in c.itertuples():
        ls, col = CONTACT_STYLE.get(r.label, ("-", "#555555"))
        ax_a.text(0.995, r.tvdss_m, f"{r.label} {r.tvdss_m:.0f} m", transform=ax_a.get_yaxis_transform(), ha="right", va="bottom", fontsize=8.5, color=col)
    n_corr = 0
    c_lab: list = []
    reg = contact_regimes(t)
    n_multi = sum(1 for m, _ in reg.values() if m == "multi")
    apex_ids = set()                                                                          # the shallowest point of every single / consistent group has an unknown apex above it
    for _, g in t.loc[[i for i, (m, _) in reg.items() if m != "multi"]].groupby(["wellbore_id", t["contact_tvdss_m"].round(0)]):
        apex_ids.add(g["tvdss_m"].idxmin())
    for r in t.itertuples():
        fluid = "gas" if str(r.fluid_hint).startswith(("gas", "condensate")) else "oil" if str(r.fluid_hint).startswith("oil") else None
        cls = (classes or {}).get((r.wellbore_id, str(r.test_number)))
        face = fcc.class_colour(cls) if cls else (hc_colour(fluid) if fluid else "#B8BEC4")
        for ax, fx in ((ax_a, xa), (ax_b, xb), (ax_c, xc)):
            ax.scatter([fx(r.reservoir_pressure_bar, r.tvdss_m)], [r.tvdss_m], s=44, color=face, edgecolor=theme.NORTH_SEA, linewidth=0.8, zorder=6)
        ax_a.annotate(f"{r.test_number}", (r.reservoir_pressure_bar, r.tvdss_m), xytext=(5, -3), textcoords="offset points", fontsize=8.5, color=theme.NORTH_SEA)
        if pd.notna(r.aquifer_pressure_bar) and str(r.buoyancy_status).startswith("corrected") and fluid:
            n_corr += 1
            h = float(r.column_below_test_m)
            g_hc = (float(r.aquifer_pressure_bar) - float(r.reservoir_pressure_bar)) / h
            line = fcc.class_colour(cls) if cls else hc_line_colour(r.buoyancy_fluid)
            zl = np.linspace(r.tvdss_m, r.contact_tvdss_m, 12)
            pl = r.reservoir_pressure_bar + g_hc * (zl - r.tvdss_m)
            key = proxy_fluid_key(cls, fluid, float(r.reservoir_pressure_bar), r.temperature_c if "temperature_c" in r._fields else None, cfg)
            dmin, dmax = envelope_deviation(env, key, zl - r.tvdss_m)
            for ax, fx in ((ax_a, xa), (ax_b, xb), (ax_c, xc)):
                if env is not None and len(env):
                    ax.fill_betweenx(zl, fx(pl + dmin, zl), fx(pl + dmax, zl), color=line, alpha=0.18, lw=0, zorder=3)
                mode, deepest = reg.get(r.Index, ("single", True))
                ax.plot(fx(pl, zl), zl, color=line, lw=2.2 if (mode == "multi" and deepest) else 1.6, ls="--" if mode == "multi" else "-", zorder=4)     # several regimes: dashed, the deepest start is the assumed connection
                if r.Index in apex_ids:
                    za = np.array([r.tvdss_m - 0.06 * (zmax - zmin), r.tvdss_m])                  # apex unknown: short dashed continuation above the measurement
                    ax.plot(fx(r.reservoir_pressure_bar + g_hc * (za - r.tvdss_m), za), za, color=line, lw=1.3, ls=(0, (2, 2)), alpha=0.8, zorder=4)
                ax.errorbar([fx(r.aquifer_pressure_bar, r.contact_tvdss_m)], [r.contact_tvdss_m], xerr=[[abs(fx(r.aquifer_pressure_bar, r.contact_tvdss_m) - fx(r.aquifer_pressure_bar + dmin[-1], r.contact_tvdss_m))],
                            [abs(fx(r.aquifer_pressure_bar + dmax[-1], r.contact_tvdss_m) - fx(r.aquifer_pressure_bar, r.contact_tvdss_m))]], fmt="D", ms=5.5, mfc="white", mec=line, ecolor=line, elinewidth=1.0, capsize=2, zorder=7)
            ax_a.annotate(f"{g_hc:.3f} bar/m ({g_hc / pu.BAR_PER_M_PER_SG:.2f} sg)", (float(np.mean(pl)), float(np.mean(zl))), xytext=(-6, 0), textcoords="offset points", fontsize=8.5, color=line, ha="right", va="center")
            c_lab.append((float(xc(r.aquifer_pressure_bar, r.contact_tvdss_m)), float(r.contact_tvdss_m), f"{float(r.aquifer_ratio):.2f} x hydrostatic, {r.aquifer_regime}"))
    if c_lab:                                                       # one label per contact value, none drawn over another (several tests can share a contact)
        from ep_physics_base.figures.figure_style import annotate_points

        annotate_points(ax_c, [c_[0] for c_ in c_lab], [c_[1] for c_ in c_lab], [c_[2] for c_ in c_lab], max_labels=6, colour=theme.NORTH_SEA, box=True)
    if len(lp):
        lp = lp[(lp["tvdss_m"] >= zmin) & (lp["tvdss_m"] <= zmax)]
        for ax, fx in ((ax_a, xa), (ax_b, xb), (ax_c, xc)):
            st.scatter_tests(ax, fx(lp["pressure_bar"].to_numpy(), lp["tvdss_m"].to_numpy()), lp["tvdss_m"], lp["test_type"], s=34, label=False)
    if len(mp):
        mp = mp[(mp["tvdss_m"] >= zmin) & (mp["tvdss_m"] <= zmax)]
        for ax, fx in ((ax_a, xa), (ax_b, xb), (ax_c, xc)):
            pm = pu.mud_pressure_bar(mp["mud_weight_sg"].to_numpy(), mp["tvdss_m"].to_numpy(), rt)
            ax.plot(fx(pm, mp["tvdss_m"].to_numpy()), mp["tvdss_m"], color="#000000", lw=1.6, drawstyle="steps-post", zorder=3)          # MW: solid black (pore pressure style guide)
    ax_c.axvline(0, color="#7EC8F0", lw=2.0)
    wd_ = float(ref.iloc[0]["water_depth_m"]) if len(ref) and pd.notna(ref.iloc[0]["water_depth_m"]) else None
    cp_ = casing_points(well_id, lot, header, cfg)
    for ax in (ax_a, ax_b, ax_c):
        ax.set_ylim(zmax, zmin)
        ax.grid(color=theme.GRIDLINE, lw=0.5)
        draw_datum_boxes(ax, wd_, rt)
        draw_casing_points(ax, cp_, label=ax is ax_a)
    ax_a.set_xlabel("Pressure [bar]"); ax_a.set_ylabel("TVD [m MSL]")
    ax_b.set_xlabel(f"Equivalent density [sg], {datum} basis"); ax_c.set_xlabel("Overpressure: pressure minus hydrostatic [bar]")
    sg_rt = pu.emw_sg(gw * zmax, zmax, rt, "RT")
    note = (f"hydrostatic {gw:.4f} bar/m = {gw / pu.BAR_PER_M_PER_SG:.3f} sg (MSL basis; {float(sg_rt):.3f} sg on the RT basis at {zmax:.0f} m); RT {rt:.0f} m above MSL; "
            "grey fan: lines of constant sg (RT basis)"
            + (f"; {n_multi} gradients do not line up: probably more than one pressure regime, so they are dashed (the deepest is assumed to connect to the contact; same sandstone package or shales between is not known)" if n_multi else ""))                     # shown under the figure title, not over the data
    return {"tests": int(len(t)), "corrected": n_corr, "contacts": int(len(c)), "lot": int(len(lp)), "mud_points": int(len(mp)), "rt_elev_m": rt, "note": note}


def triplet_figures(wells: list[tuple[int, str]], pressure: pd.DataFrame, contacts: pd.DataFrame, lot: pd.DataFrame, mud: pd.DataFrame | None, header: pd.DataFrame, cfg: dict[str, Any],
                    env: pd.DataFrame | None = None, classes: dict | None = None, datum: str = "RT"):
    """One figure per well with the three views side by side (pressure with the sg fan, equivalent density, overpressure); returns [(name, figure, summary)]."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out = []
    for wid, name in wells:
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), sharey=True)
        s = plot_triplet(axes, wid, name, pressure, contacts, lot, mud, header, cfg, env, classes, datum)
        fig.suptitle(f"{name}: {s['tests']} tests, {s['corrected']} with a contact below, {s['lot']} casing-shoe tests, {s['mud_points']} mud-weight points", fontsize=10.5, color=theme.NORTH_SEA, x=0.01, ha="left", y=0.995)
        fig.text(0.01, 0.945, "\n".join(__import__("textwrap").wrap(s["note"], 170)), fontsize=8.5, color="#4A5A66", ha="left", va="top")
        handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#B8BEC4", markeredgecolor=theme.NORTH_SEA, label="DST reservoir pressure (measured; colour = PVT class)"),
                   Line2D([0], [0], marker="D", color="w", markerfacecolor="white", markeredgecolor=theme.NORTH_SEA, label="aquifer-equivalent pressure at the contact (derived; bar = modelled envelope)"),
                   Line2D([0], [0], color=theme.NORTH_SEA, lw=1.6, label="hydrocarbon gradient (derived; band = modelled envelope; dashed = no apex or several regimes, assumed)"),
                   Line2D([0], [0], color="#7EC8F0", lw=2.0, label="hydrostatic (assumed normal pressure; dashed strong brine, dotted fresh water)"),
                   *st.legend_handles(), Line2D([0], [0], color="#000000", lw=1.6, label="MW: mud weight (measured, surface density)")]
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8.5)
        fig.tight_layout(rect=(0, 0.12, 1, 0.88))
        out.append((name, fig, s))
    return out

# --------------------------------------------------------------------------------------------------------------------------------
# The whole well in equivalent density: hydrostatic, overburden, leakage line, pore pressure, leak-off tests, mud weight, ESD / ECD bands, fracture band and the drilling window
# --------------------------------------------------------------------------------------------------------------------------------
def plot_fullwell(ax, well_id: int, pressure: pd.DataFrame, lot: pd.DataFrame, mud: pd.DataFrame | None, header: pd.DataFrame, cfg: dict[str, Any], pcfg: dict[str, Any], datum: str = "RT",
                  z_max: float | None = None, casing: bool = True) -> pd.DataFrame:
    """Equivalent density [sg] against TVD [m MSL] for the whole well.  Returns the drilling-window rows at the tested depths (for a table).

    The overburden, Poisson band, ESD and ECD bands are ASSUMPTIONS (config/pore_pressure.yaml) shown as bands; a quantity without data is not drawn."""
    from ep_pressure_pvt_fluids import pore_pressure as pp

    h = header[header["wellbore_id"] == well_id]
    ref = pr.well_depth_reference(h, cfg)
    rt = float(ref.iloc[0]["rt_elev_m"]) if len(ref) else float(cfg["depth"]["assumed_rt_elev_m"])
    wd = float(ref.iloc[0]["water_depth_m"]) if len(ref) and pd.notna(ref.iloc[0]["water_depth_m"]) else 100.0
    td = float(ref.iloc[0]["td_tvd_m"]) if len(ref) and pd.notna(ref.iloc[0]["td_tvd_m"]) else None
    gw = float(cfg["hydrostatic"]["gradient_bar_per_m"])
    t = pressure[(pressure["wellbore_id"] == well_id) & pressure["reservoir_pressure_bar"].notna() & pressure["tvdss_m"].notna() & ~pressure["depth_basis"].isin(["md_deviated", "md_no_tvd"])]
    zmax = z_max or (max([td - rt] if td else [3000.0] + list(t["tvdss_m"])) * 1.02)
    zz = np.linspace(max(wd + 20.0, 100.0), zmax, 80)

    def emw(p, z):
        return pu.emw_sg(p, z, rt, datum)

    from ep_pressure_pvt_fluids.plots import pp_style as ps

    _, gw_lo_, gw_hi_, _ = pr.water_gradient_band(cfg)
    ps.plot_curve(ax, "hydrostatic", emw(gw * zz, zz), zz)                                  # hydrostatic reference (no pore pressure prediction exists, so it is not called expected)
    ps.plot_curve(ax, "hydrostatic_max", emw(gw_hi_ * zz, zz), zz); ps.plot_curve(ax, "hydrostatic_min", emw(gw_lo_ * zz, zz), zz)
    obg = pp.overburden_pressure_bar(zz, wd, pcfg)
    ps.plot_curve(ax, "obg", emw(obg, zz), zz)
    ax.plot(emw(pp.leakage_line_bar(obg, pcfg), zz), zz, color=ps.line_style("obg")["color"], lw=1.0, ls=":", label="OBG - 100 bar (Sodir leakage line)")
    fc = ps.fracture_curves(obg, gw * zz, pcfg["poisson_ratio"]["band"])                    # Sh and FG family from the Poisson band (assumption)
    for key in ("fg_max", "fg", "fg_min", "sh", "sh_min"):
        ps.plot_curve(ax, key, emw(fc[key], zz), zz)
    mp = mud_points(well_id, mud, header, cfg) if mud is not None else pd.DataFrame()
    mc = pcfg["mud"]
    if len(mp):
        mpl = mp[mp["tvdss_m"] <= zmax]
        mwr = pu.sg_from_pressure(pu.mud_pressure_bar(mpl["mud_weight_sg"].to_numpy(), mpl["tvdss_m"].to_numpy(), rt), pu.tvd_rt_from_msl(mpl["tvdss_m"].to_numpy(), rt) if datum == "RT" else mpl["tvdss_m"].to_numpy())
        ax.step(mwr, mpl["tvdss_m"], where="post", color=ps.line_style("mw")["color"], lw=ps.line_style("mw")["lw"], ls=ps.line_style("mw")["ls"], label=ps.line_style("mw")["label"])
        esd_lo, esd_hi = (float(x) for x in mc["esd_minus_mud_weight_sg"])
        loss_lo, loss_hi = (float(x) for x in mc["annular_loss_sg"])
        ax.fill_betweenx(mpl["tvdss_m"], mwr + esd_lo, mwr + esd_hi, step="post", color="#B8C2C9", alpha=0.5, lw=0, label="ESD band (assumed)")
        ax.fill_betweenx(mpl["tvdss_m"], mwr + esd_hi + loss_lo, mwr + esd_hi + loss_hi, step="post", color="#F0C9A0", alpha=0.55, lw=0, label="ECD band (assumed annular loss)")
    lp = lot_points(well_id, lot, header, cfg) if lot is not None and len(lot) else pd.DataFrame()
    if len(lp):
        lp = lp[lp["tvdss_m"] <= zmax]
        st.scatter_tests(ax, emw(lp["pressure_bar"].to_numpy(), lp["tvdss_m"].to_numpy()), lp["tvdss_m"], lp["test_type"], s=38, label=True)
    rows = []
    for r in t.itertuples():
        z = float(r.tvdss_m)
        p_raw = float(r.reservoir_pressure_bar)
        fluid = "gas" if str(r.fluid_hint).startswith(("gas", "condensate")) else "oil" if str(r.fluid_hint).startswith("oil") else None
        ax.scatter([emw(p_raw, z)], [z], s=40, color=hc_colour(fluid) if fluid else "#B8BEC4", edgecolor=theme.NORTH_SEA, linewidth=0.8, zorder=7)
        pp_use = float(r.aquifer_pressure_bar) if pd.notna(r.aquifer_pressure_bar) and str(r.buoyancy_status).startswith("corrected") else p_raw
        z_use = float(r.contact_tvdss_m) if pd.notna(r.aquifer_pressure_bar) and str(r.buoyancy_status).startswith("corrected") else z
        if pp_use != p_raw:
            ax.scatter([emw(pp_use, z_use)], [z_use], marker="D", s=34, color="white", edgecolor=theme.NORTH_SEA, linewidth=1.0, zorder=7)
        ob = float(pp.overburden_pressure_bar(np.array([z_use]), wd, pcfg)[0])
        band = pp.fracture_band_bar(np.array([ob]), np.array([pp_use]), pcfg).iloc[0]
        lo, hi = float(emw(band["shmin_low_bar"], z_use)), float(emw(band["fg_upper_bar"], z_use))
        w = pp.drilling_window(float(emw(pp_use, z_use)), np.nan, float(emw(band["shmin_mid_bar"], z_use)), float(pcfg["mud"]["window_margin_sg"])).iloc[0]
        rows.append({"well": None, "tvdss_m": z_use, "pp_sg": float(emw(pp_use, z_use)), "shmin_low_sg": lo, "shmin_mid_sg": float(emw(band["shmin_mid_bar"], z_use)), "fg_upper_sg": hi, "obg_sg": float(emw(ob, z_use)),
                     "window_lower_sg": w["lower_sg"], "window_upper_sg": w["upper_sg"], "window_width_sg": w["width_sg"], "status": w["status"]})
    xs_ = [v for r_ in rows for v in (r_["pp_sg"],)] + [float(emw(float(q), float(zq))) for q, zq in zip(t["reservoir_pressure_bar"], t["tvdss_m"])]
    ax.set_ylim(zmax, -(rt + 0.05 * zmax))
    draw_datum_boxes(ax, wd, rt)
    if casing:
        draw_casing_points(ax, casing_points(well_id, lot, header, cfg))
    ax.set_xlabel(f"Equivalent density [sg], {datum} basis")
    ax.set_ylabel("TVD [m MSL]")
    ax.grid(color=theme.GRIDLINE, lw=0.5)
    ax.set_xlim(min(0.9, (min(xs_) - 0.05) if xs_ else 0.9), max(2.6, ax.get_xlim()[1]))
    return pd.DataFrame(rows)
