"""Matplotlib figures of the PVT charts for the notebooks and reports: the same zones, pastel colours, hollow markers and model-fluid surface as the dashboards
(the dashboards and the pressure-temperature chart with the HPHT classes live in ep-pressure-pvt-fluids, which also needs the pressure library).  EDS light theme (AGENTS.md section 10); each function draws on the axes it is given.

A hollow marker is a test whose temperature is ESTIMATED from the geothermal gradient of its well (docs/temperature.md); a solid one has a gauge temperature.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ep_pressure_pvt_fluids import eos
from ep_pressure_pvt_fluids import pvt
from ep_pressure_pvt_fluids.plots import pvt_colours as pc
from ep_physics_base.figures import theme
from ep_physics_base.config import load_yaml, resolve_path

XMIN, XMAX = 10.0, 1.0e6
GLR_TICKS = [10, 100, 1000, 10000, 100000, 1000000]


def _types_cfg() -> dict[str, Any]:
    return load_yaml("config/measurement_types.yaml")["types"]


def scatter_tests(ax: Any, x: pd.Series, y: pd.Series, colour: str, hollow: pd.Series, size: float = 20.0, label: str | None = None, types: pd.Series | None = None, **kw: Any) -> None:
    """Solid markers where `hollow` is False, hollow (coloured outline) where True; one legend entry.  `types` (a measurement type id per point, config/measurement_types.yaml)
    changes the SHAPE of the marker by measurement type (circle, square, diamond ...); a point without a type keeps the circle."""
    if types is not None:
        styles = _types_cfg()
        kinds = pd.Series(np.asarray(types, dtype=object), dtype=object).fillna("dst")
        if label:
            ax.scatter([], [], s=size, color=colour, edgecolor="white", linewidth=0.5, marker="o", label=label)          # the colour key is a neutral circle: the shape belongs to the measurement type
        for kind in kinds.unique():
            mask = (kinds == kind).to_numpy()
            scatter_tests(ax, np.asarray(x)[mask], np.asarray(y)[mask], colour, pd.Series(np.asarray(hollow)[mask]), size, None, None, marker=styles.get(kind, styles["other"])["marker"], **kw)
        return
    solid = ~pd.Series(np.asarray(hollow)).to_numpy(dtype=bool)
    if solid.any():
        ax.scatter(np.asarray(x)[solid], np.asarray(y)[solid], s=size, color=colour, edgecolor="white", linewidth=0.5, label=label, zorder=3, **kw)
    if (~solid).any():
        ax.scatter(np.asarray(x)[~solid], np.asarray(y)[~solid], s=size, facecolors="none", edgecolors=colour, linewidth=1.4, label=None if solid.any() else label, zorder=3, **kw)


def types_of(pv: pd.DataFrame) -> pd.Series | None:
    """The measurement type of every test, or None when the frame has none (old marts): the markers are then all circles."""
    return pv["measurement_type"].astype(object) if "measurement_type" in pv else None


def _type_handles(pv: pd.DataFrame) -> list[Any]:
    """Legend entries (grey markers) for the measurement types present, with their counts."""
    from matplotlib.lines import Line2D

    if "measurement_type" not in pv:
        return []
    styles, counts = _types_cfg(), pv["measurement_type"].value_counts()
    return [Line2D([], [], marker=styles[k]["marker"], linestyle="", markersize=5.5, markerfacecolor="#8A98A3", markeredgecolor="#5B6B77", label=styles[k]["label"] + f" ({int(n)})") for k, n in counts.items() if k in styles]


def measurement_wells(oil_samples: pd.DataFrame, wells: pd.DataFrame, type_id: str = "wireline_tester") -> pd.DataFrame:
    """Wells with at least one oil-sample record of a measurement type (default: wireline formation tester, MDT / RFT / FMT) with their position and number of records;
    `wells` has wellbore_id, lat_etrs89, lon_etrs89.  Used for the context layer of the maps (such a record has no GLR or pressure: it cannot be a point in the GLR chart)."""
    from ep_pressure_pvt_fluids import measurement_types as mtypes

    cfg = mtypes.load_config()
    s = oil_samples.assign(kind=mtypes.sample_type_of(oil_samples["test_type"], cfg))
    n = s[s["kind"] == type_id].groupby("wellbore_id").size().rename("n_records")
    pos = wells.drop_duplicates("wellbore_id").set_index("wellbore_id")[["lat_etrs89", "lon_etrs89"]]
    return n.to_frame().join(pos, how="inner").dropna(subset=["lat_etrs89", "lon_etrs89"]).reset_index()


def estimated_mask(pv: pd.DataFrame) -> pd.Series:
    return pv["temperature_is_estimated"].fillna(False).astype(bool) if "temperature_is_estimated" in pv else pd.Series(False, index=pv.index)


def plot_glr_chart(ax: Any, pv: pd.DataFrame, cfg: dict[str, Any], pmax: float = 1200.0, legend: bool = True) -> None:
    """GLR (log scale, on top) against reservoir pressure (increasing downwards): pastel zones blurred into each other, the dashed saturation-pressure envelope of the
    reference fluid and the tests coloured by saturation state."""
    ref = cfg["reference"]
    image, _ = pvt.zone_raster(cfg, ref["api"], ref["temperature_c"], ref["gas_gravity"], pmax, pc.ZONE_COLOURS, pc.ZONE_OF_STATE)
    ax.imshow(image, extent=[np.log10(XMIN), np.log10(XMAX), pmax, 0.0], aspect="auto", interpolation="bilinear", zorder=0)
    outline = pvt.zone_polygons(np.arange(0.0, pmax + 5.0, 5.0), ref["temperature_c"], ref["api"], ref["gas_gravity"], cfg)["forbidden"]
    ax.plot(np.log10(np.asarray(outline[0])), outline[1], linestyle="--", color=theme.TEXT_PRIMARY, linewidth=1.4, zorder=2)
    shown = pv[(pv["reservoir_pressure_bar"] <= pmax) & pv["glr_sm3_sm3"].notna()]
    for state, colour in pc.STATE_COLOURS.items():
        s = shown[shown["saturation_state"] == state]
        if len(s):
            scatter_tests(ax, np.log10(s["glr_sm3_sm3"].astype(float)), s["reservoir_pressure_bar"].astype(float), colour, estimated_mask(s), label=f"{state} ({len(s)})", types=types_of(s))
    ax.set_xlim(np.log10(XMIN), np.log10(XMAX))
    ax.set_ylim(pmax, 0.0)
    ax.set_xticks(np.log10(GLR_TICKS))
    ax.set_xticklabels([f"{t:,}" for t in GLR_TICKS])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("GLR [Sm3/Sm3] (log scale)")
    ax.set_ylabel("Reservoir pressure [bar], increasing downwards")
    ax.text(3.0, 40, "FORBIDDEN DOMAIN", ha="center", fontsize=8, fontweight="bold", color=theme.TEXT_PRIMARY)
    ax.set_title(f"Zones of the GLR-pressure plane (reference fluid {ref['api']:.0f} API, {ref['temperature_c']:.0f} degC, gas gravity {ref['gas_gravity']:.2f})", y=1.1, fontsize=9)
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        zone = [__import__("matplotlib").patches.Patch(facecolor=c, edgecolor="#B8BEC4", label=n) for n, c in pc.ZONE_COLOURS.items()]
        ax.legend(handles=zone + handles + _type_handles(shown), fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=3, frameon=False)


def plot_api_chart(ax: Any, pv: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Stock-tank API gravity against temperature with the biodegradation brackets (degradable below 60, transition 60-80, pasteurised above 80 degC)."""
    b = cfg["biodegradation"]
    edges = [0.0] + [float(x["below"]) for x in b["temperature_brackets_c"][:-1]] + [200.0]
    for i, band in enumerate(b["temperature_brackets_c"]):
        colour = pc.BRACKET_COLOURS.get(band["name"], pc.GREY)
        ax.axvspan(edges[i], edges[i + 1], color=colour, alpha=0.09, zorder=0)
        ax.text((edges[i] + min(edges[i + 1], 200)) / 2, 61, band["name"], ha="center", fontsize=8.5, color=colour)
    for v in (b["heavy_api_below"], b["light_api_above"]):
        ax.axhline(v, color=theme.TEXT_PRIMARY, linestyle="--", linewidth=0.9)
    t = pv[pv["api"].notna() & pv["temperature_used_c"].notna()]
    for state, colour in pc.STATE_COLOURS.items():
        s = t[t["saturation_state"] == state]
        if len(s):
            scatter_tests(ax, s["temperature_used_c"].astype(float), s["api"].astype(float), colour, estimated_mask(s), size=14, types=types_of(s))
    ax.set_xlim(0, 200)
    ax.set_ylim(5, 62)
    ax.set_xlabel(f"Test temperature [degC] (n={len(t)}; hollow: estimated)")
    ax.set_ylabel("Stock-tank gravity [degAPI]")


def map_layers(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Country outlines (clipped to the shelf region) and the five area outlines, read from the mid-layer; empty where a layer is missing."""
    import geopandas as gpd
    from shapely.geometry import box

    if config is None:
        raise ValueError("map_layers needs the mid-layer directories: pass config={'midlayer': {'marts_dir': ..., 'spatial_dir': ...}}")
    spatial, marts = resolve_path(config["midlayer"].get("spatial_dir", "data/midlayer/spatial")), resolve_path(config["midlayer"]["marts_dir"])
    out: dict[str, Any] = {"countries": None, "areas": {}}
    if (spatial / "countries.parquet").exists():
        countries = gpd.read_parquet(spatial / "countries.parquet")
        clipped = countries.geometry.intersection(box(-6.0, 54.0, 36.0, 82.0)).simplify(0.03)
        out["countries"] = gpd.GeoDataFrame(geometry=clipped[~clipped.is_empty])
    for area in ("southern_north_sea", "central_north_sea", "northern_north_sea", "norwegian_sea", "barents_sea"):
        path = marts / "regions" / area / "_aoi.parquet"
        if path.exists():
            out["areas"][area] = gpd.read_parquet(path).geometry.union_all().simplify(0.02)
    return out


def plot_well_map(ax: Any, pv: pd.DataFrame, wells: pd.DataFrame, layers: dict[str, Any], colour_column: str = "saturation_state", context: pd.DataFrame | None = None) -> None:
    """Wells with a PVT test on the shelf outline (marker size: tests per well, colour: the most frequent category); `wells` has wellbore_id, lat_etrs89, lon_etrs89.  `context` (from
    `measurement_wells`) adds grey hexagons for wells with wireline formation-tester sample records, which have no GLR or pressure of their own and so are not PVT tests."""
    if layers.get("countries") is not None:
        layers["countries"].plot(ax=ax, color="#EEF1F3", edgecolor="#B8BEC4", linewidth=0.5, zorder=0)
    for geom in layers.get("areas", {}).values():
        for part in getattr(geom, "geoms", [geom]):
            ax.plot(*part.exterior.xy, color="#8A98A3", linewidth=0.8, linestyle=":", zorder=1)
    pos = wells.drop_duplicates("wellbore_id").set_index("wellbore_id")[["lat_etrs89", "lon_etrs89"]]
    typed = "measurement_type" in pv
    mode_of = lambda s: s.mode().iloc[0] if s.notna().any() else None
    g = pv.groupby("wellbore_id").agg(n=("test_number", "size"), cat=(colour_column, mode_of), kind=("measurement_type" if typed else colour_column, mode_of)).join(pos).dropna(subset=["lat_etrs89", "lon_etrs89"])
    styles = _types_cfg()
    if context is not None and len(context):
        ax.scatter(context["lon_etrs89"], context["lat_etrs89"], s=16, marker=styles["wireline_tester"]["marker"], facecolors="none", edgecolors="#8A98A3", linewidth=0.7, zorder=2,
                   label=f"{styles['wireline_tester']['label']}: sample record, no GLR ({len(context)} wells)")
    for state, colour in pc.STATE_COLOURS.items():
        s = g[g["cat"] == state]
        for kind in (s["kind"].unique() if typed else [None]):
            part = s[s["kind"] == kind] if typed else s
            if len(part):
                ax.scatter(part["lon_etrs89"], part["lat_etrs89"], s=10 + 8 * np.sqrt(part["n"]), color=colour, edgecolor="white", linewidth=0.5, zorder=3, marker=styles.get(kind, styles["other"])["marker"] if typed else "o",
                           label=f"{state} ({len(s)} wells)" if (not typed or kind == s["kind"].unique()[0]) else None)
    if len(g):
        pad_x, pad_y = max((g["lon_etrs89"].max() - g["lon_etrs89"].min()) * 0.1, 1.0), max((g["lat_etrs89"].max() - g["lat_etrs89"].min()) * 0.05, 0.6)
        ax.set_xlim(g["lon_etrs89"].min() - pad_x, g["lon_etrs89"].max() + pad_x)
        ax.set_ylim(g["lat_etrs89"].min() - pad_y, g["lat_etrs89"].max() + pad_y)
        ax.set_aspect(1.0 / np.cos(np.radians(float(g["lat_etrs89"].mean()))))
    ax.set_xlabel("Longitude [deg E]")
    ax.set_ylabel("Latitude [deg N]")


def plot_pvt_surface(ax: Any, fluid: eos.Fluid, surface: dict[str, Any], tests: pd.DataFrame | None = None, zones: dict[str, str] | None = None, view: tuple[float, float] = (22.0, -128.0)) -> None:
    """The pressure-volume-temperature surface of a model fluid on 3D axes (temperature [degC], log10 molar volume [L/mol], log10 pressure [bar]): liquid, two-phase dome, vapour
    and supercritical regions in pastel colours, the dome curves, the critical isotherm and point, and the tests (hollow: estimated temperature).  `tests` has t_c, log_v, p_bar,
    regime and optionally estimated."""
    from matplotlib.colors import to_rgba

    zones = zones or pc.SURFACE_COLOURS
    y, z, region = np.array(surface["y"]), np.array(surface["z"]), np.array(surface["region"])
    x = np.repeat(np.array(surface["t_c"])[:, None], y.shape[1], axis=1)
    y_cut = np.where(y > 3.0, np.nan, y)                                         # volumes above 1,000 L/mol are cut off, as in the dashboard
    codes = ["liquid", "two-phase", "vapour", "supercritical"]
    colours = np.array([[to_rgba(zones[codes[int(r)]]) for r in row] for row in region])
    ax.plot_surface(x, y_cut, np.log10(z), facecolors=colours, shade=False, linewidth=0, antialiased=False, rstride=1, cstride=1, alpha=0.9, zorder=1)
    for key in ("dome_liquid", "dome_vapour"):
        d = surface[key]
        ax.plot(d["x"], d["y"], np.log10(d["z"]), color=theme.TEXT_PRIMARY, linewidth=1.8, zorder=6)
    ci = surface["critical_isotherm"]
    ax.plot(ci["x"], ci["y"], np.log10(ci["z"]), color=theme.ORANGE, linewidth=1.8, zorder=6)
    cp = surface["critical"]
    ax.scatter([cp["t_c"]], [cp["log_v"]], [np.log10(cp["p_bar"])], color=theme.RED, marker="D", s=46, zorder=12, label="critical point")
    if tests is not None and len(tests):
        regime_colour = {"liquid": theme.GREEN, "vapour": theme.RED, "supercritical": "#B8A100", "two-phase": theme.NORTH_SEA}
        est = tests["estimated"].fillna(False).astype(bool) if "estimated" in tests else pd.Series(False, index=tests.index)
        for regime, colour in regime_colour.items():
            s = tests[tests["regime"] == regime]
            if len(s):
                hollow = est.loc[s.index].to_numpy()
                ax.scatter(s["t_c"][~hollow], s["log_v"][~hollow], np.log10(s["p_bar"][~hollow]), color=colour, s=16, depthshade=False, zorder=10, label=f"{regime} ({len(s)})")
                ax.scatter(s["t_c"][hollow], s["log_v"][hollow], np.log10(s["p_bar"][hollow]), facecolors="none", edgecolors=colour, s=18, linewidth=1.3, depthshade=False, zorder=10)
    ax.set_xlabel("Temperature [degC]")
    ax.set_ylabel("Molar volume [L/mol] (log10)")
    ax.set_zlabel("Pressure [bar] (log10)")
    ax.set_ylim(min(np.nanmin(y_cut), -2.0), 3.0)
    ax.set_zlim(np.log10(np.nanmin(z)), np.log10(1200.0))
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_title(f"{fluid.name}: critical point {fluid.tc_k - 273.15:.0f} degC, {fluid.pc_bar:.0f} bar (Peng-Robinson, pure component)", fontsize=9)


def plot_pv_projection(ax: Any, fluid: eos.Fluid, surface: dict[str, Any], tests: pd.DataFrame | None = None, zones: dict[str, str] | None = None, every: int = 3) -> None:
    """The pressure-volume projection of the surface (log-log): the two-phase dome (pastel grey, bounded by the saturated liquid and vapour curves), isotherms below the critical
    temperature (blue) and above it (yellow), the critical isotherm (orange), the critical point and the tests (hollow: estimated temperature)."""
    from matplotlib.patches import Polygon

    zones = zones or pc.SURFACE_COLOURS
    y, z, t_c = np.array(surface["y"]), np.array(surface["z"]), np.array(surface["t_c"])
    dl, dv = surface["dome_liquid"], surface["dome_vapour"]
    dome = np.r_[np.c_[dl["y"], np.log10(dl["z"])], np.c_[dv["y"], np.log10(dv["z"])][::-1]]
    ax.add_patch(Polygon(dome, closed=True, facecolor=zones["two-phase"], edgecolor="none", zorder=0))
    tc = surface["critical"]["t_c"]
    for i in range(0, len(t_c), every):
        ax.plot(y[i], np.log10(z[i]), color=theme.BLUE if t_c[i] < tc - 1e-6 else "#B8A100", linewidth=0.8, alpha=0.7, zorder=1)
    ci = surface["critical_isotherm"]
    ax.plot(ci["y"], np.log10(ci["z"]), color=theme.ORANGE, linewidth=1.8, zorder=3, label=f"critical isotherm ({tc:.0f} degC)")
    ax.plot(dl["y"], np.log10(dl["z"]), color=theme.TEXT_PRIMARY, linewidth=1.6, zorder=3)
    ax.plot(dv["y"], np.log10(dv["z"]), color=theme.TEXT_PRIMARY, linewidth=1.6, zorder=3, label="dome (saturated liquid and vapour)")
    ax.scatter([surface["critical"]["log_v"]], [np.log10(surface["critical"]["p_bar"])], color=theme.RED, marker="D", s=46, zorder=6, label="critical point")
    if tests is not None and len(tests):
        est = tests["estimated"].fillna(False).astype(bool) if "estimated" in tests else pd.Series(False, index=tests.index)
        for regime, colour in pc.REGIME_COLOURS.items():
            s = tests[tests["regime"] == regime]
            if len(s):
                hollow = est.loc[s.index].to_numpy()
                ax.scatter(s["log_v"][~hollow], np.log10(s["p_bar"][~hollow]), color=colour, s=14, edgecolor="white", linewidth=0.4, zorder=5, label=f"{regime} ({len(s)})")
                ax.scatter(s["log_v"][hollow], np.log10(s["p_bar"][hollow]), facecolors="none", edgecolors=colour, s=16, linewidth=1.2, zorder=5)
    ax.set_xlim(-2.2, 3.0)
    ax.set_ylim(np.log10(np.nanmin(z)), np.log10(1200.0))
    ax.set_xlabel("Molar volume [L/mol] (log10)")
    ax.set_ylabel("Pressure [bar] (log10)")
    ax.set_title("Pressure-volume projection: dome, isotherms and tests", fontsize=9)
    ax.legend(fontsize=8.5, loc="lower left", frameon=False)
