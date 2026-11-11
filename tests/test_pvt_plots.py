"""PVT figures for the notebooks: the blurred zone raster agrees with the classification, and every figure function draws without error."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ep_fluids import eos, pvt
from ep_pressure import pressure as pr  # noqa: E402
from ep_fluids.plots import pvt_colours as pc  # noqa: E402
from ep_fluids.plots import pvt_plots as pp
from ep_pressure.plots import pt_chart  # noqa: E402
from ep_physics_base.config import load_yaml  # noqa: E402

CFG = pvt.load_config()
CFG3D = load_yaml("config/pvt3d.yaml")


def synthetic() -> pd.DataFrame:
    n = 10
    rng = np.random.default_rng(4)
    pressure = pd.DataFrame({"wellbore_id": pd.array(range(1, n + 1), dtype="Int64"), "well_name": [f"W{i}" for i in range(n)], "test_number": ["1"] * n, "area": ["NORTH SEA"] * n, "mid_md_m": np.linspace(2000, 4500, n),
                             "reservoir_pressure_bar": rng.uniform(150, 800, n), "temperature_c": [np.nan, np.nan] + list(rng.uniform(60, 180, n - 2))})
    pressure["temperature_est_c"] = [90.0, 110.0] + [np.nan] * (n - 2)
    dst = pd.DataFrame({"wellbore_id": pd.array(range(1, n + 1), dtype="Int64"), "test_number": ["1"] * n, "gor_sm3_sm3": [100, 500, 800, 5000, 20000, 150, 300, 6000, 90, 400],
                        "oil_density_g_cm3": [0.85, 0.8, 0.78, 0.78, 0.76, 0.93, 0.84, 0.78, 0.88, 0.81], "gas_relative_density": [0.75] * n})
    out, _ = pvt.build_well_dst_pvt(pressure, dst, CFG, use_estimated_temperature=True)
    return out


def test_zone_raster_agrees_with_the_classification_and_blurs_across_boundaries():
    ref = CFG["reference"]
    image, zone = pvt.zone_raster(CFG, ref["api"], ref["temperature_c"], ref["gas_gravity"], 1200.0, pc.ZONE_COLOURS, pc.ZONE_OF_STATE, width=120, height=100, sigma=3.0)
    assert image.shape == (100, 120, 3) and zone.shape == (100, 120) and image.min() >= 0.0 and image.max() <= 1.0
    assert set(np.unique(zone)) == set(pc.ZONE_COLOURS)                                           # all six zones occur on the reference chart
    assert zone[5, 60] == "forbidden" and zone[95, 5] == "monophasic: undersaturated oil" and zone[95, 115] == "monophasic: undersaturated gas"
    base = np.array([int(pc.ZONE_COLOURS["monophasic: undersaturated oil"][i:i + 2], 16) / 255.0 for i in (1, 3, 5)])
    assert np.allclose(image[95, 5], base, atol=0.02)                                             # far from a boundary the pastel colour is unchanged
    row = zone[95]
    edge = int(np.flatnonzero(row[:-1] != row[1:])[0])                                            # a boundary: the colour changes gradually, not in one step
    assert np.abs(image[95, edge] - image[95, edge + 1]).max() < 0.05 and np.abs(image[95, edge - 8] - image[95, edge + 8]).max() > 0.02


def test_figure_functions_draw_the_glr_pt_api_and_3d_charts():
    pv = synthetic()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    pp.plot_glr_chart(axes[0], pv, CFG)
    pt_chart.plot_pt_chart(axes[1], pv, pr.load_config())
    pp.plot_api_chart(axes[2], pv, CFG)
    assert len(axes[0].images) == 1 and len(axes[1].patches) == 6 and axes[1].collections                 # the raster, six P-T zones, test markers
    assert any(c.get_facecolor().shape[0] == 0 or c.get_facecolor()[0][3] == 0 for c in axes[1].collections)       # hollow markers for the estimated temperatures
    plt.close(fig)
    fluid = eos.fluids_from_config(CFG3D)["n_butane"]
    surface = eos.pvt_surface(fluid, {**CFG3D["grid"], "n_temperature": 12, "n_liquid": 8, "n_plateau": 4, "n_vapour": 8})
    t = pd.DataFrame({"t_c": [80.0, 170.0], "p_bar": [300.0, 300.0]})
    states = [eos.state(x, 300.0, fluid) for x in t["t_c"]]
    t["log_v"], t["regime"], t["estimated"] = [s[0] for s in states], [s[1] for s in states], [True, False]
    fig = plt.figure(figsize=(6, 5))
    pp.plot_pvt_surface(fig.add_subplot(111, projection="3d"), fluid, surface, t)
    assert list(t["regime"]) == ["liquid", "supercritical"]
    plt.close(fig)
