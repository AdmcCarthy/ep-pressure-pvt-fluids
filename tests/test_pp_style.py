"""The pore pressure line style guide (config/pore_pressure_style.yaml): the mapping and its use in the whole-well plot."""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from ep_pressure import pore_pressure as pp  # noqa: E402
from ep_pressure import pressure as pr  # noqa: E402
from ep_pressure.plots import pp_style as ps  # noqa: E402
from ep_pressure.plots import pressure_plots as ppl  # noqa: E402

GREEN, BLUE, RED, PURPLE, LIGHT = "#2E9D3E", "#12408C", "#E20337", "#7B3FA0", "#7EC8F0"


def test_every_quantity_has_the_requested_colour_and_dash():
    expect = {"sh": (GREEN, "-"), "sh_min": (GREEN, "--"), "fg": (BLUE, "-"), "fg_min": (BLUE, ":"), "fg_max": (BLUE, "--"), "sfg": (PURPLE, "-"),
              "pp": (RED, "-"), "pp_max": (RED, "--"), "pp_min": (RED, ":"), "mw": ("#000000", "-"), "hydrostatic": (LIGHT, "-"), "hydrostatic_max": (LIGHT, "--"), "hydrostatic_min": (LIGHT, ":")}
    for k, (c, ls) in expect.items():
        s = ps.line_style(k)
        assert s["color"].lower() == c.lower() and s["ls"] == ls, k
    obg = ps.line_style("obg")
    r, g, b = (int(obg["color"][i:i + 2], 16) for i in (1, 3, 5))
    assert obg["ls"] == "-" and r > g and r > b and r < 170 and g < 80                      # dark red to brown


def test_fracture_curves_are_ordered_by_the_poisson_band():
    z = np.array([1000.0, 2000.0])
    c = ps.fracture_curves(pp.overburden_pressure_bar(z, 100.0, pp.load_config()), 0.1005 * z, [0.25, 0.30, 0.35])
    assert np.all(c["sh_min"] < c["sh"]) and np.all(c["fg_min"] < c["fg"]) and np.all(c["fg"] < c["fg_max"]) and np.all(c["sh"] < c["fg"])


def test_the_whole_well_plot_draws_the_curves_in_the_guide_styles():
    cfg, pcfg = pr.load_config(), pp.load_config()
    hdr = pd.DataFrame({"wellbore_id": [1], "rt_elev_m": [25.0], "water_depth_m": [120.0], "td_md_m": [3000.0], "td_tvd_m": [2995.0], "main_area": "NORTH SEA"})
    mud = pd.DataFrame({"wellbore_id": [1, 1], "md_m": [500.0, 2500.0], "mud_weight_sg": [1.2, 1.6], "mud_type": "WATER BASED"})
    t = pd.DataFrame(columns=["wellbore_id", "reservoir_pressure_bar", "tvdss_m", "depth_basis"])
    fig, ax = plt.subplots()
    ppl.plot_fullwell(ax, 1, t, pd.DataFrame(), mud, hdr, cfg, pcfg, "RT")
    got = {ln.get_label(): (ln.get_color().lower(), ln.get_linestyle()) for ln in ax.get_lines()}
    for key in ("hydrostatic", "hydrostatic_max", "hydrostatic_min", "obg", "fg", "fg_min", "fg_max", "sh", "sh_min"):
        s = ps.line_style(key)
        assert got[s["label"]] == (s["color"].lower(), s["ls"]), key
    assert not any(ln.get_color().lower() == RED.lower() for ln in ax.get_lines())                    # no predicted pore pressure exists, so no red PP line may be drawn
    dark, light = int(ps.line_style("fg")["color"][1:], 16), int(ps.line_style("hydrostatic")["color"][1:], 16)
    assert dark < light                                                                            # FG is the dark blue, hydrostatic the light blue
    step = [ln for ln in ax.get_lines() if ln.get_label() == ps.line_style("mw")["label"]]
    assert step and step[0].get_color() == "#000000" and step[0].get_linestyle() == "-"
    plt.close(fig)