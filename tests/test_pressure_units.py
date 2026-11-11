"""Pressure, specific gravity, overpressure and the datum; fluid-class colours; the three-view pressure-depth figure (docs/pore_pressure.md)."""
from __future__ import annotations


import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from ep_pressure_pvt_fluids import pressure as pr  # noqa: E402
from ep_physics_base import units as pu  # noqa: E402
from ep_pressure_pvt_fluids.plots import pressure_plots as pp  # noqa: E402


def test_pressure_and_sg_round_trip_and_the_unit_of_one_sg():
    assert abs(pu.BAR_PER_M_PER_SG - 0.0980665) < 1e-9
    p = pu.pressure_from_sg(1.5, 3000.0)
    assert abs(p - 441.2993) < 1e-3 and abs(float(pu.sg_from_pressure(p, 3000.0)) - 1.5) < 1e-12
    assert np.isnan(pu.sg_from_pressure(10.0, 0.0))


def test_the_datum_changes_the_equivalent_density_of_the_same_pressure():
    gw, rt, z = 0.1005, 25.0, 2000.0
    p = float(pu.hydrostatic_bar(z, gw))
    sg_msl, sg_rt = float(pu.emw_sg(p, z, rt, "MSL")), float(pu.emw_sg(p, z, rt, "RT"))
    assert abs(sg_msl - gw / pu.BAR_PER_M_PER_SG) < 1e-12                                   # 1.025 sg on the water-gradient basis
    assert abs(sg_rt - sg_msl * z / (z + rt)) < 1e-12 and 1.011 < sg_rt < 1.014            # about 1.012 on the rotary-table basis
    assert float(pu.tvd_rt_from_msl(z, rt)) == 2025.0 and float(pu.tvd_msl_from_rt(2025.0, rt)) == 2000.0


def test_overpressure_has_the_hydrostatic_gradient_as_zero_in_bar_and_sg():
    gw = 0.1005
    z = np.array([2000.0, 3000.0])
    p = pu.hydrostatic_bar(z, gw)
    assert np.allclose(pu.overpressure_bar(p, z, gw), 0.0) and np.allclose(pu.overpressure_sg(p, z, gw), 0.0)
    assert np.allclose(pu.overpressure_sg(p + 50.0, z, gw), 50.0 / (pu.BAR_PER_M_PER_SG * z))


def test_ecd_is_esd_plus_the_annular_loss_over_the_depth_and_inverts():
    esd, z, rt = 1.60, 3000.0, 25.0
    loss = 12.0                                                                          # bar
    ecd = float(pu.ecd_sg(esd, loss, z, rt))
    assert abs(ecd - (esd + loss / (pu.BAR_PER_M_PER_SG * 3025.0))) < 1e-12
    assert abs(float(pu.annular_loss_bar(ecd, esd, z, rt)) - loss) < 1e-9
    assert float(pu.ecd_sg(esd, loss, z, rt, back_pressure_bar=5.0)) > ecd
    # a mud column of the ESD has the pressure ESD x g x z_RT
    assert abs(float(pu.mud_pressure_bar(esd, z, rt)) - esd * pu.BAR_PER_M_PER_SG * 3025.0) < 1e-9


def test_iso_sg_lines_are_straight_through_the_datum():
    z = np.array([1000.0, 2000.0])
    a = pu.iso_sg_line(1.4, z, 25.0)
    assert np.allclose(a, 1.4 * pu.BAR_PER_M_PER_SG * (z + 25.0)) and abs((a[1] - a[0]) / 1000.0 - 1.4 * pu.BAR_PER_M_PER_SG) < 1e-12




def _frames():
    cfg = pr.load_config()
    t = pd.DataFrame({"wellbore_id": [1, 1], "test_number": ["1", "2"], "reservoir_pressure_bar": [250.0, 262.0], "tvdss_m": [2400.0, 2500.0], "depth_basis": "md_near_vertical",
                      "fluid_hint": "oil (GOR < 1000)", "aquifer_pressure_bar": [258.0, 266.5], "buoyancy_status": "corrected", "buoyancy_fluid": "oil", "column_below_test_m": [100.0, 60.0],
                      "contact_tvdss_m": [2500.0, 2560.0], "aquifer_ratio": [1.03, 1.06], "aquifer_regime": "normal", "temperature_c": [95.0, 98.0], "contact_basis": "x"})
    con = pd.DataFrame({"wellbore_id": [1], "label": ["OWC"], "tvdss_m": [2560.0], "contact_class": "oil_water"})
    lot = pd.DataFrame({"wellbore_id": [1], "lot_emw_sg": [1.6], "casing_depth_md_m": [2420.0], "test_type": ["LOT"]})
    mud = pd.DataFrame({"wellbore_id": [1, 1], "md_m": [2400.0, 2450.0], "mud_weight_sg": [1.3, 1.45], "mud_type": "WATER BASED"})
    hdr = pd.DataFrame({"wellbore_id": [1], "rt_elev_m": [25.0], "water_depth_m": [120.0], "td_md_m": [3000.0], "td_tvd_m": [2995.0], "main_area": "NORTH SEA"})
    env = pd.DataFrame({"fluid_key": "black_oil", "dz_m": [0.0, 600.0], "delta_p_min_bar": [0.0, -3.0], "delta_p_max_bar": [0.0, 1.0]})
    return cfg, t, con, lot, mud, hdr, env


def test_triplet_figure_draws_three_views_with_envelope_annotation_and_fan():
    import matplotlib.pyplot as plt

    cfg, t, con, lot, mud, hdr, env = _frames()
    (name, fig, s), = pp.triplet_figures([(1, "TEST-1")], t, con, lot, mud, hdr, cfg, env, {(1, "1"): "black oil", (1, "2"): "black oil"})
    assert s["tests"] == 2 and s["corrected"] == 2 and s["lot"] == 1 and s["mud_points"] == 2 and len(fig.axes) == 3
    texts = " ".join(tx.get_text() for ax in fig.axes for tx in ax.texts)
    assert "bar/m" in texts and "sg" in texts and "hydrostatic" in texts and "x hydrostatic" in texts
    for ax in fig.axes:                                                                    # depth increases downwards, hydrostatic drawn
        assert ax.get_ylim()[0] > ax.get_ylim()[1]
    plt.close(fig)


def test_envelope_deviation_interpolates_and_is_zero_without_a_proxy():
    _, _, _, _, _, _, env = _frames()
    lo, hi = pp.envelope_deviation(env, "black_oil", np.array([0.0, 300.0, 600.0]))
    assert np.allclose(lo, [0.0, -1.5, -3.0]) and np.allclose(hi, [0.0, 0.5, 1.0])
    z0, z1 = pp.envelope_deviation(env, "unknown", np.array([100.0]))
    assert z0[0] == 0.0 and z1[0] == 0.0
    cfg = pr.load_config()
    assert pp.proxy_fluid_key("volatile oil", "oil", 850.0, 190.0, cfg) == "hpht_volatile_oil" and pp.proxy_fluid_key(None, "gas", 300.0, 100.0, cfg) == "rich_gas_condensate"
