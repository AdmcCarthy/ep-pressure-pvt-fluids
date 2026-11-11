"""Overburden, fracture bounds and the drilling window (docs/pore_pressure.md)."""
from __future__ import annotations

import numpy as np

from ep_pressure import pore_pressure as pp
from ep_physics_base import units as pu
from ep_physics_base.config import load_yaml

CFG = load_yaml("config/pore_pressure.yaml")


def test_bulk_density_increases_with_depth_to_the_matrix_density():
    d = pp.bulk_density_g_cm3(np.array([0.0, 1000.0, 5000.0, 50000.0]), CFG)
    assert (np.diff(d) > 0).all() and d[0] < 2.0 and abs(d[-1] - 2.65) < 0.01


def test_overburden_is_water_plus_sediment_and_tracks_a_lithostatic_gradient():
    z = np.array([100.0, 2000.0, 4000.0])
    obg = pp.overburden_pressure_bar(z, 100.0, CFG)
    assert abs(obg[0] - 0.1005 * 100.0) < 1e-9                                      # at the seabed only the water column
    g = np.diff(obg) / np.diff(z)
    assert (g > 0.1005).all() and g[1] > g[0] and 0.18 < g[1] < 0.26               # sediments are heavier than water; the gradient rises with compaction
    deeper = pp.overburden_pressure_bar(z, 400.0, CFG)
    assert (deeper[1:] < obg[1:]).all()                                              # a deeper sea means less rock above the same depth: a lower overburden


def test_fracture_bounds_follow_the_minimum_stress_relations():
    obg, p = 700.0, 450.0
    s = float(pp.shmin_lower_bound_bar(obg, p, 0.25))
    assert abs(s - (0.25 / 0.75 * 250.0 + 450.0)) < 1e-9
    assert float(pp.fracture_upper_bound_bar(s, p)) == 2 * s - p
    band = pp.fracture_band_bar(np.array([obg]), np.array([p]), CFG)
    assert band["shmin_low_bar"].iloc[0] < band["shmin_mid_bar"].iloc[0] < band["shmin_high_bar"].iloc[0] < band["fg_upper_bar"].iloc[0]
    assert abs(band["fg_most_likely_bar"].iloc[0] - 0.5 * (band["shmin_mid_bar"].iloc[0] + band["fg_upper_bar"].iloc[0])) < 1e-9
    assert float(pp.shmin_lower_bound_bar(obg, obg, 0.3)) == obg                    # with the pore pressure at the overburden the minimum stress is the overburden


def test_leakage_line_is_100_bar_below_the_lithostatic_pressure():
    assert np.allclose(pp.leakage_line_bar(np.array([800.0, 900.0]), CFG), [700.0, 800.0])


def test_drilling_window_needs_pp_and_fg_and_marks_a_missing_shear_failure_gradient():
    w = pp.drilling_window(np.array([1.2, 1.5, 1.2, np.nan]), np.array([np.nan, 1.3, np.nan, 1.1]), np.array([1.6, 1.52, 1.1, 1.6]), margin_sg=0.05)
    assert list(w["status"]) == ["open", "narrow", "closed: the fracture gradient is at or below the lower limit", "undefined (pore pressure or fracture gradient missing)"]
    assert w["lower_is_minimum"].tolist() == [True, False, True, False]               # no SFG: the lower limit is the pore pressure only, a minimum
    assert abs(w["lower_sg"].iloc[1] - 1.5) < 1e-12 and abs(w["lower_sg"].iloc[0] - 1.2) < 1e-12 and abs(w["width_sg"].iloc[0] - 0.4) < 1e-12


def test_overbalance_sign():
    assert float(pp.overbalance(1.6, 1.4)) > 0 > float(pp.overbalance(1.3, 1.4))
    assert abs(float(pu.mud_pressure_bar(1.6, 3000.0, 25.0)) - 1.6 * pu.BAR_PER_M_PER_SG * 3025.0) < 1e-9

def test_a_registered_overlay_is_applied_when_none_is_passed_and_removed_again():
    base = pp.load_config()
    marker = base["overburden"]["matrix_density_g_cm3"] + 0.01

    def overlay(cfg):
        cfg["overburden"]["matrix_density_g_cm3"] = marker
        return cfg

    pp.register_overlay(overlay)
    try:
        assert pp.load_config()["overburden"]["matrix_density_g_cm3"] == marker
    finally:
        pp.register_overlay(None)
    assert pp.load_config()["overburden"]["matrix_density_g_cm3"] == base["overburden"]["matrix_density_g_cm3"]
