"""Shoe tests against the screening band of the fracture pressure, and the pressure as a share of the modelled overburden (src/ep_pressure_pvt_fluids/pore_pressure.py, pressure.py)."""
import numpy as np
import pandas as pd

from ep_pressure_pvt_fluids import pore_pressure as pp
from ep_pressure_pvt_fluids import pressure as pr

CFG = pp.load_config()
GW = 0.1005                                                                  # hydrostatic gradient [bar/m]


def _header():
    return pd.DataFrame({"wellbore_id": [1, 2, 3], "rt_elev_m": [25.0, 25.0, np.nan], "water_depth_m": [100.0, 100.0, 100.0]})


def _lot(emw, kind="LOT", wellbore=1, md=2025.0):
    return pd.DataFrame({"wellbore_id": [wellbore], "well_name": "w", "casing_depth_md_m": [md], "test_type": [kind], "lot_emw_sg": [emw]})


def test_a_test_inside_the_band_is_inside_and_one_far_above_it_is_above_and_the_value_is_unchanged():
    inside = pp.fracture_band_for_shoes(_lot(1.65), _header(), CFG, GW).iloc[0]
    assert inside["band_position"] == "inside" and inside["band_low_bar"] < inside["test_pressure_bar"] < inside["band_high_bar"]
    above = pp.fracture_band_for_shoes(_lot(2.6), _header(), CFG, GW).iloc[0]
    assert above["band_position"] == "above" and above["lot_emw_sg"] == 2.6 and above["depth_basis"] == "md_as_tvd"


def test_a_fit_below_the_band_is_expected_and_a_leak_off_below_it_is_not():
    fit = pp.fracture_band_for_shoes(_lot(1.05, "FIT"), _header(), CFG, GW).iloc[0]
    lot = pp.fracture_band_for_shoes(_lot(1.05, "LOT"), _header(), CFG, GW).iloc[0]
    assert fit["band_position"] == lot["band_position"] == "below"
    assert "integrity test" in fit["band_note"] and "integrity test" not in lot["band_note"]


def test_a_shoe_without_an_elevation_or_close_to_the_seabed_is_not_testable_and_not_dropped():
    out = pp.fracture_band_for_shoes(pd.concat([_lot(1.6, wellbore=3), _lot(1.6, md=150.0)], ignore_index=True), _header(), CFG, GW)
    assert list(out["band_position"]) == ["not_testable", "not_testable"] and out["band_low_bar"].isna().all()


def test_the_share_of_the_overburden_rises_with_pressure_and_a_share_above_the_limit_is_flagged_not_corrected():
    z, wd = np.array([2000.0, 2000.0, np.nan]), 100.0
    obg = float(pp.overburden_pressure_bar([2000.0], wd, CFG)[0])
    out = pp.overburden_share(np.array([0.5 * obg, 0.99 * obg, 200.0]), z, wd, CFG)
    assert abs(out["overburden_share"].iloc[0] - 0.5) < 1e-9 and list(out["above_upper_limit"]) == [False, True, False] and np.isnan(out["overburden_share"].iloc[2])


def test_the_share_on_a_deviated_well_is_a_minimum_and_existing_columns_are_not_changed():
    pressure = pd.DataFrame({"wellbore_id": [1, 2], "well_name": ["a", "b"], "reservoir_pressure_bar": [250.0, 250.0], "tvdss_m": [2000.0, 2000.0], "depth_basis": ["md_near_vertical", "md_deviated"]})
    out, events = pr.add_overburden_share(pressure, _header(), CFG)
    assert list(out["overburden_share_is_minimum"]) == [False, True] and out["reservoir_pressure_bar"].tolist() == [250.0, 250.0]
    assert out["overburden_share"].between(0.3, 0.9).all() and any(e["rule"] == "overburden_share" for e in events)
