import pandas as pd
from ep_pressure.plots.pressure_plots import contact_regimes


def _t(rows):
    return pd.DataFrame(rows, columns=["wellbore_id", "tvdss_m", "contact_tvdss_m", "aquifer_pressure_bar", "buoyancy_status"])


def test_single_equivalent_is_single():
    assert contact_regimes(_t([(1, 2300, 2450, 250.0, "corrected")]))[0] == ("single", True)


def test_points_that_line_up_are_consistent():
    r = contact_regimes(_t([(1, 2300, 2450, 250.0, "corrected"), (1, 2350, 2450, 250.8, "corrected")]))
    assert r[0][0] == r[1][0] == "consistent"


def test_points_that_do_not_line_up_are_multi_and_deepest_is_flagged():
    r = contact_regimes(_t([(1, 2300, 2450, 250.0, "corrected"), (1, 2400, 2450, 262.0, "corrected")]))
    assert r[0] == ("multi", False) and r[1] == ("multi", True)
