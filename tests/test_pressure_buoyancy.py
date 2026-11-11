"""Buoyancy correction, contacts as TVDSS, free water level from gradient crossings, seal helpers (docs/pressure.md section 8)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ep_pressure import pressure as pr
from ep_pressure import pressure_gradients as pg

CFG = pr.load_config()
GW = CFG["hydrostatic"]["gradient_bar_per_m"]


def _tests(rows):
    base = {"wellbore_id": 1, "area": "NORTH SEA", "flags": pd.NA, "pressure_regime": "normal", "overpressure_bar": 0.0}
    return pd.DataFrame([{**base, **r} for r in rows])


def _contacts(*rows):
    return pd.DataFrame([{"wellbore_id": 1, "confidence": "medium", "evidence": "text", "basis": "stated TVDSS", **r} for r in rows])


def test_oil_test_above_an_owc_is_translated_to_the_contact():
    z, col = 2000.0, 100.0
    g_oil = 0.075
    p_contact = GW * (z + col)                                    # an aquifer at hydrostatic pressure
    p_test = p_contact - g_oil * col                              # the oil column puts 7.5 bar less on its own depth than the water would
    t = _tests([{"tvdss_m": z, "reservoir_pressure_bar": p_test, "fluid_hint": "oil (GOR < 1000)", "pressure_ratio": p_test / (GW * z), "overpressure_bar": p_test - GW * z}])
    c = _contacts({"label": "OWC", "contact_class": "oil_water", "tvdss_m": z + col})
    out, _ = pr.add_buoyancy_correction(t, c, None, CFG)
    r = out.iloc[0]
    assert r["buoyancy_status"] == "corrected" and abs(r["aquifer_pressure_bar"] - p_contact) < 1e-6 and abs(r["aquifer_ratio"] - 1.0) < 1e-6
    assert abs(r["buoyancy_excess_bar"] - (GW - g_oil) * col) < 1e-6 and r["buoyancy_excess_low_bar"] < r["buoyancy_excess_bar"] < r["buoyancy_excess_high_bar"]
    assert abs(r["overpressure_beyond_column_bar"] - (p_test - GW * z - (GW - g_oil) * col)) < 1e-6      # = the raw excess minus the column effect (here a normal aquifer)


def test_gas_over_oil_is_two_segments_and_gas_cap_only_is_a_minimum():
    z = 2000.0
    t = _tests([{"tvdss_m": z, "reservoir_pressure_bar": 220.0, "fluid_hint": "gas or condensate (GOR >= 1000)", "pressure_ratio": 1.09}])
    both = _contacts({"label": "GOC", "contact_class": "gas_oil", "tvdss_m": 2050.0}, {"label": "OWC", "contact_class": "oil_water", "tvdss_m": 2100.0})
    out, _ = pr.add_buoyancy_correction(t, both, None, CFG)
    r = out.iloc[0]
    expected = 220.0 + 0.025 * 50 + 0.075 * 50
    assert abs(r["aquifer_pressure_bar"] - expected) < 1e-6 and r["buoyancy_contacts"] == "GOC+OWC" and not r["buoyancy_is_minimum"]
    only_goc, _ = pr.add_buoyancy_correction(t, both.iloc[[0]], None, CFG)
    assert only_goc.iloc[0]["buoyancy_is_minimum"] and "minimum" in only_goc.iloc[0]["buoyancy_status"] or only_goc.iloc[0]["buoyancy_is_minimum"]


def test_no_contact_below_the_test_is_not_guessed_and_is_flagged():
    t = _tests([{"tvdss_m": 2000.0, "reservoir_pressure_bar": 230.0, "fluid_hint": "oil (GOR < 1000)", "pressure_ratio": 1.14},
                {"tvdss_m": 2000.0, "reservoir_pressure_bar": 230.0, "fluid_hint": "oil (GOR < 1000)", "pressure_ratio": 1.14, "wellbore_id": 2}])
    c = _contacts({"label": "OWC", "contact_class": "oil_water", "tvdss_m": 1900.0}, {"label": "OWC", "contact_class": "oil_water", "tvdss_m": 2900.0})            # one above, one 900 m below: another accumulation
    out, _ = pr.add_buoyancy_correction(t, c, None, CFG)
    assert out["buoyancy_status"].tolist() == ["nearest contact below is further than the column window", "no contact on record"]
    above, _ = pr.add_buoyancy_correction(t.iloc[[0]], c.iloc[[0]], None, CFG)
    assert above.iloc[0]["buoyancy_status"] == "test below the stated contact (no column below)"
    assert out["aquifer_pressure_bar"].isna().all() and "buoyancy_unknown_contact" in str(out.iloc[1]["flags"])


def test_gradients_follow_the_fluid_not_the_area():
    from ep_fluids import fluid_gradients as fg

    c, low, high, basis = fg.gas_gradient(300.0, 100.0, 0.70, CFG)
    assert 0.019 < c < 0.025 and low < c < high and "real-gas" in basis                                 # rho = p M / (Z R T): about 0.22 g/cm3 for a 0.7 gas at 300 bar and 100 degC
    assert fg.gas_gradient(300.0, 100.0, 0.90, CFG)[0] > c and fg.gas_gradient(500.0, 100.0, 0.70, CFG)[0] > c      # heavier gas, higher pressure: denser
    oil, _, _, ob = fg.oil_gradient(0.85, 100.0, 0.75, "black oil", CFG)
    assert abs(oil - (850.0 + 100.0 * 0.75 * fg.RHO_AIR_STD) / 1.20 * 9.80665 / 1e5) < 1e-9 and "mass balance" in ob
    assert fg.oil_gradient(0.80, 300.0, 0.75, "volatile oil", CFG)[0] < fg.oil_gradient(0.80, 300.0, 0.75, "black oil", CFG)[0]      # a volatile oil is lighter in situ
    assert fg.oil_gradient(8.77, 100.0, 0.75, "black oil", CFG)[3].startswith("configured band")              # an implausible density is not used
    assert abs(fg.RHO_AIR_STD - 1.225) < 0.005


def test_water_gradient_depends_on_salinity():
    from ep_fluids import fluid_gradients as fg

    fresh, brine = fg.water_gradient(0.0, CFG)[0], fg.water_gradient(88000.0, CFG)[0]
    assert abs(fresh - 0.0980) < 1e-9 and abs(brine - 0.1052) < 1e-9 and fg.water_gradient(55000.0, CFG)[0] == 0.1018
    assert fg.water_gradient(200000.0, CFG)[0] == brine                                                      # not extrapolated
    assert abs(fg.salinity_from_gradient(0.1018, CFG) - 55000.0) < 1e-6
    assert fg.water_gradient(None, CFG)[0] == CFG["hydrostatic"]["gradient_bar_per_m"]


def test_contacts_are_converted_to_tvdss_and_never_fabricated():
    facts = pd.DataFrame({"wellbore_id": [1, 1, 2, 3], "fact_type": "contact", "label": ["OWC", "GOC", "OWC", "ODT"], "value": [2912.0, np.nan, np.nan, np.nan], "unit": ["m TVD", None, None, None],
                          "depth_top_m": [2954.0, 3000.0, 3500.0, 2800.0], "depth_reference": ["MD RT", "MSL", "MD RT", "MD RT"], "confidence": "medium", "evidence": "t"})
    header = pd.DataFrame({"wellbore_id": [1, 2, 3], "rt_elev_m": [25.0, 25.0, 25.0], "water_depth_m": 100.0, "td_md_m": [4000.0, 4000.0, 4000.0], "td_tvd_m": [3990.0, 2500.0, np.nan], "main_area": "NORTH SEA"})
    c = pr.well_contacts_tvdss(facts, header, None, CFG).set_index(["wellbore_id", "label"])
    assert c.loc[(1, "OWC"), "tvdss_m"] == 2912.0 - 25.0 and c.loc[(1, "OWC"), "basis"].startswith("stated TVD")
    assert c.loc[(1, "GOC"), "tvdss_m"] == 3000.0                                                  # a depth below MSL is TVDSS
    assert (2, "OWC") not in c.index and (3, "ODT") not in c.index                                 # deviated / unknown TVD: no conversion, no guess


def test_free_water_level_is_where_the_gradients_cross():
    z_oil = np.array([2000.0, 2030.0, 2060.0, 2090.0])
    z_wat = np.array([2200.0, 2260.0, 2320.0])
    fwl = 2150.0
    p_fwl = GW * fwl
    pts = pd.concat([pd.DataFrame({"tvdss_m": z_oil, "pressure_bar": p_fwl + 0.075 * (z_oil - fwl), "fluid": "oil"}),
                     pd.DataFrame({"tvdss_m": z_wat, "pressure_bar": p_fwl + GW * (z_wat - fwl), "fluid": "water"})], ignore_index=True)
    r = pg.free_water_level(pts, CFG).iloc[0]
    assert r["level"].startswith("free water level") and abs(r["depth_tvdss_m"] - fwl) < 1e-6 and r["status"] == "ok"
    assert abs(r["light_density_g_cm3"] - 0.075 / 0.0980665) < 1e-6
    one_water = pts[pts["fluid"].eq("oil") | (pts.index == len(z_oil))]
    r1 = pg.free_water_level(one_water, CFG).iloc[0]
    assert abs(r1["depth_tvdss_m"] - fwl) < 1.0 and "assumed water gradient" in r1["method"]
    wrong = pts.assign(pressure_bar=np.where(pts["fluid"] == "oil", p_fwl + 0.12 * (pts["tvdss_m"] - fwl), pts["pressure_bar"]))              # oil heavier than water: no crossing
    assert "do not cross" in pg.free_water_level(wrong, CFG).iloc[0]["status"]


def test_seal_capacity_and_contact_above_fwl():
    assert abs(pg.max_column_height_m(5.0, 0.1005, 0.075) - 5.0 / 0.0255) < 1e-9
    assert abs(pg.contact_above_fwl_m(1.0, 0.1005, 0.075) - 1.0 / 0.0255) < 1e-9
