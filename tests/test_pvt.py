"""PVT: API, fluid class, GLR-pressure zones (forbidden, diphasic bands, monophasic, critical band), saturation state, Standing envelope, biodegradation brackets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ep_pressure_pvt_fluids import pvt

CFG = pvt.load_config()


def s(*values) -> pd.Series:
    return pd.Series(values, dtype="float64")


def test_api_from_density_and_plausible_range():
    api = pvt.api_from_density(s(0.85, 0.9, 5.0), CFG)
    assert abs(api[0] - 34.8) < 0.1 and abs(api[1] - 25.7) < 0.2         # 0.85 g/cm3 is about 35 API (SG = density / 0.999)
    assert np.isnan(api[2])                                              # a density of 5 is nonsense: kept in the source, not classified


def test_standing_envelope_rises_with_pressure_and_is_inverted_by_the_bubble_point():
    p = s(100, 200, 400)
    limit = pvt.standing_max_gor(p, s(90, 90, 90), s(35, 35, 35), s(0.75, 0.75, 0.75), CFG)
    assert limit.is_monotonic_increasing and 80 < limit[1] < 200          # about 125 Sm3/Sm3 at 200 bar for a 35 API oil at 90 degC
    back = pvt.standing_bubble_point(limit, s(90, 90, 90), s(35, 35, 35), s(0.75, 0.75, 0.75), CFG)
    assert np.allclose(back, p, rtol=1e-6)                                # the two directions agree


def classify(glr, pressure, t=100.0, api=35.0, gg=0.75):
    """classify_domain for tests of one reference fluid."""
    g, p = s(*glr), s(*pressure)
    ones = pd.Series(1.0, index=g.index)
    oil = pvt.standing_max_gor(p, ones * t, ones * api, ones * gg, CFG)
    gas = pvt.gas_min_glr(p, CFG)
    p_crit = pvt.critical_pressure(ones * t, ones * api, ones * gg, CFG)
    return pvt.classify_domain(g, p, oil, gas, p_crit, CFG), oil, gas, p_crit


def test_forbidden_domain_200_bar_glr_500_is_forbidden_not_a_single_phase():
    d, oil, gas, _ = classify([500.0], [200.0], t=90.0)
    assert d.loc[0, "domain"] == "forbidden" and d.loc[0, "saturation_state"] == "forbidden"
    assert oil[0] < 500 < gas[0] or oil[0] < 500                          # far above what an oil dissolves at 200 bar (about 125)


def test_monophasic_liquid_gas_and_the_diphasic_bands_beneath_the_forbidden_domain():
    p = [300.0] * 4
    _, oil, gas, _ = classify([100.0] * 4, p)
    glr = [oil[0] * 0.5, oil[0] * 0.9, gas[0] * 1.1, gas[0] * 3]
    d, *_ = classify(glr, p)
    assert list(d["saturation_state"]) == ["undersaturated oil", "saturated oil", "saturated gas", "undersaturated gas"]
    assert list(d["domain"]) == ["monophasic liquid", "diphasic", "diphasic", "monophasic gas"]          # saturated = at the saturation pressure: oil with gas cap / gas with oil ring
    above, *_ = classify([oil[0] * 1.5, gas[0] * 0.5], [300.0, 300.0])
    assert list(above["domain"]) == ["forbidden", "forbidden"]            # beyond the limits at that pressure


def test_critical_band_is_single_phase_only_above_the_critical_pressure():
    d, _, _, p_crit = classify([800.0, 800.0, 800.0], [300.0, 900.0, 5000.0])
    assert 400 < p_crit[0] < 1000                                         # the U closes inside the practical pressure range
    assert list(d["saturation_state"]) == ["forbidden", "critical fluid", "critical fluid"]
    assert list(d["domain"]) == ["forbidden", "critical fluid", "critical fluid"]


def test_critical_band_edges_follow_the_configuration():
    low, high = CFG["critical_band"]["glr_sm3_sm3"]
    d, *_ = classify([low - 1, low, high, high + 1], [3000.0] * 4)        # very high pressure: only the band itself is critical
    assert list(d["saturation_state"]) == ["undersaturated oil", "critical fluid", "critical fluid", "undersaturated gas"]


def test_missing_inputs_give_missing_domain_never_a_guess():
    d, *_ = classify([np.nan, 100.0, 100.0], [300.0, np.nan, 300.0])
    assert d["domain"].iloc[:2].isna().all() and d["domain"].iloc[2] == "monophasic liquid"


def test_zone_polygons_are_flat_disjoint_zones_that_agree_with_the_classification():
    from shapely.geometry import Point, Polygon
    grid = np.arange(0.0, 1000.5, 0.5)
    zones = pvt.zone_polygons(grid, 100.0, 35.0, 0.75, CFG)
    assert set(zones) == set(pvt.ZONES)
    polys = {name: Polygon(np.c_[x, y]).buffer(0) for name, (x, y) in zones.items()}
    rng = np.random.default_rng(1)
    glr, p = 10 ** rng.uniform(1.05, 5.9, 600), rng.uniform(5, 990, 600)
    d, *_ = classify(list(glr), list(p))
    want = d["saturation_state"].map({"undersaturated oil": "monophasic: undersaturated oil", "saturated oil": "diphasic: oil with gas cap", "forbidden": "forbidden", "critical fluid": "monophasic: critical fluid",
                                      "saturated gas": "diphasic: gas with oil ring", "undersaturated gas": "monophasic: undersaturated gas"}).to_numpy()
    hit = [[z for z, poly in polys.items() if poly.contains(Point(g, q))] for g, q in zip(glr, p)]
    assert sum(len(h) > 1 for h in hit) <= 3                              # zones do not overlap (boundary slivers only)
    wrong = sum(1 for h, w in zip(hit, want) if h and w not in h)
    assert wrong / len(hit) < 0.02                                        # the picture and the classification agree

def test_biodegradation_brackets_and_uplift():
    out = pvt.classify_biodegradation(s(50, 70, 95, 50, 95, 100), s(18, 20, 18, 38, 38, np.nan),
                                      pd.Series(["NORTH SEA"] * 5 + ["BARENTS SEA"], dtype="string"), CFG)
    assert list(out["temperature_bracket"]) == ["degradable", "transition", "pasteurised", "degradable", "pasteurised", "pasteurised"]
    text = out["biodegradation_assessment"].astype("string")
    assert text[0].startswith("likely biodegraded") and text[1].startswith("possibly") and text[2].startswith("heavy but hot")
    assert "pasteurised in the past" in text[3] and "consistent with pasteurisation" in text[4] and pd.isna(text[5])
    assert list(out["uplift_area"]) == [False] * 5 + [True]


def test_fluid_classes_follow_mccain_gor_limits():
    c = pvt._band(s(100, 400, 1000, 5000, 50000), CFG["fluid_class_by_gor_sm3_sm3"])
    assert list(c) == ["black oil", "volatile oil", "gas condensate", "wet gas", "dry gas"]


def _dst_and_pressure(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pressure = pd.DataFrame([{"wellbore_id": 1, "well_name": "A", "test_number": str(i), "area": "NORTH SEA", "mid_md_m": 3000.0, "reservoir_pressure_bar": r["p"], "temperature_c": r["t"]} for i, r in enumerate(rows)]).astype({"wellbore_id": "Int64"})
    dst = pd.DataFrame([{"wellbore_id": 1, "test_number": str(i), "gor_sm3_sm3": r["gor"], "oil_density_g_cm3": r["rho"], "gas_relative_density": r["gg"]} for i, r in enumerate(rows)]).astype({"wellbore_id": "Int64"})
    return pressure, dst


def test_build_well_dst_pvt_end_to_end_with_reference_fluid_label():
    pressure, dst = _dst_and_pressure([{"p": 200.0, "t": 90.0, "gor": 500.0, "rho": 0.85, "gg": 0.75},       # the forbidden example
                                       {"p": 350.0, "t": 110.0, "gor": 100.0, "rho": 0.85, "gg": 0.75},      # undersaturated oil
                                       {"p": 250.0, "t": np.nan, "gor": 8000.0, "rho": np.nan, "gg": np.nan},  # gas test without fluid properties: reference envelope
                                       {"p": 250.0, "t": 90.0, "gor": 0.0, "rho": 0.85, "gg": 0.75}])        # no GOR measured: not in the mart
    out, events = pvt.build_well_dst_pvt(pressure, dst, CFG)
    assert len(out) == 3 and list(out["domain"].iloc[:2]) == ["forbidden", "monophasic liquid"]
    assert out.loc[0, "flags"].startswith("forbidden_domain_no_single_phase_fluid") and out.loc[1, "saturation_state"] == "undersaturated oil"
    assert out.loc[2, "envelope_basis"] in ("reference", "partly reference") and "envelope_uses_reference_fluid" in out.loc[2, "flags"]
    assert out.loc[1, "pressure_to_bubble_point"] > 1.0 and pd.isna(out.loc[2, "bubble_point_est_bar"])
    assert any(e["rule"] == "pvt_forbidden_tests" for e in events)


def _reports(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).astype({"wellbore_id": "Int64"})




def test_stated_pressures_decide_undersaturated_saturated_and_forbidden_for_oil_and_gas():
    pressure, dst = _dst_and_pressure([{"p": 400.0, "t": 110.0, "gor": 100.0, "rho": 0.85, "gg": 0.75}, {"p": 300.0, "t": 110.0, "gor": 100.0, "rho": 0.85, "gg": 0.75},
                                       {"p": 400.0, "t": 110.0, "gor": 5000.0, "rho": 0.78, "gg": 0.75}, {"p": 300.0, "t": 110.0, "gor": 5000.0, "rho": 0.78, "gg": 0.75}])
    pressure["mid_md_m"] = [3000.0, 3200.0, 3400.0, 3600.0]
    rep = _reports([{"wellbore_id": 1, "md_m": 3000.0, "bubble_point_bar": 250.0, "source": "x"}, {"wellbore_id": 1, "md_m": 3200.0, "bubble_point_bar": 350.0, "source": "x"},     # 300 bar below Pb 350: forbidden
                    {"wellbore_id": 1, "md_m": 3400.0, "dew_point_bar": 300.0, "source": "x"}, {"wellbore_id": 1, "md_m": 3600.0, "dew_point_bar": 303.0, "source": "x"}])           # 300 vs 303: saturated gas
    out, _ = pvt.build_well_dst_pvt(pressure, dst, CFG, rep)
    assert list(out["saturation_state"]) == ["undersaturated oil", "forbidden", "undersaturated gas", "saturated gas"]
    assert out.loc[1, "domain"] == "forbidden" and "forbidden_domain_no_single_phase_fluid" in out.loc[1, "flags"]


def test_report_without_a_matching_test_is_logged_and_official_api_is_kept():
    pressure, dst = _dst_and_pressure([{"p": 350.0, "t": 110.0, "gor": 100.0, "rho": 0.85, "gg": 0.75}])
    rep = _reports([{"wellbore_id": 1, "md_m": 3000.0, "api": 20.0, "source": "x"}, {"wellbore_id": 7, "md_m": 100.0, "bubble_point_bar": 200.0, "source": "x"}])
    out, events = pvt.build_well_dst_pvt(pressure, dst, CFG, rep)
    assert abs(out.loc[0, "api"] - 34.8) < 0.1 and out.loc[0, "api_source"] == "official_dst"                     # official wins; the disagreement is logged
    assert any(e["rule"] == "pvt_report_conflict" for e in events) and any(e["rule"] == "pvt_report_unmatched" for e in events)


def test_report_api_fills_when_the_dst_has_no_density():
    pressure, dst = _dst_and_pressure([{"p": 350.0, "t": 110.0, "gor": 100.0, "rho": np.nan, "gg": 0.75}])
    out, _ = pvt.build_well_dst_pvt(pressure, dst, CFG, _reports([{"wellbore_id": 1, "md_m": 3000.0, "api": 28.0, "source": "x"}]))
    assert out.loc[0, "api"] == 28.0 and out.loc[0, "api_source"] == "pvt_report" and out.loc[0, "api_class"] == "medium"


def test_dew_point_branch_and_critical_pressure_follow_the_reference_chart():
    """The configured dew-point curve and the reference fluid reproduce the digitised branches of the reference chart (config/pvt.yaml `calibration`)."""
    cal, ref = CFG["calibration"], CFG["reference"]
    one = lambda v: pd.Series([float(v)])
    for glr, bar in cal["chart_right_branch"]:
        assert abs(float(pvt.gas_dew_pressure(one(glr), CFG).iloc[0]) - bar) <= 4          # the right branch is the table itself: a few bar
    for glr, bar in cal["chart_left_branch"][:-1]:
        model = float(pvt.bubble_point_estimate(one(glr), one(ref["temperature_c"]), one(ref["api"]), one(ref["gas_gravity"]), CFG).iloc[0])
        assert abs(model / bar - 1) < 0.20                                                  # the default branch (Glaso) is within 20 % of the chart in pressure (reference fluid 40 API, 55 degC, 1.00): -16 to +17 %
        old = float(pvt.standing_bubble_point(one(glr), one(ref["temperature_c"]), one(ref["api"]), one(ref["gas_gravity"]), CFG).iloc[0])
        assert abs(old / bar - 1) < 0.35                                                    # Standing, the earlier default, is within 35 % (27-31 % low in the mid-range)
    p_crit = float(pvt.critical_pressure(one(ref["temperature_c"]), one(ref["api"]), one(ref["gas_gravity"]), CFG).iloc[0])
    assert abs(p_crit / cal["minimum_of_u"]["pressure_bar"] - 1) < 0.10                     # the U closes within 10 % of the chart's critical pressure
    low, high = CFG["critical_band"]["glr_sm3_sm3"]
    assert low <= cal["minimum_of_u"]["glr_sm3_sm3"] <= high                                 # and its minimum lies inside the critical band


def test_estimated_temperature_feeds_the_envelope_only_when_allowed_and_is_labelled():
    pressure, dst = _dst_and_pressure([{"p": 350.0, "t": np.nan, "gor": 100.0, "rho": 0.85, "gg": 0.75},       # no gauge temperature, an estimate exists
                                       {"p": 350.0, "t": 110.0, "gor": 100.0, "rho": 0.85, "gg": 0.75}])      # gauge temperature: the estimate is not used
    pressure["temperature_est_c"], pressure["temperature_est_basis"] = [60.0, 70.0], ["well_gradient:bht_uncorrected", "well_gradient:dst_gauge"]
    on, _ = pvt.build_well_dst_pvt(pressure, dst, CFG, use_estimated_temperature=True)
    off, _ = pvt.build_well_dst_pvt(pressure, dst, CFG, use_estimated_temperature=False)
    assert list(on["temperature_used_c"]) == [60.0, 110.0] and list(on["temperature_is_estimated"]) == [True, False]          # a gauge value always wins
    assert on.loc[0, "envelope_basis"] == "test (estimated temperature)" and on.loc[1, "envelope_basis"] == "test"
    assert "temperature_estimated" in on.loc[0, "flags"] and pd.isna(on.loc[1, "flags"])
    assert not off.loc[0, "temperature_is_estimated"] and off.loc[0, "envelope_basis"] == "partly reference" and pd.isna(off.loc[0, "temperature_used_c"])    # switched off: the reference temperature is used and labelled
    assert on.loc[0, "oil_limit_glr"] != off.loc[0, "oil_limit_glr"]                                                          # the estimate changed the envelope


def test_tests_are_placed_in_the_play_that_contains_their_depth():
    tests = pd.DataFrame({"wellbore_id": pd.array([1, 1, 1, 2], dtype="Int64"), "mid_md_m": [2500.0, 3500.0, 9000.0, 100.0]})
    wp = pd.DataFrame({"wellbore_id": pd.array([1, 1, 1], dtype="Int64"), "play_id": ["a", "b", "wide"], "top_md_m": [2000.0, 3000.0, 1000.0], "base_md_m": [2900.0, 4000.0, 5000.0]})
    play = pvt.place_tests_in_plays(tests, wp)
    assert list(play.iloc[:2]) == ["a", "b"]                                                # the shortest interval wins over the wide one
    assert play.iloc[2:].isna().all()                                                       # no interval contains 9000 m, and well 2 has no play


def test_play_summary_blanks_medians_below_five_values():
    pressure, dst = _dst_and_pressure([{"p": 350.0, "t": 110.0, "gor": 100.0, "rho": 0.85, "gg": 0.75}] * 2)
    out, _ = pvt.build_well_dst_pvt(pressure, dst, CFG)
    wp = pd.DataFrame({"wellbore_id": pd.array([1], dtype="Int64"), "play_id": ["x"], "top_md_m": [2900.0], "base_md_m": [3100.0]})
    summary, _ = pvt.build_play_pvt_summary(out, wp)
    assert summary.loc[0, "n_tests"] == 2 and pd.isna(summary.loc[0, "median_api"])
    assert summary.loc[0, "n_undersaturated_oil"] == 2


def test_forbidden_glr_factor_is_the_distance_to_the_nearest_allowed_side_and_classifies_the_gor_error():
    state, oil, gas, _ = classify([300, 80, 5000], [200, 200, 200])                  # at 200 bar the Standing limit is ~125 Sm3/Sm3 and the gas limit several thousand
    assert state["domain"].tolist() == ["forbidden", "monophasic liquid", "forbidden"]
    band = tuple(CFG["critical_band"]["glr_sm3_sm3"])
    factor = pvt.forbidden_glr_factor(s(300, 80, 5000), oil, gas, state["domain"], band)
    assert abs(factor.iloc[0] - 300 / oil.iloc[0]) < 1e-9 and abs(factor.iloc[2] - gas.iloc[2] / 5000) < 1e-9 and (factor.dropna() > 1).all()
    assert np.isnan(factor.iloc[1])                                                   # not forbidden: no factor
    inside, o2, g2, pc = classify([700], [100])                                       # in the critical band below the critical pressure: forbidden although the GLR may be under the oil limit
    assert inside["domain"].iloc[0] == "forbidden" and pvt.forbidden_glr_factor(s(700), o2, g2, inside["domain"], band).iloc[0] >= 1.0
    out = pvt.classify_forbidden_resolution(pd.Series([1.04, 1.20, 1.60, np.nan]), CFG)
    assert out.tolist()[:3] == pvt.FORBIDDEN_RESOLUTIONS and pd.isna(out.iloc[3])    # +-5 % typical, +-25 % worst case (Moffatt & Williams 1998, Table 4)
    assert CFG["gor_uncertainty"]["knowledge_entry"] == "mw98-gor-error-bounds"


def test_glaso_reproduces_the_papers_table_9_and_inverts_and_the_switch_follows_the_config():
    to_sm3 = pvt._factors()[1]
    # Glaso (1980) Table 8 / 9, North Sea samples 1, 2 and 5: (GOR scf/STB, gas gravity, API, T degF, estimated Pb psig of the paper)
    for gor, gg, api, t_f, pb_paper in ((1409, 0.799, 42.5, 260, 4615), (756, 0.759, 35.1, 230, 3527), (1623, 0.761, 39.0, 260, 5613)):
        pb_bar = pvt.glaso_bubble_point(s(gor * to_sm3), s((t_f - 32) / 1.8), s(api), s(gg), CFG)
        assert abs(pb_bar[0] * pvt._factors()[0] - 14.7 - pb_paper) / pb_paper < 0.01          # within 1 % of the paper (it rounds, and reads psig where the text says psia)
    p = s(100, 250, 400)
    ones = pd.Series(1.0, index=p.index)
    limit = pvt.glaso_max_gor(p, ones * 55, ones * 40, ones * 1.0, CFG)
    assert limit.is_monotonic_increasing and np.allclose(pvt.glaso_bubble_point(limit, ones * 55, ones * 40, ones * 1.0, CFG), p, rtol=1e-6)      # the two directions agree
    standing_cfg = {**CFG, "left_branch": {"correlation": "standing"}}
    assert pvt.left_branch_name(CFG) == "glaso" and pvt.left_branch_name(standing_cfg) == "standing"                  # Glaso is the default (Adam, 2026-11-03)
    assert np.allclose(pvt.oil_limit_glr(p, ones * 55, ones * 40, ones * 1.0, CFG), limit) and not np.allclose(pvt.oil_limit_glr(p, ones * 55, ones * 40, ones * 1.0, standing_cfg), limit)
    try:
        pvt.left_branch_name({**CFG, "left_branch": {"correlation": "lasater"}})
        raise AssertionError("an unknown correlation must be refused")
    except ValueError:
        pass
