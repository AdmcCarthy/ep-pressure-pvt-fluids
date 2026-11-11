"""Cubic EOS for mixtures and compositional grading of a column (docs/column_models.md)."""
from __future__ import annotations

import numpy as np

from ep_pressure_pvt_fluids import column_grading as cg
from ep_pressure_pvt_fluids import eos_mixture as em
from ep_physics_base.config import load_yaml

CFG = load_yaml("config/column_models.yaml")
PR = em.mixture_from_config(CFG, "PR", True)


def _x(key: str) -> np.ndarray:
    x = np.array(CFG["fluids"][key]["composition"], dtype=float)
    return x / x.sum()


def _pure(eos: str = "PR", shift: bool = True) -> em.Mixture:
    return em.Mixture(["methane"], np.array([190.56]), np.array([45.99]), np.array([0.011]), np.array([em.molar_mass_g_mol(1)]), eos, None, shift)


def test_methane_is_an_ideal_gas_at_low_pressure_and_molar_mass_is_right():
    m = _pure()
    _, z = em.ln_phi(m, np.array([1.0]), 300.0, 1.0)
    assert abs(z - 1.0) < 0.005
    assert abs(em.molar_mass_g_mol(1) - 16.043) < 0.01 and abs(em.molar_mass_g_mol(10) - 142.29) < 0.05
    for eos in ("PR", "SRK"):
        assert em.EOS_FORMS[eos]["u"] in (1.0, 2.0)


def test_pure_component_column_has_constant_composition_and_dp_dz_equals_rho_g():
    m = _pure()
    pr = cg.grade_column(m, np.array([1.0]), 100.0, 300.0, np.array([0.0, 50.0, 100.0, 200.0]), "isothermal", step_m=10.0, saturation=False)
    assert pr["converged"].all()
    d = pr.sort_values("dz_m")
    mid_rho = 0.5 * (d["rho_kg_m3"].iloc[0] + d["rho_kg_m3"].iloc[1])
    assert abs((d["p_bar"].iloc[1] - d["p_bar"].iloc[0]) - mid_rho * em.G * 50.0 / em.BAR) < 0.02            # dp = rho g dz (Gibbs-Duhem), trapezoid
    assert np.allclose(d["x_methane"], 1.0) and (np.diff(d["p_bar"]) > 0).all()


def test_pressure_rises_with_depth_and_heavy_components_settle_in_the_isothermal_column():
    f = CFG["fluids"]["volatile_oil"]
    pr = cg.grade_column(PR, _x("volatile_oil"), f["t_c"], f["p_ref_bar"], np.array([0.0, 100.0, 300.0]), "isothermal", step_m=20.0, saturation=False)
    assert pr["converged"].all() and (np.diff(pr["p_bar"]) > 0).all()
    assert pr["x_n_eicosane"].iloc[-1] > pr["x_n_eicosane"].iloc[0] and pr["x_methane"].iloc[-1] < pr["x_methane"].iloc[0]       # gravity pulls the heavy fraction down
    up = cg.grade_column(PR, _x("volatile_oil"), f["t_c"], f["p_ref_bar"], np.array([-200.0, 0.0]), "isothermal", step_m=20.0, saturation=False)
    assert up["x_methane"].iloc[0] > up["x_methane"].iloc[1]                                                          # and the light one rises


def test_black_oil_and_lean_gas_barely_grade_but_the_near_critical_fluid_does():
    def spread(key, h=300.0):
        f = CFG["fluids"][key]
        p = cg.grade_column(PR, _x(key), f["t_c"], f["p_ref_bar"], np.array([0.0, h]), "isothermal", step_m=20.0, saturation=False)
        return float(abs(p["x_methane"].iloc[-1] - p["x_methane"].iloc[0]) / p["x_methane"].iloc[0])
    assert spread("black_oil") < 0.05 and spread("lean_gas") < 0.03 and spread("near_critical") > 0.10


def test_mixture_column_satisfies_dp_dz_equals_rho_g_with_and_without_translation():
    for key in ("black_oil", "volatile_oil", "hpht_volatile_oil"):
        f = CFG["fluids"][key]
        for shift in (True, False):
            m = PR.with_options("PR", shift)
            pr = cg.grade_column(m, _x(key), f["t_c"], f["p_ref_bar"], np.array([0.0, 20.0, 40.0]), "isothermal", step_m=10.0, saturation=False).sort_values("dz_m")
            dp = pr["p_bar"].diff().iloc[1:].to_numpy()
            rho = 0.5 * (pr["rho_kg_m3"].to_numpy()[:-1] + pr["rho_kg_m3"].to_numpy()[1:])
            assert np.allclose(dp, rho * em.G * 20.0 / em.BAR, rtol=0.01), (key, shift)                              # Gibbs-Duhem: dp = rho g dz, with the density of the same (translated) model


def test_volume_translation_changes_density_not_phase_equilibrium():
    x = _x("black_oil")
    t_k = 90.0 + 273.15
    a, b = PR.with_options("PR", True), PR.with_options("PR", False)
    assert em.density_kg_m3(a, x, t_k, 300.0) > em.density_kg_m3(b, x, t_k, 300.0)                                  # a liquid becomes denser with the translation
    la, lb = em.ln_phi(a, x, t_k, 300.0)[0], em.ln_phi(b, x, t_k, 300.0)[0]
    assert np.allclose(la - lb, -a.c_m3_mol * 300.0e5 / (em.R * t_k))                                                  # ln phi* = ln phi - c p / (R T): the same shift for every phase at one pressure
    assert abs(em.saturation_pressure(a, x, t_k, "bubble") - em.saturation_pressure(b, x, t_k, "bubble")) < 1e-6


def test_srk_and_pr_differ_and_both_converge():
    x = _x("rich_gas_condensate")
    t_k = 140.0 + 273.15
    d_pr = em.density_kg_m3(PR, x, t_k, 350.0)
    d_srk = em.density_kg_m3(PR.with_options("SRK", True), x, t_k, 350.0)
    assert d_pr > 0 and d_srk > 0 and d_pr != d_srk


def test_saturation_pressures_of_the_example_fluids():
    t_k = 140.0 + 273.15
    pb = em.saturation_pressure(PR, _x("volatile_oil"), t_k, "bubble")
    assert 200.0 < pb < 280.0                                                                                          # the example volatile oil is built for a bubble point near 239 bar
    assert not np.isfinite(em.saturation_pressure(PR, _x("lean_gas"), 100.0 + 273.15, "bubble"))                       # a lean dry gas has none


def test_linear_gradient_is_fine_for_a_short_column_and_fails_for_a_tall_near_critical_one():
    f = CFG["fluids"]["near_critical"]
    short = cg.grade_column(PR, _x("near_critical"), f["t_c"], f["p_ref_bar"], np.array([0.0, 10.0, 300.0]), "isothermal", step_m=10.0, saturation=False)
    d = short.set_index("dz_m")["delta_p_bar"]
    assert abs(d[10.0]) < 0.1 and abs(d[300.0]) > 5 * abs(d[10.0]) and abs(d[300.0]) > 1.0
    assert short["converged"].all()


def test_envelope_orders_its_percentiles_and_contains_the_base_case():
    cfg = {**CFG, "ensemble": {**CFG["ensemble"], "eos": ["PR"], "volume_shift": [True, False], "temperature_modes": ["isothermal"], "n_composition_draws": 1}, "grid": {**CFG["grid"], "step_m": 25.0}}
    env, mem = cg.column_envelope(cfg, "volatile_oil", 100.0, n_points=3)
    assert (env["p_min_bar"] <= env["p_p05_bar"] + 1e-9).all() and (env["p_p05_bar"] <= env["p_median_bar"] + 1e-9).all() and (env["p_median_bar"] <= env["p_max_bar"] + 1e-9).all()
    assert ((env["p_base_bar"] >= env["p_min_bar"] - 1e-9) & (env["p_base_bar"] <= env["p_max_bar"] + 1e-9)).all()
    assert len(mem) == 4 and mem["converged"].all()


def test_contact_offset_interpolates_the_crossing():
    import pandas as pd
    prof = pd.DataFrame({"dz_m": [0.0, 10.0, 20.0], "p_minus_psat_bar": [-4.0, -1.0, 3.0]})
    assert abs(cg.contact_offset(prof) - 12.5) < 1e-9 and np.isnan(cg.contact_offset(prof.assign(p_minus_psat_bar=[1.0, 2.0, 3.0])))
