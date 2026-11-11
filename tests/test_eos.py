"""Peng-Robinson equation of state: saturation pressure, critical point, volumes, regimes and the closed pressure-volume-temperature surface."""

from __future__ import annotations

import numpy as np
import pytest

from ep_pressure_pvt_fluids import eos
from ep_physics_base.config import load_yaml

CFG = load_yaml("config/pvt3d.yaml")
FLUIDS = eos.fluids_from_config(CFG)


def test_saturation_pressure_of_known_fluids_within_equation_accuracy():
    # reference vapour pressures (NIST / DIPPR, bar); Peng-Robinson is accurate to a few per cent away from the critical point
    cases = [("propane", 300.0, 10.02), ("n_butane", 350.0, 9.43), ("n_hexane", 341.9, 1.013), ("methane", 150.0, 10.4), ("n_heptane", 371.6, 1.013)]
    for name, t_k, expected in cases:
        assert eos.saturation_pressure(t_k, FLUIDS[name]) == pytest.approx(expected, rel=0.06), name


def test_critical_point_is_where_the_dome_closes():
    for fluid in FLUIDS.values():
        near = fluid.tc_k * (1.0 - 1e-3)
        ps = eos.saturation_pressure(near, fluid)
        assert ps == pytest.approx(fluid.pc_bar, rel=0.01)                                           # Psat(T -> Tc) = Pc
        vl, vv = eos.volume_l_mol(near, ps, fluid, "liquid"), eos.volume_l_mol(near, ps, fluid, "vapour")
        assert vl < fluid.vc_l_mol < vv and (vv - vl) / fluid.vc_l_mol < 0.5                         # liquid and vapour volumes converge on Vc
    assert eos.saturation_pressure(FLUIDS["propane"].tc_k + 1.0, FLUIDS["propane"]) != eos.saturation_pressure(FLUIDS["propane"].tc_k + 1.0, FLUIDS["propane"])   # NaN above Tc


def test_saturation_pressure_rises_with_temperature_and_fugacities_are_equal_on_it():
    f = FLUIDS["n_butane"]
    t = np.linspace(250.0, f.tc_k * 0.999, 12)
    ps = np.array([eos.saturation_pressure(x, f) for x in t])
    assert (np.diff(ps) > 0).all()
    a_, b_ = eos.ab(350.0, eos.saturation_pressure(350.0, f), f)
    z = eos.z_roots(a_, b_)
    assert len(z) == 3 and eos.ln_phi(z[0], a_, b_) == pytest.approx(eos.ln_phi(z[-1], a_, b_), abs=1e-7)


def test_regimes_of_a_point():
    f = FLUIDS["n_butane"]                                           # Tc 152 degC, Pc 38 bar
    assert eos.state(100.0, 200.0, f)[1] == "liquid"                 # below Tc and above the saturation pressure
    assert eos.state(100.0, 2.0, f)[1] == "vapour"                   # below Tc and below the saturation pressure
    assert eos.state(200.0, 300.0, f)[1] == "supercritical"          # above Tc: one phase at any pressure
    ps = eos.saturation_pressure(373.15, f)
    assert eos.state(100.0, ps, f)[1] == "two-phase"
    liquid, vapour = eos.state(100.0, 200.0, f)[0], eos.state(100.0, 2.0, f)[0]
    assert liquid < np.log10(f.vc_l_mol) < vapour                    # a compressed liquid is denser, a vapour thinner, than at the critical point
    assert np.isnan(eos.state(float("nan"), 100.0, f)[0])            # missing temperature: no point, never a guess


def test_methane_reservoir_conditions_are_supercritical_and_decane_is_liquid():
    assert eos.state(100.0, 400.0, FLUIDS["methane"])[1] == "supercritical"
    assert eos.state(100.0, 400.0, FLUIDS["n_decane"])[1] == "liquid"
    assert 0.15 < 10 ** eos.state(100.0, 400.0, FLUIDS["n_decane"])[0] < 0.30                                  # liquid decane about 0.2 L/mol (a large molecule)
    assert 0.05 < 10 ** eos.state(100.0, 400.0, FLUIDS["methane"])[0] < 0.12                                   # a dense methane gas about 0.08 L/mol


def test_surface_is_one_closed_mesh_with_the_dome_and_the_critical_point():
    f = FLUIDS["n_butane"]
    s = eos.pvt_surface(f, CFG["grid"])
    n_cols = CFG["grid"]["n_liquid"] + CFG["grid"]["n_plateau"] + CFG["grid"]["n_vapour"]
    assert len(s["x"]) == len(s["t_c"]) and all(len(r) == n_cols for r in s["x"] + s["y"] + s["z"] + s["region"])       # a regular mesh
    x, y, z, r = (np.array(s[k]) for k in ("x", "y", "z", "region"))
    sub = x[:, 0] < f.tc_k - 273.15 - 1e-6
    assert (np.diff(y[sub], axis=1) >= -1e-9).all()                                                # volume rises along every subcritical isotherm
    assert (np.diff(z[sub][:, :CFG["grid"]["n_liquid"]], axis=1) <= 1e-6).all()                      # pressure falls along the liquid branch
    plateau = z[sub][:, CFG["grid"]["n_liquid"]:CFG["grid"]["n_liquid"] + CFG["grid"]["n_plateau"]]
    assert np.allclose(plateau, plateau[:, :1])                                                    # the two-phase plateau is flat: constant pressure (Maxwell)
    assert set(np.unique(r[sub])) == {0.0, 1.0, 2.0} and (r[~sub] == 3.0).all()                    # liquid, two-phase, vapour below Tc; supercritical above
    assert s["critical"]["p_bar"] == f.pc_bar and abs(s["critical"]["t_c"] - (f.tc_k - 273.15)) < 1e-9
    dl, dv = s["dome_liquid"], s["dome_vapour"]
    gap = np.array(dv["y"]) - np.array(dl["y"])
    assert (gap > 0).all() and gap[-1] < gap[0] / 3                                                # the dome narrows towards the critical point
    assert max(s["z"][0]) <= CFG["grid"]["p_max_bar"] + 1e-6
