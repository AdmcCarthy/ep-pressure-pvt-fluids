"""3D PVT page: model-fluid surfaces and test points in the page configuration, valid JSON, self-contained, regime of a point on the surface."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ep_fluids import eos, pvt
from ep_physics_base.config import load_yaml

from ep_pressure.plots import pvt3d_dashboard as page3d  # noqa: E402

CFG = pvt.load_config()
CFG3D = load_yaml("config/pvt3d.yaml")
SMALL_GRID = {**CFG3D["grid"], "n_temperature": 12, "n_liquid": 8, "n_plateau": 4, "n_vapour": 8}          # a coarse mesh keeps the test fast


def synthetic() -> pd.DataFrame:
    n = 8
    pressure = pd.DataFrame({"wellbore_id": pd.array(range(1, n + 1), dtype="Int64"), "well_name": [f"W{i}" for i in range(n)], "test_number": ["1"] * n, "area": ["NORTH SEA"] * n,
                             "mid_md_m": np.linspace(2000, 4500, n), "reservoir_pressure_bar": np.linspace(180, 700, n), "temperature_c": [np.nan] + list(np.linspace(60, 190, n - 1))})
    pressure["temperature_est_c"] = [95.0] + [np.nan] * (n - 1)
    dst = pd.DataFrame({"wellbore_id": pd.array(range(1, n + 1), dtype="Int64"), "test_number": ["1"] * n, "gor_sm3_sm3": [100, 500, 800, 5000, 20000, 150, 300, 6000],
                        "oil_density_g_cm3": [0.85, 0.8, 0.78, 0.78, 0.76, 0.93, 0.84, 0.78], "gas_relative_density": [0.75] * n})
    out, _ = pvt.build_well_dst_pvt(pressure, dst, CFG, use_estimated_temperature=True)
    return out


def test_point_states_place_every_test_on_every_model_fluid_and_leave_missing_empty():
    tests = synthetic()
    fluids = eos.fluids_from_config(CFG3D)
    states = page3d.point_states(tests, fluids, 0.002)
    assert set(states) == set(fluids) and all(len(v["v"]) == len(tests) == len(v["r"]) for v in states.values())
    assert set(states["methane"]["r"]) == {"supercritical"}                                  # every reservoir temperature is above the critical temperature of methane (-83 degC)
    assert set(states["n_decane"]["r"]) == {"liquid"}                                        # and below that of decane (345 degC), at these pressures
    assert {"liquid", "supercritical"} <= set(states["n_butane"]["r"])                        # butane (152 degC) splits the data
    nan_tests = tests.assign(temperature_used_c=np.nan)
    assert all(v is None for v in page3d.point_states(nan_tests, fluids, 0.002)["methane"]["v"])      # no temperature: no point, never a guess


def test_page_embeds_the_surfaces_the_points_and_valid_json():
    cfg3d = {**CFG3D, "grid": SMALL_GRID, "fluids": CFG3D["fluids"][:2] + [f for f in CFG3D["fluids"] if f["id"] == "n_butane"], "default_fluid": "n_butane"}
    html = page3d.build_html(synthetic(), CFG, cfg3d=cfg3d)
    data = json.loads(re.search(r"const D = (\[.*?\]), C = ", html, re.S).group(1))                  # invalid JSON (NaN) would raise
    cfg = json.loads(re.search(r", C = (\{.*?\});\r?\nD\.forEach", html, re.S).group(1))
    assert len(data) == 8 and cfg["variant"] == "3d" and cfg["eos"]["default"] == "n_butane"
    assert [f["id"] for f in cfg["eos"]["fluids"]] == ["methane", "ethane", "n_butane"]
    s = cfg["eos"]["surfaces"]["n_butane"]
    assert "x" not in s and len(s["y"]) == len(s["t_c"]) == len(s["z"]) == len(s["region"]) and s["critical"]["p_bar"] == 37.96
    assert len(cfg["eos"]["points"]["n_butane"]["v"]) == 8 and "id=\"f-model\"" in html and "mesh3d" in html
    assert data[0]["ti"] is True                                                             # the estimated temperature of the first test is carried (hollow marker)
    assert "http://" not in html and "https://" not in html.replace("https://www.w3.org", "")  # self-contained


def test_page_keeps_the_shared_controls_and_replaces_the_glr_chart():
    html = page3d.build_html(synthetic(), CFG, cfg3d={**CFG3D, "grid": SMALL_GRID, "fluids": CFG3D["fluids"][:3], "default_fluid": "propane"})
    for element in ('id="f-areas"', 'id="f-plays"', 'id="c-map"', 'id="c-pt"', 'id="c-api"', 'id="f-colour"', 'value="re"'):
        assert element in html                                                                # areas, plays, map, P-T, API and the regime colour-by are all there
    assert "GLR (log scale, on top)" not in html and "Temperature, molar volume and pressure" in html
