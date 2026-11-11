"""PVT dashboard page: valid embedded JSON (no NaN), self-contained (no network), placeholder for an empty mart."""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from ep_pressure_pvt_fluids import pvt

from ep_pressure_pvt_fluids.plots import pvt_dashboard as dash  # noqa: E402

CFG = pvt.load_config()


def synthetic() -> pd.DataFrame:
    n = 12
    pressure = pd.DataFrame({"wellbore_id": pd.array(range(1, n + 1), dtype="Int64"), "well_name": [f"W{i}" for i in range(n)], "test_number": ["1"] * n, "area": ["NORTH SEA"] * 8 + ["BARENTS SEA"] * 4,
                             "mid_md_m": np.linspace(2000, 4500, n), "reservoir_pressure_bar": np.linspace(180, 700, n), "temperature_c": [np.nan] + list(np.linspace(60, 150, n - 1))})
    dst = pd.DataFrame({"wellbore_id": pd.array(range(1, n + 1), dtype="Int64"), "test_number": ["1"] * n, "gor_sm3_sm3": [100, 500, 800, 5000, 20000, 150, 300, 6000, 90, 400, 7000, 60],
                        "oil_density_g_cm3": [0.85, 0.8, np.nan, 0.78, 0.76, 0.93, 0.84, 0.78, 0.88, 0.81, 0.77, 0.95], "gas_relative_density": [0.75, 0.75, np.nan, 0.7, 0.65, 0.8, 0.75, 0.7, 0.8, 0.75, 0.7, 0.85]})
    out, _ = pvt.build_well_dst_pvt(pressure, dst, CFG)
    return out


def test_page_embeds_valid_json_without_nan_and_loads_no_remote_script():
    html = dash.build_html(synthetic(), CFG)
    data = re.search(r"const D = (\[.*?\]), C = ", html, re.S).group(1)
    rows = json.loads(data)                                       # invalid JSON (NaN, Infinity) would raise here
    assert len(rows) == 12 and {"w", "p", "g", "st", "dm", "ar", "pl", "ti"} <= set(rows[0]) and rows[0]["ar"] == "outside" and rows[0]["pl"] == "none"
    assert any(r["t"] is None for r in rows) and any(r["st"] == "forbidden" for r in rows)
    cfg = json.loads(re.search(r", C = (\{.*?\});\r?\nD\.forEach", html, re.S).group(1))
    assert cfg["gas"]["glr"][0] == CFG["gas_envelope"]["min_glr"][0] and cfg["colours"]["st"]["forbidden"]
    assert cfg["ymax"] == 1200.0 and [z["name"] for z in cfg["pt"]["zones"]] == ["normal", "high temperature", "high pressure", "HPHT", "ultra-HPHT", "extreme-HPHT"]
    assert "central_north_sea" in cfg["areaLabel"] and "northern_north_sea" in cfg["areaLabel"] and cfg["areaLabel"]["northern_north_sea"] != cfg["areaLabel"]["central_north_sea"]   # two separate areas
    assert "example" not in cfg and "200 bar, GLR 500" not in html                        # the example marker is gone
    assert "http://" not in html and "https://" not in html.replace("https://www.w3.org", "")          # nothing is fetched when the page is viewed
    assert 'src="assets/plotly.min.js"' in html and "DRAFT" in html


def test_empty_mart_gives_a_placeholder_page():
    html = dash.build_html(pd.DataFrame(), CFG)
    assert "No DST tests" in html and "plotly" not in html


def test_rows_get_position_area_and_play_and_unknowns_stay_explicit():
    tests = synthetic().assign(play_id=pd.array(["a", None] + [None] * 10, dtype="string"))
    header = pd.DataFrame({"wellbore_id": pd.array([1, 2], dtype="Int64"), "lat_etrs89": [58.1, 71.2], "lon_etrs89": [2.2, 20.5]})
    area = pd.DataFrame({"wellbore_id": pd.array([1], dtype="Int64"), "region_id": ["central_north_sea"]})
    out = dash.prepare_rows(tests, header, area)
    first, second, third = out.iloc[0], out.iloc[1], out.iloc[2]
    assert (first["lat"], first["lon"], first["region_id"], first["play_id"]) == (58.1, 2.2, "central_north_sea", "a")
    assert second["region_id"] == "outside" and second["play_id"] == "none" and second["lat"] == 71.2          # in no area and no play: stated, not guessed
    assert pd.isna(third["lat"]) and third["region_id"] == "outside"                                              # no position: kept, left off the map


def test_map_outline_rings_are_valid_for_a_plotly_line():
    from shapely.geometry import MultiPolygon, Polygon
    x, y = dash._rings(MultiPolygon([Polygon([(0, 0), (1, 0), (1, 1)]), Polygon([(5, 5), (6, 5), (6, 6)])]))
    assert x.count(None) == 1 and len(x) == len(y) and x[0] == x[3] and y[0] == y[3]                              # two closed rings separated by one gap