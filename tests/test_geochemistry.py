"""FIS anomalies and Rock-Eval before / after extraction: screening results with assumed limits."""

import numpy as np
import pandas as pd

from ep_pressure_pvt_fluids import geochemistry as gc


def test_the_packaged_limits_load_and_are_numbers():
    cfg = gc.load_config()
    assert cfg["anomaly_k"] > 0 and 0 < cfg["s1_removed_share"] < 1 and "ch4" in cfg["anomaly_species"]


def test_fis_anomalies_find_a_spike_against_the_campaign_background():
    depth = np.arange(1000.0, 1300.0, 10.0)
    ch4 = np.full(len(depth), 1e5) * (1 + 0.1 * np.sin(depth))
    ch4[10:12] = 5e7                                                                   # two neighbouring samples far above the background
    p = pd.DataFrame({"campaign": "A", "well_name_file": "W", "depth_md_m": depth, "ch4": ch4})
    a, _ = gc.fis_anomalies(p, {"anomaly_k": 3.0, "max_gap_m": 25.0, "anomaly_species": ["ch4", "absent"]}, "s")
    assert len(a) == 1 and a["samples"].iloc[0] == 2 and a["top_md_m"].iloc[0] == 1100.0 and a["peak_over_background"].iloc[0] > 100


def test_rock_eval_pairs_flag_contaminated_s1_and_unreliable_tmax():
    cols = ["well_name_file", "depth_md_m", "lab_id", "sample_type", "s1_mg_g", "s2_mg_g", "tmax_c", "pp_mg_g", "pi", "hi_mg_hc_g_toc", "toc_pct"]
    rows = pd.DataFrame([["W", 3760.0, "18660C", "DC", 26.46, 9.51, 333, 35.97, 0.74, 277, 3.43],
                         ["W", 3760.0, "18660C", "DC, extr.", 0.03, 0.6, 426, 0.63, 0.05, 154, 0.39]], columns=cols)
    p, ev = gc.rock_eval_pairs(rows, {"s1_removed_share": 0.9, "s2_min_for_tmax": 0.5}, "s")
    row = p.iloc[0]
    assert row["flag_s1_removed"] and row["s1_removed_share"] > 0.99 and row["toc_change_pct_points"] < 0
    assert not row["flag_pi_differs"] and row["flag_tmax_unreliable_after"] == (0.6 < 0.5)
    assert any(e["rule"] == "s1_removed_by_extraction" for e in ev)