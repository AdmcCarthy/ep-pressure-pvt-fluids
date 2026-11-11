"""Geochemistry: fluid inclusion stratigraphy (FIS) profiles and anomalies, Rock-Eval / TOC with the effect of solvent pre-extraction.  Pure functions; they return `(result, events)` where they change or flag data.

What the data are: FIS volatiles released by crushing cuttings (intensities of species and ratios by depth, unit not stated in the files); Rock-Eval pyrolysis of cuttings before and after solvent pre-extraction.  Rules: values are shown as delivered
(no zero is nulled: an intensity of 0 is a reading); a FIS anomaly is a screening result with an ASSUMED threshold (adapter), never a hydrocarbon or fluid-contact statement; cuttings volatiles carry mud and cavings (an oil-based mud
adds hydrocarbons), so a FIS or Rock-Eval value is a bounding observation; a pair of Rock-Eval analyses before and after extraction is compared, and a sample whose S1 falls by most of its value is flagged as contaminated (assumed limits),
kept, and its pre-extraction values are not read as source-rock properties.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ep_physics_base.events import dq_event

def load_config() -> dict[str, Any]:
    """config/geochemistry.yaml (the packaged defaults, or a project file of that name): the thresholds of the screening rules, all ASSUMPTIONS."""
    from ep_physics_base.config import load_yaml

    return load_yaml("config/geochemistry.yaml")


SCHEMA_VERSIONS = {"geochem_fis_param": "1.0", "geochem_fis_amu": "1.0", "geochem_fis_sample": "1.0", "geochem_fis_anomaly": "1.0", "geochem_rockeval": "1.0", "geochem_rockeval_pair": "1.0"}


def fis_anomalies(params: pd.DataFrame, cfg: dict[str, Any], sid: str) -> tuple[pd.DataFrame, list[dict]]:
    """Intervals where a species stands out from the campaign's own background.  Per campaign the background is the median of log10 of the species and its spread the median absolute deviation (MAD, scaled by 1.4826); a sample is
    anomalous when its log10 value is above `median + k x spread` (k = `anomaly_k`, an ASSUMPTION), consecutive anomalous samples (no gap above `max_gap_m`) form one interval.  Returned per species: the interval, the number of
    samples, the peak depth and value and the ratio of the peak to the background median.  A screening result, not a statement about fluids."""
    rows, events = [], []
    k, gap, species = float(cfg["anomaly_k"]), float(cfg["max_gap_m"]), cfg["anomaly_species"]
    for camp, g in params.groupby("campaign"):
        g = g.sort_values("depth_md_m")
        for sp in species:
            if sp not in g:
                continue
            v = g[sp].astype("float64")
            lv = np.log10(v.where(v > 0))
            med = lv.median()
            spread = 1.4826 * (lv - med).abs().median()
            if not np.isfinite(spread) or spread == 0:
                continue
            hit = (lv > med + k * spread).fillna(False).to_numpy()
            depth = g["depth_md_m"].to_numpy()
            i = 0
            while i < len(g):
                if not hit[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < len(g) and hit[j + 1] and depth[j + 1] - depth[j] <= gap:
                    j += 1
                seg = g.iloc[i:j + 1]
                pk = seg[sp].astype("float64").idxmax()
                rows.append({"campaign": camp, "well_name_file": g["well_name_file"].iloc[0], "species": sp, "top_md_m": float(seg["depth_md_m"].min()), "base_md_m": float(seg["depth_md_m"].max()), "samples": len(seg),
                             "peak_md_m": float(g.loc[pk, "depth_md_m"]), "peak_value": float(g.loc[pk, sp]), "background_median": float(10 ** med), "peak_over_background": float(g.loc[pk, sp] / 10 ** med),
                             "threshold_log10": float(med + k * spread)})
                i = j + 1
    out = pd.DataFrame(rows)
    return out, events


def rock_eval_pairs(rows: pd.DataFrame, cfg: dict[str, Any], sid: str) -> tuple[pd.DataFrame, list[dict]]:
    """One row per sample with the values before (`DC`) and after (`DC, extr.`) solvent pre-extraction side by side, the change, `pi_check = S1 / (S1 + S2)` against the delivered PI, and the flags: `flag_s1_removed` (S1 falls by more than
    `s1_removed_share` of its value: the free hydrocarbons were mostly extractable, which in cuttings points to mud or migrated oil, not to kerogen), `flag_tmax_unreliable` (S2 below `s2_min_for_tmax` mg/g: Tmax is not read), `flag_pi_differs`.
    Limits are ASSUMPTIONS (adapter)."""
    raw = rows[rows["sample_type"] == "DC"].set_index(["well_name_file", "depth_md_m", "lab_id"])
    ext = rows[rows["sample_type"] == "DC, extr."].copy()
    ext["lab_id"] = ext["lab_id"].astype(str)
    cols = ["s1_mg_g", "s2_mg_g", "tmax_c", "pp_mg_g", "pi", "hi_mg_hc_g_toc", "toc_pct"]
    ext = ext.set_index(["well_name_file", "depth_md_m", "lab_id"])
    d = raw[cols].merge(ext[cols], left_index=True, right_index=True, how="outer", suffixes=("_before", "_after")).reset_index()
    d["s1_removed_share"] = 1.0 - d["s1_mg_g_after"] / d["s1_mg_g_before"]
    d["toc_change_pct_points"] = d["toc_pct_after"] - d["toc_pct_before"]
    d["pi_check_before"] = d["s1_mg_g_before"] / (d["s1_mg_g_before"] + d["s2_mg_g_before"])
    d["flag_s1_removed"] = d["s1_removed_share"] > float(cfg["s1_removed_share"])
    d["flag_tmax_unreliable_before"] = d["s2_mg_g_before"] < float(cfg["s2_min_for_tmax"])
    d["flag_tmax_unreliable_after"] = d["s2_mg_g_after"] < float(cfg["s2_min_for_tmax"])
    d["flag_pi_differs"] = (d["pi_check_before"] - d["pi_before"]).abs() > 0.02
    events = []
    n = int(d["flag_s1_removed"].sum())
    if n:
        events.append(dq_event(sid, "geochem_rockeval", "s1_removed_by_extraction", n, "warning", [f"{r.well_name_file} {r.depth_md_m:.0f}" for r in d[d["flag_s1_removed"]].head(5).itertuples()],
                               "S1 falls by most of its value after solvent pre-extraction: the free hydrocarbons are extractable (mud or migrated oil); the pre-extraction values are not source-rock properties; kept"))
    return d, events
