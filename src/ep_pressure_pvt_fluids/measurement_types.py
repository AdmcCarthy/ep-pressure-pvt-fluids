"""Measurement types of the PVT tests: which kind of measurement a data point comes from, classified from the official Sodir records only (docs/measurement_types.md).

A DST of `well_dst` is linked to the oil-sample records (`well_oil_sample`, one per sample taken in a test, with the test type DST / MDT / RFT / PROD ...) of its wellbore: the same
test number is strong evidence (high confidence), an overlapping interval weaker (medium), no record is `dst_unsampled` (medium: absence of a record is not proof of no sample).
A stated saturation pressure of a PVT report makes the point a laboratory type (`pvt_report`).  Nothing is guessed: without evidence the type stays `dst` or `other`.
Pure functions (no I/O) returning frames and events (AGENTS.md section 14).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ep_physics_base.events import dq_event
from ep_physics_base.config import load_yaml

CONFIG_FILE = "config/measurement_types.yaml"
CONFIDENCES = ["high", "medium"]


def load_config() -> dict[str, Any]:
    return load_yaml(CONFIG_FILE)


def sample_type_of(raw: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    """Measurement type of an oil-sample record from its Sodir test type (`sample_link.sodir_test_type`); an unknown or empty test type is `other`."""
    mapping = {str(k).upper(): v for k, v in cfg["sample_link"]["sodir_test_type"].items()}
    key = raw.astype("string").str.strip().str.upper()
    return key.map(mapping).fillna("other").astype("string")


def _test_key(value: Any) -> str | None:
    """'DST1' -> '1', '3B' -> '3B', 1 -> '1': the number part of a test reference, upper case (None when there is none)."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return None
    text = str(value).strip().upper()
    if not text or text in ("NAN", "<NA>", "NONE"):
        return None
    digits = "".join(ch for ch in text if ch.isalnum())
    start = next((i for i, ch in enumerate(digits) if ch.isdigit()), None)
    return digits[start:] if start is not None else None


def link_dst_samples(tests: pd.DataFrame, oil_samples: pd.DataFrame | None, cfg: dict[str, Any]) -> pd.DataFrame:
    """For every test (columns wellbore_id, test_number, top_md_m, base_md_m) the best link to a DST oil-sample record of its wellbore.
    Columns (same index as `tests`): sample_link (same test number | depth overlap | other test in well | no record), sample_fluid_type (Sodir fluid type of the linked records),
    sample_types_in_well (text such as 'DST x3; MDT x2'), wireline_samples_in_well (bool)."""
    out = pd.DataFrame({"sample_link": "no record", "sample_fluid_type": pd.NA, "sample_types_in_well": pd.NA, "wireline_samples_in_well": False}, index=tests.index)
    out["sample_link"] = out["sample_link"].astype("string")
    out["sample_fluid_type"] = out["sample_fluid_type"].astype("string")
    out["sample_types_in_well"] = out["sample_types_in_well"].astype("string")
    if oil_samples is None or oil_samples.empty or "test_type" not in oil_samples:
        return out
    s = oil_samples.copy()
    s["sample_type"] = sample_type_of(s["test_type"], cfg)
    s["_tn"] = s["test_number"].map(_test_key) if "test_number" in s else None
    tol = float(cfg["sample_link"]["depth_tolerance_m"])
    by_well = {w: g for w, g in s.groupby("wellbore_id")}
    link, fluid, inwell, wire = [], [], [], []

    def record(status: str, fluid_type: Any, in_well: Any, wireline: bool) -> None:
        link.append(status)
        fluid.append(fluid_type)
        inwell.append(in_well)
        wire.append(wireline)

    for row in tests[["wellbore_id", "test_number", "top_md_m", "base_md_m"]].itertuples(index=False):
        g = by_well.get(row.wellbore_id)
        if g is None:
            record("no record", pd.NA, pd.NA, False)
            continue
        counts = g["test_type"].astype("string").str.upper().value_counts()
        in_well, wireline = "; ".join(f"{k} x{v}" for k, v in counts.items()), bool((g["sample_type"] == "wireline_tester").any())
        d = g[g["sample_type"] == "dst"]
        key = _test_key(row.test_number)
        same = d[d["_tn"] == key] if key is not None else d.iloc[0:0]
        overlap = d.iloc[0:0]
        if len(d) and pd.notna(row.top_md_m) and pd.notna(row.base_md_m):
            d_top = d["top_md_m"].astype("float64")
            d_base = d["base_md_m"].astype("float64").fillna(d_top)
            overlap = d[(d_top <= float(row.base_md_m) + tol) & (d_base >= float(row.top_md_m) - tol)]
        if len(same):
            status, picked = "same test number", same
        elif len(overlap):
            status, picked = "depth overlap", overlap
        else:
            record("other test in well" if len(d) else "no record", pd.NA, in_well, wireline)
            continue
        kinds = sorted({str(x) for x in picked["fluid_type"].dropna()}) if "fluid_type" in picked else []
        record(status, "/".join(kinds) if kinds else pd.NA, in_well, wireline)
    out["sample_link"] = pd.array(link, dtype="string")
    out["sample_fluid_type"] = pd.array(fluid, dtype="string")
    out["sample_types_in_well"] = pd.array(inwell, dtype="string")
    out["wireline_samples_in_well"] = wire
    return out


def drawdown_ratio(sibhp_bar: pd.Series, fbhp_bar: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    """1 - flowing / shut-in bottom-hole pressure (a proxy for how hard the well was pulled).  Missing unless both are positive and the flowing value is not above the static one
    by more than `drawdown.max_flow_to_static` (that is a gauge or wellhead mix-up, not a drawdown)."""
    si = pd.to_numeric(sibhp_bar, errors="coerce").astype("float64")
    fl = pd.to_numeric(fbhp_bar, errors="coerce").astype("float64")
    ok = (si > 0) & (fl > 0) & (fl <= si * float(cfg["drawdown"]["max_flow_to_static"]))
    return (1.0 - fl / si).clip(lower=0.0).where(ok)


def drawdown_class(ratio: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    edges, labels = [-np.inf] + [float(e) for e in cfg["drawdown"]["class_edges"]] + [np.inf], cfg["drawdown"]["class_labels"]
    return pd.cut(pd.to_numeric(ratio, errors="coerce"), bins=edges, labels=labels, right=False).astype("string")


def classify_measurements(tests: pd.DataFrame, oil_samples: pd.DataFrame | None, cfg: dict[str, Any] | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Measurement type, evidence and confidence of every test, plus the indicators the issues of its type can be read from (interval length, drawdown ratio).
    `tests` needs wellbore_id, test_number, top_md_m, base_md_m and optionally saturation_basis, sibhp_bar, fbhp_bar.  Returns (columns with the index of `tests`, events)."""
    cfg = cfg or load_config()
    ctx = ("all", "measurement_types")
    links = link_dst_samples(tests, oil_samples, cfg)
    have_samples = oil_samples is not None and not oil_samples.empty
    report = tests["saturation_basis"].eq("pvt_report") if "saturation_basis" in tests else pd.Series(False, index=tests.index)
    mtype = pd.Series("dst", index=tests.index, dtype="string")
    conf = pd.Series("medium", index=tests.index, dtype="string")
    evidence = pd.Series("official DST record; the oil-sample table was not available to look for a sample record", index=tests.index, dtype="string")
    if have_samples:
        same, over = links["sample_link"].eq("same test number"), links["sample_link"].eq("depth overlap")
        other, none = links["sample_link"].eq("other test in well"), links["sample_link"].eq("no record")
        mtype[same | over], mtype[other | none] = "dst_sampled", "dst_unsampled"
        conf[same], conf[over | other | none] = "high", "medium"
        evidence[same] = "Sodir oil-sample record of type DST for this test number"
        evidence[over] = "Sodir oil-sample record of type DST whose interval overlaps the test interval"
        evidence[other] = "no oil-sample record for this test; the well has DST sample records for other tests"
        evidence[none] = "no DST oil-sample record in the well (absence of a record is not proof that no sample was taken)"
    mtype[report], conf[report] = "pvt_report", "high"
    evidence[report] = "stated saturation pressure of a matched PVT report (" + evidence[report].astype("string") + ")"
    top = pd.to_numeric(tests["top_md_m"], errors="coerce").astype("float64") if "top_md_m" in tests else pd.Series(np.nan, index=tests.index)
    base = pd.to_numeric(tests["base_md_m"], errors="coerce").astype("float64") if "base_md_m" in tests else pd.Series(np.nan, index=tests.index)
    dd = drawdown_ratio(tests["sibhp_bar"] if "sibhp_bar" in tests else pd.Series(np.nan, index=tests.index), tests["fbhp_bar"] if "fbhp_bar" in tests else pd.Series(np.nan, index=tests.index), cfg)
    group_of = {k: v["group"] for k, v in cfg["types"].items()}
    out = pd.DataFrame({"measurement_type": mtype, "measurement_group": mtype.map(group_of).astype("string"), "measurement_confidence": conf, "measurement_evidence": evidence,
                        "sample_link": links["sample_link"], "sample_fluid_type": links["sample_fluid_type"], "sample_types_in_well": links["sample_types_in_well"],
                        "wireline_samples_in_well": links["wireline_samples_in_well"], "test_interval_m": (base - top).where((base - top) >= 0), "drawdown_ratio": dd, "drawdown_class": drawdown_class(dd, cfg)},
                       index=tests.index)
    events = [dq_event(*ctx, "measurement_types", len(out), "info", [], f"types: {mtype.value_counts().to_dict()}; sample links: {links['sample_link'].value_counts().to_dict()}; drawdown ratio for {int(dd.notna().sum())} tests"
                       + ("" if have_samples else " (no oil-sample table: DST types are not split by sample record)"))]
    return out, events


def sample_record_timeline(oil_samples: pd.DataFrame, cfg: dict[str, Any], anchor: int = 1965, window: int = 5) -> pd.DataFrame:
    """Oil-sample records per 5-year window (anchor 1965, label '1980–84') and measurement type of the sample: how the kind of fluid sampling changed over time.
    Records without a test date are counted in the row 'undated'.  Columns are the type ids that occur."""
    s = oil_samples.copy()
    s["type"] = sample_type_of(s["test_type"], cfg)
    year = pd.to_datetime(s["test_date"], errors="coerce").dt.year
    start = anchor + ((year - anchor) // window) * window
    label = start.map(lambda v: "undated" if pd.isna(v) else f"{int(v)}–{str(int(v) + window - 1)[-2:]}")
    table = pd.crosstab(label, s["type"])
    order = sorted((i for i in table.index if i != "undated"), key=lambda v: int(v[:4])) + (["undated"] if "undated" in table.index else [])
    return table.reindex(order)


def type_catalogue(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """The configured types as a table (group, label, symbol, classifiable, evidence, issues) for the notebook and the docs."""
    cfg = cfg or load_config()
    rows = [{"type": k, "group": cfg["groups"][v["group"]], "label": v["label"], "symbol": v["glyph"], "classifiable": v["classifiable"], "evidence": v["evidence"], "issues": ", ".join(v["issues"])} for k, v in cfg["types"].items()]
    return pd.DataFrame(rows)
