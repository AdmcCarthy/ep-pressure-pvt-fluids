"""Back-stripping, first form: decompaction and the weight of a lithology column.  Pure functions (docs/knowledge_architecture.md section 5).

Model (Wikipedia 'Back-stripping' states the law; the grain-conservation step below is derived here, it is arithmetic): porosity falls exponentially with depth below the seabed,
phi(z) = phi0 exp(-c z).  The thickness of SOLID grains in a layer between z1 and z2 is  S = (z2 - z1) - (phi0 / c) (exp(-c z1) - exp(-c z2))  and is conserved when the layer is moved to another depth,
so a decompacted thickness follows from S by solving for the new base depth.  The weight of a column is the sum over layers of grains (S x grain density) plus the water in the pores.

LEFT OUT and named: eroded rock, the burial history through time, sea-level and the isostatic (tectonic subsidence) step; lithology shares and compaction constants are assumptions (placeholders, grade
`placeholder` in the knowledge pack).  The overburden function compares with the generic trend of src/processing/pore_pressure.py.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ep_physics_base import units as pu

G_BAR_PER_M_PER_GCM3 = pu.BAR_PER_M_PER_SG                    # 1 g/cm3 over 1 m = 0.0980665 bar


def shares_from_text(text: str | None, keywords: dict[str, list[str]], default: dict[str, float], unmodelled: list[str] | None = None) -> tuple[dict[str, float], bool, str]:
    """Lithology shares of a unit from the words of its description: the keyword counts per lithology, normalised.  Returns (shares, flagged, basis).  No usable word = the default shares.
    `flagged` is True when the text names a rock that is not modelled (salt, coal ...), so the result is a lower-confidence estimate."""
    low = (text or "").lower()
    import re

    # longest word first, non-overlapping: "claystone" counts once, not as "clay" and "claystone"
    counts = {k: len(re.findall("|".join(re.escape(w) for w in sorted(ws, key=len, reverse=True)), low)) for k, ws in keywords.items()}
    flagged = any(w in low for w in (unmodelled or []))
    tot = sum(counts.values())
    if tot == 0:
        return dict(default), flagged, "default (no usable description)"
    return {k: v / tot for k, v in counts.items() if v}, flagged, "inferred from the lexicon description"


def solid_thickness_m(z_top_m, z_base_m, phi0: float, c_per_m: float):
    """Thickness of solid grains between two depths below the seabed for phi = phi0 exp(-c z)."""
    z1, z2 = np.asarray(z_top_m, dtype=float), np.asarray(z_base_m, dtype=float)
    return (z2 - z1) - phi0 / c_per_m * (np.exp(-c_per_m * z1) - np.exp(-c_per_m * z2))


def base_depth_for_solid_m(z_top_m, solid_m, phi0: float, c_per_m: float, z_max_m: float = 50000.0):
    """Depth of the base of a layer that holds `solid_m` of grains, when its top is at `z_top_m` (bisection; solid thickness grows with the base depth)."""
    z1, s = np.broadcast_arrays(np.asarray(z_top_m, dtype=float), np.asarray(solid_m, dtype=float))
    lo, hi = z1.copy(), np.full(z1.shape, float(z_max_m))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        less = solid_thickness_m(z1, mid, phi0, c_per_m) < s
        lo, hi = np.where(less, mid, lo), np.where(less, hi, mid)
    return 0.5 * (lo + hi)


def decompacted_thickness_m(solid_m: float, new_top_m: float, phi0: float, c_per_m: float) -> float:
    """Thickness a layer of `solid_m` grains has when its top sits at `new_top_m` (used to move a layer to shallower depth when the rock above is removed)."""
    return float(base_depth_for_solid_m(new_top_m, solid_m, phi0, c_per_m) - new_top_m)


def column_overburden_bar(layers: list[dict[str, Any]], water_depth_m: float, params: dict[str, dict[str, float]], grain_density: float, water_density: float,
                          seawater_gradient_bar_per_m: float) -> float:
    """Overburden at the base of a column of PRESENT-DAY layers.  Each layer: {top_m, base_m (below the seabed), lithology: {name: share}}.  Within a layer every lithology share is integrated with its own
    exponential porosity law; the weight is the water column to the seabed plus (grains x grain density + pore water x water density) x g."""
    w = seawater_gradient_bar_per_m * water_depth_m
    total = 0.0
    for lay in layers:
        for name, share in lay["lithology"].items():
            p = params[name]
            c = p["c_per_km"] / 1000.0
            solid = float(solid_thickness_m(lay["top_m"], lay["base_m"], p["phi0"], c))
            water = (lay["base_m"] - lay["top_m"]) - solid
            total += share * (solid * grain_density + water * water_density)
    return w + total * G_BAR_PER_M_PER_GCM3


def strip_top_layers(layers: list[dict[str, Any]], n_remove: int, params: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    """Remove the `n_remove` youngest layers and let the rest rebound: each remaining layer keeps its grains and gets the (thicker) decompacted thickness at its new, shallower position.  First form: one
    effective lithology per layer = the share-weighted constants."""
    keep = layers[n_remove:]
    out, top = [], 0.0
    for lay in keep:
        phi0 = sum(s * params[n]["phi0"] for n, s in lay["lithology"].items())
        c = sum(s * params[n]["c_per_km"] for n, s in lay["lithology"].items()) / 1000.0
        solid = float(solid_thickness_m(lay["top_m"], lay["base_m"], phi0, c))
        th = decompacted_thickness_m(solid, top, phi0, c)
        out.append({**lay, "top_m": top, "base_m": top + th})
        top += th
    return out
