"""Leak-off / integrity test symbols, one set for every figure (config/shoe_test_symbols.yaml).

LOT, FIT, XLOT and an undifferentiated test (type not stated) each have their own marker shape; colour stays the project orange.  Pure helpers; plotting only adds markers."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from ep_physics_base.figures import theme
from ep_physics_base.config import load_yaml

ORDER = ["LOT", "FIT", "XLOT", "UNDIFFERENTIATED"]


@lru_cache(maxsize=1)
def _cfg() -> dict[str, Any]:
    return load_yaml("config/shoe_test_symbols.yaml")


def norm_type(types) -> pd.Series:
    """Map raw test types to LOT / FIT / XLOT / UNDIFFERENTIATED (aliases and the missing value come from the config)."""
    c = _cfg()
    s = pd.Series(types, copy=True).astype("string").str.strip().str.upper().fillna("")
    s = s.replace({str(k).upper(): str(v).upper() for k, v in c["aliases"].items()})
    return s.where(s.isin(c["types"].keys()), "UNDIFFERENTIATED").astype(str)


def scatter_tests(ax, x, y, types, s: float = 34.0, label: bool = True, zorder: int = 6) -> dict[str, int]:
    """Draw the tests with the shape of their type; returns the count per type.  `label=True` adds one legend entry per type present."""
    c = _cfg()["types"]
    x, y, t = np.asarray(x, dtype=float), np.asarray(y, dtype=float), norm_type(types).to_numpy()
    counts: dict[str, int] = {}
    for key in ORDER:
        m = t == key
        if not m.any():
            continue
        counts[key] = int(m.sum())
        ax.scatter(x[m], y[m], marker=c[key]["marker"], s=s, facecolor=theme.ORANGE if c[key]["filled"] else "white", edgecolor=theme.NORTH_SEA, linewidth=0.8 if c[key]["filled"] else 1.3,
                   zorder=zorder, label=c[key]["label"] if label else None)
    return counts


def legend_handles(present: list[str] | None = None):
    """Legend handles for the test types (all of them, or those in `present`), in the fixed order."""
    from matplotlib.lines import Line2D

    c = _cfg()["types"]
    keys = [k for k in ORDER if present is None or k in present]
    return [Line2D([0], [0], marker=c[k]["marker"], color="w", markerfacecolor=theme.ORANGE if c[k]["filled"] else "white", markeredgecolor=theme.NORTH_SEA, markeredgewidth=0.8 if c[k]["filled"] else 1.3,
                   markersize=7, label=c[k]["label"] + " (measured)") for k in keys]