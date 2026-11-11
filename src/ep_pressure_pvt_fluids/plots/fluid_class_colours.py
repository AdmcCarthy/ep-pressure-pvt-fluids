"""Colours of the fluid classes for figures that compare them (config/fluid_class_colours.yaml, docs/geostandard.md).

    class_colour("volatile oil")            # "#7CB518"
    fluid_style("hpht_volatile_oil")        # ("#7CB518", "--")   the same hue as the class, dashed for the HPHT variant
    CLASS_ORDER                             # black oil ... dry gas, the order of the fluid continuum

The plain Geostandard oil / gas / condensate colours (`hc_colours`) are still the colours of a single hydrocarbon type; this scale is for telling the six classes apart.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from ep_physics_base.config import load_yaml

CONFIG_FILE = "config/fluid_class_colours.yaml"       # read through load_yaml: the project's file, else the packaged default of a library


@lru_cache(maxsize=1)
def load_fluid_class_colours() -> dict[str, Any]:
    return load_yaml(CONFIG_FILE)


CLASS_ORDER = [c["name"] for c in load_fluid_class_colours()["classes"]]


def class_colour(name: str, default: str = "#6083A8") -> str:
    """Colour of a fluid class (the Geostandard `unknown` blue-grey for a name that is not a class)."""
    for c in load_fluid_class_colours()["classes"]:
        if c["name"] == str(name).lower():
            return c["colour"]
    return default


def fluid_style(key: str) -> tuple[str, str]:
    """(colour, linestyle) of an example fluid of config/column_models.yaml: the class colour, dashed for the HPHT variants."""
    spec = load_fluid_class_colours()["fluid_keys"][key]
    return class_colour(spec["class"]), "--" if spec["dashed"] else "-"


def fluid_label_class(key: str) -> str:
    return load_fluid_class_colours()["fluid_keys"][key]["class"]