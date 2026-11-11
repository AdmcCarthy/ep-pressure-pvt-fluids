"""Colours of the PVT charts, shared by the dashboards (scripts/build_pvt_dashboard.py, build_pvt3d_dashboard.py) and the notebook figures.

Zone fills are pastel and flat per zone (the pages blend them across the boundaries); the points use the stronger EDS colours.  Light theme.
"""

from __future__ import annotations

from ep_physics_base.figures import theme

YELLOW = "#F2C14E"
GREY = "#8A98A3"
STATE_COLOURS = {"undersaturated oil": theme.GREEN, "saturated oil": theme.MOSS_GREEN, "critical fluid": "#B8A100", "saturated gas": theme.ORANGE, "undersaturated gas": theme.RED, "forbidden": theme.NORTH_SEA}
ZONE_COLOURS = {"forbidden": "#E4E8EB", "diphasic: oil with gas cap": "#EBF3C4", "diphasic: gas with oil ring": "#FBE2DA", "monophasic: undersaturated oil": "#CFE8C6",
                "monophasic: critical fluid": "#FBF0B4", "monophasic: undersaturated gas": "#F7CBC3"}
ZONE_OF_STATE = {"forbidden": "forbidden", "saturated oil": "diphasic: oil with gas cap", "saturated gas": "diphasic: gas with oil ring", "undersaturated oil": "monophasic: undersaturated oil",
                 "critical fluid": "monophasic: critical fluid", "undersaturated gas": "monophasic: undersaturated gas"}
PT_COLOURS = {"normal": "#DDEFD6", "high temperature": "#FBE0B8", "high pressure": "#CBE3F2", "HPHT": "#F6B7A6", "ultra-HPHT": "#E88B8B", "extreme-HPHT": "#C95F7A"}
PT_LABELS = {"normal": "Normal pressure", "high temperature": "High temperature", "high pressure": "High pressure", "HPHT": "HPHT", "ultra-HPHT": "UHPHT", "extreme-HPHT": "Extreme HPHT"}
FLUID_COLOURS = {"black oil": theme.GREEN, "volatile oil": theme.MOSS_GREEN, "near-critical": theme.RED, "gas condensate": theme.ORANGE, "wet gas": YELLOW, "dry gas": theme.BLUE}
API_COLOURS = {"extra-heavy": theme.NORTH_SEA, "heavy": "#5B6B77", "medium": theme.BLUE, "light": theme.GREEN, "condensate-like": theme.ORANGE}
BRACKET_COLOURS = {"degradable": theme.BLUE, "transition": YELLOW, "pasteurised": theme.RED}
BASIS_COLOURS = {"test": theme.GREEN, "test (estimated temperature)": theme.MOSS_GREEN, "partly reference": theme.ORANGE, "reference": GREY}
REGIME_COLOURS = {"liquid": theme.GREEN, "vapour": theme.RED, "supercritical": "#B8A100", "two-phase": theme.NORTH_SEA}          # regime of a point on the surface of a model fluid
SURFACE_COLOURS = {"liquid": "#CFE8C6", "two-phase": "#E4E8EB", "vapour": "#F7CBC3", "supercritical": "#FBF0B4"}                  # pastel colours of the 3D surface regions
