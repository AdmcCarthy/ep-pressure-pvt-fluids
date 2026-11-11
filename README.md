# ep-pressure-pvt-fluids: what the pores push back with and what fills them

Pore pressure, pressure regimes, the drilling window and their figures; PVT classification, equations of state, column grading, fluid gradients and measurement types; and the pages and charts that need both (the PVT dashboards, the HPHT pressure-temperature chart). One library because a pressure in a reservoir cannot be read without the fluid in it, and a fluid cannot be placed without its pressure.

**Status: private, built public-ready.** The library follows the rules of a public one (no host-repository imports, no data, no private paths, an automated publication gate in `scripts/publication_gate.py`) and is published only when that gate passes. Licence: MIT with acknowledgement (`LICENSE`).

## Install

```
uv add git+https://github.com/AdmcCarthy/ep-pressure-pvt-fluids
```

Depends on: `ep-physics-base`. Optional extras:

* `plots`: matplotlib, geopandas, shapely

## What is in it

Package `ep_pressure_pvt_fluids`.

| Module | What it does |
| :--- | :--- |
| `pressure` | Pressure regimes, HPHT classes, hydrostatic gradients and fracture gradients from drill stem tests, leak-off tests and the wellbore header. |
| `pore_pressure` | Pore pressure, overburden, fracture pressure and the drilling window (`register_overlay` lets a host replace configured values). |
| `pressure_gradients` | Pressure-depth gradients by fluid, the free water level where a hydrocarbon gradient meets the water gradient, and the column a seal can hold. |
| `backstrip` | Back-stripping, first form: decompaction and the weight of a lithology column. |
| `pvt` | PVT fluid classification of drill stem tests: API class, fluid class, the zones of the GLR-pressure plane (forbidden domain, diphasic bands, monophasic domains), saturation state, biodegradation bracket. |
| `eos` | Peng-Robinson equation of state of a pure component: saturation pressure, volumes, regime of a (T, P) point and the pressure-volume-temperature surface. |
| `eos_mixture` | Cubic equations of state for mixtures (Peng-Robinson, Soave-Redlich-Kwong, van der Waals mixing, binary interaction parameters, volume translation). |
| `column_grading` | Compositional grading of a hydrocarbon column and the envelope of its pressure profile. |
| `fluid_gradients` | In-situ gradients of gas, oil and formation water from the fluid and not from the area. |
| `measurement_types` | Which kind of measurement a PVT data point comes from, classified from official records only, with the issues of each type. |
| `plots.pressure_plots`, `plots.pp_style`, `plots.pretest_plots`, `plots.gauge_plots`, `plots.shoe_tests` | Pressure-depth figures, the pore pressure line style guide, pretest and permanent-gauge figures, leak-off test symbols. |
| `plots.pvt_plots`, `plots.pvt_colours`, `plots.fluid_class_colours` | GLR, API, map and 3D surface figures of the PVT tests, with their colours and test-marker helpers. |
| `plots.pt_chart` | Pressure against temperature with the HPHT classes. |
| `plots.pvt_dashboard`, `plots.pvt3d_dashboard` | The self-contained PVT pages (GLR and pressure; 3D pressure-volume-temperature surface), as pure functions from frames to HTML. |

Functions are pure: they take frames and parameters and return `(result, events)` where they change or flag data; nothing reads or writes files except the configuration loader.

## Settings and assumptions

Every setting has a packaged default in `src/ep_pressure_pvt_fluids/defaults/`. A project can override a file by providing `config/<same name>` in its working directory; the project file wins. A default is an assumption unless it says otherwise: its level (`placeholder`, `inferred`, `published`, `official`, `exact`) and a one-line basis travel with it, and a citation (author and year) is given where one exists. Libraries never import a knowledge store: a host that has one passes its values in as an overlay.

## Develop

```
uv sync --all-extras
uv run pytest
uv run ruff check .
uv run python scripts/publication_gate.py
```

Tests use small synthetic data generated on the fly. No network, no real data.

## History

Formed on 2026-11-11 from `ep-pressure` and `ep-fluids`; the earlier repository `ep-fluids` is archived.
