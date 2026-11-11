# ep-pressure: What the pores push back with: pressure regimes, the window and how to draw them

Pore pressure and pressure regimes, HPHT classes, buoyancy and column effects, free water level, compaction (back-stripping), and the figures and style rules of these: the pore-pressure line guide, shoe-test symbols, pretest and gauge plots.

**Status: private, built public-ready.** The library follows the rules of a public one (no host-repository imports, no data, no private paths, an automated publication gate in `scripts/publication_gate.py`) and is published only when that gate passes. Licence: MIT with acknowledgement (`LICENSE`).

## Install

```
uv add git+https://github.com/AdmcCarthy/ep-pressure
```

Depends on: `ep-physics-base`, `ep-fluids`. Optional extras:

* `plots`: matplotlib

## What is in it

| Module | What it does |
| :--- | :--- |
| `pressure` | Pressure regimes, HPHT classes, hydrostatic gradients and fracture gradients from drill stem tests, leak-off tests and the wellbore header. |
| `pore_pressure` | Pore pressure, overburden, fracture pressure and the drilling window.  Pure functions (docs/pore_pressure.md). |
| `pressure_gradients` | Pressure-depth gradients by fluid, the free water level where a hydrocarbon gradient meets the water gradient, and the column a seal can hold.  Pure f |
| `backstrip` | Back-stripping, first form: decompaction and the weight of a lithology column.  Pure functions (docs/knowledge_architecture.md section 5). |
| `plots.pressure_plots` | Pressure-depth figures with contacts (docs/pressure.md section 10; knowledge `sw05-gradients-and-contacts`).  One axis per well. |
| `plots.pp_style` | The pore pressure line style guide (config/pore_pressure_style.yaml): one place, used by every pore pressure figure. |
| `plots.pretest_plots` | Plots of formation pretests on the pressure-depth figures (notebooks 22 and 24): one marker per test, its shape and fill say how far the draw-down can |
| `plots.gauge_plots` | Figures of the permanent-gauge shut-in pressures (notebooks 22 and 24).  A shut-in value is the LAST daily average of a run of fully shut-in days: sha |
| `plots.shoe_tests` | Leak-off / integrity test symbols, one set for every figure (config/shoe_test_symbols.yaml). |

Functions are pure: they take frames and parameters and return `(result, events)` where they change or flag data; nothing reads or writes files except the configuration loader.

## Settings and assumptions

Every setting has a packaged default in `src/ep_pressure/defaults/`. A project can override a file by providing `config/<same name>` in its working directory; the project file wins. A default is an assumption unless it says otherwise: its level (`placeholder`, `inferred`, `published`, `official`, `exact`) and a one-line basis travel with it, and a citation (author and year) is given where one exists. Libraries never import a knowledge store: a host that has one passes its values in as an overlay.

## Develop

```
uv sync --all-extras
uv run pytest            # 6 test files
uv run ruff check .
uv run python scripts/publication_gate.py
```

Tests use small synthetic data generated on the fly. No network, no real data.
