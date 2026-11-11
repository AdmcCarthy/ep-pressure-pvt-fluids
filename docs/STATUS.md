# Status of ep-pressure-pvt-fluids

**State.** Formed on 2026-11-11 by merging `ep-fluids` into `ep-pressure` (package `ep_pressure_pvt_fluids`, version 0.3.0): 11 modules, 14 plot modules and the tests of both. The engine depends on this library (its old module paths are aliases of the modules here), and this repository is the source of truth.

## Decisions (Adam, 2026-11-11)
* Domain libraries, one per topic; pressure, PVT and fluids form one library (Adam: merge `ep-fluids` in, name `pressure-pvt-fluids`). Figures belong to their domain.
* Private until the publication gate passes; built public-ready. Licence MIT with acknowledgement.
* One-way design: the library never imports a knowledge store; assumptions travel as packaged defaults with a level, a one-line basis and an author-and-year citation where there is one.
* The hydrocarbon colours of the Sodir Geostandard are used as they are.

## Changes after the first cut
* The HPHT pressure-temperature chart (`plots.pt_chart`) and the PVT dashboard pages (`plots.pvt_dashboard`, `plots.pvt3d_dashboard`) were added; `plots.pvt_plots` keeps the GLR, API, map and 3D surface figures with public marker helpers (`scatter_tests`, `types_of`, `estimated_mask`).
* `pore_pressure.register_overlay` lets a host replace configured values without the library importing a knowledge store.
* `ep-fluids` was merged in; it is archived and its content lives here (the earlier import names `ep_fluids` and `ep_pressure` are gone).

## Scope not yet covered
* Geochemistry (Rock-Eval, FIS screening, gas wetness) is not in this library; its engine code is a small Volve module (`ep-volve`). Adding it needs a source-independent form first.

## Next steps
* Add the knowledge identifiers used here to `KNOWLEDGE.md` and export the defaults from the knowledge store.
* Run `scripts/publication_gate.py` and clear the warnings before publishing.
