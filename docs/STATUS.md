# Status of ep-pressure

**State.** Cut out of the E&P data engine on the date of the first commit: 9 modules, 6 test files. The engine switched over on 2026-11-11: it depends on this library (its old module paths are aliases of the modules here), and this repository is the source of truth.

## Decisions (Adam, 2026-11-11)
* Domain libraries, one per topic; this one is `ep-pressure`. Figures belong to their domain.
* Private until the publication gate passes; built public-ready. Licence MIT with acknowledgement.
* One-way design: the library never imports a knowledge store; assumptions travel as packaged defaults with a level, a one-line basis and an author-and-year citation where there is one.
* The hydrocarbon colours of the Sodir Geostandard are used as they are.

## Changes made when the code was cut out
* `processing.pore_pressure`: def load_config(overlay: Any = None) -> dict[str, Any]:

## Tests that moved only in part (the rest use the engine and stay there)
* `test_pressure_units.py: kept 7, 1 stay in the engine`

## Couplings still to cut
* none: every `src` import was mapped.

## Next steps
* Add the knowledge identifiers used here to `KNOWLEDGE.md` and export the defaults from the knowledge store.
* Run `scripts/publication_gate.py` and clear the warnings before publishing.

## Changes after the first cut (2026-11-11)
* ep-fluids 0.2.0 / ep-pressure 0.2.0: the HPHT pressure-temperature chart (ep_pressure.plots.pt_chart) and the PVT dashboard pages (ep_pressure.plots.pvt_dashboard, pvt3d_dashboard) live in ep-pressure because they need both libraries; ep_fluids.plots.pvt_plots keeps the GLR, API, map and 3D surface figures with public marker helpers; ep-pressure has egister_overlay for a host's configuration overlay.

