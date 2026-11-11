# Pressure, HPHT and fracture gradient

Screening of reservoir pressure, temperature and fracture gradient from the Sodir drill stem tests (DST), leak-off / formation integrity tests (LOT / FIT) and the wellbore header. Config: `config/pressure.yaml`; code: `pressure`; marts `well_dst_pressure`, `well_pressure_summary`, `play_pressure_summary`; the analysis notebooks section 4b. The set-up is ready for enrichment (TVD, seabed depth, formation pressures): see `docs/enrichment.md`.

## 1. Units: MPa versus bar (DQ-075 resolved)

The Sodir DST headers carry no units. The pressures are **MPa** and are converted to bar:
- The median bottom-hole pressure / MD gradient is **0.103 bar/m** (259 tests). Hydrostatic is 0.098-0.108 bar/m; read as bar the gradient would be 0.010 bar/m, converted twice 1.0 bar/m. The check runs on every build (`pressure_unit_check`; `pressure_unit_suspect` warns outside 0.085-0.125).
- 83 % of the raw values are whole MPa, so they are rounded field values; a few rows are probably delivered in bar (for example 34/7-4 test 1: 385 delivered, 3,850 bar after conversion, but 385 bar at 2,555 m is 1.5 x hydrostatic). Rows like this are kept, flagged `rejected: possibly delivered in bar (not MPa)` and not used.

## 2. Which pressure is the reservoir pressure?

Only the **bottom-hole pressure** is. The final **shut-in** and **flowing** pressures are mostly **wellhead** values: where a test has both, shut-in / bottom-hole is a median **0.36 for oil tests and 0.77 for gas tests** (the lighter the fluid column above the gauge, the closer to bottom hole), which is exactly what a wellhead value gives; a subset has a ratio near 1 and is a bottom-hole value. Each shut-in is classified from its own test:
- `bottom_hole` (ratio to the bottom-hole pressure at least 0.92) or `wellhead` (lower), with the evidence;
- with no bottom-hole pressure: `bottom_hole_probable` when the gradient is at least 0.085 bar/m (low confidence), else `wellhead_probable`.
Reservoir pressure = bottom-hole pressure (high confidence), else a shut-in classified as a bottom-hole value (medium or low). A wellhead pressure is never a reservoir pressure. The text-derived flowing pressure disagreement (median 0.46 x official, DQ-075) is the same effect.

**Screen:** a reservoir pressure below 0.5 x or above 2.2 x the hydrostatic pressure is kept in `rejected_pressure_bar` with the reason and is not used (ASSUMPTION limits). Of 262 bottom-hole values, 21 are far too low for a static pressure (70-100 bar at 3,000-4,700 m: wellhead or flowing values entered in the bottom-hole column), 5 are far too high and 1 is probably in bar.

### Four pressures, and what Sodir does not say

The pipeline separates **SIWHP** (shut-in wellhead), **SIBHP** (static shut-in bottom-hole), **FWHP** (flowing wellhead) and **FBHP** (flowing bottom-hole): `well_dst_pressure.siwhp_bar / sibhp_bar / fwhp_bar / fbhp_bar`. For the shut-in and flowing columns the class is inferred from the test (ratio to the bottom-hole value, else the gradient; `pressure_definition_source = inferred from ratio / gradient`). For the **bottom-hole** column Sodir does not say whether it is a static or a flowing pressure: `bhp_kind = unspecified (static or flowing)` and `static_bhp_confirmed = False`. A flowing value is lower than the static one, so it can mimic underpressure: that question **waits for enrichment**. The sheet `dst_pressure_detail` takes SIWHP, SIBHP, FWHP, FBHP and `bhp_kind` from the DST report: a stated SIBHP or static bottom-hole pressure becomes the reservoir pressure (`high`, manual), a bottom-hole pressure stated as flowing is never used as a reservoir pressure.

## 3. Overpressure and regimes

Hydrostatic pressure = 0.1005 bar/m (1.025 sg, ASSUMPTION) x depth below **sea level** (RT elevation removed; water depth is part of the path, so seabed depth is not needed for the regime, only for the geothermal gradient). `pressure_ratio` = pressure / hydrostatic, `overpressure_bar` = difference, `pore_emw_sg` = pressure / (0.0981 x depth) on the same RT basis as a leak-off test.

| Regime | Ratio |
|---|---|
| underpressured (Barents Sea only) / subnormal (elsewhere) | below 0.95 |
| normal | 0.95-1.08 (North Sea formation water is 1.03-1.07 sg) |
| mildly overpressured | 1.08-1.30 |
| overpressured | 1.30-1.60 |
| highly overpressured | above 1.60 |

**Underpressure in the Barents Sea.** Pore pressure below hydrostatic is a real regime in the Barents Sea (uplift and erosion; for example the Hoop area and the Triassic intervals). There a ratio below 0.95 is labelled `underpressured` (`underpressure.areas` in the config, matched on the Sodir main area), and `underpressure_static_unconfirmed` marks the values whose bottom-hole pressure is not known to be static. Elsewhere a low value is a **screening suspect** (wellhead or flowing value, depletion, MD depth): below 0.80 the confidence is lowered to low and the flag `low_pressure_outside_underpressure_area` is set; the regime stays `subnormal`. Current data: 4 of the 12 Barents Sea tests with a pressure are underpressured (ratio 0.67-0.84), 62 + 16 North Sea and Norwegian Sea tests are screening suspects.

A test inside a hydrocarbon column also reads above the water line (buoyancy, up to +0.05 to 0.1 sg for a gas column of some hundred metres), so the lowest overpressure class is not proof of real overpressure. The common feature of hydrostatic and overpressured fields is exactly this: a normal regime at the water leg and an apparent overpressure in the column; compare tests below the contact when they exist.

## 4. Depth basis (no survey, no true vertical depth)

TVD is never fabricated. `depth_used_m` is TVD from a survey or manual point (`tvd_survey`, high confidence), else MD with a basis: `md_near_vertical` (reported TD TVD / TD MD at least 0.98: MD is a TVD proxy within 2 %, medium), `md_deviated` or `md_no_tvd` (low: MD overestimates depth, so hydrostatic is overestimated and **overpressure is a minimum**, `overpressure_is_minimum`). A missing RT elevation is replaced by 25 m and flagged `rt_assumed`. In the current data 213 of the 287 tests with a reservoir pressure are near vertical, 26 are in deviated wells and 48 in wells with no TD TVD (74 minima).

## 5. HPHT and temperature classes

Pressure and temperature are classed separately (HSE-type thresholds: 690 / 1,034 / 1,379 bar = 10,000 / 15,000 / 20,000 psi and 149 / 177 / 204 degC = 300 / 350 / 400 F; ASSUMPTION) and a test or well is **HPHT** only when both exceed the class threshold; `hp_ht_pattern` also says *high pressure only*, *high temperature only* or *unknown*. Temperature is the DST downhole temperature, else the header bottom-hole temperature, which is uncorrected (a minimum, labelled `header_bht_uncorrected_minimum`). Result: 7 HPHT wells, 18 hot-only, 1 high-pressure-only; most wells with a high bottom-hole temperature have no pressure, so they cannot be classed.

## 6. Hydrostatic gradient per well and geothermal gradient

Per well with at least 3 reservoir pressures over at least 150 m: slope and r2 of pressure against depth below sea level (`gradient_bar_per_m`), with an indicative fluid hint (gas below 0.05, oil 0.05-0.092, water 0.092-0.115, overpressured water above) only when r2 is at least 0.7. Only 16 wells qualify, so most wells have no gradient. Geothermal gradient = (T - 4 degC) / depth below seabed (needs water depth; MD proxy).

## 7. Fracture gradient and mud window

Source: casing shoes with a leak-off (LOT) or integrity (FIT) test (`well_casing_lot`, equivalent mud density, 1,554 wellbores, median 1.79 sg; values outside 0.9-2.6 sg are listed and not used). `fracture_gradient_bar_per_m` = equivalent density x 0.0981 at the **deepest** test. A FIT is only a lower bound. The **mud window** = leak-off density minus the pore-pressure density of the deepest test; it is a **minimum** because the shoe is shallower than the test and fracture gradients rise with depth. 106 wellbores have both; `narrow_window` below 0.10 sg. LOT density is a gradient valid at TVD, so the lack of TVD matters less here than for pore pressure.

## 8. What would improve it (set-up for enrichment)

`well_pressure_summary.missing` lists, per well, what is missing: survey (TVD), `td_tvd_m`, water depth, bottom-hole or formation pressure, downhole temperature, leak-off test. Fill the sheets `well_depth` and `formation_pressure` of the data-gap workbook or provide a canonical `well_survey` (MD, TVD): the next build uses them with no code change, and the depth basis changes to `tvd_survey`.

## 9. Limits

Pressures at one depth in one well are not a pressure profile; DST bottom-hole pressure may be a final flowing rather than a static value (the data do not say). No formation-pressure (RFT / MDT) table exists in the open Sodir data. Regimes are screening classes, not pore-pressure prediction.

## 10. Buoyancy: a test in a hydrocarbon column against the aquifer pressure (added 2026-11-04)
A test in a gas or oil leg reads above the water pressure of the same depth because the light fluid has a smaller gradient than water; the excess over hydrostatic is then the column, not overpressure (Zhang, knowledge `zh-hydraulic-connection`). `add_buoyancy_correction` (`pressure`, config `buoyancy`) translates the test pressure along the hydrocarbon gradient to the contact below it:

* `aquifer_pressure_bar = p_test + g_hc x (z_contact - z_test)` (a gas leg over an oil leg is two segments: the GOC, then the oil-water contact); `aquifer_ratio` is that pressure over the hydrostatic pressure at the contact and `aquifer_regime` its regime;
* `buoyancy_excess_bar = (g_water - g_hc) x (z_contact - z_test)` with a low-high range from the gradient range; `overpressure_beyond_column_bar` = the raw excess minus the column effect (the overpressure of the aquifer itself);
* **contacts** are the ones stated in the well-history text (`well_history_fact`, OWC, GOC, GWC, HWC; ODT / GDT are down-to depths and give a MINIMUM column, `buoyancy_is_minimum`), converted to TVDSS (`well_contacts_tvdss`: a stated TVDSS or MSL depth as it is, a stated TVD read as below RT, an MD only with a survey or for a near-vertical well, never fabricated); the nearest contact below the test within `max_column_m` (400 m) is used, one further down is another accumulation;
* **fluid and gradients (from the fluid, not the area):** the fluid is the GOR hint of the test (or the fluid of a manual point); the gradient is the in-situ density of that fluid times g (`fluid_gradients`, config `fluid_gradients`). Gas: the real-gas law, density = p M / (Z R T), with M from the DST gas gravity (default 0.70), the test pressure and temperature and a Z band of 0.80-1.00 (0.90 central). Oil: mass balance, (stock-tank density + Rs x gas density at standard conditions) / Bo, with the DST stock-tank density, Rs = the test GLR and a Bo band per PVT fluid class (black oil 1.05-1.40, volatile oil 1.45-2.50, near-critical 2.0-3.5). Where a test has none of these the configured band is the fallback (`hc_gradient_basis` says which). These gradients are set by PVT and so by the source rock and its maturity (marine type II sources give oil, coal-rich type III gas and condensate: `fluid_source_rock_literature`, cited not read), which is why they follow the PVT class and not the area. Result: in-situ gradients of about 0.077 (black oil), 0.067 (volatile oil), 0.031-0.033 (condensate, wet gas) and 0.014 bar/m (dry gas).
* **water gradient and salinity:** the central water gradient is the hydrostatic gradient (1.025 sg, about seawater); the sensitivity band is fresh water to a strong brine (0.0990-0.1052 bar/m), which moves the buoyancy excess range and, if used as the aquifer gradient, the aquifer ratio by up to 7 %. See section 13.
* **No contact, no correction:** `buoyancy_status` says why (`no contact on record`, `nearest contact below is further than the column window`, `test below the stated contact`, `fluid of the test unknown`); a hydrocarbon test at or above ratio 1.04 with no contact is flagged `buoyancy_unknown_contact`. Nothing is corrected by a guess; the raw `pressure_regime` stays next to `aquifer_regime`.
* **Result (2026-11-04):** of 287 reservoir pressures, 69 are in a gas or oil leg with a contact below the test (65 full, 4 minimum); the column effect is 0.04-12 bar (median 1.2 bar) and moves 5 tests to a lower regime. The effect is small because the columns below a test are short (median 42 m); it matters for tests high in tall columns (gas columns of 150-250 m: 11-12 bar, 0.05 sg).

## 11. Free water level and seal capacity (pressure-depth plots)
`pressure_gradients` (config `fwl`, mart `well_free_water_level`): the pressure data indicate the **free water level** (FWL), the depth where a hydrocarbon gradient crosses the water gradient (Swarbrick et al. 2005, knowledge `sw05-gradients-and-contacts`). It is the most robust placement of a contact because it needs no log or sample reading. `free_water_level` fits one line per fluid to formation-pressure points (depth TVDSS, pressure bar, fluid), reports the crossing with a 5-95 % bootstrap range, the gradients as densities, and a status (`gradients do not cross`, `outside the tested range`, a gradient outside the plausible band). A fluid with one point only (water) uses the assumed gradient; two points per fluid is the rule of the literature.

* The oil-water contact seen on logs lies ABOVE the FWL by the height the entry pressure of the rock holds up (`contact_above_fwl_m`); the FWL is where the capillary pressure is zero (AAPG wiki, cited not reviewed, `aapg_fwl_pressure`).
* **Seal capacity** as a column: `max_column_height_m = seal entry pressure / (water gradient - hydrocarbon gradient)`; the mechanical limit is the fracture pressure minus the pore pressure (cited, not reviewed: `seal_capacity_literature`). A leak-off test (`well_casing_lot`) gives the fracture side; the entry pressure needs mercury-injection or similar data.
* **Data:** Sodir has no formation-pressure table, so the results exist where points are supplied (sheet `formation_pressure`, column `fluid`; docs/enrichment.md). The function and tests are ready; the mart is written when points exist.
* **Reading rules** (knowledge `sw05-*`): a pressure transition zone (gradient above any fluid gradient), a regression (overpressure falling with depth: lateral escape), compartments (same level, different aquifer pressure), hydrodynamic tilt (one hydrocarbon gradient, offset water gradients: a tilted contact, not an error).

## 12. Planned ingestion of FIT, LOT and XLOT and of formation pressures (decision of 2026-11-04)
The two papers in `knowledge/pressure.yaml` (Al Ramadan 2024, Zhang) prepare a richer source than the Sodir casing table. When the data arrive they are new canonical entities (additive, never a change of an existing column):

| Entity | One row per | Columns it needs (from the literature) |
| :--- | :--- | :--- |
| `well_formation_strength_test` | FIT / LOT / XLOT cycle | wellbore, test type (FIT, LOT, XLOT), shoe depth MD and TVD (m RT), mud type (WBM / OBM / SBM), mud weight at surface and downhole (sg), where the pressure was recorded (surface at cement unit, downhole PWD), leak-off point, fracture initiation, breakdown (ultimate), propagation, ISIP, closure and reopening pressures (bar), permeable or impermeable formation, temperature, comment, source |
| `well_formation_pressure` | RFT / MDT / XPT / gauge point | wellbore, depth MD and TVD (m RT), pressure (bar), fluid (water, oil, gas), tool, quality (supercharged, tight), temperature, source (already the manual sheet `formation_pressure`) |

Rules ready for them: the minimum horizontal stress is the closure pressure, read with the right plot (square root of time in permeable rock, flowback or added volume in impermeable rock); pressures are corrected for gel strength and for the cement-unit-to-reference head, and the mud type is kept (`ar24-*`); fracture pressure is bracketed by the lower bound (minimum stress) and the upper bound (breakdown) of `zh-fracture-bounds`; the aquifer-equivalent pressure of section 10 and the FWL of section 11 are the pressure-depth plot and the seal screen (fracture pressure minus aquifer pressure, and the column the seal can hold). Burial history (compaction, uplift) explains overpressure and underpressure (`zh-overpressure-mechanisms`, `zh-unloading-methods`). Until then: no minimum stress, no tensile strength and no seal capacity is derived from the casing table.

> Pore pressure, overburden, fracture gradient, the drilling window, bar against sg and the datum: see `docs/pore_pressure.md`.

## 13. The water gradient and salinity (researched 2026-11-04)
Formation water is denser the more salt it holds, so its gradient rises with salinity: the anchor values in `fluid_gradients.water.salinity_anchors` (fresh water 0.433 psi/ft = 0.0980 bar/m, 55,000 ppm 0.450 psi/ft = 0.1018, 88,000 ppm 0.465 psi/ft = 0.1052 bar/m) come from the AAPG wiki as seen in a search result (`aapg_formation_water_density`, page not fetched: **to be confirmed**). Formation-water salinity on the shelf spans about 1,000 to 400,000 ppm against about 30,000 ppm for seawater (`dy20-formation-water-density`); temperature lowers the density and pressure raises it, so a hot deep aquifer is lighter than the same brine at the surface, and a small gas content has no visible effect. The full correlations (Batzle and Wang, McCain, Rowe and Chou) could not be read, so none is implemented: `fluid_gradients.water_gradient(salinity_ppm)` interpolates between the anchors (not extrapolated) and `salinity_from_gradient` inverts it (a water line fitted in an aquifer gives its salinity).

Why it matters (the analysis notebooks, section 4): the same aquifer pressure reads a regime lower against a brine than against fresh water (median aquifer ratio 1.19 against 1.11 over the 69 corrected tests, a shift of 0.08 that crosses the normal limit of 1.08), and a free water level found from a water point 200 m away moves by 27 m in an oil leg and 9 m in a gas leg for a water gradient error of 0.0035 bar/m: a water sample is worth more where the leg is oil. Salinity is not in the Sodir data; a formation-water analysis or an RFT / MDT water sample would give it per aquifer.
