# Pore pressure, overburden, fracture gradient and the drilling window

Code `src/processing/pore_pressure.py`, `src/processing/pressure_units.py`, `src/utils/pressure_plots.py`, config `config/pore_pressure.yaml`, notebooks 22 (column plots) and 24 (units, mud weight, the window), knowledge `knowledge/pore_pressure.yaml`. Requested by Adam (2026-11-04): learn how each quantity is defined, calculated and estimated, what data it needs, and set up pore pressure plots that combine with the column plots at reservoir level.

## 1. Definitions
| Quantity | Definition | Source |
| :--- | :--- | :--- |
| **OBG** overburden gradient (vertical stress, sigma_v) | the weight of the water column and all the rock above a depth, a gradient when divided by the depth; about 0.20-0.24 bar/m for rock against 0.10 bar/m for water | Zhang (read, 2011?) and the offshore integration recipe in search results; Sodir slides: lithostatic from compaction trends |
| **PP** pore pressure | the fluid pressure in the pore space (or in fractures and cavities); hydrostatic when connected to the surface, overpressured above, underpressured below | Zhang (read), Swarbrick et al. 2005 (read), Sodir slides (read) |
| **FG** fracture gradient | the pressure that induces fractures at a depth; bounded below by the minimum stress (the pressure to reopen a fracture, the closure pressure) and above by the breakdown pressure (tensile strength, temperature) | Zhang (read), Everhard et al. (read), Al Ramadan (read) |
| **SHmin** minimum horizontal stress | the lowest principal stress in a normal-faulting regime; a fracture grows once the mud pressure exceeds it | Everhard et al., Zhang |
| **FCP** fracture closure pressure / stop of losses | the pressure at which an open fracture closes; equals SHmin; the conservative limit of mud pressure | Everhard et al., Al Ramadan |
| **SFG** shear failure gradient (collapse gradient) | the lowest mud pressure that keeps the borehole wall from shear failure: Mohr-Coulomb strength against the Kirsch stresses around the hole | Everhard et al. (data needed); formula from search results, not read |
| **Drilling window** | the safe mud weight: from max(PP, SFG) up to the fracture gradient (FCP / SHmin as the conservative upper limit) | Everhard et al. (read); search results: the lower bound is the larger of PP and SFG |
| **ESD** equivalent static density | the density of a uniform mud column that gives the static pressure of the real column (the surface mud weight corrected for compressibility and thermal expansion) | not defined on the IADC page; used here as stated |
| **ECD** equivalent circulating density | ESD plus the annular friction pressure loss and any back pressure, over the depth of interest (API RP 59: hydrostatic + drilled solids + annular friction losses, divided by the depth) | IADC Lexicon (read) |
| **Overbalance** | mud pressure minus pore pressure; positive = the mud holds the formation | Everhard et al. |

## 2. How they are calculated or estimated, and what data each needs
* **Overburden.** Integrate the bulk density: sigma_v(z) = rho_w g z_w + the integral of rho_b g dz, with the seawater column above the seabed (and the air gap from the rotary table as a separate term), a corrected density log (spikes removed) and an estimated trend between the seabed and the top of the log, since logs rarely reach the seabed (search results). Needs a density log (or a sonic-derived density), water depth and air gap. **Here:** water depth and RT are in the header; with no density log the density follows an exponential compaction trend of the porosity (`config/pore_pressure.yaml`, assumptions), the kind of trend the Sodir slides use for their lithostatic line.
* **Pore pressure.** Direct: formation pressure (RFT, MDT, XPT) and DST, read in sands. Indirect, in shales: Eaton (resistivity or sonic against a normal compaction trend, exponents 1.2 and 3 as printed in Zhang), Bowers and Miller (velocity, with the unloading branch for uplifted rock), the Tau model, a theoretical porosity-based model, and basin modelling (compaction disequilibrium, hydrocarbon generation, unloading; Zhang, Sodir slides). Needs shale logs and a normal-compaction trend, or pressure points. **Here:** DST pressures (about 290) and the aquifer-equivalent pressure after the column effect (notebook 22); no log-based trend.
* **Fracture gradient and minimum stress.** Measured: leak-off test (a lower-ish value of the curve), extended leak-off test (closure pressure = SHmin, read with the square-root-of-time plot in permeable rock and flowback or added-volume plots in impermeable rock), mini-frac. Predicted: lower bound sigma_min = nu/(1-nu)(sigma_v - p) + p, upper bound 2 sigma_min - p, mean = most likely (Zhang; Hubbert-Willis, Matthews-Kelly and Eaton are the classic forms, not read). Needs sigma_v, pore pressure and Poisson's ratio (from compressional and shear velocity). **Here:** one LOT / FIT equivalent density per casing shoe (the point of the curve is not stated) and a Poisson band of 0.25-0.35 (assumption).
* **Shear failure gradient.** Mohr-Coulomb strength sigma_1 = UCS + sigma_3 tan^2(45 + phi/2) against the Kirsch stresses at the wall; for a vertical well the critical wellbore pressure is a closed form of the horizontal stresses, UCS and the friction coefficient (a form was seen garbled in search results: **check against Zoback or Fjaer before use**); the Mohr-Coulomb and Tresca criteria give the highest (most conservative) minimum mud pressure. Needs UCS, friction angle, the three stresses, pore pressure, the azimuth and inclination of the well. **Here:** none of it: SFG is not computed, the window lower limit is the pore pressure only (a minimum).
* **Window, overbalance, ESD, ECD.** Window = fracture limit minus max(PP, SFG). ESD needs the mud PVT and temperature; ECD needs the hydraulics (flow rate, rheology, geometry) or a downhole annular pressure measurement (PWD). **Here:** the Sodir mud table (surface density against measured depth, 36,743 rows, canonical `well_mud`) with ESD and ECD bands (assumptions).

## 3. The datum
Formation pressure and the water column stand on mean sea level; the mud column and the leak-off tests stand on the rotary table. `src/processing/pressure_units.py` converts between the bases (`EMW_RT = p / (g (z_MSL + rt))`, `EMW_MSL = p / (g z_MSL)`); seawater is 1.025 sg on the MSL basis and 1.012 sg on the RT basis at 2000 m with a 25 m RT. Every figure names its basis (notebooks 22, 24).

## 4. The plots
* **Column plot (notebook 22):** pressure in bar with the fan of constant-sg lines, equivalent density, and overpressure with the hydrostatic gradient as the zero line; hydrocarbon gradients annotated in bar/m and sg; the modelled envelope of a proxy fluid around every leg (notebook 23, `column_envelope_proxy`), shown as a band and a bar on the aquifer-equivalent point.
* **Pore pressure plot (notebook 24):** the whole well in equivalent density (hydrostatic, modelled overburden, the Sodir leakage line = overburden minus about 100 bar, pore pressure, leak-off tests, mud weight with ESD and ECD bands, the fracture band at the tested depths) next to the reservoir-level column plot. The window is the gap between the pore pressure and the fracture band; SFG is not drawn.
* **What the Sodir slides add:** multi-well pressure-depth plots with the lithostatic and fracture-gradient lines; a leakage line about 100 bar below lithostatic for water depths of 250-400 m (closer to lithostatic in shallow water); the dependence of lithostatic and fracture pressure on water depth; a maximum hydrocarbon column fixed by the top of the pressure cell; hydrodynamic aquifers with a common gas gradient and different water pressures per well (Ormen Lange) and different pressures and contacts per segment (Valemon, Goliat): a free water level can differ between wells and is not an error (knowledge `riis-overpressure-regimes`).

## 5. Limits
The overburden is modelled, not measured; the Poisson band and the ESD / ECD bands are assumptions; the shear failure gradient and the minimum stress from closure pressure are not available; the leak-off point of Sodir's LOT value is not stated (a lower-ish estimate of the fracture pressure); mud weight is the surface density; DST pressures are at the middle of the tested interval. The window of notebook 24 is therefore an upper bound of the safe range. The reading list that would remove these limits is `docs/literature_to_acquire.md`.

## 14. Pore pressure line style guide (Adam, 2026-11-05; amended the same day)

Every pore pressure figure draws these quantities in these styles (`config/pore_pressure_style.yaml`, `src/utils/pp_style.py`; the whole-well plot, the pressure-depth triplet and the pore pressure figures of notebooks 24, 25 and 27 use it):

| Quantity | Style | Definition used |
| :--- | :--- | :--- |
| Sh | green line | minimum horizontal stress, central Poisson's ratio |
| Sh min | green dashed | the same with the lowest Poisson's ratio of the band |
| FG | **dark blue** line | most likely fracture pressure (mean of the lower and upper Zhang bound), central ratio |
| FG min | dark blue dotted | most likely fracture pressure with the lowest ratio |
| FG max | dark blue dashed | most likely fracture pressure with the highest ratio |
| SFG | purple line | shear failure gradient (rock strength and stresses: not held, so not drawn until data exist) |
| OBG | dark red-brown line | overburden gradient (modelled trend, or a density log when held) |
| MW | solid black line | mud weight (surface density, RT basis) |
| **Hydrostatic** | **light blue** line (dashed strong brine, dotted fresh water) | the water gradient: the normal-pressure reference |
| PP expected | red line | a **predicted** pore pressure away from hydrostatic (trend from logs, basin model) |
| PP max | red dashed | the high end of that prediction |
| PP min | red dotted | the low end of that prediction |

**Hydrostatic is not the expected pore pressure.** Until a prediction away from hydrostatic exists (shale logs, a basin model), the reference is called and drawn as *hydrostatic* (light blue); the red PP family is reserved for a real prediction and is not drawn today. Test pressures and the aquifer-equivalent points are data, not predictions. A new pore pressure figure takes its styles from `pp_style.line_style(key)`; a test checks the mapping.
