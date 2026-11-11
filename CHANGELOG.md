# Changelog

## 1.0.0 (2026-11-11)

First public release.

## What is in it

- Pore pressure, overburden, fracture bounds and the drilling window; HPHT classes; buoyancy and the free water level.
- PVT classification of drill stem tests, Peng-Robinson and SRK equations of state, compositional column grading, in-situ fluid gradients.
- Fluid inclusion stratigraphy anomalies and Rock-Eval before and after solvent extraction (screening, with placeholder limits).
- Pressure-depth figures, the pore pressure line style guide, pretest and gauge plots, and the self-contained PVT pages (GLR, 3D surface).

## Things to know

- Limits and settings that the code reads are packaged defaults and are assumptions unless their level says otherwise (see `KNOWLEDGE.md` and the defaults folder); a project file of the same name overrides them.
- Functions are pure: frames and parameters in, `(result, events)` out; nothing reads or writes files except the configuration loader.
- Licence: MIT with acknowledgement (see `LICENSE`).
- Data names from the Norwegian Offshore Directorate (Sodir, formerly NPD) appear as the sources the code was built against.
