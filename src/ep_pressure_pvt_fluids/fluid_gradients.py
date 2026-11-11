"""In-situ pressure gradients of gas, oil and formation water (density x g), from the fluid and not from the area.  Pure functions.

Why (decision of 2026-11-04): the gradient of a hydrocarbon leg is the in-situ density of its fluid, which is set by PVT (the stock-tank density, the gas-oil ratio, the gas gravity, the formation
volume factor, pressure and temperature) and through them by the source rock and the maturity that made the fluid (marine type II kerogen gives oil, coal-rich type III gives gas and
condensate), not by the geographic area.  So the gradient follows the PVT fluid class of the test (`well_dst_pvt.fluid_class`: black oil, volatile oil, gas condensate, wet gas, dry gas) and
its measured API, GLR and gas gravity where they exist.

    gas  :  rho_g = p M / (Z R T)             M = gas gravity x M_air   (real-gas law; Z is a configured band: the Z factor itself needs a composition)
    oil  :  rho_o = (rho_sto + Rs x rho_gas_sc) / Bo      mass balance of stock-tank oil plus its dissolved gas, per reservoir volume; Bo >= 1 by class band
    water:  a gradient between fresh water and a strong brine (salinity); the temperature and pressure of the aquifer move it a little

A gradient is rho [kg/m3] x 9.80665 / 1e5 [bar/m].  Every band in `config/pressure.yaml` `fluid_gradients` is an ASSUMPTION or a unit conversion and says so.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

R_J_MOL_K = 8.314462618
G_M_S2 = 9.80665
M_AIR_G_MOL = 28.9647
P_STD_PA, T_STD_K = 101325.0, 288.15                      # standard conditions of the Norwegian shelf (15 degC, 1 atm)
RHO_AIR_STD = P_STD_PA * M_AIR_G_MOL / 1000.0 / (R_J_MOL_K * T_STD_K)      # kg/m3 (about 1.225)


def _gradient_from_density(rho_kg_m3: float) -> float:
    return rho_kg_m3 * G_M_S2 / 1.0e5


def gas_gradient(p_bar: float, t_c: float, gas_gravity: float | None, cfg: dict[str, Any]) -> tuple[float, float, float, str]:
    """(central, low, high, basis) for a gas leg at the test pressure and temperature; the band comes from the Z factor (central, and the Z that gives the low / high gradient)."""
    g = cfg["fluid_gradients"]["gas"]
    ok = gas_gravity is not None and pd.notna(gas_gravity) and g["gravity_plausible"][0] <= float(gas_gravity) <= g["gravity_plausible"][1]
    gravity = float(gas_gravity) if ok else float(g["gravity_default"])
    if not (pd.notna(p_bar) and pd.notna(t_c)):
        return (*[float(x) for x in cfg["buoyancy"]["fluid_gradient_bar_per_m"]["gas"]], "configured band (no pressure or temperature)")
    m_kg = gravity * M_AIR_G_MOL / 1000.0
    t_k = float(t_c) + 273.15

    def grad(z: float) -> float:
        return _gradient_from_density(float(p_bar) * 1e5 * m_kg / (z * R_J_MOL_K * t_k))

    zc, z_low_grad, z_high_grad = (float(x) for x in g["z"])
    lo, hi = float(g["clip_bar_per_m"][0]), float(g["clip_bar_per_m"][1])
    c, l, h = (float(np.clip(grad(z), lo, hi)) for z in (zc, z_low_grad, z_high_grad))
    return c, min(l, c), max(h, c), ("real-gas law, DST gas gravity" if ok else "real-gas law, default gas gravity")


def oil_gradient(sto_density_g_cm3: float | None, glr_sm3_sm3: float | None, gas_gravity: float | None, fluid_class: str | None, cfg: dict[str, Any]) -> tuple[float, float, float, str]:
    """(central, low, high, basis) for an oil leg: stock-tank density from the DST, the dissolved gas from the gas-liquid ratio, Bo from the band of the PVT fluid class."""
    o = cfg["fluid_gradients"]["oil"]
    if not (sto_density_g_cm3 is not None and pd.notna(sto_density_g_cm3) and o["sto_density_plausible"][0] <= float(sto_density_g_cm3) <= o["sto_density_plausible"][1]):
        return (*[float(x) for x in cfg["buoyancy"]["fluid_gradient_bar_per_m"]["oil"]], "configured band (no plausible stock-tank density)")
    rs = float(glr_sm3_sm3) if glr_sm3_sm3 is not None and pd.notna(glr_sm3_sm3) and 0 <= float(glr_sm3_sm3) <= float(o["rs_max_sm3_sm3"]) else 0.0
    gg = float(gas_gravity) if gas_gravity is not None and pd.notna(gas_gravity) and 0.4 <= float(gas_gravity) <= 1.5 else float(cfg["fluid_gradients"]["gas"]["gravity_default"])
    bo_c, bo_lo, bo_hi = (float(x) for x in o["bo_by_class"].get(str(fluid_class), o["bo_by_class"]["default"]))
    rho_sto = float(sto_density_g_cm3) * 1000.0

    def grad(bo: float) -> float:
        return _gradient_from_density((rho_sto + rs * gg * RHO_AIR_STD) / bo)

    lo_clip, hi_clip = (float(x) for x in o["clip_bar_per_m"])
    c = float(np.clip(grad(bo_c), lo_clip, hi_clip))
    return c, float(np.clip(grad(bo_hi), lo_clip, hi_clip)), float(np.clip(grad(bo_lo), lo_clip, hi_clip)), f"mass balance, DST stock-tank density, Rs = GLR, Bo band of the PVT class ({fluid_class or 'unclassified'})"      # a larger Bo = a lighter oil = a smaller gradient


def hydrocarbon_gradients(fluid: str, test: dict[str, Any], cfg: dict[str, Any]) -> tuple[float, float, float, str]:
    """Gradient of the leg a test sits in: `fluid` is `oil` or `gas`; `test` carries pressure_bar, temperature_c, gas_gravity, sto_density_g_cm3, glr, fluid_class (any may be missing)."""
    if fluid == "gas":
        return gas_gradient(test.get("pressure_bar"), test.get("temperature_c"), test.get("gas_gravity"), cfg)
    return oil_gradient(test.get("sto_density_g_cm3"), test.get("glr"), test.get("gas_gravity"), test.get("fluid_class"), cfg)


def water_gradient(salinity_ppm: float | None, cfg: dict[str, Any]) -> tuple[float, float, float, str]:
    """(central, low, high, basis) of the water gradient: with a salinity, interpolated between the anchor points (fresh water to a strong brine); without, the hydrostatic gradient of the
    configuration with the fresh-to-brine range as the band."""
    w = cfg["fluid_gradients"]["water"]
    central = float(cfg["hydrostatic"]["gradient_bar_per_m"])
    low, high = (float(x) for x in w["band_bar_per_m"])
    if salinity_ppm is not None and pd.notna(salinity_ppm):
        xs, ys = zip(*[(float(a), float(b)) for a, b in w["salinity_anchors"]])
        g = float(np.interp(float(salinity_ppm), xs, ys))                 # beyond the last anchor the last value (not extrapolated)
        return g, min(g, low), max(g, high), f"salinity {float(salinity_ppm):,.0f} ppm"
    return central, low, high, "hydrostatic gradient of the configuration, fresh-to-brine band"


def salinity_from_gradient(gradient_bar_per_m: float, cfg: dict[str, Any]) -> float:
    """Salinity [ppm] that gives an observed water gradient (the inverse of the anchor table: a free-water-level water line tells the salinity of the aquifer)."""
    xs, ys = zip(*[(float(a), float(b)) for a, b in cfg["fluid_gradients"]["water"]["salinity_anchors"]])
    return float(np.interp(gradient_bar_per_m, ys, xs))
