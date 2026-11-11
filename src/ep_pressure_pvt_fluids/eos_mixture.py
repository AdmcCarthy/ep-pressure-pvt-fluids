"""Cubic equations of state for MIXTURES: Peng-Robinson and Soave-Redlich-Kwong with van der Waals one-fluid mixing, binary interaction parameters and the Péneloux volume translation.  Pure.

Used by `column_grading.py` (compositional grading of a hydrocarbon column) and for saturation pressures of a composition.  Both equations are written in one form,

    p = R T / (v - b)  -  a(T) / (v^2 + u b v + w b^2)         PR: u = 2, w = -1          SRK: u = 1, w = 0

    Z^3 - (1 + B - u B) Z^2 + (A + w B^2 - u B - u B^2) Z - (A B + w B^2 + w B^3) = 0,        A = a p / (R T)^2,  B = b p / (R T)
    ln phi_i = (B_i / B)(Z - 1) - ln(Z - B) - A / (B sqrt(u^2 - 4 w)) (2 sum_j x_j A_ij / A - B_i / B) ln[(2 Z + B (u + sqrt(u^2 - 4 w))) / (2 Z + B (u - sqrt(u^2 - 4 w)))]

with a_i = Oa R^2 Tc^2 / Pc alpha(T), b_i = Ob R Tc / Pc (PR: Oa = 0.45724, Ob = 0.07780, alpha = [1 + kappa (1 - sqrt(Tr))]^2, kappa = 0.37464 + 1.54226 w - 0.26992 w^2, above w = 0.491 the
Robinson-Peng polynomial; SRK: Oa = 0.42748, Ob = 0.08664, alpha = [1 + m (1 - sqrt(Tr))]^2, m = 0.480 + 1.574 w - 0.176 w^2; coefficients as given in the Wikipedia article on cubic equations
of state, read 2026-11-04, knowledge `cubic-eos-forms`).  Volume translation (Péneloux et al. 1982): v_true = v_eos - sum x_i c_i with c_i = 0.40768 (R Tc / Pc)(0.29441 - Z_RA,i), Z_RA = 0.29056 - 0.08775 w:
it changes liquid densities and the fugacity coefficients by ln phi_i* = ln phi_i - c_i p / (R T); the change cancels in K values at one pressure, so a saturation pressure is unaffected, but it does not cancel
between depths: a gravity-grading calculation that uses translated densities MUST use the translated fugacities (otherwise dp/dz differs from rho g).  Wikipedia gives this correlation for SRK; applying it to PR is an ASSUMPTION (flagged by the caller).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import brentq

R = 8.314462618                                    # J / (mol K)
BAR = 1.0e5                                        # Pa per bar
G = 9.80665                                        # m / s2
EOS_FORMS = {"PR": {"u": 2.0, "w": -1.0, "oa": 0.45724, "ob": 0.07780}, "SRK": {"u": 1.0, "w": 0.0, "oa": 0.42748, "ob": 0.08664}}
CARBON_NUMBER = {"methane": 1, "ethane": 2, "propane": 3, "n_butane": 4, "n_pentane": 5, "n_hexane": 6, "n_heptane": 7, "n_octane": 8, "n_decane": 10, "n_eicosane": 20}


def molar_mass_g_mol(carbon_number: int) -> float:
    """n-alkane CnH(2n+2) from standard atomic weights (C 12.011, H 1.008)."""
    return 12.011 * carbon_number + 1.008 * (2 * carbon_number + 2)


def _alpha_kappa(eos: str, omega: float) -> float:
    if eos == "PR":
        return 0.379642 + 1.48503 * omega - 0.164423 * omega ** 2 + 0.016666 * omega ** 3 if omega > 0.491 else 0.37464 + 1.54226 * omega - 0.26992 * omega ** 2
    return 0.480 + 1.574 * omega - 0.176 * omega ** 2


@dataclass
class Mixture:
    """Components (arrays in the order of `ids`), the equation and its binary interaction parameters."""
    ids: list[str]
    tc_k: np.ndarray
    pc_bar: np.ndarray
    omega: np.ndarray
    mw_g_mol: np.ndarray
    eos: str = "PR"
    kij: np.ndarray | None = None
    volume_shift: bool = True
    c_m3_mol: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        n = len(self.ids)
        self.kij = np.zeros((n, n)) if self.kij is None else np.asarray(self.kij, dtype=float)
        zra = 0.29056 - 0.08775 * self.omega
        self.c_m3_mol = 0.40768 * (R * self.tc_k / (self.pc_bar * BAR)) * (0.29441 - zra)

    def with_options(self, eos: str | None = None, volume_shift: bool | None = None) -> "Mixture":
        return Mixture(self.ids, self.tc_k, self.pc_bar, self.omega, self.mw_g_mol, eos or self.eos, self.kij, self.volume_shift if volume_shift is None else volume_shift)

    @property
    def n(self) -> int:
        return len(self.ids)

    def mw(self, x: np.ndarray) -> float:
        return float(x @ self.mw_g_mol)


def mixture_from_config(cfg: dict[str, Any], eos: str = "PR", volume_shift: bool = True) -> Mixture:
    """Build the component set of `config/column_models.yaml` (constants from `components`, binary interaction parameters from `kij` where given)."""
    comps = cfg["components"]
    ids = [c["id"] for c in comps]
    n = len(ids)
    kij = np.zeros((n, n))
    for pair, val in (cfg.get("kij") or {}).items():
        a, b = pair.split("-")
        if a in ids and b in ids:
            kij[ids.index(a), ids.index(b)] = kij[ids.index(b), ids.index(a)] = float(val)
    return Mixture(ids, np.array([float(c["tc_k"]) for c in comps]), np.array([float(c["pc_bar"]) for c in comps]), np.array([float(c["omega"]) for c in comps]),
                   np.array([molar_mass_g_mol(CARBON_NUMBER[c["id"]]) for c in comps]), eos, kij, volume_shift)


def _pure_ab(m: Mixture, t_k: float) -> tuple[np.ndarray, np.ndarray]:
    f = EOS_FORMS[m.eos]
    kappa = np.array([_alpha_kappa(m.eos, w) for w in m.omega])
    alpha = (1.0 + kappa * (1.0 - np.sqrt(t_k / m.tc_k))) ** 2
    a = f["oa"] * R ** 2 * m.tc_k ** 2 / (m.pc_bar * BAR) * alpha
    b = f["ob"] * R * m.tc_k / (m.pc_bar * BAR)
    return a, b


def _roots(m: Mixture, a_: float, b_: float) -> np.ndarray:
    f = EOS_FORMS[m.eos]
    u, w = f["u"], f["w"]
    r = np.roots([1.0, -(1.0 + b_ - u * b_), a_ + w * b_ ** 2 - u * b_ - u * b_ ** 2, -(a_ * b_ + w * b_ ** 2 + w * b_ ** 3)])
    real = np.sort(r[np.abs(r.imag) < 1e-9 * np.maximum(1.0, np.abs(r.real))].real)
    return real[real > b_]


def ln_phi(m: Mixture, x: np.ndarray, t_k: float, p_bar: float, phase: str = "stable") -> tuple[np.ndarray, float]:
    """(ln fugacity coefficients, Z) of a phase of composition x at T [K], p [bar]; phase = 'liquid' (smallest Z), 'vapour' (largest Z) or 'stable' (the root of the lowest Gibbs energy)."""
    f = EOS_FORMS[m.eos]
    u, w = f["u"], f["w"]
    sq = np.sqrt(u * u - 4.0 * w)
    ai, bi = _pure_ab(m, t_k)
    aij = (1.0 - m.kij) * np.sqrt(np.outer(ai, ai))
    a_mix, b_mix = float(x @ aij @ x), float(x @ bi)
    p = p_bar * BAR
    A, B = a_mix * p / (R * t_k) ** 2, b_mix * p / (R * t_k)
    roots = _roots(m, A, B)

    shift = (m.c_m3_mol * p / (R * t_k)) if m.volume_shift else 0.0              # volume translation: ln phi_i* = ln phi_i - c_i p / (R T); it cancels in K values (same p in both phases) but not between depths

    def lnphi_of(z: float) -> np.ndarray:
        return -shift + (bi / b_mix) * (z - 1.0) - np.log(z - B) - A / (B * sq) * (2.0 * (aij @ x) / a_mix - bi / b_mix) * np.log((2.0 * z + B * (u + sq)) / (2.0 * z + B * (u - sq)))

    if len(roots) == 0:
        raise ValueError("no physical root of the cubic")
    if phase == "liquid":
        z = roots[0]
    elif phase == "vapour":
        z = roots[-1]
    else:
        z = roots[0] if len(roots) == 1 else min(roots, key=lambda zz: float(x @ lnphi_of(zz)))
    return lnphi_of(z), float(z)


def molar_volume_m3_mol(m: Mixture, x: np.ndarray, t_k: float, p_bar: float, z: float) -> float:
    v = z * R * t_k / (p_bar * BAR)
    return v - (float(x @ m.c_m3_mol) if m.volume_shift else 0.0)


def density_kg_m3(m: Mixture, x: np.ndarray, t_k: float, p_bar: float, phase: str = "stable") -> float:
    """Density of the phase (volume-translated when `m.volume_shift`)."""
    _, z = ln_phi(m, x, t_k, p_bar, phase)
    return m.mw(x) / 1000.0 / molar_volume_m3_mol(m, x, t_k, p_bar, z)


def wilson_k(m: Mixture, t_k: float, p_bar: float) -> np.ndarray:
    return m.pc_bar / p_bar * np.exp(5.373 * (1.0 + m.omega) * (1.0 - m.tc_k / t_k))


def _equilibrium_sum(m: Mixture, z: np.ndarray, t_k: float, p_bar: float, kind: str, n_iter: int = 60) -> float:
    """Sum of the incipient-phase mole numbers minus 1 at (T, p): bubble (vapour incipient) sum K x - 1, dew (liquid incipient) sum x / K - 1 (NaN for the trivial solution K = 1)."""
    k = wilson_k(m, t_k, p_bar)
    for _ in range(n_iter):
        if kind == "bubble":
            y = k * z
            s = y.sum()
            y = y / s
            lnk_new = ln_phi(m, z, t_k, p_bar, "liquid")[0] - ln_phi(m, y, t_k, p_bar, "vapour")[0]
        else:
            xl = z / k
            s = xl.sum()
            xl = xl / s
            lnk_new = ln_phi(m, xl, t_k, p_bar, "liquid")[0] - ln_phi(m, z, t_k, p_bar, "vapour")[0]
        k_new = np.exp(lnk_new)
        if np.max(np.abs(np.log(k_new / k))) < 1e-9:
            k = k_new
            break
        k = k_new
    if np.max(np.abs(np.log(k))) < 1e-4:
        return float("nan")
    return float((k * z).sum() - 1.0) if kind == "bubble" else float((z / k).sum() - 1.0)


def saturation_pressure(m: Mixture, z: np.ndarray, t_k: float, kind: str, p_lo: float = 5.0, p_hi: float = 1500.0, n_scan: int = 45) -> float:
    """Bubble-point or dew-point pressure [bar] of composition z at T [K]: the root of the equilibrium sum found on a pressure scan (NaN where there is none, for example a supercritical mixture)."""
    ps = np.geomspace(p_lo, p_hi, n_scan)
    vals = []
    for p in ps:
        try:
            vals.append(_equilibrium_sum(m, z, t_k, float(p), kind))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            vals.append(float("nan"))
    vals = np.array(vals)
    roots = []
    for i in range(len(ps) - 1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] < 0:
            try:
                roots.append(brentq(lambda p: _equilibrium_sum(m, z, t_k, p, kind), float(ps[i]), float(ps[i + 1]), xtol=1e-6))
            except (ValueError, RuntimeError):
                continue
    if not roots:
        return float("nan")
    return float(max(roots)) if kind == "dew" else float(roots[-1])
