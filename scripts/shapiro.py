"""Two mechanisms for half-integer Shapiro steps, and how to tell them apart.

The problem
-----------
Zhu, Wang, Zhang and Xue (National Science Review 13, nwaf569, 2026;
arXiv:2512.01586) measured half-integer and odd-denominator Shapiro steps in
twisted cuprate junctions and concluded they came from trapped vortices rather
than from anything intrinsic to the pairing.  Their evidence was operational:
the steps vanished on thermal cycling, came back after small-field or
large-current training.  That battery works, but every test in it perturbs the
sample — cycle through Tc, train with a field, anneal with a current — and none
of them is a direct measurement of the mechanism.

Since then the community standard for any fractional-step claim is to run that
battery up front.  The question this module asks is whether there is a
*non-invasive* discriminator instead: a sweep the experiment already knows how
to do, whose shape differs between the two mechanisms.

Two models
----------
Intrinsic.  A single overdamped junction whose current-phase relation carries a
second harmonic:

    dphi/dtau = i_dc + i_ac cos(Omega tau) - [c1 sin(phi) + c2 sin(2 phi)] + eta

Half-integer steps here come from the c2 term.  This is the Tsarev-Fominov
two-harmonic RSJ baseline (SciPost Phys. 19, 148) that any intrinsic claim has
to be shown to exceed.

Extrinsic.  A trapped vortex turns the junction into two parallel weak links
sharing a loop.  With finite loop inductance the two phases are independent and
the dynamics is two-dimensional:

    dphi1/dtau = i_b + 2 j_circ - 2 a1 sin(phi1) + eta1
    dphi2/dtau = i_b - 2 j_circ - 2 a2 sin(phi2) + eta2
    j_circ     = [(phi1 - phi2) - 2 pi f] / beta_L

with f the trapped flux in units of Phi_0, beta_L = 2 pi L Ic / Phi_0 the
screening parameter, and a1 + a2 = 1 setting the branch asymmetry.  No harmonic
is put in by hand: the half-steps, when they appear, come from the internal
loop dynamics.

The point of the comparison is that the second model has an internal timescale
(set by beta_L) and the first one does not.  A quantity that depends on drive
*frequency* should therefore separate them, while a quantity that depends only
on drive amplitude may not.

Conventions follow the RSJ module of the TBG photonic-junction project
(`rsj_reference.py` there): reduced time tau = omega_c t, currents in units of
Ic, reduced voltage v = <dphi/dtau> = V/(Ic Rn), steps at v = (n/m) Omega.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import math

import numpy as np


# --------------------------------------------------------------------------
# Current-phase relation
# --------------------------------------------------------------------------
def cpr(phi: np.ndarray, c: np.ndarray) -> np.ndarray:
    out = np.zeros_like(phi)
    for m, cm in enumerate(c, start=1):
        if cm != 0.0:
            out = out + cm * np.sin(m * phi)
    return out


def normalize_cpr(c_raw) -> np.ndarray:
    """Rescale harmonics so max_phi I(phi) = 1, i.e. currents are in units of Ic."""
    c_raw = np.asarray(c_raw, float)
    phi = np.linspace(0.0, 2.0 * np.pi, 40001)
    return c_raw / cpr(phi, c_raw).max()


# --------------------------------------------------------------------------
# Model I — single junction, two-harmonic CPR
# --------------------------------------------------------------------------
def run_intrinsic(i_dc, i_ac, c, omega: float, gamma: float = 0.0,
                  n_transient: int = 150, n_avg: int = 300, spp: int = 60,
                  seed: int = 0, phi0_offset: float = 0.0) -> np.ndarray:
    """Reduced dc voltage v for each i_dc. Vectorised fixed-step RK4."""
    i_dc = np.atleast_1d(np.asarray(i_dc, float))
    # i_ac may be a scalar or an array broadcasting against i_dc, so a whole
    # (drive amplitude, bias) grid integrates in one call.
    i_ac = np.asarray(i_ac, float)
    c = np.asarray(c, float)
    period = 2.0 * np.pi / omega
    dt = period / spp
    phi = np.full_like(i_dc, phi0_offset)
    rng = np.random.default_rng(seed)
    noise_amp = np.sqrt(2.0 * gamma / dt) if gamma > 0 else 0.0

    def rhs(tau, ph, xi):
        return i_dc + i_ac * np.cos(omega * tau) - cpr(ph, c) + xi

    tau = 0.0
    for _ in range(int(n_transient * spp)):
        xi = noise_amp * rng.standard_normal(phi.shape) if noise_amp else 0.0
        k1 = rhs(tau, phi, xi)
        k2 = rhs(tau + 0.5 * dt, phi + 0.5 * dt * k1, xi)
        k3 = rhs(tau + 0.5 * dt, phi + 0.5 * dt * k2, xi)
        k4 = rhs(tau + dt, phi + dt * k3, xi)
        phi = phi + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        tau += dt

    phi_start, tau_start = phi.copy(), tau
    for _ in range(int(n_avg * spp)):
        xi = noise_amp * rng.standard_normal(phi.shape) if noise_amp else 0.0
        k1 = rhs(tau, phi, xi)
        k2 = rhs(tau + 0.5 * dt, phi + 0.5 * dt * k1, xi)
        k3 = rhs(tau + 0.5 * dt, phi + 0.5 * dt * k2, xi)
        k4 = rhs(tau + dt, phi + dt * k3, xi)
        phi = phi + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        tau += dt
    return (phi - phi_start) / (tau - tau_start)


def run_langevin(i_dc, i_ac, c, omega: float, gamma: float,
                 n_transient: int = 150, n_avg: int = 300, spp: int = 100,
                 seed: int = 0, rng=None) -> np.ndarray:
    """Stochastic Heun integration of the RSJ-Langevin equation.

        dphi/dtau = i_dc + i_ac cos(Omega tau) - sum_m c_m sin(m phi) + eta
        <eta(tau) eta(tau')> = 2 Gamma delta(tau - tau'),  Gamma = k_B T / E_J

    The deterministic runners above bolt a noise term onto an RK4 frame, which
    is fine for a deterministic sweep and wrong for a thermal one: RK4's
    intermediate stages are not defined for a Wiener increment, so reusing one
    draw across them does not integrate the right process. Heun does — additive
    noise makes Ito and Stratonovich coincide, so a single increment used in
    both the predictor and the corrector is correct. Same convention as the
    reference implementation this track validates against, including its
    spp >= 100 requirement for noisy runs.
    """
    if spp < 100:
        raise ValueError("noisy runs need spp >= 100")
    i_dc = np.atleast_1d(np.asarray(i_dc, float))
    i_ac = np.asarray(i_ac, float)
    c = np.asarray(c, float)
    if rng is None:
        rng = np.random.default_rng(seed)
    period = 2.0 * np.pi / omega
    dt = period / spp
    sig = math.sqrt(2.0 * gamma * dt)
    phi = np.zeros_like(i_dc)

    def drift(tau, ph):
        return i_dc + i_ac * np.cos(omega * tau) - cpr(ph, c)

    n1, n2 = int(n_transient * spp), int(n_avg * spp)
    tau = 0.0
    phi_a, tau_a = phi.copy(), tau
    for step in range(n1 + n2):
        if step == n1:
            phi_a, tau_a = phi.copy(), tau
        xi = sig * rng.standard_normal(phi.shape) if sig > 0 else 0.0
        f1 = drift(tau, phi)
        pred = phi + dt * f1 + xi
        f2 = drift(tau + dt, pred)
        phi = phi + 0.5 * dt * (f1 + f2) + xi
        tau += dt
    return (phi - phi_a) / (tau - tau_a)


# --------------------------------------------------------------------------
# Model E — two parallel junctions sharing an inductive loop
# --------------------------------------------------------------------------
def run_extrinsic(i_dc, i_ac, omega: float, beta_l: float = 2.0,
                  flux=0.25, asym: float = 0.3, gamma: float = 0.0,
                  n_transient: int = 150, n_avg: int = 300, spp: int = 60,
                  seed: int = 0, phi0_offset: float = 0.0) -> np.ndarray:
    """Reduced dc voltage for the trapped-flux two-junction loop.

    `asym` is (a1 - a2) with a1 + a2 = 1; `flux` is Phi_ext/Phi_0; `beta_l` is
    the screening parameter 2 pi L Ic / Phi_0.  beta_l -> 0 collapses the loop
    to a single junction with a flux-modulated critical current and no
    half-steps, which `selfcheck` uses as a control.
    """
    i_dc = np.atleast_1d(np.asarray(i_dc, float))
    i_ac = np.asarray(i_ac, float)
    # flux may also be an array broadcasting against i_dc, so a whole
    # (trapped flux, bias) sweep integrates in one call.
    flux = np.asarray(flux, float)
    a1 = 0.5 * (1.0 + asym)
    a2 = 0.5 * (1.0 - asym)
    period = 2.0 * np.pi / omega
    # The difference mode relaxes at rate 4/beta_L, so the step has to resolve
    # it as well as the drive period; RK4 needs roughly lambda*dt < 1.
    dt = min(period / spp, 0.25 * beta_l)
    # Offsetting phi1 only starts the loop with a different initial circulating
    # current, which is the knob that selects between coexisting attractors if
    # there are any.
    phi1 = np.full_like(i_dc, phi0_offset)
    phi2 = np.zeros_like(i_dc)
    rng = np.random.default_rng(seed)
    noise_amp = np.sqrt(2.0 * gamma / dt) if gamma > 0 else 0.0
    two_pi_f = 2.0 * np.pi * flux

    def rhs(tau, p1, p2, xi1, xi2):
        i_b = i_dc + i_ac * np.cos(omega * tau)
        j_circ = ((p1 - p2) - two_pi_f) / beta_l
        # The screening current opposes the flux change: it must enter the two
        # branches with opposite signs and in the sense that relaxes
        # (phi1 - phi2) towards 2 pi f. With the opposite sign the difference
        # mode grows exponentially instead of relaxing, which is easy to write
        # by accident and shows up immediately as an overflow.
        d1 = i_b - 2.0 * j_circ - 2.0 * a1 * np.sin(p1) + xi1
        d2 = i_b + 2.0 * j_circ - 2.0 * a2 * np.sin(p2) + xi2
        return d1, d2

    def step(tau, p1, p2):
        xi1 = noise_amp * rng.standard_normal(p1.shape) if noise_amp else 0.0
        xi2 = noise_amp * rng.standard_normal(p2.shape) if noise_amp else 0.0
        a1k, a2k = rhs(tau, p1, p2, xi1, xi2)
        b1k, b2k = rhs(tau + 0.5 * dt, p1 + 0.5 * dt * a1k,
                       p2 + 0.5 * dt * a2k, xi1, xi2)
        c1k, c2k = rhs(tau + 0.5 * dt, p1 + 0.5 * dt * b1k,
                       p2 + 0.5 * dt * b2k, xi1, xi2)
        d1k, d2k = rhs(tau + dt, p1 + dt * c1k, p2 + dt * c2k, xi1, xi2)
        p1 = p1 + dt / 6.0 * (a1k + 2 * b1k + 2 * c1k + d1k)
        p2 = p2 + dt / 6.0 * (a2k + 2 * b2k + 2 * c2k + d2k)
        return p1, p2

    n_tr = int(np.ceil(n_transient * period / dt))
    n_av = int(np.ceil(n_avg * period / dt))
    tau = 0.0
    for _ in range(n_tr):
        phi1, phi2 = step(tau, phi1, phi2)
        tau += dt
    s1, s2, tau_start = phi1.copy(), phi2.copy(), tau
    for _ in range(n_av):
        phi1, phi2 = step(tau, phi1, phi2)
        tau += dt
    dtau = tau - tau_start
    return 0.5 * ((phi1 - s1) + (phi2 - s2)) / dtau


# --------------------------------------------------------------------------
# Step-width detector
# --------------------------------------------------------------------------
def step_width(i_dc: np.ndarray, v: np.ndarray, v_step: float,
               tol: float = 0.025) -> float:
    """Total i_dc measure spent within `tol` of the target reduced voltage."""
    on = np.abs(v - v_step) < tol
    if not on.any():
        return 0.0
    di = np.gradient(i_dc)
    return float(np.sum(di[on]))


def kernel_width(i_dc: np.ndarray, v: np.ndarray, v_step: float,
                 sigma_k: float = 0.025) -> float:
    """Grid-continuous step width: the Gaussian-kernel transit integral.

        W = \int exp[-(v(i_dc) - v_step)^2 / 2 sigma_k^2] d i_dc

    Counting grid points inside a plateau, as `step_width` does, quantises the
    answer in units of the bias step — and when the bias grid is scaled with
    drive frequency (as a dome scan across I_c(nu) must be), the quantum itself
    changes from filling to filling, so a *shape* read off such widths is
    partly an artefact of the grid. This estimator has no such quantum.

    The kernel is sigma_k wide in v and dv/di_dc is of order 1 on the ohmic
    branch, so neither sigma_k nor the bias step should be scaled with Omega.
    A pure ohmic transit through the target voltage contributes
    sigma_k*sqrt(2*pi)/|dv/di| even with no step present, so this estimator is
    only meaningful after subtracting the same quantity computed for a control
    with no fractional step (see `net_kernel_width`).
    """
    return float(np.trapezoid(
        np.exp(-(v - v_step) ** 2 / (2.0 * sigma_k ** 2)), i_dc))


def analytic_transit(sigma_k: float, slope: float) -> float:
    """Kernel integral for an exactly linear transit v = slope * i_dc.

    Used as the estimator's self-check: sigma_k*sqrt(2*pi)/slope.
    """
    return sigma_k * math.sqrt(2.0 * math.pi) / slope


def sweep_widths(runner, i_dc: np.ndarray, omega: float,
                 rel_tol: float = 0.005, **kw) -> dict:
    """Half-step and first-integer-step widths for one drive setting.

    The plateau tolerance is scaled to the drive frequency, `tol = rel_tol *
    omega`, because the steps themselves sit omega/2 apart: a fixed absolute
    tolerance would be loose at small omega and impossibly tight at large
    omega. At omega = 5 and rel_tol = 0.005 this reproduces the 0.025 used by
    the reference implementation.
    """
    v = runner(i_dc, omega=omega, **kw)
    tol = rel_tol * omega
    return {
        "w_half": step_width(i_dc, v, 0.5 * omega, tol),
        "w_one": step_width(i_dc, v, 1.0 * omega, tol),
        "w_zero": step_width(i_dc, v, 0.0, tol),
        "tol": tol,
        "v": v,
    }
