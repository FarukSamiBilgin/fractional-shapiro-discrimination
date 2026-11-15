"""E0b — the axis experiments actually use is degenerate.

Shapiro maps are measured against rf power, not drive frequency. The literature
map points out that three independent published frameworks give the *same*
functional shape for the half-to-integer step ratio on that axis:

  * intrinsic two-harmonic CPR   (Renne-Polder / Luebbig-Luther / Sellier):
        w_half ~ |J_2n(2a)| ,  w_1 ~ |J_n(a)|
  * inductive two-junction loop  (Mori-Maekawa):  |J_2q(2b)| vs |J_k(b)|
  * microscopic point contact    (Duprat & Levy Yeyati Eq. 11):
        |J_1(4 alpha)| vs |J_1(2 alpha)|

all of which reduce to

        w_half / w_1  ∝  |J_1(2a) / J_1(a)|,      a = 2 alpha

with zeros at a = 1.9159 and 3.5078 (from J_1(2a) = 0) and a pole at
a = 3.8317 (from J_1(a) = 0) — identical in all three, differing only by a
constant prefactor set by the harmonic content.

If that holds numerically, the drive-amplitude axis measures the harmonic ratio
and says nothing about where the harmonic came from. That is a stronger
negative result than the drive-frequency one this track reported earlier,
because it closes the measurement experiments actually run rather than a
hypothetical one.

Sellier et al. (2004) used this axis quantitatively in the intrinsic direction,
fitting a harmonic ratio from node positions — so a demonstrated degeneracy has
an immediate experimental reading.

Convention: w_half is control-subtracted against a sinusoidal CPR (which has no
half step); w_1 is the raw kernel width, since the integer step exists in both
signal and control and subtracting would remove it.
"""

from __future__ import annotations

import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.special import jv, jn_zeros
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from shapiro import (  # noqa: E402
    normalize_cpr, run_intrinsic, run_extrinsic, kernel_width,
)

RESULTS = os.path.join(_HERE, "..", "results")
FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
SIGMA_K = 0.025
HALF_WINDOW = 0.5
D_I = 0.005
OMEGA = 5.0
N_AVG = 300


def _width(kind, i_ac, v_target, cpr=None, beta_l=1.0, flux=0.5, asym=0.3):
    """Kernel width of the step at v_target for one drive amplitude."""
    coarse = np.linspace(0.05 * OMEGA, 1.5 * OMEGA, 120)

    def solve(grid, n_avg):
        if kind == "cpr":
            return run_intrinsic(grid, i_ac, cpr, OMEGA, n_transient=150,
                                 n_avg=n_avg, spp=60)
        return run_extrinsic(grid, i_ac, OMEGA, beta_l=beta_l, flux=flux,
                             asym=asym, n_transient=100, n_avg=n_avg, spp=60)

    v = solve(coarse, max(60, N_AVG // 3))
    centre = coarse[int(np.argmin(np.abs(v - v_target)))]
    fine = centre + np.arange(-HALF_WINDOW, HALF_WINDOW + D_I, D_I)
    return kernel_width(fine, solve(fine, N_AVG), v_target, SIGMA_K)


def ratio_curve(kind, alphas, cpr=None, ctrl=None, beta_l=1.0):
    """w_half (control-subtracted) over w_1 (raw) versus drive amplitude."""
    rows = []
    for a in alphas:
        i_ac = a * OMEGA
        if kind == "cpr":
            wh = (_width("cpr", i_ac, 0.5 * OMEGA, cpr=cpr)
                  - _width("cpr", i_ac, 0.5 * OMEGA, cpr=ctrl))
            w1 = _width("cpr", i_ac, 1.0 * OMEGA, cpr=cpr)
        else:
            wh = (_width("loop", i_ac, 0.5 * OMEGA, beta_l=beta_l, flux=0.5)
                  - _width("loop", i_ac, 0.5 * OMEGA, beta_l=beta_l, flux=0.0))
            w1 = _width("loop", i_ac, 1.0 * OMEGA, beta_l=beta_l, flux=0.5)
        r = wh / w1 if w1 > 1e-6 else float("nan")
        rows.append({"alpha": float(a), "a_bessel": float(2 * a),
                     "w_half": wh, "w_one": w1, "ratio": r})
        print(f"  {kind:5} alpha={a:5.3f} a={2*a:5.3f}  w1/2={wh:8.5f}  "
              f"w1={w1:8.5f}  ratio={r:7.4f}", flush=True)
    return rows


def main() -> None:
    j1 = jn_zeros(1, 2)
    zeros_a = j1 / 2.0            # J_1(2a) = 0
    pole_a = j1[0]                # J_1(a) = 0
    print(f"analytic envelope |J1(2a)/J1(a)|: zeros at a = "
          f"{zeros_a[0]:.4f}, {zeros_a[1]:.4f}; pole at a = {pole_a:.4f}")
    print(f"in alpha = a/2: zeros at {zeros_a[0]/2:.4f}, {zeros_a[1]/2:.4f}; "
          f"pole at {pole_a/2:.4f}\n")

    alphas = np.arange(0.10, 2.15, 0.05)
    cpr = normalize_cpr([1.0, -0.25])
    ctrl = normalize_cpr([1.0])
    print("intrinsic two-harmonic CPR, c2/c1 = -0.25:")
    intr = ratio_curve("cpr", alphas, cpr=cpr, ctrl=ctrl)
    print("\nextrinsic inductive loop, beta_L = 1.0 (E_J2/E_J1 = beta_L/4 = 0.25):")
    extr = ratio_curve("loop", alphas, beta_l=1.0)

    a = 2 * alphas
    env = np.abs(jv(1, 2 * a) / jv(1, a))

    def find_min(rows):
        """Locate the ratio's minima in a, ignoring the divergence region."""
        r = np.array([x["ratio"] for x in rows])
        av = np.array([x["a_bessel"] for x in rows])
        ok = np.isfinite(r)
        mins = []
        for k in range(1, len(r) - 1):
            if ok[k] and ok[k-1] and ok[k+1] and r[k] < r[k-1] and r[k] < r[k+1]:
                mins.append(float(av[k]))
        return mins

    out = {"omega": OMEGA, "convention": "a = 2*alpha = 2*i_ac/Omega",
           "analytic": {"zeros_in_a": zeros_a.tolist(), "pole_in_a": float(pole_a),
                        "envelope": env.tolist()},
           "intrinsic": intr, "extrinsic": extr,
           "intrinsic_minima_in_a": find_min(intr),
           "extrinsic_minima_in_a": find_min(extr)}
    print(f"\nratio minima in a — intrinsic: {out['intrinsic_minima_in_a']}")
    print(f"                    extrinsic: {out['extrinsic_minima_in_a']}")
    print(f"       analytic zeros of the envelope: {zeros_a.tolist()}")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    ax.plot(a, [x["w_half"] for x in intr], "o-", ms=3.5, label=r"intrinsic $w_{1/2}$")
    ax.plot(a, [x["w_one"] for x in intr], "o--", ms=3.5, alpha=0.6, label=r"intrinsic $w_1$")
    ax.plot(a, [x["w_half"] for x in extr], "s-", ms=3.5, label=r"loop $w_{1/2}$")
    ax.plot(a, [x["w_one"] for x in extr], "s--", ms=3.5, alpha=0.6, label=r"loop $w_1$")
    ax.set_xlabel(r"$a = 2\alpha$"); ax.set_ylabel("kernel width")
    ax.set_title("Step widths vs drive amplitude")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1]
    ri = np.array([x["ratio"] for x in intr]); re = np.array([x["ratio"] for x in extr])
    ax.plot(a, ri, "o-", ms=3.5, color="tab:blue", label=r"intrinsic ($c_2/c_1=-0.25$)")
    ax.plot(a, re, "s-", ms=3.5, color="tab:orange", label=r"loop ($\beta_L=1$)")
    scale = np.nanmedian(ri[np.isfinite(ri) & (a < 1.8)]) / np.nanmedian(env[a < 1.8])
    ax.plot(a, scale * env, "--", lw=2, color="k",
            label=r"$|J_1(2a)/J_1(a)|$, scaled")
    for z in zeros_a:
        ax.axvline(z, color="0.5", ls=":", lw=1)
    ax.axvline(pole_a, color="tab:red", ls=":", lw=1)
    ax.text(zeros_a[0], ax.get_ylim()[1]*0.92, "  envelope zeros", fontsize=7, color="0.4")
    ax.text(pole_a, ax.get_ylim()[1]*0.72, "  pole", fontsize=7, color="tab:red")
    ax.set_xlabel(r"$a = 2\alpha$"); ax.set_ylabel(r"$w_{1/2}/w_1$")
    ax.set_ylim(0, 1.2); ax.set_title("The drive axis cannot tell the two apart")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.13, wspace=0.26)
    path = os.path.join(FIGURES, "e0b_drive_degeneracy.png")
    fig.savefig(path, dpi=140); plt.close(fig)
    with open(os.path.join(RESULTS, "drive_degeneracy.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path} and drive_degeneracy.json")


if __name__ == "__main__":
    main()
