"""Sweep drive frequency, not the sample: a non-invasive mechanism test.

The intrinsic mechanism (a second harmonic in the current-phase relation) has
no internal timescale.  At fixed alpha = i_ac/Omega the reduced step widths are
set by Bessel functions of alpha alone, so w_half should be nearly flat in
Omega.

The extrinsic mechanism (a trapped vortex splitting the junction into two weak
links sharing an inductive loop) does have one: the difference mode relaxes at
rate 4/beta_L in reduced units.  When the drive period approaches that
relaxation time the loop can no longer follow, and the half-step should
disappear — giving a roll-off in Omega with no counterpart in the intrinsic
model.

If that holds, an experiment can distinguish the two by sweeping microwave
frequency at fixed alpha, which perturbs nothing: no thermal cycling, no field
training, no current annealing.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from shapiro import (  # noqa: E402
    normalize_cpr, run_intrinsic, run_extrinsic, sweep_widths,
)

RESULTS = os.path.join(_HERE, "..", "results")
FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
C_SKEW = normalize_cpr([1.0, -0.25])
FLOOR = 0.05          # half-step detector floor for a purely sinusoidal CPR


def intrinsic_widths(omega: float, alpha: float, i_dc: np.ndarray,
                     **kw) -> dict:
    return sweep_widths(
        lambda idc, omega, **k: run_intrinsic(idc, alpha * omega, C_SKEW,
                                              omega, **k),
        i_dc, omega, **kw)


def extrinsic_widths(omega: float, alpha: float, i_dc: np.ndarray,
                     beta_l: float = 1.0, flux: float = 0.5,
                     asym: float = 0.3, **kw) -> dict:
    return sweep_widths(
        lambda idc, omega, **k: run_extrinsic(idc, alpha * omega, omega,
                                              beta_l=beta_l, flux=flux,
                                              asym=asym, **k),
        i_dc, omega, **kw)


def current_grid(omega: float, alpha: float, d_i: float = 0.02) -> np.ndarray:
    """A window wide enough to contain both the n=1/2 and n=1 steps.

    The n-th step sits near i_dc ~ n*omega for an overdamped junction, so the
    window has to grow with omega; a fixed [0, 6] window silently loses the
    integer step above omega ~ 6 and makes the ratio undefined.
    """
    i_max = 1.3 * omega + alpha * omega + 2.0
    return np.arange(0.0, i_max + d_i, d_i)


def frequency_scan(alpha: float = 0.5,
                   omegas=(1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0),
                   beta_l: float = 1.0, flux: float = 0.5,
                   d_i: float = 0.02) -> dict:
    out = {"alpha": alpha, "omegas": list(omegas), "beta_l": beta_l,
           "flux": flux, "d_i": d_i, "intrinsic": [], "extrinsic": []}
    print(f"  alpha={alpha}, beta_L={beta_l}, flux={flux}")
    print("   Omega | intrinsic w1/2  w1   ratio | extrinsic w1/2  w1   ratio")
    for om in omegas:
        i_dc = current_grid(om, alpha, d_i)
        a = intrinsic_widths(om, alpha, i_dc, n_transient=120, n_avg=300)
        b = extrinsic_widths(om, alpha, i_dc, beta_l=beta_l, flux=flux,
                             n_transient=80, n_avg=300)
        ra = a["w_half"] / a["w_one"] if a["w_one"] > 0 else float("nan")
        rb = b["w_half"] / b["w_one"] if b["w_one"] > 0 else float("nan")
        out["intrinsic"].append({"omega": om, "w_half": a["w_half"],
                                 "w_one": a["w_one"], "ratio": ra})
        out["extrinsic"].append({"omega": om, "w_half": b["w_half"],
                                 "w_one": b["w_one"], "ratio": rb})
        print(f"   {om:5g} | {a['w_half']:11.4f} {a['w_one']:6.4f} {ra:6.3f}"
              f" | {b['w_half']:11.4f} {b['w_one']:6.4f} {rb:6.3f}", flush=True)
    return out


def plot(data: dict, path: str) -> None:
    om = [r["omega"] for r in data["intrinsic"]]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    ax = axes[0]
    ax.plot(om, [r["w_half"] for r in data["intrinsic"]], "o-",
            label="intrinsic (CPR harmonic)")
    ax.plot(om, [r["w_half"] for r in data["extrinsic"]], "s-",
            label=r"extrinsic (trapped flux, $\beta_L$="
                  f"{data['beta_l']:g})")
    ax.axhline(FLOOR, color="0.5", ls=":", label="detector floor")
    ax.set_xlabel(r"drive frequency $\Omega = \omega/\omega_c$")
    ax.set_ylabel(r"half-step width $w_{1/2}$")
    ax.set_title(rf"Fixed $\alpha=i_{{ac}}/\Omega$={data['alpha']:g}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(om, [r["ratio"] for r in data["intrinsic"]], "o-",
            label="intrinsic")
    ax.plot(om, [r["ratio"] for r in data["extrinsic"]], "s-",
            label="extrinsic")
    ax.set_xlabel(r"$\Omega$")
    ax.set_ylabel(r"$w_{1/2}\,/\,w_1$")
    ax.set_title("Ratio removes the overall current scale")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Sweeping the drive frequency separates the two mechanisms",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    print("frequency scan at fixed alpha:")
    data = frequency_scan()
    plot(data, os.path.join(FIGURES, "d1_frequency_scan.png"))
    with open(os.path.join(RESULTS, "frequency_scan.json"), "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"\nwrote {RESULTS}/frequency_scan.json")


if __name__ == "__main__":
    main()
