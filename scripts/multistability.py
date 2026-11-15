"""A second candidate test: does the half-step depend on where you start?

The frequency-trend test did not separate the mechanisms (see verify.py, V4:
the extrinsic loop's ratio falls with drive frequency for beta_L ~ 1-10 but
rises for beta_L >~ 30, so the sign of the trend carries no information without
independently knowing beta_L).

This tries a structural difference instead. The intrinsic model is a single
overdamped phase variable: for a given (i_dc, i_ac, Omega) its attractor is
unique, so the measured voltage cannot depend on the initial condition. The
extrinsic loop has two coupled phases and can support several coexisting
attractors, so it can. Experimentally the corresponding statement is that the
half-step should show hysteresis between an up-sweep and a down-sweep of the
bias current — a bidirectional IV sweep, which every experiment already does,
and which perturbs nothing.

The test here is the cheap proxy for that: run the same sweep from several
initial phases and see whether the measured plateau width moves.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from shapiro import normalize_cpr, run_intrinsic, run_extrinsic, step_width  # noqa: E402
from discriminator import current_grid  # noqa: E402

RESULTS = os.path.join(_HERE, "..", "results")
ALPHA = 0.5


def _widths(v, i_dc, omega, rel_tol=0.005):
    tol = rel_tol * omega
    return (step_width(i_dc, v, 0.5 * omega, tol),
            step_width(i_dc, v, 1.0 * omega, tol))


def scan_initial_conditions(omega: float = 5.0, d_i: float = 0.01,
                            phases=(0.0, np.pi / 2, np.pi, 3 * np.pi / 2)) -> dict:
    """Same sweep, different starting phase. Unique attractor => no change."""
    i_dc = current_grid(omega, ALPHA, d_i)
    c = normalize_cpr([1.0, -0.25])
    out = {"omega": omega, "d_i": d_i, "phases": list(phases),
           "intrinsic": [], "extrinsic": {}}

    print(f"  intrinsic (single phase variable), Omega={omega}:")
    for p0 in phases:
        v = run_intrinsic(i_dc, ALPHA * omega, c, omega,
                          n_transient=120, n_avg=300, phi0_offset=p0)
        wh, w1 = _widths(v, i_dc, omega)
        out["intrinsic"].append({"phi0": p0, "w_half": wh, "w_one": w1})
        print(f"    phi0={p0:6.3f}  w_half={wh:.4f}  w_one={w1:.4f}", flush=True)
    ih = [r["w_half"] for r in out["intrinsic"]]
    out["intrinsic_spread"] = float(max(ih) - min(ih))
    print(f"    -> w_half spread {out['intrinsic_spread']:.4f}", flush=True)

    for beta_l in (1.0, 4.0, 30.0):
        rows = []
        print(f"  extrinsic loop, beta_L={beta_l}:")
        for p0 in phases:
            v = run_extrinsic(i_dc, ALPHA * omega, omega, beta_l=beta_l,
                              flux=0.5, asym=0.3, n_transient=80, n_avg=300,
                              phi0_offset=p0)
            wh, w1 = _widths(v, i_dc, omega)
            rows.append({"phi0": p0, "w_half": wh, "w_one": w1})
            print(f"    phi0={p0:6.3f}  w_half={wh:.4f}  w_one={w1:.4f}",
                  flush=True)
        spread = (max(r["w_half"] for r in rows)
                  - min(r["w_half"] for r in rows))
        mean = np.mean([r["w_half"] for r in rows])
        out["extrinsic"][str(beta_l)] = {
            "rows": rows, "w_half_spread": float(spread),
            "relative_spread": float(spread / mean) if mean else float("nan")}
        print(f"    -> w_half spread {spread:.4f} "
              f"({spread / mean:.1%} of mean)" if mean else "", flush=True)
    return out


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    out = {}
    for om in (2.0, 5.0, 8.0):
        print(f"\nOmega = {om}")
        out[str(om)] = scan_initial_conditions(omega=om)
    with open(os.path.join(RESULTS, "multistability.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {RESULTS}/multistability.json")


if __name__ == "__main__":
    main()
