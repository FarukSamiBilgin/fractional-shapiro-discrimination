"""Is the flat extrinsic ratio real, or is it grid resolution?

The frequency scan reports w_half/w_one rising by a factor ~4 for the intrinsic
mechanism and staying flat for the extrinsic one. Both widths are measured by
counting current-grid points inside a plateau, so a coarse grid could fake
flatness. Three checks:

V1  Refine the current grid by 4x at three frequencies and see whether the
    numbers move.
V2  Vary the loop parameter beta_L. If the flat ratio is an accident of one
    beta_L it is not a usable signature; if it survives, it is.
V3  Vary the intrinsic harmonic content c2/c1. The rise should be a property of
    the mechanism, not of one particular skew.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from shapiro import normalize_cpr, run_intrinsic, run_extrinsic, sweep_widths  # noqa: E402
from discriminator import current_grid  # noqa: E402

RESULTS = os.path.join(_HERE, "..", "results")
ALPHA = 0.5


def _intr(om, c, d_i, **kw):
    grid = current_grid(om, ALPHA, d_i)
    return sweep_widths(lambda idc, omega, **k: run_intrinsic(
        idc, ALPHA * omega, c, omega, **k), grid, om,
        n_transient=120, n_avg=300, **kw)


def _extr(om, beta_l, d_i, flux=0.5, asym=0.3, **kw):
    grid = current_grid(om, ALPHA, d_i)
    return sweep_widths(lambda idc, omega, **k: run_extrinsic(
        idc, ALPHA * omega, omega, beta_l=beta_l, flux=flux, asym=asym, **k),
        grid, om, n_transient=80, n_avg=300, **kw)


def v1_grid_refinement() -> dict:
    c = normalize_cpr([1.0, -0.25])
    rows = []
    print("V1 — grid refinement (ratio should not move):")
    for om in (1.5, 5.0, 10.0):
        for d_i in (0.02, 0.005):
            a = _intr(om, c, d_i)
            b = _extr(om, 1.0, d_i)
            ra = a["w_half"] / a["w_one"] if a["w_one"] else float("nan")
            rb = b["w_half"] / b["w_one"] if b["w_one"] else float("nan")
            rows.append({"omega": om, "d_i": d_i, "intrinsic_ratio": ra,
                         "extrinsic_ratio": rb,
                         "intrinsic_w_half": a["w_half"],
                         "extrinsic_w_half": b["w_half"]})
            print(f"   Omega={om:<5} d_i={d_i:<6} intrinsic {ra:.3f}   "
                  f"extrinsic {rb:.3f}", flush=True)
    return {"rows": rows}


def v2_beta_dependence() -> dict:
    rows = []
    print("\nV2 — does the flat extrinsic ratio survive other beta_L?")
    for beta_l in (0.5, 1.0, 2.0, 4.0):
        vals = {}
        for om in (2.0, 8.0):
            b = _extr(om, beta_l, 0.01)
            vals[om] = b["w_half"] / b["w_one"] if b["w_one"] else float("nan")
        change = (vals[8.0] - vals[2.0]) / vals[2.0] if vals[2.0] else float("nan")
        rows.append({"beta_l": beta_l, "ratio_omega2": vals[2.0],
                     "ratio_omega8": vals[8.0], "relative_change": change})
        print(f"   beta_L={beta_l:<5} ratio(2)={vals[2.0]:.3f}  "
              f"ratio(8)={vals[8.0]:.3f}  change={change:+.1%}", flush=True)
    return {"rows": rows}


def v3_skew_dependence() -> dict:
    rows = []
    print("\nV3 — is the intrinsic rise generic in c2/c1?")
    for c2 in (-0.15, -0.25, -0.40):
        c = normalize_cpr([1.0, c2])
        vals = {}
        for om in (2.0, 8.0):
            a = _intr(om, c, 0.01)
            vals[om] = a["w_half"] / a["w_one"] if a["w_one"] else float("nan")
        change = (vals[8.0] - vals[2.0]) / vals[2.0] if vals[2.0] else float("nan")
        rows.append({"c2_over_c1": c2, "ratio_omega2": vals[2.0],
                     "ratio_omega8": vals[8.0], "relative_change": change})
        print(f"   c2/c1={c2:<6} ratio(2)={vals[2.0]:.3f}  "
              f"ratio(8)={vals[8.0]:.3f}  change={change:+.1%}", flush=True)
    return {"rows": rows}


def v4_large_beta() -> dict:
    """Push beta_L into the regime a real twisted-cuprate device sits in.

    beta_L = 2 pi L Ic / Phi_0. Xue's devices are driven with current pulses of
    ~10 mA peak at ~10x Ic, so Ic is of order 1 mA; a micron-scale trapped-flux
    loop has L of order 1 pH, giving beta_L ~ 10^3. The V2 scan only reached 4,
    where the ratio already falls by 60%; if the fall persists and deepens the
    test becomes two-sided for exactly the devices that motivated it.
    """
    rows = []
    print("\nV4 — large beta_L, the regime of a mA-scale junction:")
    for beta_l in (4.0, 10.0, 30.0, 100.0):
        vals = {}
        for om in (2.0, 8.0):
            b = _extr(om, beta_l, 0.01)
            vals[om] = b["w_half"] / b["w_one"] if b["w_one"] else float("nan")
        change = ((vals[8.0] - vals[2.0]) / vals[2.0]
                  if vals[2.0] else float("nan"))
        rows.append({"beta_l": beta_l, "ratio_omega2": vals[2.0],
                     "ratio_omega8": vals[8.0], "relative_change": change})
        print(f"   beta_L={beta_l:<6} ratio(2)={vals[2.0]:.3f}  "
              f"ratio(8)={vals[8.0]:.3f}  change={change:+.1%}", flush=True)
    return {"rows": rows}


def v5_intrinsic_bound() -> dict:
    """How reliably does the intrinsic ratio rise? Wider skew range."""
    rows = []
    print("\nV5 — intrinsic rise across a wide harmonic content:")
    for c2 in (-0.10, -0.15, -0.20, -0.25, -0.30, -0.40, -0.49):
        c = normalize_cpr([1.0, c2])
        vals = {}
        for om in (2.0, 8.0):
            a = _intr(om, c, 0.01)
            vals[om] = a["w_half"] / a["w_one"] if a["w_one"] else float("nan")
        change = ((vals[8.0] - vals[2.0]) / vals[2.0]
                  if vals[2.0] else float("nan"))
        rows.append({"c2_over_c1": c2, "ratio_omega2": vals[2.0],
                     "ratio_omega8": vals[8.0], "relative_change": change})
        print(f"   c2/c1={c2:<6} ratio(2)={vals[2.0]:.3f}  "
              f"ratio(8)={vals[8.0]:.3f}  change={change:+.1%}", flush=True)
    changes = [r["relative_change"] for r in rows
               if np.isfinite(r["relative_change"])]
    print(f"   -> intrinsic change range: {min(changes):+.1%} to {max(changes):+.1%}")
    return {"rows": rows, "min_change": min(changes), "max_change": max(changes)}


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    out = {"v1": v1_grid_refinement(), "v2": v2_beta_dependence(),
           "v3": v3_skew_dependence(), "v4": v4_large_beta(),
           "v5": v5_intrinsic_bound()}
    with open(os.path.join(RESULTS, "verify.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {RESULTS}/verify.json")


if __name__ == "__main__":
    main()


