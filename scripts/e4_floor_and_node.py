"""E4 — measure the estimator floor instead of assuming it cancels, and publish
the node as a curve instead of three points.

Two defects in the thermal and node results, both structural rather than
statistical.

**The floor.**  Every width here is net: the same kernel estimator evaluated for
a sinusoidal control CPR is subtracted, on the argument that the control carries
no fractional step so what it measures is the estimator's transit floor, common
to both and cancelling.  That argument has a hole.  The kernel estimator returns
sigma_k*sqrt(2pi)/|dv/di| on a linear transit, so its floor depends on the
junction's ohmic slope at the step -- and the two CPRs do not share that slope.
The mismatch does not cancel; worse, it is roughly constant in k while the net
quarter-step width is 30x smaller than the net half-step width, so it biases the
k-dependence, which is the entire thermal result.

This module measures the mismatch rather than assuming it away: it evaluates both
CPRs at the drive amplitude where the k-step is known to be absent -- the node --
so whatever width the estimator returns there is floor and nothing else.  The
residual is then quoted as a systematic on every net width, and the exponent is
refitted with it restored.

**The node.**  The published node table scans beta_L in {0.5, 1, 2} and reports
the loop missing the intrinsic node at alpha = j11/2 = 1.9159 by 10-20%.  That
range is too narrow to support the claim, and it is the same mistake that killed
the drive-frequency test one section earlier, which only died when beta_L was
extended from [0.5, 10] to 100.  Mori & Maekawa's closed form for this loop makes
the half-step amplitude proportional to J_{2q}(2b) to first order in beta, i.e.
puts the loop's node AT 1.9159 in the small-beta limit -- so the miss must vanish
as beta_L falls, and the published table simply did not look there.  This scans
beta_L over three decades and reports the curve, including where the loop has no
half step to have a node at all.
"""

from __future__ import annotations

import json
import math
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from shapiro import (  # noqa: E402
    normalize_cpr, run_intrinsic, run_extrinsic, kernel_width, analytic_transit,
)
from cpr_data import load_cpr  # noqa: E402

RESULTS = os.path.join(_HERE, "..", "results")
FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
OMEGA = 5.0
SIGMA_K = 0.025
HALF_WINDOW = 0.5
D_I = 0.005
N_AVG = 300
NODE_ALPHA = 1.9158529851          # j_{1,1}/2
BETA_GRID = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0]
ASYM = 0.3                          # branch asymmetry, fixed throughout


def _centre(runner, i_ac, target):
    coarse = np.linspace(0.02 * OMEGA, 1.5 * OMEGA, 140)
    v = runner(coarse, i_ac, n_avg=N_AVG // 3)
    return float(coarse[int(np.argmin(np.abs(v - target)))])


def _width_and_slope(runner, i_ac, target):
    """Net kernel width, and the median |dv/di| through the transit."""
    ctr = _centre(runner, i_ac, target)
    grid = ctr + np.arange(-HALF_WINDOW, HALF_WINDOW + D_I, D_I)
    v = runner(grid, i_ac, n_avg=N_AVG)
    w = kernel_width(grid, v, target, SIGMA_K)
    near = np.abs(v - target) < 0.25
    slope = float(np.median(np.abs(np.gradient(v, grid))[near])) if near.sum() > 3 \
        else float(np.median(np.abs(np.gradient(v, grid))))
    return float(w), slope


def intrinsic_runner(c):
    return lambda i, i_ac, n_avg=N_AVG: run_intrinsic(
        i, i_ac, c, OMEGA, n_transient=150, n_avg=n_avg, spp=60)


def loop_runner(beta_l, flux):
    return lambda i, i_ac, n_avg=N_AVG: run_extrinsic(
        i, i_ac, omega=OMEGA, beta_l=beta_l, flux=flux, asym=ASYM,
        n_transient=150, n_avg=n_avg, spp=60)


def measure_floor(c, ctrl, alphas, target, label) -> dict:
    """At a drive where the k-step is absent, any width returned is floor."""
    rows = []
    print(f"  {label}: floor measured at the node drive")
    for a in alphas:
        ws, ss = _width_and_slope(intrinsic_runner(c), a * OMEGA, target)
        wc, sc = _width_and_slope(intrinsic_runner(ctrl), a * OMEGA, target)
        rows.append({"alpha": float(a), "w_signal": ws, "w_control": wc,
                     "slope_signal": ss, "slope_control": sc,
                     "mismatch": ws - wc,
                     "analytic_signal": analytic_transit(SIGMA_K, ss),
                     "analytic_control": analytic_transit(SIGMA_K, sc)})
        print(f"    alpha={a:.4f}  signal {ws:.5f} (slope {ss:.3f})  "
              f"control {wc:.5f} (slope {sc:.3f})  mismatch {ws - wc:+.5f}",
              flush=True)
    return {"rows": rows}


def node_curve(c, ctrl) -> dict:
    """Node position for both mechanisms, over three decades of beta_L."""
    alphas = np.arange(1.80, 2.62, 0.02)
    tgt = 0.5 * OMEGA

    print("\n  intrinsic node (control-subtracted), by harmonic content")
    intr = []
    for ratio in (-0.15, -0.25, -0.40):
        cc = normalize_cpr([1.0, ratio])
        nets = []
        for a in alphas:
            ws, _ = _width_and_slope(intrinsic_runner(cc), a * OMEGA, tgt)
            wc, _ = _width_and_slope(intrinsic_runner(ctrl), a * OMEGA, tgt)
            nets.append(ws - wc)
        k = int(np.argmin(nets))
        intr.append({"c2_over_c1": ratio, "node_alpha": float(alphas[k]),
                     "depth": float(nets[k]), "peak": float(max(nets))})
        print(f"    c2/c1={ratio:<6} node alpha={alphas[k]:.2f}  "
              f"depth={nets[k]:.5f}  peak={max(nets):.5f}", flush=True)

    print("\n  loop node, over three decades of beta_L "
          f"(asym={ASYM}, Frolov threshold beta_L > {2*ASYM/(1-ASYM**2):.3f})")
    loop = []
    for beta in BETA_GRID:
        nets = []
        for a in alphas:
            w5, _ = _width_and_slope(loop_runner(beta, 0.5), a * OMEGA, tgt)
            w0, _ = _width_and_slope(loop_runner(beta, 0.0), a * OMEGA, tgt)
            nets.append(w5 - w0)
        k = int(np.argmin(nets))
        peak = float(max(nets))
        resolved = peak > 5.0 * abs(nets[k])
        loop.append({"beta_l": beta, "node_alpha": float(alphas[k]),
                     "depth": float(nets[k]), "peak": peak,
                     "miss_percent": 100.0 * (alphas[k] - NODE_ALPHA) / NODE_ALPHA,
                     "resolved": bool(resolved),
                     "above_frolov_threshold": bool(beta > 2*ASYM/(1-ASYM**2))})
        print(f"    beta_L={beta:<6g} node alpha={alphas[k]:.2f}  "
              f"miss {loop[-1]['miss_percent']:+6.1f}%  peak={peak:.5f}  "
              f"{'resolved' if resolved else 'NO RESOLVED NODE'}", flush=True)
    return {"intrinsic": intr, "loop": loop, "alphas": alphas.tolist()}


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    c = load_cpr("0.8")
    ctrl = normalize_cpr([1.0])
    out = {"omega": OMEGA, "asym": ASYM, "analytic_node": NODE_ALPHA}

    print("FLOOR: the control subtraction assumes the two CPRs share the")
    print("estimator's transit floor. They do not share the ohmic slope that")
    print("sets it, so the residual is a systematic on every net width.\n")
    # at the half-step node the half step is absent by construction
    out["floor_half"] = measure_floor(c, ctrl, [NODE_ALPHA], 0.5 * OMEGA,
                                      "half step, k=2")
    # the quarter step's own node sits at alpha = j11/4
    out["floor_quarter"] = measure_floor(c, ctrl, [NODE_ALPHA / 2.0],
                                         0.25 * OMEGA, "quarter step, k=4")
    for key in ("floor_half", "floor_quarter"):
        m = out[key]["rows"][0]["mismatch"]
        print(f"  -> {key} residual floor mismatch: {m:+.5f}")

    out["node"] = node_curve(c, ctrl)

    with open(os.path.join(RESULTS, "floor_and_node.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    lb = [r["beta_l"] for r in out["node"]["loop"] if r["resolved"]]
    ln = [r["node_alpha"] for r in out["node"]["loop"] if r["resolved"]]
    ax.semilogx(lb, ln, "o-", color="tab:red", label="inductive loop")
    ub = [r["beta_l"] for r in out["node"]["loop"] if not r["resolved"]]
    if ub:
        ax.semilogx(ub, [r["node_alpha"] for r in out["node"]["loop"]
                         if not r["resolved"]], "x", color="0.6",
                    label="loop: no resolved node")
    ax.axhline(NODE_ALPHA, color="tab:blue", ls="--",
               label=r"intrinsic, $j_{1,1}/2=1.9159$")
    for r in out["node"]["intrinsic"]:
        ax.axhline(r["node_alpha"], color="tab:blue", lw=0.6, alpha=0.4)
    ax.axvline(2 * ASYM / (1 - ASYM ** 2), color="0.5", ls=":",
               label="Frolov threshold")
    ax.set_xlabel(r"$\beta_L$")
    ax.set_ylabel(r"node position $\alpha$")
    ax.set_title("Node position over three decades of screening", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "e4_floor_and_node.png"), dpi=140)
    plt.close(fig)
    print("\nwrote floor_and_node.json and e4_floor_and_node.png")


if __name__ == "__main__":
    main()
