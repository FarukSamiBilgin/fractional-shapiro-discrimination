"""E5 — the control subtraction's residual, measured where the result is read.

Every width in this track is net: the kernel estimator evaluated for a sinusoidal
control CPR is subtracted from the same estimator evaluated for the CPR under
test.  The justification is that the control carries no fractional step, so what
it returns is the estimator's transit floor, common to both and cancelling.

The floor is not common.  On a linear transit the kernel estimator returns
sigma_k*sqrt(2*pi)/|dv/di|, so it is set by the junction's ohmic slope at the
step -- and the two CPRs do not share that slope.  What survives the subtraction
is therefore

    net = (true step) + [floor_signal(alpha) - floor_control(alpha)]

and the bracket is a systematic, not zero.

Where it has to be measured matters, and a first attempt got this wrong.  The
natural idea is to evaluate both CPRs at the drive where the step is absent --
the node -- since any width returned there is floor and nothing else.  But the
floor depends on the drive, and the thermal result is read at a different drive
(the step's own optimum), so a floor measured at the node is the right quantity
in the wrong place.  This module measures the slopes at the operating drives and
forms the floor difference there.

The damage is in the k-dependence, which is the whole thermal result: the bias is
roughly constant in magnitude across orders while the net width falls by 30x, so
it is 1.5% of the half-step net and nearly 60% of the quarter-step net.
"""

from __future__ import annotations

import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from shapiro import (  # noqa: E402
    normalize_cpr, run_intrinsic, kernel_width, analytic_transit,
)
from cpr_data import load_cpr  # noqa: E402

RESULTS = os.path.join(_HERE, "..", "results")
OMEGA = 5.0
SIGMA_K = 0.025
HALF_WINDOW = 0.5
D_I = 0.005
N_AVG = 300
# matched effective drive k*alpha = 2.4, as used for the thermal comparison
OPERATING = ((2, 1.20), (3, 0.80), (4, 0.60))


def probe(c, alpha: float, target: float) -> tuple[float, float]:
    """Kernel width and the median |dv/di| through the transit."""
    coarse = np.linspace(0.02 * OMEGA, 1.5 * OMEGA, 140)
    v = run_intrinsic(coarse, alpha * OMEGA, c, OMEGA, n_transient=150,
                      n_avg=100, spp=60)
    centre = coarse[int(np.argmin(np.abs(v - target)))]
    grid = centre + np.arange(-HALF_WINDOW, HALF_WINDOW + D_I, D_I)
    vv = run_intrinsic(grid, alpha * OMEGA, c, OMEGA, n_transient=150,
                       n_avg=N_AVG, spp=60)
    w = kernel_width(grid, vv, target, SIGMA_K)
    d = np.abs(np.gradient(vv, grid))
    near = np.abs(vv - target) < 0.25
    slope = float(np.median(d[near])) if near.sum() > 3 else float(np.median(d))
    return float(w), slope


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    c = load_cpr("0.8")
    ctrl = normalize_cpr([1.0])
    out = {"omega": OMEGA, "sigma_k": SIGMA_K, "k_alpha": 2.4, "rows": []}

    print("floor mismatch at the operating drive (k*alpha = 2.4)")
    print(f"  {'k':>2} {'alpha':>5} {'w_sig':>8} {'w_ctl':>8} {'net':>8} "
          f"{'slope_s':>7} {'slope_c':>7} {'bias':>9} {'bias/net':>9}")
    for k, alpha in OPERATING:
        target = OMEGA / k
        w_s, s_s = probe(c, alpha, target)
        w_c, s_c = probe(ctrl, alpha, target)
        floor_s = analytic_transit(SIGMA_K, s_s)
        floor_c = analytic_transit(SIGMA_K, s_c)
        net = w_s - w_c
        bias = floor_s - floor_c
        out["rows"].append({"k": k, "alpha": alpha, "w_signal": w_s,
                            "w_control": w_c, "net": net,
                            "slope_signal": s_s, "slope_control": s_c,
                            "floor_signal": floor_s, "floor_control": floor_c,
                            "bias": bias, "bias_over_net": bias / net})
        print(f"  {k:>2} {alpha:>5.2f} {w_s:>8.5f} {w_c:>8.5f} {net:>8.5f} "
              f"{s_s:>7.3f} {s_c:>7.3f} {bias:>+9.5f} {bias / net:>+8.1%}",
              flush=True)

    n2 = out["rows"][0]["net"]
    n4 = out["rows"][2]["net"]
    b2 = out["rows"][0]["bias"]
    b4 = out["rows"][2]["bias"]
    print("\n  the deterministic width ratio, with a fraction of the bias "
          "restored")
    out["sensitivity"] = []
    for f in (0.0, 0.25, 0.5, 1.0):
        r = (n4 - f * b4) / (n2 - f * b2)
        out["sensitivity"].append({"fraction_restored": f, "ratio": r})
        print(f"    {f:4.0%}  net(k=2)={n2 - f * b2:.5f}  "
              f"net(k=4)={n4 - f * b4:.5f}  ratio={r:.4f}")
    r0 = out["sensitivity"][0]["ratio"]
    r1 = out["sensitivity"][-1]["ratio"]
    print(f"\n  VERDICT: restoring the measured bias moves the deterministic "
          f"width ratio by {abs(r1 - r0) / r0:.0%}. Survival is net/det, so the "
          f"thermal exponent inherits this directly; it is not covered by the "
          f"statistical band.")
    with open(os.path.join(RESULTS, "floor_systematic.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote floor_systematic.json")


if __name__ == "__main__":
    main()
