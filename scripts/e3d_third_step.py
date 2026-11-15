"""E3d — the third step, which turns two points into a power law.

E3b/E3c measure the thermal visibility threshold at k = 2 and k = 4 and reject
both published laws:

    Gamma*_4 / Gamma*_2 = 0.1245 +/- 0.0062
        Duprat  k^2 T   predicts 0.2500   -> 20 sigma away
        barrier |c_k|   predicts 0.0897   -> 5.7 sigma away

and land, unasked, on (2/4)^3 = 0.125, i.e. Gamma*_k proportional to k^-3.
Two points cannot establish an exponent -- any law through both fits -- so the
claim is worth nothing until a third k is measured.  k = 3 is available and it
discriminates:

    Gamma*_3 / Gamma*_2 :   k^-3      0.2963
                            k^-2      0.4444   (Duprat)
                            |c3/c2|   0.2834   (barrier)

k^-3 and the barrier reading sit 4.5% apart here, so this point cannot separate
those two -- the quarter step already did that at 5.7 sigma.  What it can do is
kill or confirm the k^-3 reading itself: if the exponent is real, the three
crossings fall on a straight line in log Gamma* against log k with slope -3,
and Duprat's slope -2 is excluded by a lever arm the two-point comparison did
not have.

Everything is held at the same matched drive as E3c, k*alpha = 2.4, so the
J_1(2 k alpha) factor is common to all three orders: alpha = 1.20, 0.80, 0.60.
"""

from __future__ import annotations

import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from shapiro import normalize_cpr  # noqa: E402
from cpr_data import load_cpr  # noqa: E402
import e3b_thermal_refine as E  # noqa: E402

RESULTS = os.path.join(_HERE, "..", "results")
FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
K_ALPHA = 2.4
ALPHA_THIRD = K_ALPHA / 3.0   # 0.80
N_THIRD = 120


def main() -> None:
    c = load_cpr("0.8")
    ctrl = normalize_cpr([1.0])

    matched = json.load(open(os.path.join(RESULTS, "thermal_laws_matched.json")))
    gs2 = matched["gamma_star_half"]
    gs4 = matched["gamma_star_quarter"]
    b2 = matched["half"]["gamma_star_band"]
    prev = json.load(open(os.path.join(RESULTS, "thermal_laws_refined.json")))
    b4 = prev["quarter"]["gamma_star_band"]

    print(f"matched drive k*alpha = {K_ALPHA}: third step at alpha = {ALPHA_THIRD:.2f}")
    print(f"anchors: Gamma*_2 = {gs2:.6f}   Gamma*_4 = {gs4:.6f}")
    for name, r in (("k^-3", (2 / 3) ** 3), ("k^-2 (Duprat)", (2 / 3) ** 2),
                    ("barrier |c3/c2|", abs(c[2] / c[1]))):
        print(f"   {name:18s} predicts Gamma*_3 = {gs2 * r:.6f}  (ratio {r:.4f})")
    print(flush=True)

    # bracket every prediction: 0.0054 (k^-3) .. 0.0085 (Duprat)
    g3 = [0.0035, 0.0045, 0.0054, 0.0062, 0.0070, 0.0085, 0.0100]
    with ProcessPoolExecutor(max_workers=E.N_WORKERS) as pool:
        third = E.sweep(pool, f"third step, alpha={ALPHA_THIRD:.2f}", c, ctrl,
                        E.OMEGA / 3.0, ALPHA_THIRD, g3, N_THIRD, 910001)
    third["gamma_star"] = E._crossing([r["gamma"] for r in third["rows"]],
                                      [r["survival"] for r in third["rows"]])
    third["gamma_star_band"] = E._mc(third)
    gs3, b3 = third["gamma_star"], third["gamma_star_band"]

    def sd(b):
        return 0.5 * (b["hi68"] - b["lo68"])

    ks = np.array([2.0, 3.0, 4.0])
    gs = np.array([gs2, gs3, gs4])
    es = np.array([sd(b2), sd(b3), sd(b4)])
    # weighted least squares of log Gamma* against log k
    x, y, w = np.log(ks), np.log(gs), 1.0 / (es / gs) ** 2
    sw, sx, sy = w.sum(), (w * x).sum(), (w * y).sum()
    sxx, sxy = (w * x * x).sum(), (w * x * y).sum()
    det = sw * sxx - sx * sx
    slope = (sw * sxy - sx * sy) / det
    inter = (sxx * sy - sx * sxy) / det
    slope_err = math.sqrt(sw / det)
    resid = y - (inter + slope * x)
    chi2 = float((w * resid ** 2).sum())

    print(f"\n  Gamma*_3 = {gs3:.6f}  68% [{b3['lo68']:.6f}, {b3['hi68']:.6f}]")
    print(f"  ratio Gamma*_3/Gamma*_2 = {gs3/gs2:.4f}")
    print("\n  POWER-LAW FIT  Gamma*_k ~ k^p   (three orders, matched drive)")
    print(f"    p = {slope:.3f} +/- {slope_err:.3f}   chi2 = {chi2:.2f} (1 dof)")
    for name, pv in (("k^-3", -3.0), ("k^-2 (Duprat)", -2.0)):
        print(f"      {name:16s} p = {pv:+.1f}  -> "
              f"{abs(slope - pv)/slope_err:.1f} sigma")

    out = {"k_alpha": K_ALPHA, "alpha_third": ALPHA_THIRD, "third": third,
           "gamma_star": {"k2": gs2, "k3": gs3, "k4": gs4},
           "gamma_star_sd": {"k2": sd(b2), "k3": sd(b3), "k4": sd(b4)},
           "ratio_3_over_2": gs3 / gs2,
           "pred_ratio_3_over_2": {"k_minus_3": (2 / 3) ** 3,
                                   "duprat": (2 / 3) ** 2,
                                   "barrier": abs(c[2] / c[1])},
           "fit": {"slope": slope, "slope_err": slope_err,
                   "intercept": inter, "chi2_1dof": chi2}}
    with open(os.path.join(RESULTS, "thermal_laws_power.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.errorbar(ks, gs, yerr=es, fmt="o", ms=6, capsize=3, color="k",
                label="measured (matched $k\\alpha=2.4$)")
    kk = np.linspace(1.9, 4.2, 60)
    ax.plot(kk, np.exp(inter) * kk ** slope, "-", color="tab:blue", lw=1.4,
            label=rf"fit $p={slope:.2f}\pm{slope_err:.2f}$")
    ax.plot(kk, gs2 * (kk / 2.0) ** -3, "--", color="tab:green", lw=1.2,
            label=r"$k^{-3}$")
    ax.plot(kk, gs2 * (kk / 2.0) ** -2, "-.", color="tab:red", lw=1.2,
            label=r"$k^{-2}$ (Duprat)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([2, 3, 4]); ax.set_xticklabels(["2", "3", "4"])
    ax.set_xlabel("step denominator $k$")
    ax.set_ylabel(r"$\Gamma^*_k = (k_BT/E_J)_{50\%}$")
    ax.set_title("Thermal visibility threshold vs step order", fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIGURES, "e3d_power_law.png")
    fig.savefig(path, dpi=140); plt.close(fig)
    print(f"\nwrote {path} and thermal_laws_power.json")


if __name__ == "__main__":
    main()
