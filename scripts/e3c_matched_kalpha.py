"""E3c — close the one loophole in E3b's rejection of the Duprat scaling.

E3b measured the half and quarter steps each at its OWN drive optimum:
alpha = 0.90 (k*alpha = 1.8) and alpha = 0.60 (k*alpha = 2.4).  Duprat & Levy
Yeyati's k^2 T collapse is not stated at arbitrary drive -- it is stated at
FIXED k*alpha, because the order-n/k step width carries a J_1(2 k alpha) factor
that has to be held constant before the temperature scaling is read off.  So
E3b's 10-sigma rejection of that law is, strictly, a rejection of it applied
outside its own stated condition.

This run removes the loophole.  Holding k*alpha = 2.4 puts the half step at
alpha = 1.20 and the quarter step at alpha = 0.60 -- and alpha = 0.60 is
exactly where E3b already measured the quarter step, so only the half step has
to be re-run.  The deterministic penalty is small (net width 0.13329 at
alpha = 1.20 against 0.14881 at the optimum 0.90, an 10% loss), which keeps the
survival curve as well resolved as before.

The barrier reading carries no such condition -- it names |c_k| E_J and nothing
else -- so its E3b test stands either way, and is repeated here against the
matched-drive anchor for consistency.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from shapiro import normalize_cpr  # noqa: E402
from cpr_data import load_cpr  # noqa: E402
import e3b_thermal_refine as E  # noqa: E402

RESULTS = os.path.join(_HERE, "..", "results")
K_ALPHA = 2.4
ALPHA_HALF_MATCHED = K_ALPHA / 2.0     # 1.20
ALPHA_QUARTER_MATCHED = K_ALPHA / 4.0  # 0.60


def main() -> None:
    c = load_cpr("0.8")
    ctrl = normalize_cpr([1.0])
    r42 = abs(c[3] / c[1])

    prev = json.load(open(os.path.join(RESULTS, "thermal_laws_refined.json")))
    quarter = prev["quarter"]
    assert abs(quarter["alpha"] - ALPHA_QUARTER_MATCHED) < 1e-9, quarter["alpha"]
    print(f"matched drive: k*alpha = {K_ALPHA}")
    print(f"  half    alpha = {ALPHA_HALF_MATCHED:.2f}   (re-run here)")
    print(f"  quarter alpha = {ALPHA_QUARTER_MATCHED:.2f}   (reused from E3b, "
          f"Gamma* = {quarter['gamma_star']:.6f})\n", flush=True)

    g_half = [0.010, 0.014, 0.017, 0.020, 0.024, 0.028]
    with ProcessPoolExecutor(max_workers=E.N_WORKERS) as pool:
        half = E.sweep(pool, f"half step, alpha={ALPHA_HALF_MATCHED:.2f}",
                       c, ctrl, 0.5 * E.OMEGA, ALPHA_HALF_MATCHED,
                       g_half, E.N_HALF, 909001)

    half["gamma_star"] = E._crossing([r["gamma"] for r in half["rows"]],
                                     [r["survival"] for r in half["rows"]])
    half["gamma_star_band"] = E._mc(half)
    gs2, gs4 = half["gamma_star"], quarter["gamma_star"]
    b2, b4 = half["gamma_star_band"], quarter["gamma_star_band"]

    pred = {"duprat": gs2 / 4.0, "barrier": gs2 * r42}
    tests = [E._test_at(quarter, pred["duprat"], "Duprat k^2 T"),
             E._test_at(quarter, pred["barrier"], "barrier |c_k|")]

    # ratio and its error, from the Monte-Carlo bands on both crossings
    s2 = 0.5 * (b2["hi68"] - b2["lo68"])
    s4 = 0.5 * (b4["hi68"] - b4["lo68"])
    ratio = gs4 / gs2
    ratio_err = ratio * float(np.hypot(s4 / gs4, s2 / gs2))

    print("\n  ADJUDICATION AT MATCHED k*alpha")
    print(f"    Gamma*_2 = {gs2:.6f}  68% [{b2['lo68']:.6f}, {b2['hi68']:.6f}]")
    print(f"    Gamma*_4 = {gs4:.6f}  68% [{b4['lo68']:.6f}, {b4['hi68']:.6f}]")
    print(f"    ratio    = {ratio:.4f} +/- {ratio_err:.4f}")
    for name, r in (("Duprat k^2 T", 0.25), ("barrier |c_k|", r42)):
        print(f"      {name:16s} predicts ratio {r:.4f}  "
              f"-> {abs(ratio - r) / ratio_err:.1f} sigma")
    print("\n    direct test -- survival measured AT each predicted Gamma*")
    for t in tests:
        if not t.get("in_range"):
            print(f"      {t['law']:16s} Gamma={t['gamma_pred']:.6f}  outside grid")
            continue
        print(f"      {t['law']:16s} Gamma={t['gamma_pred']:.6f}  "
              f"survival={t['survival_at_pred']:.3f}+/-{t['sem_at_pred']:.3f}  "
              f"|z|={abs(t['z']):.1f}")

    out = {"k_alpha": K_ALPHA, "alpha_half": ALPHA_HALF_MATCHED,
           "alpha_quarter": ALPHA_QUARTER_MATCHED, "cpr": c.tolist(),
           "half": half, "quarter_reused_from": "thermal_laws_refined.json",
           "gamma_star_half": gs2, "gamma_star_quarter": gs4,
           "ratio_measured": ratio, "ratio_err": ratio_err,
           "ratio_duprat": 0.25, "ratio_barrier": r42,
           "sigma_duprat": abs(ratio - 0.25) / ratio_err,
           "sigma_barrier": abs(ratio - r42) / ratio_err,
           "tests": tests}
    with open(os.path.join(RESULTS, "thermal_laws_matched.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote thermal_laws_matched.json")


if __name__ == "__main__":
    main()
