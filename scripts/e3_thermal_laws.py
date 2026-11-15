"""E3 — adjudicate two published thermal-scaling laws at the quarter step.

Two laws exist for how a fractional Shapiro step of order n/k survives thermal
phase noise, and nothing in the literature has decided between them.

  * Duprat & Levy Yeyati (PRB 71, 054510 (2005)) collapse the order-n/k step
    against k^2 T at fixed k*alpha: the visibility threshold scales as
        Gamma*_k  ∝  1/k^2 .
  * The barrier reading used by the MATBG manuscript makes the relevant energy
    the harmonic's own well depth |c_k| E_J, giving
        Gamma*_k  ∝  |c_k| .

At the half step the two agree too closely to separate. At the third step they
differ by 1.57x. At the QUARTER step, with the Beenakker tau = 0.8 harmonics
(c2 = -0.18419, c4 = -0.016530), they differ by

        Gamma*_4 / Gamma*_2  =  1/4      (Duprat)
                            =  |c4/c2| = 0.0897   (barrier)

a factor of 2.8 — enough to decide, and cheap, because both are anchored to
Gamma*_2 measured on the same solver in the same run rather than imported.

Protocol. Find each step's own drive optimum deterministically, then sweep
Gamma with an ensemble of stochastic-Heun realisations, subtracting at every
Gamma the same width computed for a sinusoidal CPR — a control that carries no
fractional step by construction, so what it measures is the estimator's transit
floor. Survival is the control-subtracted width divided by its deterministic
value; Gamma* is where survival crosses 1/2.
"""

from __future__ import annotations

import json
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
    normalize_cpr, run_intrinsic, run_langevin, kernel_width,
)
from cpr_data import load_cpr  # noqa: E402

RESULTS = os.path.join(_HERE, "..", "results")
FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
OMEGA = 5.0
SIGMA_K = 0.025
HALF_WINDOW = 0.5
D_I = 0.005
N_AVG = 300


def _centre(c, i_ac, target, gamma=0.0):
    coarse = np.linspace(0.02 * OMEGA, 1.5 * OMEGA, 140)
    v = run_intrinsic(coarse, i_ac, c, OMEGA, n_transient=150,
                      n_avg=N_AVG // 3, spp=60)
    return coarse[int(np.argmin(np.abs(v - target)))]


def det_width(c, ctrl, i_ac, target):
    """Deterministic control-subtracted width at one drive."""
    out = []
    for cc in (c, ctrl):
        ctr = _centre(cc, i_ac, target)
        fine = ctr + np.arange(-HALF_WINDOW, HALF_WINDOW + D_I, D_I)
        v = run_intrinsic(fine, i_ac, cc, OMEGA, n_transient=150,
                          n_avg=N_AVG, spp=60)
        out.append(kernel_width(fine, v, target, SIGMA_K))
    return out[0] - out[1]


def noisy_width(c, i_ac, target, gamma, n_real, seed):
    """Ensemble-mean kernel width under thermal noise, one CPR."""
    ctr = _centre(c, i_ac, target)
    fine = ctr + np.arange(-HALF_WINDOW, HALF_WINDOW + D_I, D_I)
    idc = np.tile(fine, n_real)
    v = run_langevin(idc, i_ac, c, OMEGA, gamma, n_transient=150,
                     n_avg=N_AVG, spp=100, seed=seed).reshape(n_real, len(fine))
    w = np.array([kernel_width(fine, v[k], target, SIGMA_K) for k in range(n_real)])
    return float(w.mean()), float(w.std(ddof=1) / np.sqrt(n_real))


def find_alpha_opt(c, ctrl, target, alphas):
    best = (None, -1.0)
    for a in alphas:
        w = det_width(c, ctrl, a * OMEGA, target)
        if w > best[1]:
            best = (float(a), w)
        print(f"    alpha={a:5.3f}  det net={w:.5f}", flush=True)
    return best


def gamma_star(gammas, survivals):
    """Linear interpolation in log-Gamma of the 50% survival crossing."""
    g, s = np.array(gammas, float), np.array(survivals, float)
    for k in range(len(s) - 1):
        if (s[k] - 0.5) * (s[k + 1] - 0.5) <= 0 and s[k] != s[k + 1]:
            t = (0.5 - s[k]) / (s[k + 1] - s[k])
            return float(np.exp(np.log(g[k]) + t * (np.log(g[k + 1]) - np.log(g[k]))))
    return None


def sweep(c, ctrl, target, alpha, gammas, n_sig, n_ctl, label, seed0):
    i_ac = alpha * OMEGA
    det = det_width(c, ctrl, i_ac, target)
    print(f"  {label}: alpha={alpha:.3f}, deterministic net = {det:.5f}")
    rows = []
    for j, g in enumerate(gammas):
        ws, es = noisy_width(c, i_ac, target, g, n_sig, seed0 + j)
        wc, ec = noisy_width(ctrl, i_ac, target, g, n_ctl, seed0 + 500 + j)
        net = ws - wc
        surv = net / det if det > 0 else float("nan")
        sem = float(np.hypot(es, ec) / det) if det > 0 else float("nan")
        rows.append({"gamma": float(g), "signal": ws, "control": wc,
                     "net": net, "survival": surv, "survival_sem": sem})
        print(f"    Gamma={g:.5f}  sig={ws:.5f} ctl={wc:.5f} net={net:.5f}  "
              f"survival={surv:.3f} +/- {sem:.3f}", flush=True)
    gs = gamma_star([r["gamma"] for r in rows], [r["survival"] for r in rows])
    print(f"  -> Gamma*({label}) = {gs}")
    return {"alpha": alpha, "det_net": det, "rows": rows, "gamma_star": gs}


def main() -> None:
    c = load_cpr("0.8")
    ctrl = normalize_cpr([1.0])
    print(f"Beenakker tau=0.8: c2={c[1]:.6f}  c4={c[3]:.6f}  "
          f"|c4/c2|={abs(c[3]/c[1]):.4f}\n")

    print("locating drive optima (deterministic):")
    print("  half step (v = Omega/2):")
    a2, w2 = find_alpha_opt(c, ctrl, 0.5 * OMEGA, np.arange(0.5, 1.35, 0.1))
    print("  quarter step (v = Omega/4):")
    a4, w4 = find_alpha_opt(c, ctrl, 0.25 * OMEGA, np.arange(0.3, 1.25, 0.1))
    print(f"\n  half:    alpha_opt={a2:.2f}  net={w2:.5f}")
    print(f"  quarter: alpha_opt={a4:.2f}  net={w4:.5f}\n")

    print("thermal sweeps:")
    g2 = [0.004, 0.007, 0.010, 0.014, 0.020, 0.028, 0.040, 0.056]
    half = sweep(c, ctrl, 0.5 * OMEGA, a2, g2, 60, 40, "half", 202609)
    g4 = [0.0006, 0.0010, 0.0016, 0.0024, 0.0034, 0.0045, 0.0064, 0.0090]
    quarter = sweep(c, ctrl, 0.25 * OMEGA, a4, g4, 60, 40, "quarter", 302609)

    out = {"omega": OMEGA, "cpr": c.tolist(), "half": half, "quarter": quarter}
    if half["gamma_star"] and quarter["gamma_star"]:
        gs2, gs4 = half["gamma_star"], quarter["gamma_star"]
        pred_duprat = gs2 / 4.0
        pred_barrier = gs2 * abs(c[3] / c[1])
        out["predictions"] = {"gamma_star_half_measured": gs2,
                              "gamma_star_quarter_measured": gs4,
                              "duprat_k2T": pred_duprat,
                              "barrier_ck_EJ": pred_barrier,
                              "ratio_measured": gs4 / gs2,
                              "ratio_duprat": 0.25,
                              "ratio_barrier": abs(c[3] / c[1])}
        print("\n  ADJUDICATION")
        print(f"    measured  Gamma*_2 = {gs2:.5f}")
        print(f"    measured  Gamma*_4 = {gs4:.5f}   (ratio {gs4/gs2:.4f})")
        print(f"    Duprat  k^2 T  predicts Gamma*_4 = {pred_duprat:.5f}  (ratio 0.2500)")
        print(f"    barrier |c_k|  predicts Gamma*_4 = {pred_barrier:.5f}  (ratio {abs(c[3]/c[1]):.4f})")
        d_dup = abs(gs4 - pred_duprat) / pred_duprat
        d_bar = abs(gs4 - pred_barrier) / pred_barrier
        print(f"    relative miss: Duprat {d_dup:.1%}, barrier {d_bar:.1%}")
        out["predictions"]["favours"] = "duprat" if d_dup < d_bar else "barrier"
        print(f"    -> favours: {out['predictions']['favours']}")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, res, name in ((axes[0], half, "half"), (axes[1], quarter, "quarter")):
        g = [r["gamma"] for r in res["rows"]]
        s = [r["survival"] for r in res["rows"]]
        e = [r["survival_sem"] for r in res["rows"]]
        ax.errorbar(g, s, yerr=e, fmt="o-", ms=4, capsize=2)
        ax.axhline(0.5, color="0.5", ls=":")
        if res["gamma_star"]:
            ax.axvline(res["gamma_star"], color="tab:blue", ls="--", lw=1.2,
                       label=rf"$\Gamma^*$={res['gamma_star']:.4f}")
        ax.set_xscale("log"); ax.set_xlabel(r"$\Gamma = k_BT/E_J$")
        ax.set_ylabel("survival"); ax.set_title(f"{name}-integer step")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    if "predictions" in out:
        p = out["predictions"]
        axes[1].axvline(p["duprat_k2T"], color="tab:red", ls="-.", lw=1.2,
                        label=rf"Duprat $k^2T$: {p['duprat_k2T']:.4f}")
        axes[1].axvline(p["barrier_ck_EJ"], color="tab:green", ls="-.", lw=1.2,
                        label=rf"barrier $|c_k|$: {p['barrier_ck_EJ']:.4f}")
        axes[1].legend(fontsize=8)
    fig.suptitle("Two published thermal-scaling laws, decided at the quarter step",
                 fontsize=10)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.87, bottom=0.14, wspace=0.24)
    path = os.path.join(FIGURES, "e3_thermal_laws.png")
    fig.savefig(path, dpi=140); plt.close(fig)
    with open(os.path.join(RESULTS, "thermal_laws.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path} and thermal_laws.json")


if __name__ == "__main__":
    main()
