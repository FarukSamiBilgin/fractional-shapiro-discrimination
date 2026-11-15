"""E3b — the same adjudication as E3, but with the statistics it needed.

E3 measured Gamma*_4 / Gamma*_2 = 0.130 against predictions of 0.250 (Duprat &
Levy Yeyati's k^2 T collapse) and 0.0897 (the barrier reading, Gamma*_k prop
|c_k|).  Comparing those three numbers as *relative misses on Gamma** made the
result look undecidable: 48.1% from one law, 44.5% from the other.  That
statistic is the wrong one twice over.  It compares a quantity read off a steep
survival curve on a logarithmic axis, and it ignores the error bars, which at
the quarter step were +/- 0.10 in survival -- large enough that the reported
crossing sat 0.3 sigma from 1/2.

This script fixes both.

  * The right test is not "how far is Gamma* from the prediction" but "is the
    measured survival at the predicted Gamma* equal to 1/2?".  Each law names a
    specific Gamma at which the quarter step should be half-dead; we go and
    measure the survival there and ask how many sigma it misses.

  * The error bars come down by pairing.  Signal and control are run with
    identical noise realisations (common random numbers): the same rng seed and
    the same array shape means grid point j of realisation k feels the same
    Wiener increments under the tau = 0.8 CPR and under the sinusoidal control.
    The estimator's transit floor is then largely common to both and cancels in
    the difference.  Measured pair correlation is 0.78, which halves the SEM
    for free; the ensemble is then raised 4x on top of that.

Everything else -- solver, kernel estimator, drive optima, control subtraction
-- is unchanged from E3, so the two runs are directly comparable.
"""

from __future__ import annotations

import json
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
N_WORKERS = 4

# Drive optima carried over from E3 (deterministic, re-verified below).
ALPHA_HALF = 0.90
ALPHA_QUARTER = 0.60

# Ensemble sizes, per Gamma, per CPR.  Split into N_WORKERS chunks.
N_HALF = 96
N_QUARTER = 240


def _grid(centre):
    return centre + np.arange(-HALF_WINDOW, HALF_WINDOW + D_I, D_I)


def _centre(c, i_ac, target):
    coarse = np.linspace(0.02 * OMEGA, 1.5 * OMEGA, 140)
    v = run_intrinsic(coarse, i_ac, c, OMEGA, n_transient=150,
                      n_avg=N_AVG // 3, spp=60)
    return float(coarse[int(np.argmin(np.abs(v - target)))])


def _det_width(c, grid, i_ac, target):
    v = run_intrinsic(grid, i_ac, c, OMEGA, n_transient=150,
                      n_avg=N_AVG, spp=60)
    return float(kernel_width(grid, v, target, SIGMA_K))


def _chunk(task):
    """One worker: n paired realisations at one Gamma.

    Signal and control share `seed` and array shape, so the noise draws are
    identical element by element -- the common-random-number pairing.
    """
    (cs, cc, gs, gc, i_ac, target, gamma, n, seed) = task
    out = []
    for c_arr, grid in ((np.asarray(cs), np.asarray(gs)),
                        (np.asarray(cc), np.asarray(gc))):
        v = run_langevin(np.tile(grid, n), i_ac, c_arr, OMEGA, gamma,
                         n_transient=150, n_avg=N_AVG, spp=100,
                         seed=seed).reshape(n, -1)
        out.append(np.array([kernel_width(grid, v[k], target, SIGMA_K)
                             for k in range(n)]))
    return out[0], out[1]


def sweep(pool, label, c, ctrl, target, alpha, gammas, n_real, seed0):
    i_ac = alpha * OMEGA
    g_sig = _grid(_centre(c, i_ac, target))
    g_ctl = _grid(_centre(ctrl, i_ac, target))
    det = _det_width(c, g_sig, i_ac, target) - _det_width(ctrl, g_ctl, i_ac, target)
    print(f"\n{label}: alpha={alpha:.2f}  deterministic net = {det:.5f}", flush=True)

    per = n_real // N_WORKERS
    rows = []
    for j, g in enumerate(gammas):
        tasks = [(c.tolist(), ctrl.tolist(), g_sig.tolist(), g_ctl.tolist(),
                  i_ac, target, float(g), per, seed0 + 1000 * j + m)
                 for m in range(N_WORKERS)]
        parts = list(pool.map(_chunk, tasks))
        sig = np.concatenate([r[0] for r in parts])
        ctl = np.concatenate([r[1] for r in parts])
        d = sig - ctl
        n = len(d)
        net, sem = float(d.mean()), float(d.std(ddof=1) / np.sqrt(n))
        rows.append({"gamma": float(g), "signal": float(sig.mean()),
                     "control": float(ctl.mean()), "net": net,
                     "net_sem": sem, "n": n,
                     "pair_corr": float(np.corrcoef(sig, ctl)[0, 1]),
                     "survival": net / det, "survival_sem": sem / det})
        print(f"  Gamma={g:.6f}  n={n:3d}  net={net:.5f}+/-{sem:.5f}  "
              f"survival={net/det:.3f}+/-{sem/det:.3f}  rho={rows[-1]['pair_corr']:.2f}",
              flush=True)
    return {"alpha": alpha, "det_net": det, "rows": rows}


def _interp_log(gammas, surv, g):
    """Survival at Gamma=g, linear in log Gamma between bracketing points."""
    lg, ls = np.log(np.asarray(gammas, float)), np.asarray(surv, float)
    return float(np.interp(np.log(g), lg, ls))


def _crossing(gammas, surv):
    lg, s = np.log(np.asarray(gammas, float)), np.asarray(surv, float)
    for k in range(len(s) - 1):
        if (s[k] - 0.5) * (s[k + 1] - 0.5) <= 0 and s[k] != s[k + 1]:
            t = (0.5 - s[k]) / (s[k + 1] - s[k])
            return float(np.exp(lg[k] + t * (lg[k + 1] - lg[k])))
    return None


def _mc(res, n_draw=20000, seed=11):
    """Monte-Carlo band on Gamma* from the survival SEMs."""
    rng = np.random.default_rng(seed)
    g = [r["gamma"] for r in res["rows"]]
    mu = np.array([r["survival"] for r in res["rows"]])
    sd = np.array([r["survival_sem"] for r in res["rows"]])
    draws = []
    for _ in range(n_draw):
        x = _crossing(g, mu + rng.normal(0.0, sd))
        if x is not None:
            draws.append(x)
    if not draws:
        return None
    d = np.array(draws)
    return {"median": float(np.median(d)),
            "lo68": float(np.percentile(d, 16)),
            "hi68": float(np.percentile(d, 84)),
            "lo95": float(np.percentile(d, 2.5)),
            "hi95": float(np.percentile(d, 97.5)),
            "n_ok": len(d)}


def _test_at(res, g_pred, label):
    """Is survival at the predicted Gamma* consistent with 1/2?"""
    g = [r["gamma"] for r in res["rows"]]
    mu = [r["survival"] for r in res["rows"]]
    sd = [r["survival_sem"] for r in res["rows"]]
    if not (min(g) <= g_pred <= max(g)):
        return {"law": label, "gamma_pred": g_pred, "in_range": False}
    s = _interp_log(g, mu, g_pred)
    e = _interp_log(g, sd, g_pred)          # SEMs vary slowly across the grid
    z = (s - 0.5) / e
    return {"law": label, "gamma_pred": float(g_pred), "in_range": True,
            "survival_at_pred": s, "sem_at_pred": e, "z": float(z)}


def main() -> None:
    c = load_cpr("0.8")
    ctrl = normalize_cpr([1.0])
    r42 = abs(c[3] / c[1])
    print(f"Beenakker tau=0.8: c2={c[1]:.6f} c4={c[3]:.6f} |c4/c2|={r42:.4f}")
    print(f"pairing: common random numbers, {N_WORKERS} workers\n", flush=True)

    # Gamma grid for the quarter step brackets BOTH predictions anchored on the
    # E3 value of Gamma*_2 (0.019242): barrier 0.001727, Duprat 0.004810.
    g_half = [0.010, 0.014, 0.017, 0.020, 0.024, 0.028]
    g_quarter = [0.0012, 0.0017269, 0.0020, 0.0024, 0.0028, 0.0034,
                 0.0040, 0.0048105]

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        half = sweep(pool, "half step (v=Omega/2)", c, ctrl, 0.5 * OMEGA,
                     ALPHA_HALF, g_half, N_HALF, 907001)
        quarter = sweep(pool, "quarter step (v=Omega/4)", c, ctrl, 0.25 * OMEGA,
                        ALPHA_QUARTER, g_quarter, N_QUARTER, 908001)

    half["gamma_star"] = _crossing([r["gamma"] for r in half["rows"]],
                                   [r["survival"] for r in half["rows"]])
    half["gamma_star_band"] = _mc(half)
    quarter["gamma_star"] = _crossing([r["gamma"] for r in quarter["rows"]],
                                      [r["survival"] for r in quarter["rows"]])
    quarter["gamma_star_band"] = _mc(quarter)

    out = {"omega": OMEGA, "cpr": c.tolist(), "pairing": "common random numbers",
           "half": half, "quarter": quarter}

    gs2 = half["gamma_star"]
    print("\n  ADJUDICATION")
    print(f"    Gamma*_2 = {gs2:.6f}  68% [{half['gamma_star_band']['lo68']:.6f},"
          f" {half['gamma_star_band']['hi68']:.6f}]")
    if quarter["gamma_star"]:
        b = quarter["gamma_star_band"]
        print(f"    Gamma*_4 = {quarter['gamma_star']:.6f}  68% [{b['lo68']:.6f},"
              f" {b['hi68']:.6f}]")
    if gs2:
        pred = {"duprat": gs2 / 4.0, "barrier": gs2 * r42}
        tests = [_test_at(quarter, pred["duprat"], "Duprat k^2 T"),
                 _test_at(quarter, pred["barrier"], "barrier |c_k|")]
        out["predictions"] = {"gamma_star_half": gs2,
                              "gamma_star_quarter": quarter["gamma_star"],
                              "duprat_k2T": pred["duprat"],
                              "barrier_ck_EJ": pred["barrier"],
                              "ratio_measured": (quarter["gamma_star"] / gs2
                                                 if quarter["gamma_star"] else None),
                              "ratio_duprat": 0.25, "ratio_barrier": r42,
                              "tests": tests}
        print("\n    direct test -- survival measured AT each predicted Gamma*")
        print("    (each law says the quarter step should be half dead there)")
        for t in tests:
            if not t["in_range"]:
                print(f"      {t['law']:16s} Gamma={t['gamma_pred']:.6f}  outside grid")
                continue
            print(f"      {t['law']:16s} Gamma={t['gamma_pred']:.6f}  "
                  f"survival={t['survival_at_pred']:.3f}+/-{t['sem_at_pred']:.3f}  "
                  f"|z|={abs(t['z']):.1f}")
        ok = [t for t in tests if t.get("in_range")]
        if len(ok) == 2:
            a, bb = sorted(ok, key=lambda t: abs(t["z"]))
            out["predictions"]["verdict"] = (
                f"{a['law']} survives at |z|={abs(a['z']):.1f}; "
                f"{bb['law']} rejected at |z|={abs(bb['z']):.1f}"
                if abs(bb["z"]) > 3.0 and abs(a["z"]) < 3.0 else "inconclusive")
            print(f"    -> {out['predictions']['verdict']}")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, res, name in ((axes[0], half, "half"), (axes[1], quarter, "quarter")):
        g = [r["gamma"] for r in res["rows"]]
        s = [r["survival"] for r in res["rows"]]
        e = [r["survival_sem"] for r in res["rows"]]
        ax.errorbar(g, s, yerr=e, fmt="o-", ms=4, capsize=2, color="k")
        ax.axhline(0.5, color="0.5", ls=":")
        if res["gamma_star"]:
            bd = res["gamma_star_band"]
            if bd:
                ax.axvspan(bd["lo68"], bd["hi68"], color="tab:blue", alpha=0.15)
            ax.axvline(res["gamma_star"], color="tab:blue", ls="--", lw=1.2,
                       label=rf"$\Gamma^*$={res['gamma_star']:.4f}")
        ax.set_xscale("log")
        ax.set_xlabel(r"$\Gamma = k_BT/E_J$")
        ax.set_ylabel("survival (control-subtracted, paired)")
        ax.set_title(f"{name}-integer step")
        ax.grid(alpha=0.3, which="both")
    if "predictions" in out:
        p = out["predictions"]
        axes[1].axvline(p["duprat_k2T"], color="tab:red", ls="-.", lw=1.2,
                        label=rf"Duprat $k^2T$: {p['duprat_k2T']:.4f}")
        axes[1].axvline(p["barrier_ck_EJ"], color="tab:green", ls="-.", lw=1.2,
                        label=rf"barrier $|c_k|$: {p['barrier_ck_EJ']:.4f}")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle("Quarter-step thermal law, paired noise and 4x ensemble",
                 fontsize=10)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.87, bottom=0.14, wspace=0.24)
    path = os.path.join(FIGURES, "e3b_thermal_refine.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    with open(os.path.join(RESULTS, "thermal_laws_refined.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path} and thermal_laws_refined.json")


if __name__ == "__main__":
    main()
