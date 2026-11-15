"""E0 — is the two-order ratio R23 drive-calibration-insensitive, or is it not?

The MATBG manuscript introduces R23 = W_net(Omega/3) / W_net(Omega/2) as "a
second, drive-calibration-insensitive ratio fixed by the harmonic content
alone", reports 0.154-0.166 across fillings, and says it measured it at a drive
optimum alpha ~ 0.6-0.7.

Duprat & Levy Yeyati (Phys. Rev. B 71, 054510 (2005), Eq. 11) give, in the
voltage-biased limit, the order-n/k step amplitude proportional to
|c_k J_1(2 k alpha)|, so

    R23(alpha) = |c3 J_1(6 alpha)| / |c2 J_1(4 alpha)|

which is a function of the harmonic content AND the reduced drive — not of the
harmonic content alone. Evaluated at the manuscript's own Beenakker tau = 0.8
harmonics it swings by a factor ~600 over alpha in [0.3, 0.8] and passes through
an exact zero at alpha = j_11/6 = 0.6386, inside the quoted drive window.

The manuscript's simulation is current-biased at finite Omega, where that
identity need not hold. So this script settles it the only way that counts: it
scans R23(alpha) on a current-biased RSJ solver with the manuscript's own CPR
and overlays the closed form. Three possible outcomes, all worth reporting:

  * the numerics track the closed form, node and all -> the manuscript's
    "drive-calibration-insensitive" claim is wrong and R23 must be quoted with
    its alpha;
  * the numerics are flat where the closed form has a node -> current bias
    washes the node out, which is a real and publishable difference between the
    two bias limits;
  * the numerics are flat at a different alpha -> the two alpha conventions
    differ and the manuscript's operating point needs restating.
"""

from __future__ import annotations

import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.special import jv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from shapiro import normalize_cpr, run_intrinsic, kernel_width  # noqa: E402
from cpr_data import load_cpr, OMEGA_0  # noqa: E402

RESULTS = os.path.join(_HERE, "..", "results")
FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
SIGMA_K = 0.025
HALF_WINDOW = 0.5
D_I = 0.005


def net_width(cpr, ctrl, omega, alpha, v_target, n_avg, gamma=0.0):
    """Control-subtracted kernel width of the step at v_target, one drive."""
    i_ac = alpha * omega
    coarse = np.linspace(0.05 * omega, 1.4 * omega, 120)
    out = []
    for c in (cpr, ctrl):
        v = run_intrinsic(coarse, i_ac, c, omega, gamma=gamma,
                          n_transient=150, n_avg=max(60, n_avg // 3), spp=60)
        centre = coarse[int(np.argmin(np.abs(v - v_target)))]
        fine = centre + np.arange(-HALF_WINDOW, HALF_WINDOW + D_I, D_I)
        vf = run_intrinsic(fine, i_ac, c, omega, gamma=gamma,
                           n_transient=150, n_avg=n_avg, spp=60)
        out.append(kernel_width(fine, vf, v_target, SIGMA_K))
    return out[0] - out[1], out[0], out[1]


def main() -> None:
    cpr = load_cpr("0.8")
    ctrl = normalize_cpr([1.0])
    c1, c2, c3 = cpr[0], cpr[1], cpr[2]
    omega = OMEGA_0
    n_avg = 300
    alphas = np.arange(0.20, 1.01, 0.025)

    print(f"Beenakker tau=0.8: c1={c1:.5f} c2={c2:.5f} c3={c3:.5f}")
    print(f"Omega={omega}, current-biased RSJ, kernel estimator, control subtracted\n")
    print(" alpha   net(O/2)   net(O/3)    R23_num    R23_closed")
    rows = []
    for a in alphas:
        n2, s2, k2 = net_width(cpr, ctrl, omega, a, 0.5 * omega, n_avg)
        n3, s3, k3 = net_width(cpr, ctrl, omega, a, omega / 3.0, n_avg)
        r_num = n3 / n2 if n2 > 1e-9 else float("nan")
        r_cl = (abs(c3 * jv(1, 6 * a)) / abs(c2 * jv(1, 4 * a))
                if abs(jv(1, 4 * a)) > 1e-12 else float("nan"))
        rows.append({"alpha": float(a), "net_half": n2, "net_third": n3,
                     "R23_numeric": r_num, "R23_closed_form": float(r_cl),
                     "signal_half": s2, "control_half": k2,
                     "signal_third": s3, "control_third": k3})
        print(f" {a:5.3f}  {n2:9.5f}  {n3:9.5f}  {r_num:9.4f}  {r_cl:9.4f}",
              flush=True)

    out = {"omega": omega, "cpr_tau": "0.8", "c": cpr.tolist(),
           "estimator": "gaussian kernel sigma_k=0.025, sinusoidal control subtracted",
           "bias": "current-biased RSJ (manuscript convention)",
           "closed_form": "|c3 J1(6 alpha)| / |c2 J1(4 alpha)|  (Duprat & Levy Yeyati PRB 71 054510 (2005) Eq. 11, voltage-biased)",
           "manuscript_claim": {"R23": [0.154, 0.166], "alpha_quoted": [0.6, 0.7],
                                "wording": "drive-calibration-insensitive ratio fixed by the harmonic content alone"},
           "rows": rows}

    valid = [r for r in rows if np.isfinite(r["R23_numeric"]) and r["net_half"] > 1e-4]
    if valid:
        rn = [r["R23_numeric"] for r in valid]
        out["numeric_range"] = [min(rn), max(rn)]
        out["numeric_swing"] = max(rn) / max(min(rn), 1e-9)
        in_band = [r["alpha"] for r in valid if 0.154 <= r["R23_numeric"] <= 0.166]
        out["alpha_reproducing_manuscript_band"] = in_band
        print(f"\n  numeric R23 range {min(rn):.4f}-{max(rn):.4f} "
              f"(swing {out['numeric_swing']:.1f}x)")
        print(f"  alpha values reproducing the quoted 0.154-0.166: {in_band}")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    a = np.array([r["alpha"] for r in rows])
    ax = axes[0]
    ax.plot(a, [r["net_half"] for r in rows], "o-", ms=4, label=r"net $w(\Omega/2)$")
    ax.plot(a, [r["net_third"] for r in rows], "s-", ms=4, label=r"net $w(\Omega/3)$")
    ax.set_xlabel(r"$\alpha = i_{ac}/\Omega$")
    ax.set_ylabel("control-subtracted kernel width")
    ax.set_title(r"Half- and third-integer steps vs drive, $\Omega=5$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(a, [r["R23_numeric"] for r in rows], "o-", ms=4,
            label="current-biased RSJ (this work)")
    ax.plot(a, [r["R23_closed_form"] for r in rows], "--", lw=2,
            label=r"$|c_3J_1(6\alpha)|/|c_2J_1(4\alpha)|$ (Duprat, voltage-biased)")
    ax.axhspan(0.154, 0.166, color="tab:green", alpha=0.2,
               label="manuscript's quoted 0.154-0.166")
    ax.axvspan(0.6, 0.7, color="tab:orange", alpha=0.15,
               label=r"manuscript's quoted $\alpha$")
    ax.axvline(3.8317 / 6, color="k", ls=":", lw=1)
    ax.text(3.8317 / 6, ax.get_ylim()[1] * 0.9, r"  $J_1$ node", fontsize=7)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$R_{23}$")
    ax.set_ylim(0, 0.6)
    ax.set_title("Is $R_{23}$ drive-independent?")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIGURES, "e0_r23_alpha.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    with open(os.path.join(RESULTS, "r23_alpha.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path} and r23_alpha.json")


if __name__ == "__main__":
    main()
