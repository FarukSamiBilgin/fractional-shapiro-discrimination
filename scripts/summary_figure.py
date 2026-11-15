"""One figure for the whole result: what the frequency trend can and cannot say."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, "..", "results")
FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")


def main() -> None:
    with open(os.path.join(RESULTS, "verify.json")) as fh:
        v = json.load(fh)

    intr = v["v5"]["rows"]
    ext = {r["beta_l"]: r for r in v["v2"]["rows"]}
    ext.update({r["beta_l"]: r for r in v["v4"]["rows"]})
    betas = sorted(ext)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))

    ax = axes[0]
    lo = min(r["relative_change"] for r in intr) * 100
    hi = max(r["relative_change"] for r in intr) * 100
    ax.axhspan(lo, hi, color="tab:blue", alpha=0.18,
               label=f"intrinsic band, any $c_2/c_1$\n({lo:.0f}% to {hi:.0f}%)")
    ax.axhline(0, color="0.4", lw=1)
    ax.semilogx(betas, [ext[b]["relative_change"] * 100 for b in betas],
                "s-", color="tab:red", label=r"extrinsic, vs $\beta_L$")
    ax.set_xlabel(r"loop screening parameter $\beta_L = 2\pi L I_c/\Phi_0$")
    ax.set_ylabel(r"change in $w_{1/2}/w_1$ from $\Omega$=2 to $\Omega$=8  (%)")
    ax.set_title("The extrinsic trend crosses the intrinsic band")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.plot([-r["c2_over_c1"] for r in intr],
            [r["ratio_omega2"] for r in intr], "o-", label=r"$\Omega=2$")
    ax.plot([-r["c2_over_c1"] for r in intr],
            [r["ratio_omega8"] for r in intr], "o-", label=r"$\Omega=8$")
    for b in (1.0, 4.0, 30.0, 100.0):
        if b in ext:
            ax.axhline(ext[b]["ratio_omega2"], ls=":", lw=1, color="0.6")
            ax.text(0.11, ext[b]["ratio_omega2"], rf" $\beta_L$={b:g}",
                    fontsize=7, color="0.4", va="bottom")
    ax.set_xlabel(r"harmonic content $|c_2/c_1|$")
    ax.set_ylabel(r"$w_{1/2}/w_1$")
    ax.set_title("What harmonic content a given ratio implies")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Half-integer Shapiro steps: intrinsic CPR harmonic vs "
                 "trapped-flux loop", fontsize=10)
    fig.tight_layout()
    path = os.path.join(FIGURES, "d2_trend_vs_beta.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")
    print(f"  intrinsic band: {lo:+.1f}% to {hi:+.1f}%")
    print("  extrinsic:", {b: f"{ext[b]['relative_change']*100:+.0f}%"
                           for b in betas})


if __name__ == "__main__":
    main()
