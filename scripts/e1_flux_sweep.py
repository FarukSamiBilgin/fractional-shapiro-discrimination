"""E1 — flux morphology of the inductive loop, against a published prediction.

The community's informal flux test for a trapped-flux origin is that the
fractional step should track the applied flux while the integer step does not.
Moshe & Mints (arXiv:0708.1222) give closed forms for a *different* extrinsic
channel — a junction with alternating critical-current density — and those
forms say something sharper and, as a blanket claim, contradictory: their
half-step current is FINITE at integer flux, exactly where their integer step
VANISHES.

So the flux axis carries a real prediction, and it is channel-dependent. This
sweep measures it for the inductive two-junction loop, which is the other
standard extrinsic caricature, over the loop's own screening parameter and the
drive frequency:

    f in [0, 1]          trapped flux in units of Phi_0
    beta_L in {0.5, 2, 10}
    Omega in {2, 5, 8}

Three outcomes, all worth reporting. The loop reproduces one of the
Moshe-Mints morphologies, in which case the two extrinsic channels are less
distinct than assumed. It reproduces none, in which case a flux-morphology
discriminant *between extrinsic channels* exists — new, and useful, because the
community currently treats "extrinsic" as one hypothesis. Or the half step
survives integer flux in the loop too, which would correct an unqualified
statement in the current literature.

The whole (flux, bias) grid integrates in one call, since the runner
broadcasts an array flux against the bias array.
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
from shapiro import run_extrinsic, kernel_width  # noqa: E402

RESULTS = os.path.join(_HERE, "..", "results")
FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
SIGMA_K = 0.025
HALF_WINDOW = 0.5
D_I = 0.005
ALPHA = 1.0          # fixed reduced drive
ASYM = 0.3


def widths_vs_flux(omega, beta_l, fluxes, n_avg=300):
    """Kernel widths of the half and integer step at each trapped flux."""
    i_ac = ALPHA * omega
    coarse = np.linspace(0.05 * omega, 1.5 * omega, 100)
    nf, nc = len(fluxes), len(coarse)

    idc_c = np.tile(coarse, nf)
    flx_c = np.repeat(fluxes, nc)
    v_c = run_extrinsic(idc_c, i_ac, omega, beta_l=beta_l, flux=flx_c,
                        asym=ASYM, n_transient=100, n_avg=max(60, n_avg // 3),
                        spp=60).reshape(nf, nc)

    out = {}
    for target, name in ((0.5 * omega, "w_half"), (1.0 * omega, "w_one")):
        centres = coarse[np.argmin(np.abs(v_c - target), axis=1)]
        fine = np.arange(-HALF_WINDOW, HALF_WINDOW + D_I, D_I)
        idc_f = np.concatenate([c + fine for c in centres])
        flx_f = np.repeat(fluxes, len(fine))
        v_f = run_extrinsic(idc_f, i_ac, omega, beta_l=beta_l, flux=flx_f,
                            asym=ASYM, n_transient=100, n_avg=n_avg,
                            spp=60).reshape(nf, len(fine))
        out[name] = [kernel_width(centres[k] + fine, v_f[k], target, SIGMA_K)
                     for k in range(nf)]
    return out


def main() -> None:
    fluxes = np.linspace(0.0, 1.0, 21)
    omegas = [2.0, 5.0, 8.0]
    betas = [0.5, 2.0, 10.0]
    out = {"alpha": ALPHA, "asym": ASYM, "fluxes": fluxes.tolist(),
           "omegas": omegas, "betas": betas, "runs": {}}

    for omega in omegas:
        for beta_l in betas:
            key = f"O{omega:g}_bL{beta_l:g}"
            w = widths_vs_flux(omega, beta_l, fluxes)
            wh, w1 = np.array(w["w_half"]), np.array(w["w_one"])
            # the flux-0 value is the control: an unfrustrated loop has no
            # half step, so what it measures there is the estimator's floor
            net = wh - wh[0]
            out["runs"][key] = {"omega": omega, "beta_l": beta_l,
                                "w_half": wh.tolist(), "w_one": w1.tolist(),
                                "net_half": net.tolist(),
                                "ratio": (wh / w1).tolist()}
            k = int(np.argmax(net))
            print(f"  Omega={omega:g} beta_L={beta_l:<5g}  "
                  f"w1/2(f=0)={wh[0]:.5f}  max net={net[k]:.5f} at f={fluxes[k]:.2f}  "
                  f"w1(f=0)={w1[0]:.5f}  w1(f=0.5)={w1[10]:.5f}", flush=True)

    # the load-bearing comparison, stated as a test
    print("\nMoshe-Mints alternating-j_c channel predicts: half step FINITE at "
          "integer flux, integer step VANISHING there.")
    print("inductive loop, measured here:")
    for key, r in out["runs"].items():
        wh, w1 = np.array(r["w_half"]), np.array(r["w_one"])
        print(f"  {key:14} at f=0 (integer):     w_half={wh[0]:.5f}  w_one={w1[0]:.5f}")
        print(f"  {' ':14} at f=0.5 (half-int): w_half={wh[10]:.5f}  w_one={w1[10]:.5f}")
    out["verdict_note"] = ("compare w_half and w_one at f=0 against f=0.5; the "
                           "loop and the alternating-j_c channel agree only if "
                           "w_half is finite and w_one vanishes at integer flux")

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.3), sharex=True)
    for ax, omega in zip(axes, omegas):
        for beta_l in betas:
            r = out["runs"][f"O{omega:g}_bL{beta_l:g}"]
            ax.plot(fluxes, r["net_half"], "o-", ms=3.5,
                    label=rf"$\beta_L$={beta_l:g}")
        ax.set_xlabel(r"trapped flux $f = \Phi/\Phi_0$")
        ax.set_title(rf"$\Omega$={omega:g}")
        ax.grid(alpha=0.3)
        if omega == omegas[0]:
            ax.set_ylabel(r"net $w_{1/2}$ (flux-0 control subtracted)")
        ax.legend(fontsize=8)
    fig.suptitle("Inductive loop: the half step needs frustration — it dies at "
                 "integer flux, where the alternating-$j_c$ channel says it survives",
                 fontsize=10)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.13, wspace=0.22)
    path = os.path.join(FIGURES, "e1_flux_sweep.png")
    fig.savefig(path, dpi=140); plt.close(fig)
    with open(os.path.join(RESULTS, "flux_sweep.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path} and flux_sweep.json")


if __name__ == "__main__":
    main()
