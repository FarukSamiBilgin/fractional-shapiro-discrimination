# Adversarial referee pass

Before the manuscript in `paper/` reached its current form, an independent
adversarial referee was given the code and every data artifact, instructed to
audit each number against the JSON files and to hunt for overclaims. The
findings are reproduced below at full severity, including where they overturned
the draft's headline claims. **The manuscript in `paper/` is the revised
version**: every item below was either corrected in the text or is stated there
as an open limitation.

Notes for the reader of this public repository:

- Section numbering ("S-1", "R-1", ...) is the docket's own. "The paper" refers
  to the *pre-referee draft*; tables and numbers quoted as "published" are the
  draft's, not the current manuscript's.
- References to `NOTES.md` and `research/literature/...` point to the author's
  private working notes (unreviewed literature maps and lab-notebook entries),
  which are deliberately not part of this repository; the manuscript cites the
  primary sources directly.
- Original date of the referee pass: 2026-09-12.

## 1. `shapiro/` — the "parameter-free test" does not survive a wider scan

**S-1 (CONFIRMED by my own rerun). The drive-node test is β_L-conditional, and
the paper scanned too narrow a range.** Table I scans β_L ∈ {0.5, 1, 2} and
reports the loop missing α = 1.9159 by 10–20%. Extending downward:

| β_L | 0.2 | 0.5 | 1 | 2 |
|---|---|---|---|---|
| node α | **2.00** | 2.10 | 2.20 | 2.30 |
| miss | **+4.4%** | +9.6% | +14.8% | +20.0% |

At β_L = 0.2 the miss is 4.4% and the "node" is a shallow modulation — depth
0.00197 against a scan maximum of 0.00426, i.e. barely a node to locate. The
loop's node runs toward the intrinsic one as β_L falls.

This is not a surprise in hindsight: `research/literature/shapiro-intrinsic-extrinsic.md`
line 552 records **Mori & Maekawa (2021)**'s closed form for this exact loop,
giving the half-step amplitude as 2β[α/(1+α)]² J_{2q}(2b) — so to first order in
β the loop's node sits at 1.9159 too. The paper neither cites it nor reconciles
with it.

**And it is the same mistake twice.** `NOTES.md` §"Sonuç 1" records that the
drive-*frequency* test was killed by extending β_L from [0.5, 10] to 30 and 100,
and says so explicitly: *"Aralığı genişletmemiş olsam yanlış bir test önermiş
olacaktım."* [Turkish: "Had I not widened the range, I would have proposed a wrong test."] The node test was then built on a 4× range instead of a 20× one.

*Disposition: the abstract's "no fitted parameter" and the title's framing come
out. What survives is a one-sided, β_L-conditional statement, and it needs the
full node(β_L) curve plus the Mori–Maekawa reconciliation.*

**S-2. The thermal control subtraction does not cancel at the quarter step.**
The net quarter-step width is a 9% difference of two ~0.05 numbers. The two CPRs
have different ohmic slopes at the transit (|dv/di| = 1.378 signal vs 1.282
control), so the estimator's analytic floor differs between them by −0.0034 —
**73% of the net signal** — and the mismatch is near-constant in k while the net
falls 30×, so it biases the k-dependence directly. Restoring a fraction of it:

| fraction restored | Γ*₄/Γ*₂ | exponent |
|---|---|---|
| 0 (as published) | 0.1278 | −2.96 |
| 0.25 | 0.1642 | −2.61 |
| 0.50 | 0.1985 | −2.35 |
| 1.00 | > 0.252 | Duprat consistent |

So "−2 is outside any plausible allowance" is not supported: the one systematic
the paper asserts cancels reaches −2 on its own. *Disposition: measure the floor
(evaluate both CPRs at a drive where the k-step is known absent) and rerun. The
thermal verdict is suspended until then.*

**S-3. "Two published laws" — one of them is mine and is uncited.** Γ*_k ∝ |c_k|
has no citation anywhere in the paper. `NOTES.md` attributes it to the author's
own unpublished MATBG manuscript. Half the title is false as written.

**S-4. "The integer step is maximal at f = 0" is false in all nine runs.**
`flux_sweep.json` argmax of w_one sits at f = 0.15, 0.30, 0.90, 0.75, 0.05, 0.15,
0.80, 0.80, 0.10 — never at 0. What is true in all nine: w_one is *minimal* near
f = 0.50–0.60. The Moshe–Mints contrast survives; the sentence does not.

**S-5. Section III uses the estimator the paper disowns two pages earlier.** The
+56%/+102%/+31%/−61%/+647% all come from `verify.json`, produced by the
grid-counting `step_width`, not the kernel estimator §II C says "all widths
reported here" use. The values are visibly rational multiples of the bias step
(3/7, 9/16, 1/18, 1/42).

**S-6. The acknowledgment is false.** Table I and Fig. 1 have no generating
script in `sim/`; `node_positions.json` and `loop_node_extension.json` were
committed without a producer.

**S-7. Statistics.** χ² = 4.17 on 1 dof (p = 0.041) is quoted next to an unscaled
parameter error. With PDG scaling by √(χ²/dof) = 2.04 the exponent error becomes
±0.089, and k⁻³ is 1.6σ (not 3.4), k⁻² is 9.6σ (not 19.5). Separately the quoted
"σ" are percentile-band quotients from a 20 000-draw MC, which resolves nothing
past ~4σ — "19.4σ" is an extrapolation into a region never sampled.

**S-8. Initial-condition null misstated.** `multistability.json` has two nonzero
spreads, not one: Ω=5, β_L=1 has spread 0.010 on mean 0.095 — a 10% dependence on
initial phase, which is the hysteresis signature the section says is absent.

Also: Frolov's own existence threshold β_L > 2α/(1−α²) = 0.659 at α = 0.3 is never
stated, and Table I's β_L = 0.5 row sits below it. The "600×" R₂₃ swing is a grid
artifact (the quantity has an exact zero; no grid gives 600×).

---

## Disposition

Nothing here is being filed as "reviewer disagreement". Every item above was
checked against the artifacts, and the ones that overturn a headline were
re-derived independently before being accepted. The four documents as committed
overstate their results; the revision has to start from that.
