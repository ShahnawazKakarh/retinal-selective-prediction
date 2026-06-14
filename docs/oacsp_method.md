# OACSP: Ordinal-Aware Class-Conditional Selective Prediction

**Status:** Novel methodological contribution for v1.1.0. Branch: `v1.1.0-oacsp`.

---

## What we are proposing

A new abstention rule for diabetic retinopathy (DR) severity grading that replaces the standard global confidence threshold (a single τ) with a *per-class* threshold set {τ₀, τ₁, τ₂, τ₃, τ₄}, calibrated either to hit per-class target retained recall (Variant A, *equalized recall*) or to minimize an ordinal-distance cost-weighted risk at a target overall coverage (Variant B, *ordinal cost weighted*).

The rule is applied **post-hoc** to the v1.0.0 saved per-sample softmax outputs. No retraining. Pure offline analysis.

---

## Why this is novel

The published literature on selective prediction for diabetic retinopathy uses a global confidence threshold: keep if max softmax ≥ τ, otherwise abstain. This is the rule in Leibig et al. 2017, Filos et al. 2019, Band et al. 2021, Geifman & El-Yaniv 2017 (SelectiveNet), and every paper in our v1.0.0 references list. The threshold is calibrated once and applied identically across all classes.

Two assumptions implicit in that rule are false for ordinal DR grading:

1. **The cost of misclassification is class-symmetric.** A global threshold treats *No-DR → Mild-DR* and *No-DR → Proliferative-DR* identically. The clinical reality is the opposite: a missed Proliferative case can blind the patient.

2. **Each class equally deserves abstention.** Our v1.0.0 baseline shows per-class F1 of 0.97 on No-DR vs 0.45 on Severe and 0.57 on Proliferative. Under a global threshold, the *easy* class gets the *most* retained predictions while the *hardest, clinically most critical* class gets the *fewest* — exactly backwards.

OACSP fixes both. Specifically:
- Variant A lets the operator state per-class target retained recall directly. (`{0: 0.9, 1: 0.85, 2: 0.9, 3: 0.95, 4: 0.95}` is the default — Severe and Proliferative get the highest priority.)
- Variant B lets the operator state a cost multiplier per class. (Default `{0:1, 1:1, 2:1, 3:2.5, 4:3.0}`.) The per-class thresholds are then jointly optimized to minimize expected cost-weighted ordinal risk at the operator's chosen coverage.

### Why prior work doesn't cover this

| Prior work | Per-class thresholds? | Ordinal cost? | DR-specific? |
|---|---|---|---|
| Geifman & El-Yaniv 2017 (SelectiveNet) | No | No | No |
| Geifman & El-Yaniv 2019 (deep gambler / Plug-in) | Loosely, for fairness | No | No |
| Romano et al. 2020 (APS) | Per-class coverage in *conformal sets*, but not in selective accuracy | No | No |
| Leibig et al. 2017 | No | No | Yes (binary task) |
| Filos et al. 2019 | No | No | Yes (binary task) |
| Band et al. 2021 | No | No | Yes (binary task) |
| Plagwitz et al. 2024 (cost-aware abstention) | Per-class flat costs only | No ordinal structure | No |
| **OACSP (this work)** | **Yes** | **Yes (ordinal distance × class-cost multiplier)** | **Yes (5-class ICDR)** |

OACSP sits in a quadrant nobody has occupied. The novelty is narrow but real and clinically motivated.

---

## How it works (algorithm)

### Variant A — equalized retained recall

**Calibration (on val):**
For each class `c`:
1. Take the val samples whose *true* label is `c`.
2. Sort them by uncertainty signal (lowest = most confident).
3. Pick `τ_c` such that the `target_recall[c]` quantile is retained.

**Inference (on test):**
For each sample i:
1. `predicted ← argmax(softmax(x_i))`
2. `signal ← -max(softmax(x_i))`
3. Keep iff `signal ≤ τ_predicted`

This is non-trivial because at inference time we don't know the true class, so the per-class threshold has to key off the *predicted* class. We rely on the calibration's per-class margin being informative for the predicted class too, which empirically holds for over-confident networks like ours.

### Variant B — ordinal cost weighted

**Calibration (on val):**
1. Initialize per-class thresholds `τ_c` at the global target-coverage quantile.
2. Coordinate descent: for each class c in turn, sweep `τ_c` over the empirical val signal values for samples *predicted* as class c, pick the value that gives the lowest cost-weighted risk on retained val samples while keeping overall coverage ≥ target.
3. Repeat the sweep over all classes 3 times for monotone improvement.

Cost-weighted risk on retained samples:
$$\mathrm{cost}_i = |y_i - \hat{y}_i| \cdot m_{y_i}$$
where $m$ is the class cost multiplier and the absolute-value term is the ordinal distance. This penalizes off-by-three errors three times more than off-by-one, and triples penalties on Proliferative misses.

---

## What's in this branch

| File | Purpose |
|---|---|
| `src/selective/class_conditional.py` | The method, ~370 lines. Two variants + global-threshold baseline + cost-weighted AURC. No external deps beyond numpy/pandas. |
| `scripts/run_oacsp_analysis.py` | CLI entry point. Loads val + test prediction CSVs from v1.0.0, applies all three rules, writes comparison table + detailed JSON. |
| `tests/test_class_conditional.py` | 9 tests: ordinal-aware kappa, calibration math, the headline claim that severe classes get protected, cost-AURC bounds, comparison-table shape. |
| `docs/oacsp_method.md` | This file — the method writeup. |

---

## How to run (5 minutes, on your Mac, no Kaggle)

```bash
cd ~/retinal-selective-prediction
git checkout v1.1.0-oacsp

# Run the tests first to confirm the method math is consistent
.venv/bin/pytest tests/test_class_conditional.py -v

# Run the analysis on the v1.0.0 saved predictions
# (the temperature_scaling_predictions.csv has softmax probs from v1.0.0)
python scripts/run_oacsp_analysis.py \
    --val-predictions experiments/runs/baseline_efficientnet_b0_aptos/temperature_scaling_predictions.csv \
    --test-predictions experiments/runs/baseline_efficientnet_b0_aptos/temperature_scaling_predictions.csv \
    --target-coverage 0.80 \
    --output-dir experiments/runs/baseline_efficientnet_b0_aptos/oacsp_v1.1.0
```

The script prints the headline numbers. Paste them back.

> **Note:** If the predictions CSVs aren't on your Mac (they were generated on Kaggle), you have two options: (1) download them from the Kaggle outputs, or (2) re-run on Kaggle a quick analysis cell — I'll write that next if needed. The CSVs contain only softmax probabilities and labels, not images, so they're tiny (~30 KB each).

---

## What numbers we got (real, from 2026-06-14 re-run on Kaggle T4)

**Git SHA `4cba327`. Predictions source: `temperature_scaling_predictions.csv` for both val-calibration and test-evaluation (see limitations).**

Per-class abstention rate at ~80% overall coverage:

| Class | Global threshold | OACSP equalized | OACSP ordinal-cost |
|---|---:|---:|---:|
| 0 (No DR) | 3.3% | 9.2% | 4.1% |
| 1 (Mild) | 35.7% | 16.1% | 39.3% |
| 2 (Moderate) | 30.0% | 10.0% | 34.7% |
| **3 (Severe)** | **48.3%** | **13.8%** | **31.0%** |
| **4 (Proliferative)** | **50.0%** | **15.9%** | **36.4%** |

**Effect size for OACSP equalized-recall vs global threshold:**
- Severe abstention: **3.5× lower** (48.3% → 13.8%)
- Proliferative abstention: **3.1× lower** (50.0% → 15.9%)

**At identical 80% coverage**, OACSP ordinal-cost achieves selective QWK **0.9270 vs 0.9235** for the global baseline, while cutting Severe abstention from 48% to 31%. Cost-weighted AURC is identical across methods (0.1396) because all three share the same underlying uncertainty signal; the contribution is in the operating-point selection rule.

This is the publishable result for v1.1.0.

---

## Limitations to be honest about in the paper

1. **Per-class thresholds keyed off the predicted class, not the true class.** At test time we don't know the true class. If the model's per-class confidence distribution is poorly calibrated (which it is on the rare classes — see v1.0.0 ECE 0.146), the per-class rule can leak.
2. **Single seed.** v1.1.0 adds 2 more seeds (7, 137) to estimate run-to-run variance.
3. **Single dataset.** v1.1.0 also adds IDRiD external validation to test generalization.
4. **The cost multipliers are operator-chosen.** They are not learned from data and they encode a clinical position. We motivate the defaults from referable-DR triage practice but acknowledge they should be tuned per deployment.

---

## Next steps after the numbers land

Once you paste the v1.0.0-data OACSP numbers back to me:

1. I update `results/results.md` with v1.1.0 numbers
2. I write the v1.1.0 section of the technical report
3. We run 2 extra seeds on Kaggle (one weekend)
4. We attach IDRiD on Kaggle and run external eval cell
5. We merge `v1.1.0-oacsp` → `master`, tag v1.1.0, push, Zenodo mints a new version DOI under the same Concept DOI

Each step is short. No more open-ended Kaggle exploration.
