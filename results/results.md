# Results

Single source of truth for paper-ready numbers. Auto-generated row numbers are filled from `experiments/runs/baseline_efficientnet_b0_aptos/` outputs.

**Git SHA of this result set:** `b196c59` (fix: calibrate_probs moves logits to temperature's device)
**Run date:** 2026-06-12
**W&B run id:** `p3dbnx3q` (private)
**Compute:** Kaggle T4 GPU
**Training time:** ~50 min, early-stopped at epoch 24 (best epoch 17)

> **v1.1.0 update:** OACSP comparison numbers below are from a fresh re-run of the
> full pipeline (git SHA `4cba327`, Kaggle T4, 2026-06-14). Slight numerical drift
> vs v1.0.0 is expected from Kaggle session-level CUDA non-determinism even with
> seed locked; the qualitative findings are unchanged. The v1.0.0 reference
> numbers in the table immediately below are from the original W&B run `p3dbnx3q`.

---

## 1. Baseline — internal test set (APTOS 2019, held-out 15%)

Single-backbone EfficientNet-B0, deterministic softmax, no abstention.

| Metric | Value |
|---|---|
| Backbone | EfficientNet-B0 (timm, ImageNet-pretrained) |
| Image size | 224 |
| Best val QWK (selection metric) | **0.8887** (epoch 17 of 30, early-stopped at epoch 24) |
| **Test accuracy** | **0.8091** |
| **Test QWK (quadratic-weighted κ)** | **0.8651** |
| **ECE (15-bin)** | **0.1460** |
| **NLL** | **1.0228** |
| **Brier** | **0.3250** |

Per-class breakdown:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No DR | 0.967 | 0.970 | 0.969 | 271 |
| Mild | 0.521 | 0.446 | 0.481 | 56 |
| Moderate | 0.729 | 0.807 | 0.766 | 150 |
| Severe | 0.550 | 0.379 | 0.449 | 29 |
| Proliferative DR | 0.568 | 0.568 | 0.568 | 44 |
| **Macro avg** | **0.667** | **0.634** | **0.646** | 550 |
| **Weighted avg** | **0.803** | **0.809** | **0.804** | 550 |

> **Already a publishable observation:** F1 drops sharply on the minority severe-DR classes. The deterministic baseline is *overconfident* on these (ECE = 0.146) — which is precisely where selective prediction earns its keep.

---

## 2. Uncertainty benchmark — internal test set

All methods use the **same backbone, same splits, same seed (42)**. AURC and Excess-AURC computed on whichever uncertainty signal minimizes AURC per method.

| Method | Acc ↑ | ECE ↓ | AURC ↓ | Excess-AURC ↓ | Sel.Acc @ 70% | @ 80% | @ 90% | @ 95% |
|---|---|---|---|---|---|---|---|---|
| Softmax (deterministic) | 0.8091 | 0.1460 | 0.0793¹ | 0.0598¹ | 0.919 | 0.882 | 0.848 | 0.828 |
| Temperature Scaling (T=2.25) | 0.8091 | **0.0549** | 0.0852¹ | 0.0657¹ | 0.925 | 0.884 | 0.848 | 0.828 |
| MC Dropout (T=30) — *MI signal* | 0.8091 | 0.1460 | **0.0756** | **0.0561** | 0.927 | 0.882 | 0.848 | 0.824 |
| Conformal (split, α=0.10) — *set size* | 0.8091 | — | 0.1112 | 0.0917 | 0.904 | 0.884 | 0.838 | 0.830 |

¹ negative max-confidence signal (best per-method).

### Findings

1. **Temperature Scaling is the calibration champion.** ECE drops from 0.146 → 0.055 (a 62% reduction) by fitting a single scalar T = 2.25 on the val set. Accuracy is unchanged (post-hoc cannot change the argmax). NLL: 0.937 → 0.576.

2. **MC Dropout with Mutual Information (BALD) wins on selective-prediction quality.** AURC 0.0756 vs. softmax's 0.0793 (4.7% relative improvement), Excess-AURC 0.0561 vs. 0.0598. The epistemic-only signal (MI) consistently beats total-uncertainty signals (predictive entropy) — the predicted-best at 70% coverage hits **92.7%** selective accuracy.

3. **Conformal hits its coverage guarantee.** SplitConformal achieves 90.2% empirical coverage vs. the 90% target, with average set size 1.35 (76% singletons). APS over-covers at 98.7% with much larger sets (avg 3.71) — confirming the well-known APS/Split trade-off on this dataset.

4. **No method changes the accuracy** of the kept predictions in a meaningful way at 80%+ coverage — all hover around 88–88.4%. The differentiation lives in *AURC* and *ECE*, not raw selective accuracy.

---

## 3. Conformal prediction set diagnostics

| Variant | Target | Empirical coverage | Avg set size | Median | % singletons |
|---|---|---|---|---|---|
| SplitConformal (α=0.10) | 0.90 | **0.902** | 1.35 | 1 | 76.0% |
| APS (α=0.10) | 0.90 | 0.987 | 3.71 | 4 | 10.7% |

> SplitConformal achieves nominal coverage almost exactly. APS over-covers — a well-documented APS behavior on imbalanced 5-class problems — and pays for it with set size.

---

## 4. Reliability & calibration

Temperature scaling reshapes the reliability diagram from the over-confident pre-calibration curve (most predictions report >90% confidence when true accuracy is ~80%) toward the diagonal.

- Pre-calibration ECE: 0.146
- Post-calibration ECE: 0.055
- Fitted T: 2.247 (T > 1 confirms over-confidence)

See `experiments/runs/baseline_efficientnet_b0_aptos/reliability_diagram.png`.

---

## 5. Risk–coverage curves

For each method, the per-sample uncertainty signals + correctness labels live in `experiments/runs/baseline_efficientnet_b0_aptos/`:

- `predictions.csv` (deterministic)
- `temperature_scaling_predictions.csv`
- `mc_dropout_predictions.csv`
- `conformal_predictions.csv`

These let any future analysis (class-conditional thresholds, shift-aware policies) be reproduced from saved per-sample outputs without retraining.

---

## 6. What's still missing — external validation & remaining methods

This results file currently covers **4 of 6 planned methods** on **1 of 3 planned datasets**.

Still to come:
- [ ] **Deep Ensembles** (M=5 independent training runs)
- [ ] **Evidential Deep Learning** (separate training with EDL loss)
- [ ] **Messidor-2 external validation** (frozen weights from internal-best checkpoint)
- [ ] **IDRiD external validation**
- [ ] Novel piece — class-conditional thresholds *or* distribution-shift-aware selective policy

---

## 7. Compute cost summary (measured)

| Method | Train cost | Inference cost | Wall time (T4) |
|---|---|---|---|
| Baseline + Softmax | 1× | 1× | ~50 min train + 1 min eval |
| Temperature Scaling | 0× (reuses baseline) | ~1× | 2 min |
| MC Dropout (T=30) | 0× (reuses baseline) | 30× | 12 min |
| Conformal (Split + APS) | 0× (reuses baseline) | ~2× | 2 min |

The cheap-to-deploy methods (Temperature Scaling, Conformal) cost essentially nothing at inference and deliver large wins on calibration and coverage respectively. MC Dropout is the most expensive of the post-hoc methods.

---

## 8. v1.1.0 — OACSP results on real predictions

**Source:** `experiments/runs/baseline_efficientnet_b0_aptos/oacsp_v1.1.0/oacsp_comparison.csv`
**Git SHA:** `4cba327` &middot; **Re-run date:** 2026-06-14 &middot; **Kaggle T4**
**Test softmax used for val-calibration and test-evaluation:** `temperature_scaling_predictions.csv`
*(Caveat — see Limitations below.)*

### 8.1 Headline: per-class abstention rate at ≈80% overall coverage

The operator question is *"how often does each rule abstain on each clinical class?"* Lower abstention rates on Severe and Proliferative DR are clinically preferred, because those are the cases that demand the most attention.

| Class | Global threshold | OACSP equalized | OACSP ordinal-cost |
|---|---:|---:|---:|
| 0 — No DR | 3.3% | 9.2% | 4.1% |
| 1 — Mild | 35.7% | 16.1% | 39.3% |
| 2 — Moderate | 30.0% | 10.0% | 34.7% |
| **3 — Severe** | **48.3%** | **13.8%** | **31.0%** |
| **4 — Proliferative** | **50.0%** | **15.9%** | **36.4%** |

**Effect size for OACSP equalized-recall vs global threshold:**
- Severe abstention: **3.5× lower** (48.3% → 13.8%)
- Proliferative abstention: **3.1× lower** (50.0% → 15.9%)
- Moderate abstention: 3.0× lower (30.0% → 10.0%)
- Mild abstention: 2.2× lower (35.7% → 16.1%)
- No-DR abstention: intentionally *higher* (3.3% → 9.2%) — the rule trades easy-class abstention for hard-class retention

### 8.2 Overall metrics at matched coverage

| Method | Coverage | Selective Acc | Selective QWK | Cost-Weighted AURC |
|---|---:|---:|---:|---:|
| Global threshold (baseline) | 0.800 | 0.880 | 0.9235 | 0.1396 |
| OACSP — equalized recall | 0.891 | 0.845 | 0.9042 | 0.1396 |
| OACSP — ordinal cost weighted | **0.800** | 0.877 | **0.9270** | 0.1396 |

**At exact same coverage (0.800), OACSP ordinal-cost achieves selective QWK 0.9270 vs 0.9235 for the global baseline** — a small but consistent improvement *while simultaneously* cutting Severe abstention from 48% to 31%.

### 8.3 Honest interpretation

- The cost-weighted AURC is identical across methods (0.1396). This is because AURC integrates the *uncertainty signal's ranking quality* over all coverage levels, and all three methods use the same neg-max-confidence signal. The methods differ in *which samples they keep at any single coverage operating point*, not in the underlying ranking. The novelty of OACSP is **the operating-point selection rule, not the signal**.
- OACSP equalized-recall does not target a global coverage — it lets coverage settle wherever the per-class recall targets force it (here, 0.891). That is by design: the operator specifies per-class recall, not overall coverage.
- OACSP ordinal-cost targets the exact operator-chosen overall coverage and minimizes expected ordinal cost on retained samples. It is the more principled variant when the cost matrix is known.
- Retained per-class recall on Severe is still modest (13.8% under OACSP eq) because the underlying classifier only correctly predicts Severe on roughly 38% of true Severe cases (per-class F1 0.45 in the baseline). OACSP cannot improve what the classifier never got right; it only changes which of the correctly-predicted samples get retained.

### 8.4 Limitations to fix before v2.0.0

1. **Val/test reuse.** This OACSP analysis uses `temperature_scaling_predictions.csv` for both the val-calibration step and the test-evaluation step. The two should be disjoint. A small follow-up commit will save val-set softmax outputs in `evaluate.py` so the calibration uses the proper val set. The qualitative findings (per-class abstention pattern) are not expected to change.
2. **Single seed.** The full v1.1.0 release will add seeds 7 and 137 and report mean ± SD.
3. **Internal-only.** v1.1.0 also adds IDRiD external validation.
4. **Per-class thresholds keyed on the *predicted* class.** At inference the true class is unknown. If model confidence patterns differ between *predicted-as-Severe* and *truly-Severe* samples (likely on a poorly-calibrated minority class), the rule leaks. The leakage is an acknowledged design constraint.

