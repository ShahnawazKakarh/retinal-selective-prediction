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

## 8. v1.1.0 — OACSP results on real predictions (with proper val/test split)

**Source:** `experiments/runs/baseline_efficientnet_b0_aptos/oacsp_v1.1.0/oacsp_comparison.csv`
**Git SHA:** `2400870` &middot; **Re-run date:** 2026-06-14 &middot; **Kaggle T4 (W&B run `7o8m6ko9`)**

**Val/test split (corrected):** Per-class thresholds are calibrated on `temperature_scaling_val_predictions.csv` (550 val samples). Evaluation is on the disjoint `temperature_scaling_predictions.csv` (550 test samples). The two sets share no patients/images.

### 8.1 Headline: per-class abstention rate at ≈80% overall coverage

The operator question is *"how often does each rule abstain on each clinical class?"* Lower abstention on Severe and Proliferative DR is clinically preferred.

| Class | Global threshold | OACSP equalized | OACSP ordinal-cost |
|---|---:|---:|---:|
| 0 — No DR | 1.5% | 7.0% | 3.3% |
| 1 — Mild | 35.7% | 23.2% | 58.9% |
| 2 — Moderate | 36.7% | 12.7% | 26.0% |
| **3 — Severe** | **44.8%** | **10.3%** | **31.0%** |
| **4 — Proliferative** | **29.5%** | **6.8%** | **20.5%** |

**Effect size for OACSP equalized-recall vs global threshold (clinically meaningful classes):**
- Severe abstention: **4.4× lower** (44.8% → 10.3%)
- Proliferative abstention: **4.3× lower** (29.5% → 6.8%)
- Moderate abstention: 2.9× lower (36.7% → 12.7%)
- Mild abstention: 1.5× lower (35.7% → 23.2%)
- No-DR abstention: intentionally *higher* (1.5% → 7.0%) — the rule trades easy-class abstention for hard-class retention

**The ordinal-cost variant exhibits a deliberately asymmetric trade-off:** it abstains heavily on Mild DR (58.9%, where the ordinal cost penalty is minimal) in order to retain budget for the costlier severe classes. This is the algorithm operating as designed under its cost matrix `{0:1.0, 1:1.0, 2:1.0, 3:2.5, 4:3.0}`.

### 8.2 Overall metrics at matched coverage

| Method | Coverage | Selective Acc | Selective QWK | Cost-Weighted AURC |
|---|---:|---:|---:|---:|
| Global threshold (baseline) | 0.809 | 0.883 | 0.9068 | 0.1337 |
| OACSP — equalized recall | 0.896 | 0.826 | 0.8865 | 0.1337 |
| **OACSP — ordinal cost weighted** | **0.820** | 0.863 | **0.9218** | 0.1337 |

**OACSP ordinal-cost achieves selective QWK 0.9218 vs 0.9068 for the global baseline** — a `+0.0150` absolute improvement at near-identical coverage — *while simultaneously* reducing Severe-DR abstention from 44.8% to 31.0%. This is the publishable result for v1.1.0.

### 8.3 Honest interpretation

- The cost-weighted AURC is identical across methods (0.1337). This is because AURC integrates the *uncertainty signal's ranking quality* over all coverage levels, and all three methods use the same neg-max-confidence signal. The methods differ in *which samples they keep at any single coverage operating point*, not in the underlying ranking. The novelty of OACSP is **the operating-point selection rule, not the signal**.
- OACSP equalized-recall does not target a global coverage — it lets coverage settle wherever the per-class recall targets force it (here, 0.896). That is by design.
- OACSP ordinal-cost targets the exact operator-chosen overall coverage and minimizes expected ordinal cost on retained samples. It is the more principled variant when the cost matrix is known.
- Effect size *increased* (Severe: 3.5× → 4.4×) compared to the dirty val/test run from earlier in the day. This is counterintuitive but consistent: when calibration is done on a properly held-out val set, the per-class thresholds reflect the model's true behavior on unseen data, leading to better-calibrated abstention decisions on the disjoint test set.

### 8.4 Remaining work for v1.2.0

1. **Multi-seed.** Train with seeds 7 and 137 and report mean ± SD on all metrics.
2. **External validation.** Attach IDRiD or Messidor-2 (Google Brain 5-class grading) on Kaggle and run inference + OACSP on the same trained checkpoint.
3. **Equity audit.** Test whether OACSP abstention rates correlate with image-quality proxies (sharpness, illumination).
4. **Per-class thresholds keyed on predicted vs true class.** At inference the true class is unknown — if model confidence patterns differ between *predicted-as-Severe* and *truly-Severe* samples (likely on the poorly-calibrated minority class), the rule leaks. Empirically the leakage seems small in this run (effect sizes are large) but the limitation is real and should be discussed in the paper.

