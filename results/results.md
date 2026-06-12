# Results

Single source of truth for paper-ready numbers. Auto-generated row numbers are filled from `experiments/runs/baseline_efficientnet_b0_aptos/` outputs.

**Git SHA of this result set:** `b196c59` (fix: calibrate_probs moves logits to temperature's device)
**Run date:** 2026-06-12
**W&B run:** [`p3dbnx3q`](https://wandb.ai/qapulsebysk/retinal-selective-prediction/runs/p3dbnx3q)
**Compute:** Kaggle T4 GPU
**Training time:** ~50 min, early-stopped at epoch 24 (best epoch 17)

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
