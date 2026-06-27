# Research Notes

Working document tracking the scientific framing, novelty claims, prior art, and version plan for this project. Updated as the work progresses.

---

## 1. Scientific framing

**Headline claim (v1.0.0, current):**
We provide an open-source, single-backbone, single-seed, fully-reproducible benchmark of five well-known selective-prediction techniques on the APTOS 2019 diabetic retinopathy (DR) screening task. The benchmark answers the practical question: *for a deployed single-backbone DR screening model, which post-hoc uncertainty method gives the best risk-coverage trade-off, the best calibration, and the best coverage guarantee — and which is the cheapest to deploy?*

This is a **benchmark report**, not a novel methodological contribution. We are explicit about this.

**Headline claim (v1.1.0 and beyond):**
We additionally propose *class-conditional selective thresholds* for ordinal DR severity grading — a method that adapts the abstention rule to the per-class clinical importance and confidence distribution, rather than using a single threshold across all severity grades. This is the novel methodological piece (see §4).

---

## 2. What is *not* novel about v1.0.0

To survive peer review, we have to be honest about prior work. Every technique we use here has been published. The contribution of v1.0.0 is **integration, reproducibility, and head-to-head comparison on a single open dataset**, not new methods.

- **EfficientNet-B0 backbone:** Tan & Le 2019 (ICML). Standard.
- **Temperature Scaling:** Guo et al. 2017 (ICML), "On Calibration of Modern Neural Networks." Single-scalar post-hoc calibration.
- **MC Dropout:** Gal & Ghahramani 2016 (ICML). Stochastic forward passes for epistemic uncertainty.
- **BALD / Mutual Information signal:** Houlsby et al. 2011. Information-theoretic acquisition for active learning, repurposed as an uncertainty signal.
- **Split Conformal Prediction:** Romano et al. 2020; Angelopoulos & Bates 2021 (tutorial). Distribution-free coverage guarantees.
- **APS (Adaptive Prediction Sets):** Romano et al. 2020 (NeurIPS).
- **Risk-Coverage curves / AURC / Excess-AURC:** El-Yaniv & Wiener 2010 (JMLR); Geifman & El-Yaniv 2017 (NeurIPS).
- **ECE (Expected Calibration Error):** Naeini et al. 2015 (AAAI).
- **Selective prediction for DR specifically:** Leibig et al. 2017 (Nature Scientific Reports) used dropout-based uncertainty on Kaggle DR. Filos et al. 2019 ("Benchmarking BDL on Diabetic Retinopathy") evaluated several BDL methods on the same Kaggle dataset. Band et al. 2021 (NeurIPS) "Benchmarking Bayesian Deep Learning on Diabetic Retinopathy" pushed this further.

**Our differentiation for v1.0.0** is therefore narrow but real:
- *Single seed, single backbone, single open dataset, full source code + W&B run* — for direct reproducibility. Prior benchmark papers typically don't release the exact training notebook with locked seeds.
- *Same evaluation harness applied identically to all methods* — controls for an annoying confound in prior benchmarks where each method was reported with its own preferred evaluation script.
- *Excess-AURC reported alongside AURC* — separates the uncertainty signal quality from the underlying accuracy.

This is a defensible v1.0.0. We will not over-claim.

---

## 3. What other groups have already done on DR with uncertainty

Brief literature scan so the report can position itself honestly:

| Year | Authors | Venue | Dataset | Methods covered | Notes |
|---|---|---|---|---|---|
| 2017 | Leibig et al. | Nature SciRep | EyePACS / Kaggle DR | MC Dropout | First major BDL-on-DR paper. Binary referable/non-referable. |
| 2019 | Filos et al. | NeurIPS BDL workshop | EyePACS / Kaggle DR | MC Dropout, Ensembles, Mean-Field VI | Binary task, EfficientNet-B0 used. |
| 2021 | Band et al. | NeurIPS Datasets & Benchmarks | EyePACS | Many BDL methods + active learning | Distribution shift + selective. |
| 2020 | Ayhan et al. | Med. Image Anal. | EyePACS, IDRiD | Test-time aug, dropout | Reliable uncertainty quantification. |
| 2022 | Linmans et al. | MIDL | Various | Evidential DL | EDL on medical imaging. |

**Gap we can credibly claim:**
- Most prior DR uncertainty work targets the *binary* "referable / not referable" task. The full 5-class ICDR ordinal grading is comparatively under-explored for selective prediction.
- Most prior work uses EyePACS or Kaggle-DR (the 2015 dataset). APTOS 2019 is newer, smaller, and label distribution is different. Replicating findings on APTOS is itself a useful contribution.
- Most prior work does not report class-conditional behavior of abstention — i.e., do we systematically abstain *more* on rare severe-DR classes? (This sets up the v1.1.0 novel piece.)

---

## 4. The novel piece for v1.1.0 — Class-conditional Selective Thresholds

**Motivation.** In our v1.0.0 results, per-class F1 ranges from 0.97 (No DR) down to 0.45 (Severe DR). The Severe and Proliferative-DR classes — clinically the most important to catch — have the lowest baseline F1. A global confidence threshold for abstention treats all five severity classes as equivalent, but they are not: missing a Severe-DR or Proliferative-DR case has very different consequences from over-calling Mild DR.

**Proposed method.** Replace the single global threshold τ with per-class thresholds {τ₀, τ₁, τ₂, τ₃, τ₄}, calibrated on the validation set such that:
- *Variant A — equalized-recall:* per-class abstention rate is set so that retained per-class recall ≥ a target (e.g., 0.90 for No DR, 0.95 for Severe/Proliferative-DR).
- *Variant B — risk-aware:* an explicit cost matrix is multiplied through the per-class confidence before thresholding (ordinal-distance-aware).

**Why this is novel:**
- Class-conditional thresholds have been studied generally (Geifman & El-Yaniv 2019 "SelectiveNet"), but to our knowledge nobody has applied them to 5-class DR selective prediction.
- The variant that respects ordinal distance (B) is genuinely new — prior selective-prediction work mostly treats DR as a flat classification problem.
- The clinical motivation is concrete and easy to justify in a paper.

**How we will evaluate:**
- Compare class-conditional vs. global thresholds at matched overall coverage (e.g., 80% coverage).
- Report per-class selective F1 and per-class abstention rates.
- Report whether the equalized-recall variant *also* improves AURC and Excess-AURC, or whether it trades one for the other.

**Code already in place to support this:**
- `src/selective/risk_coverage.py` — handles arbitrary signal arrays. The class-conditional logic is a small addition.
- Per-sample predictions + uncertainty signals saved in `*_predictions.csv` from v1.0.0 runs. We can prototype this *without retraining*.

This is the credible methodological contribution we add in v1.1.0.

---

## 5. The further extensions for v2.0.0 (J-BHI submission)

Each of these strengthens the paper but is *not* required for v1.0.0 or v1.1.0.

- **External validation on IDRiD and Messidor-2.** Tests how well APTOS-trained calibration transfers to other DR datasets — direct evidence of generalization. The infrastructure for this is sketched in `src/data/idrid.py` and `src/data/messidor2.py` (to be added).
- **Deep Ensembles (M=5).** Strong baseline; expected to dominate single-model MC Dropout on AURC.
- **Evidential Deep Learning.** Dirichlet-parameterized head + custom loss. Provides a single forward pass with calibrated uncertainty.
- **Equity audit.** Do uncertainty-based abstention rates correlate with image quality proxies (sharpness, illumination, retinal coverage)? Does the class-conditional method reduce demographic disparities in abstention?

---

## 6. Reproducibility plan (matches what is actually in the repo)

- All seeds locked: `GLOBAL_SEED = 42` (training, splits, MC Dropout sampling).
- All scripts are pure Python with explicit CLI args, no notebook hidden state.
- One Kaggle notebook contains the complete pipeline (clone → install → train → evaluate → all uncertainty methods → save outputs). Re-running it on Kaggle T4 produces all v1.0.0 numbers in ~45 minutes.
- The git SHA used to produce each result is saved inside the result file (`outputs/git_sha.txt`).
- W&B run id is recorded in the result files for traceability.

---

## 7. Version plan summary

| Version | Scope | Novelty claim | DOI | Status |
|---|---|---|---|---|
| **v1.0.0** | Single-seed benchmark of 4 uncertainty methods on APTOS 2019 | Reproducibility + head-to-head comparison only | `10.5281/zenodo.20681524` | ✅ Released 2026-06-13 |
| **v1.1.0** | + OACSP (Ordinal-Aware Class-Conditional Selective Prediction) + proper val/test split | Methodological: class-conditional ordinal-aware abstention rule, occupies previously-empty quadrant in selective-prediction literature | `10.5281/zenodo.20695855` | ✅ Released 2026-06-14 |
| v1.2.0 | + multi-seed variance (seeds 7, 137) + IDRiD external validation + OACSP transfer test + paper-quality figures + restyled "RESEARCH ARTICLE" PDF for SSRN | Empirical: cross-dataset generalization of both backbone and abstention rule | pending | 🟡 In progress |
| v2.0.0 | + Deep Ensembles + Evidential DL + Messidor-2 + equity audit + image-quality stratified analysis | Full benchmark + methodological + clinical contribution; consolidated paper | pending | 🔵 Target: IEEE J-BHI |

Each Zenodo version increments under the same Zenodo Concept DOI (`10.5281/zenodo.20681415`), so the ORCID entry shows the project as a single evolving artifact.

---

## 8. v1.1.0 result snapshot (real numbers)

The OACSP novel contribution was demonstrated on APTOS 2019 with the proper val/test split. Headline result at ~80% overall coverage:

| Class | Global threshold abstention | OACSP equalized recall | Improvement |
|---|---:|---:|---:|
| 3 — Severe | 44.8% | 10.3% | **4.4× lower** |
| 4 — Proliferative | 29.5% | 6.8% | **4.3× lower** |

OACSP ordinal-cost-weighted variant achieves **selective QWK 0.9218 vs. 0.9068** for the global baseline at near-identical coverage. Full breakdown in `results/results.md` §8 and `docs/oacsp_method.md`.

---

## 9. v1.2.0 concrete deliverables (locked scope)

1. **Multi-seed runs.** Re-train baseline with seeds 7 and 137 on Kaggle. Re-run all four uncertainty methods + OACSP on each. Report mean ± SD on every published metric.
2. **IDRiD external validation.** Attach Kaggle dataset `mariaherrerot/idrid-dataset` (images + 5-class ICDR labels matching APTOS scheme). New module `src/data/idrid.py`, new script `scripts/run_external_validation.py`. Evaluate the APTOS-trained checkpoint zero-shot on IDRiD.
3. **OACSP transfer test.** Apply OACSP per-class thresholds calibrated on APTOS val to IDRiD test set. Does the per-class threshold transfer? This is a stronger novelty claim than v1.1.0.
4. **Paper-quality figures.** Bar charts of per-class abstention, risk-coverage curves, reliability diagrams, OACSP novelty quadrant. Produced by `scripts/generate_paper_figures.py` from saved JSON/CSV outputs, no GPU needed.
5. **Restyled "RESEARCH ARTICLE" PDF.** Clean journal-style cover (RESEARCH ARTICLE eyebrow, big serif title, author block, date+version, code+licence footer). Suitable for SSRN upload.

