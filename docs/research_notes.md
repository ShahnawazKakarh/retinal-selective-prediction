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
- W&B run is permanent and public: `qapulsebysk/retinal-selective-prediction`.

---

## 7. Version plan summary

| Version | Scope | Novelty claim | DOI |
|---|---|---|---|
| **v1.0.0** | Single-seed benchmark of 4 uncertainty methods on APTOS 2019 | Reproducibility + head-to-head comparison only | Yes (Zenodo) |
| v1.1.0 | + class-conditional thresholds (novel piece) + IDRiD external validation | Methodological: class-conditional ordinal-aware selective thresholds | Yes (Zenodo, new version DOI) |
| v2.0.0 | + Deep Ensembles + Evidential DL + Messidor-2 + equity audit | Full benchmark + methodological + clinical contribution | J-BHI submission |

Each Zenodo version increments under the same Zenodo Concept DOI, so the ORCID entry shows the project as a single evolving artifact.
