# 👁️ Retinal Selective Prediction

> Calibrated uncertainty and **selective prediction** for retinal disease screening — *"when should a deep model abstain and route to a clinician, and how much does it actually help?"*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![timm](https://img.shields.io/badge/timm-1.x-1F77B4.svg)](https://github.com/huggingface/pytorch-image-models)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebooks-20BEFF.svg?logo=kaggle)](https://www.kaggle.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Target venue: IEEE J-BHI](https://img.shields.io/badge/target%20venue-IEEE%20J--BHI-c8102e.svg)](https://www.embs.org/jbhi/)

---

## 🧭 Overview

Deep learning models for diabetic retinopathy (DR) screening routinely report 85–95% accuracy on standard benchmarks. The number that *doesn't* appear in those papers is the one a clinician actually needs: **what fraction of confidently-wrong predictions slip through, and how do we catch them?**

This repository benchmarks five families of **uncertainty quantification** methods on the same retinal-screening backbone and same fixed splits, then evaluates each through the lens of **selective prediction** — the right to abstain and refer hard cases to a human.

| Method | Family | What it estimates | Cost vs. baseline |
|---|---|---|---|
| **Softmax confidence** | Deterministic | Aleatoric ≈ epistemic, mixed | 1× (free) |
| **MC Dropout** | Bayesian approx. | Mean + epistemic spread across T passes | T× inference |
| **Deep Ensembles** | Frequentist Bayesian | Predictive variance across M independent models | M× train + inference |
| **Temperature Scaling** | Post-hoc calibration | Sharpens / softens softmax to match accuracy | ~0× (1-param fit) |
| **Evidential Deep Learning** | Subjective logic | Dirichlet over class probs, explicit uncertainty | 1× (custom loss) |
| **Conformal Prediction** | Distribution-free | Calibrated prediction sets with coverage guarantee | ~0× (post-hoc) |

> **Why this matters clinically:** screening tools live or die on **false negatives**. A 95% accurate model that confidently mis-classifies the 5% of severe-DR cases it misses is *worse* than an 85% accurate model that abstains on its hardest 15% and surfaces them for human review. Selective prediction reframes "how accurate is the model" as **"how accurate is the model on the cases it chose to answer."**

---

## 📈 Results

> ✅ **Internal test results live** (`b196c59`, 2026-06-12). See [`results/results.md`](results/results.md) for full breakdown and reproducibility notes. External validation on Messidor-2 / IDRiD coming next.

### Internal test set — APTOS 2019 (held-out 15%, stratified)

| Method | Acc ↑ | QWK ↑ | ECE ↓ | NLL ↓ | AURC ↓ | Excess-AURC ↓ | Sel. Acc @ 80% coverage ↑ |
|---|---|---|---|---|---|---|---|
| Softmax (deterministic) | 0.809 | 0.865 | 0.146 | 1.023 | 0.0793 | 0.0598 | 0.882 |
| Temperature Scaling (T=2.25) | 0.809 | 0.865 | **0.055** | 0.576 | 0.0852 | 0.0657 | 0.884 |
| MC Dropout (T=30, MI signal) | 0.809 | 0.865 | 0.146 | — | **0.0756** | **0.0561** | 0.882 |
| Conformal (split, α=0.10) | 0.809 | 0.865 | — | — | 0.1112 | 0.0917 | 0.884 |
| Deep Ensembles (M=5) | – | – | – | – | – | – | – |
| Evidential DL | – | – | – | – | – | – | – |

QWK = quadratic-weighted Cohen's κ (APTOS official metric). ECE = Expected Calibration Error (15 bins). AURC = Area Under Risk–Coverage curve. Excess-AURC isolates the uncertainty signal from baseline accuracy.

**Three different methods, three different wins.** Temperature Scaling cuts ECE by **62%** (0.146 → 0.055) with one scalar. MC Dropout with mutual information has the lowest AURC, meaning its uncertainty signal best identifies which predictions to abstain on. Conformal hits its 90% coverage target almost exactly (empirical 90.2%) with average set size 1.35.

### External validation — Messidor-2

| Method | Acc ↑ | QWK ↑ | ECE ↓ | AURC ↓ | Excess-AURC ↓ | Sel. Acc @ 80% coverage ↑ |
|---|---|---|---|---|---|---|
| Softmax | – | – | – | – | – | – |
| Temperature Scaling | – | – | – | – | – | – |
| MC Dropout | – | – | – | – | – | – |
| Deep Ensembles | – | – | – | – | – | – |
| Evidential DL | – | – | – | – | – | – |
| Conformal | – | – | – | – | – | – |

### External validation — IDRiD

| Method | Acc ↑ | QWK ↑ | ECE ↓ | AURC ↓ | Excess-AURC ↓ |
|---|---|---|---|---|---|
| (same rows) | – | – | – | – | – |

### Comparison vs. published baselines

| Reference | Backbone | Dataset | Reported QWK | Notes |
|---|---|---|---|---|
| APTOS 2019 winner (kaggle, 2019) | Inception-ResNet v2 ensemble | APTOS (test) | 0.936 | TTA + 4× ensemble, no uncertainty / no abstention |
| Krause et al. 2018 (Google) | Inception-v4 | Proprietary | – | Closed dataset; not directly comparable |
| *(this work)* | EfficientNet-B0 + selective | APTOS (internal test) | – | Single-backbone for clean ablations |

> The point of this work is **not** to chase the absolute best QWK number — that race is already over. The contribution is the **selective-prediction comparison on a fixed backbone**, which the literature has not done cleanly.

---

## 📊 Datasets

| Dataset | Role | Size | Classes | License | Notes |
|---|---|---|---|---|---|
| [**APTOS 2019**](https://www.kaggle.com/competitions/aptos2019-blindness-detection) | Train / val / internal test | 3,662 images, 5-class DR | 5 (No DR → Proliferative) | Kaggle competition (free) | Indian population, smartphone fundus camera |
| [**Messidor-2**](https://www.adcis.net/en/third-party/messidor2/) | External validation #1 | 1,748 images | Re-mapped to 5-class DR | Free with registration | French, multiple camera vendors → real distribution shift |
| [**IDRiD**](https://idrid.grand-challenge.org/) | External validation #2 | 516 images | 5-class DR + DME grading | Free | Indian Diabetic Retinopathy Image Dataset |

All datasets are public and require either Kaggle competition acceptance or a free registration. **No private or hospital data.** This is a deliberate design choice — every result is reproducible by anyone with a Kaggle account.

Stratified train/val/test splits are computed once and the CSVs committed to `data/splits/aptos2019/`, guaranteeing identical splits across every uncertainty method.

---

## 🏗️ Architecture

```
                  ┌───────────────────────┐
                  │   Fundus image (RGB)  │
                  └───────────┬───────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │  Backbone (EffNet-B0) │
                  │   ImageNet-pretrained │
                  └───────────┬───────────┘
                              ▼
                  ┌───────────────────────┐
                  │   Dropout (p=0.3)     │   ← kept active at inference
                  │                       │     for MC Dropout sampling
                  └───────────┬───────────┘
                              ▼
                  ┌───────────────────────┐
                  │   Linear → 5 classes  │
                  └───────────┬───────────┘
                              ▼
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼                                           ▼
┌──────────────────┐                  ┌──────────────────────────┐
│ Softmax / Temp.  │                  │ MC Dropout / Ensembles / │
│   Scaling /      │                  │ Evidential / Conformal   │
│   Conformal      │                  │ → predictive + epistemic │
└────────┬─────────┘                  └────────────┬─────────────┘
         │                                         │
         └─────────────────┬───────────────────────┘
                           ▼
                ┌──────────────────────┐
                │ Selective Prediction │
                │  AURC, risk–coverage │
                │  selective accuracy  │
                └──────────────────────┘
```

Single backbone, multiple uncertainty heads — this is the design choice that makes the comparison clean. Implementations live in:

- `src/models/backbone.py` — `RetinalClassifier` (timm backbone + dropout-friendly head)
- `src/uncertainty/` — one module per uncertainty method
- `src/selective/risk_coverage.py` — AURC, excess-AURC, selective accuracy (modality-agnostic)
- `src/evaluation/` — calibration metrics + reliability diagrams

---

## 🚀 Quickstart

### Install

```bash
git clone https://github.com/ShahnawazKakarh/retinal-selective-prediction.git
cd retinal-selective-prediction

python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pre-commit install

cp .env.example .env   # then fill in WANDB_API_KEY (optional)
```

### Train the baseline (Kaggle — recommended)

```bash
# See docs/kaggle_setup.md for the full walkthrough.
# Short version: open notebooks/kaggle_train_baseline.py on Kaggle,
# attach the APTOS 2019 dataset, set Internet=On + GPU T4, run all.
```

### Train locally (small smoke test)

```bash
# Make stratified splits (one-time)
python scripts/prepare_aptos_splits.py \
    --train-csv /path/to/aptos/train.csv \
    --output-dir data/splits/aptos2019

# Train baseline
python scripts/train.py \
    --config configs/baseline.yaml \
    --splits-dir data/splits/aptos2019 \
    --images-dir /path/to/aptos/train_images
```

### Evaluate (deterministic — softmax + calibration)

```bash
python scripts/evaluate.py \
    --run-dir experiments/runs/baseline_efficientnet_b0_aptos \
    --splits-dir data/splits/aptos2019 \
    --images-dir /path/to/aptos/train_images
# → metrics.json, predictions.csv, confusion_matrix.png, reliability_diagram.png
```

### Run uncertainty methods

```bash
# MC Dropout — T stochastic forward passes
python scripts/run_mc_dropout.py \
    --run-dir experiments/runs/baseline_efficientnet_b0_aptos \
    --splits-dir data/splits/aptos2019 \
    --images-dir /path/to/aptos/train_images \
    --n-samples 30
```

More methods land as the project progresses — see [Roadmap](#-roadmap).

---

## 📁 Repository structure

```
retinal-selective-prediction/
├── configs/                  # one YAML per reproducible experiment
├── src/
│   ├── data/                 # APTOS dataset, stratified splits, augmentations
│   ├── models/               # backbones with dropout hooks
│   ├── uncertainty/          # MC dropout, ensembles, temp scaling, EDL, conformal
│   ├── selective/            # risk–coverage, AURC, selective policies
│   ├── training/             # training loop, AMP, cosine+warmup LR, W&B
│   ├── evaluation/           # ECE, reliability diagram, confusion matrices
│   └── utils/                # seeding, config loading
├── scripts/                  # CLI entry points (train, evaluate, run_*)
├── notebooks/                # Kaggle / Colab notebooks (jupytext .py)
├── data/splits/              # committed CSVs — same splits across all runs
├── experiments/runs/         # run outputs (gitignored)
├── results/                  # paper-ready tables + plots
├── docs/                     # Kaggle setup, dataset notes
└── tests/                    # pytest
```

---

## 🔬 Research notes

Full writeup of scientific framing, prior work, novelty positioning, and the v1.0.0 → v1.1.0 → v2.0.0 plan lives in [`docs/research_notes.md`](docs/research_notes.md).

A few design decisions called out so they don't surprise future-me or reviewers:

- **Single backbone for all uncertainty methods.** Mixing backbones across methods is the silent killer in this literature — any AURC difference becomes "is it the uncertainty method or the model?" Fixed backbone makes the comparison genuinely apples-to-apples.
- **Splits committed to git.** Every method reads the *exact same* train.csv / val.csv / test.csv. No accidental data shuffling between experiments.
- **Excess-AURC is the headline.** Raw AURC is dominated by overall accuracy — a stronger model has lower AURC almost regardless of uncertainty quality. Excess-AURC subtracts the oracle's AURC and isolates "how good is the uncertainty *signal*."
- **External validation is non-negotiable.** Messidor-2 + IDRiD are pre-committed as held-out test sets — never used for hyperparameter selection, never used for calibration. They exist solely to answer "does the method generalize?"
- **Negative results stay in.** If MC Dropout loses to plain softmax confidence on this backbone (it sometimes does), the paper says so. The contribution is the honest comparison, not a flattering hero method.

---

## 🛣️ Roadmap

- [x] Repo scaffolding (gitignore, pre-commit secret scanning, deps, pyproject)
- [x] APTOS dataset loader, stratified split utility, train/eval augmentations
- [x] EfficientNet-B0 backbone with MC-Dropout-friendly head
- [x] Training loop (AMP, cosine+warmup LR, early stopping on QWK, W&B logging)
- [x] Deterministic evaluation (ECE, reliability diagram, confusion matrix)
- [x] Kaggle notebook for baseline training
- [x] MC Dropout uncertainty + risk–coverage / AURC selective analysis
- [x] **Baseline trained — val QWK 0.889, test QWK 0.865** ([W&B run](https://wandb.ai/qapulsebysk/retinal-selective-prediction))
- [x] **Temperature Scaling — ECE 0.146 → 0.055**
- [x] **MC Dropout (T=30) — best AURC across all methods**
- [x] **Conformal Prediction (Split + APS) — 90.2% empirical coverage @ α=0.10**
- [ ] Deep Ensembles (M=5 independent seeds)
- [ ] Evidential Deep Learning (Dirichlet head + custom loss)
- [ ] External validation harness — Messidor-2 + IDRiD
- [ ] **Novel piece:** class-conditional selective thresholds *(or)* distribution-shift-aware selective prediction
- [ ] Paper draft — target IEEE J-BHI submission Oct/Nov 2026
- [ ] Blog writeup on [skakarh.com](https://www.skakarh.com/blog/)

---

## 📦 Versioned releases & how to cite

This project is published as **versioned Zenodo releases**, each with its own DOI. The Concept DOI on Zenodo resolves to the latest version; the version-specific DOI pins to an exact artifact. The same project also progresses toward an IEEE J-BHI submission for v2.0.0.

| Version | Scope | Status |
|---|---|---|
| **v1.0.0** | Single-seed benchmark on APTOS 2019 (4 uncertainty methods) | 🟢 Released → Zenodo (this release) |
| v1.1.0 | + Class-conditional selective thresholds (novel) + IDRiD external validation | 🟡 Planned |
| v2.0.0 | + Deep Ensembles + Evidential DL + Messidor-2 + equity audit | 🔵 Target: IEEE J-BHI |

**Step-by-step Zenodo release & ORCID flow:** [`docs/zenodo_release.md`](docs/zenodo_release.md)
**Built technical report PDF source:** [`report/`](report/) (run `python report/build.py` to rebuild)
**Machine-readable citation:** [`CITATION.cff`](CITATION.cff)

If you use this benchmark in your work, please cite it via the `CITATION.cff` metadata (DOI added once Zenodo mints v1.0.0).

---

## 📚 Citations & references

Core papers this work builds on:

- Gal & Ghahramani, *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*, ICML 2016
- Lakshminarayanan, Pritzel & Blundell, *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*, NeurIPS 2017
- Guo, Pleiss, Sun & Weinberger, *On Calibration of Modern Neural Networks*, ICML 2017
- Geifman & El-Yaniv, *Selective Classification for Deep Neural Networks*, NeurIPS 2017
- Sensoy, Kaplan & Kandemir, *Evidential Deep Learning to Quantify Classification Uncertainty*, NeurIPS 2018
- Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction*, FnT ML 2023
- Tan & Le, *EfficientNet: Rethinking Model Scaling for CNNs*, ICML 2019

If this code is useful for your work, please cite it via [`CITATION.cff`](CITATION.cff).

---

## ⚠️ Disclaimer

This is **research code**. It is *not* a medical device. It is *not* validated for clinical use. Outputs must never be used to make screening or diagnostic decisions in a clinical workflow. The point of selective prediction research is precisely to acknowledge that ML models, deployed naively, fail in clinically dangerous ways.

---

## 🤝 Contributing

Bug reports, suggestions, and pull requests are welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) for development setup, code style, and how to add a new uncertainty method or dataset. For substantial changes, open an issue first so we can align.

---

## 📄 License

MIT © [Shahnawaz Khan](https://github.com/ShahnawazKakarh)

---

## 🌐 More from SK

| | |
|---|---|
| 🌐 **Website** | [www.skakarh.com](https://www.skakarh.com) |
| ✍️ **Blog** | [skakarh.com/blog](https://www.skakarh.com/blog/) |
| 🛠️ **Services** | [skakarh.com/services](https://www.skakarh.com/services/) |
| 💼 **LinkedIn** | [linkedin.com/in/skakarh](https://www.linkedin.com/in/skakarh) |
| 📦 **More projects** | [github.com/ShahnawazKakarh](https://github.com/ShahnawazKakarh) |
