# Retinal Selective Prediction

Selective prediction with calibrated uncertainty for retinal disease screening — research code accompanying the paper *(in preparation)*.

> **Target venue:** IEEE Journal of Biomedical and Health Informatics (J-BHI)
> **Status:** Scaffold — week 1 of ~16-week plan

---

## Research question

When should a deep model abstain from diagnosing retinal disease and route the case to a clinician? How much does selective prediction actually help — on calibration, on missed diagnoses, and under distribution shift?

## Contributions (planned)

1. **Benchmark.** A clean, single-backbone comparison of five uncertainty estimators on retinal screening — MC Dropout, Deep Ensembles, Temperature Scaling, Evidential Deep Learning, and Conformal Prediction.
2. **Selective prediction analysis.** Risk–coverage curves, AURC, and selective accuracy at fixed coverage. Quantifies the clinical trade-off between automation and safety.
3. **Novel piece *(TBD after baseline):*** either (a) class-conditional selective thresholds for asymmetric-cost diseases, or (b) distribution-shift-aware selective prediction validated on external datasets.

## Datasets (all public)

| Role                    | Dataset                | Size      | Notes |
|-------------------------|------------------------|-----------|-------|
| Train / val / internal test | APTOS 2019 (Kaggle)    | ~3.6k     | 5-class diabetic retinopathy |
| External validation #1  | Messidor-2             | 1,748     | Different cameras / population |
| External validation #2  | IDRiD                  | 516       | Distribution shift test |

## Repository layout

```
src/
  data/         dataset loaders, splits, augmentations
  models/       backbones with uncertainty-friendly hooks
  uncertainty/  MC Dropout, ensembles, temperature scaling, EDL, conformal
  selective/    risk–coverage, AURC, threshold policies
  training/     training loop, callbacks, W&B logging
  evaluation/   calibration (ECE), AUC, external-eval harness
  utils/        seeds, config loading, logging
configs/        YAML configs — one file per reproducible run
experiments/    run outputs (gitignored)
notebooks/      Kaggle / Colab notebooks
scripts/        CLI entry points (train, evaluate, calibrate)
tests/          pytest
```

## Setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pre-commit install
cp .env.example .env   # then fill in WANDB_API_KEY etc.
```

## Reproducibility

- All training is seeded (`GLOBAL_SEED=42`) and configs are committed under `configs/`.
- W&B logs every run; run IDs are referenced in the paper.
- External validation runs use frozen weights from internal-test checkpoints — never retrained on external data.

## Disclaimer

Research code. Not a medical device. Not for clinical use.

## License

MIT
