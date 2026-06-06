# Kaggle setup

This project trains on Kaggle's free T4/P100 GPUs. APTOS 2019 already lives on Kaggle, so we don't download or store the dataset locally.

## One-time setup

1. **Sign in to Kaggle** and accept the competition rules for [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules). Without this you cannot attach the dataset.

2. **Verify your phone number** on Kaggle (Settings → Account). Required to use GPU/internet on notebooks.

3. **Get a Kaggle API token** (only needed if you want to push notebooks via CLI, not required for the web UI):
   - Kaggle → Settings → API → "Create New Token" → downloads `kaggle.json`
   - Move it: `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json`

## Workflow for each experiment

1. Push code changes to GitHub (you're already doing this).
2. On Kaggle, open the notebook `kaggle_train_baseline.ipynb` (created from `notebooks/kaggle_train_baseline.py`).
3. In the right-hand panel:
   - **Accelerator** → GPU T4 x2 (or P100; T4 is usually faster to schedule).
   - **Internet** → On (needed to `pip install` and `wandb`).
   - **Add input** → search "APTOS 2019" → add the competition dataset.
4. In the first cell, set `GITHUB_BRANCH = "master"` (or your working branch).
5. Add your W&B key as a Kaggle Secret named `WANDB_API_KEY` (Add-ons → Secrets).
6. Run all. The notebook clones the repo at the chosen branch, installs deps, makes splits, trains, and uploads the best checkpoint as a Kaggle dataset (optional).

## Why this layout

- **No data download.** The APTOS images are mounted read-only at `/kaggle/input/aptos2019-blindness-detection/` on every notebook run. No 9 GB transfer.
- **Reproducibility.** The notebook clones a *specific git SHA* and saves the SHA into the run config. Anyone re-running gets identical code.
- **No secrets in code.** `WANDB_API_KEY` comes from Kaggle Secrets, never the notebook source.

## Converting `.py` → `.ipynb`

The notebook source of truth is `notebooks/kaggle_train_baseline.py` in jupytext "percent" format (cells separated by `# %%`). Two ways to convert:

```bash
# Option A: jupytext (recommended; preserves cell structure)
pip install jupytext
jupytext --to ipynb notebooks/kaggle_train_baseline.py

# Option B: upload the .py directly to Kaggle; it auto-converts.
```

The `.py` file is what gets reviewed in PRs. The `.ipynb` is a generated artifact and is gitignored.
