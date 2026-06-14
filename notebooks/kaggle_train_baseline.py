# %% [markdown]
# # APTOS 2019 — Full Pipeline (Train + Uncertainty + OACSP)
#
# This is the **canonical Kaggle notebook** for this project. It is a plain
# Python file (`.py`) that Kaggle and Jupyter both understand via the `# %%`
# cell markers. The file lives in the GitHub repo so if Kaggle ever loses the
# notebook, you can re-upload this file as a new Kaggle notebook and continue.
#
# This single notebook runs the complete v1.1.0 pipeline end-to-end:
# clone repo → train baseline → evaluate → temperature scaling → MC dropout
# → conformal prediction → OACSP analysis (with proper val/test split).
#
# **Total wall time on Kaggle T4: approximately 75 minutes.**
#
# ## Before running
# 1. Right panel → **Add input** → attach the **APTOS 2019 Blindness Detection**
#    competition dataset.
# 2. Right panel → **Settings** → **Accelerator** → **GPU T4 x2** (or T4 x1).
# 3. Right panel → **Settings** → **Internet** → **ON**.
# 4. (Optional) Add a Kaggle Secret named `WANDB_API_KEY` under
#    *Add-ons → Secrets* for W&B logging. The pipeline works without it
#    (falls back to offline mode).
#
# ## Full runbook
# See `docs/kaggle_setup.md` in the repo for troubleshooting and design notes.

# %% [markdown]
# ## 1. Configuration

# %%
GITHUB_USER = "ShahnawazKakarh"
GITHUB_REPO = "retinal-selective-prediction"
GITHUB_BRANCH = "v1.1.0-oacsp"  # current working branch; switch to "master" for v1.0.0 frozen release

APTOS_INPUT_DIR = "/kaggle/input/competitions/aptos2019-blindness-detection"
WORKDIR = f"/kaggle/working/{GITHUB_REPO}"
SPLITS_DIR = f"{WORKDIR}/data/splits/aptos2019"
RUN_DIR = f"{WORKDIR}/experiments/runs/baseline_efficientnet_b0_aptos"


# %% [markdown]
# ## 2. Clone repo + install dependencies

# %%
import os
import subprocess
import sys


def run(cmd: str, check: bool = True) -> None:
    """Run a shell command and stream output. Raises on non-zero exit if check=True."""
    print(f"$ {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True, check=False)
    if check and result.returncode != 0:
        raise SystemExit(f"Command failed (exit {result.returncode}): {cmd}")


# Fresh clone (idempotent — removes any stale checkout from a previous session)
os.chdir("/kaggle/working")
run(f"rm -rf {GITHUB_REPO}", check=False)
run(
    f"git clone --depth 1 --branch {GITHUB_BRANCH} "
    f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}.git"
)
os.chdir(WORKDIR)
sys.path.insert(0, WORKDIR)

# Record the exact git SHA used for this run — paste this into results.md
git_sha = subprocess.check_output("git rev-parse HEAD", shell=True, text=True).strip()
print(f"\nRunning git SHA: {git_sha[:7]}")


# %% [markdown]
# ## 3. Load W&B key (optional)

# %%
try:
    from kaggle_secrets import UserSecretsClient

    os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
    print("W&B key loaded — runs will sync to wandb.ai")
except Exception as exc:
    os.environ["WANDB_MODE"] = "offline"
    print(f"W&B offline mode (reason: {exc})")


# %% [markdown]
# ## 4. Install pinned dependencies

# %%
# Quiet install. Kaggle warns about RAPIDS dependency conflicts — those are
# harmless, RAPIDS is pre-installed for other notebooks and we don't use it.
run(
    "pip install --quiet -q "
    "'timm>=1.0.11' 'albumentations>=1.4.18' 'omegaconf>=2.3.0' "
    "'wandb>=0.18.0' 'python-dotenv>=1.0.1' 'netcal>=1.3.5'"
)


# %% [markdown]
# ## 5. Prepare stratified train/val/test splits (70/15/15)

# %%
run(
    f"python scripts/prepare_aptos_splits.py "
    f"--train-csv {APTOS_INPUT_DIR}/train.csv "
    f"--output-dir {SPLITS_DIR}"
)


# %% [markdown]
# ## 6. Train EfficientNet-B0 baseline
# Approximately 50 minutes on T4. Mixed precision, early stopping on val QWK,
# seed locked at 42.

# %%
run(
    f"python scripts/train.py "
    f"--config configs/baseline.yaml "
    f"--splits-dir {SPLITS_DIR} "
    f"--images-dir {APTOS_INPUT_DIR}/train_images"
)


# %% [markdown]
# ## 7. Evaluate on internal test set
# Saves `metrics.json`, `predictions.csv`, **`val_predictions.csv`** (added in
# commit `2400870` so downstream OACSP can use a proper val/test split),
# confusion matrix and reliability diagram.

# %%
run(
    f"python scripts/evaluate.py "
    f"--run-dir {RUN_DIR} "
    f"--splits-dir {SPLITS_DIR} "
    f"--images-dir {APTOS_INPUT_DIR}/train_images"
)


# %% [markdown]
# ## 8. Temperature Scaling
# Fits a single scalar T on val logits, applies to test. Saves both calibrated
# val and test predictions for downstream OACSP.

# %%
run(
    f"python scripts/run_temperature_scaling.py "
    f"--run-dir {RUN_DIR} "
    f"--splits-dir {SPLITS_DIR} "
    f"--images-dir {APTOS_INPUT_DIR}/train_images"
)


# %% [markdown]
# ## 9. MC Dropout (T=30)
# ~12 minutes — 30 stochastic forward passes through the dropout-active model.
# Saves four uncertainty signals (max-conf, predictive entropy, expected
# entropy, mutual information / BALD).

# %%
run(
    f"python scripts/run_mc_dropout.py "
    f"--run-dir {RUN_DIR} "
    f"--splits-dir {SPLITS_DIR} "
    f"--images-dir {APTOS_INPUT_DIR}/train_images "
    f"--n-samples 30"
)


# %% [markdown]
# ## 10. Split Conformal Prediction (α=0.10)
# Calibrates on val, evaluates on test. Reports empirical coverage,
# average set size, singleton fraction.

# %%
run(
    f"python scripts/run_conformal.py "
    f"--run-dir {RUN_DIR} "
    f"--splits-dir {SPLITS_DIR} "
    f"--images-dir {APTOS_INPUT_DIR}/train_images "
    f"--alpha 0.10"
)


# %% [markdown]
# ## 11. OACSP — the v1.1.0 novel contribution
# Ordinal-Aware Class-Conditional Selective Prediction. Calibrates per-class
# thresholds on **val** predictions, evaluates on **test** predictions.
# Methodologically correct val/test split (fixed in commit `2400870`).
#
# Outputs the headline comparison table for the v1.1.0 paper.

# %%
run(
    f"python scripts/run_oacsp_analysis.py "
    f"--val-predictions  {RUN_DIR}/temperature_scaling_val_predictions.csv "
    f"--test-predictions {RUN_DIR}/temperature_scaling_predictions.csv "
    f"--target-coverage  0.80 "
    f"--output-dir       {RUN_DIR}/oacsp_v1.1.0"
)


# %% [markdown]
# ## 12. Done — what to paste back
#
# The full pipeline is complete. The numbers that matter live in:
#
# * `experiments/runs/baseline_efficientnet_b0_aptos/metrics.json`
#   — baseline accuracy, QWK, ECE, NLL, Brier, per-class P/R/F1
# * `experiments/runs/baseline_efficientnet_b0_aptos/temperature_scaling.json`
#   — fitted T, ECE before/after
# * `experiments/runs/baseline_efficientnet_b0_aptos/mc_dropout_selective.json`
#   — AURC / excess-AURC / selective accuracy for all 4 uncertainty signals
# * `experiments/runs/baseline_efficientnet_b0_aptos/conformal_summary.json`
#   — empirical coverage, set size diagnostics
# * `experiments/runs/baseline_efficientnet_b0_aptos/oacsp_v1.1.0/oacsp_comparison.csv`
#   — OACSP headline table (the v1.1.0 novel result)
# * `experiments/runs/baseline_efficientnet_b0_aptos/oacsp_v1.1.0/oacsp_detailed.json`
#   — full per-class metrics for global / equalized / ordinal-cost rules
#
# The cell output above already prints the "HEADLINE CHECK" block. Copy that block
# back to the project lead — it gets committed to `results/results.md`.

# %%
import json
from pathlib import Path

print("=" * 78)
print("HEADLINE NUMBERS — paste this entire block back to the project lead")
print("=" * 78)

for fname, label in [
    ("metrics.json", "metrics.json"),
    ("temperature_scaling.json", "temperature_scaling.json"),
    ("temperature_scaling_selective.json", "temperature_scaling_selective.json"),
    ("mc_dropout_selective.json", "mc_dropout_selective.json"),
    ("conformal_summary.json", "conformal_summary.json"),
]:
    fpath = Path(RUN_DIR) / fname
    if fpath.exists():
        print(f"\n--- {label} ---")
        print(json.dumps(json.loads(fpath.read_text()), indent=2))

oacsp_csv = Path(RUN_DIR) / "oacsp_v1.1.0" / "oacsp_comparison.csv"
if oacsp_csv.exists():
    print(f"\n--- oacsp_comparison.csv ---")
    print(oacsp_csv.read_text())

oacsp_json = Path(RUN_DIR) / "oacsp_v1.1.0" / "oacsp_detailed.json"
if oacsp_json.exists():
    print(f"\n--- oacsp_detailed.json ---")
    print(json.dumps(json.loads(oacsp_json.read_text()), indent=2))

print()
print(f"git SHA of this run: {git_sha}")
print()
print("=" * 78)
