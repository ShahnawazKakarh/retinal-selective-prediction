# %% [markdown]
# # APTOS 2019 — Baseline Training on Kaggle
#
# This notebook clones the `retinal-selective-prediction` repo at a pinned branch,
# installs dependencies, prepares stratified splits, and trains the EfficientNet-B0
# baseline defined in `configs/baseline.yaml`.
#
# **Before running:**
# 1. Attach the APTOS 2019 competition dataset (right panel → Add input).
# 2. Set Accelerator to GPU T4 x2 (or P100).
# 3. Turn Internet ON.
# 4. Add a Kaggle Secret named `WANDB_API_KEY` (Add-ons → Secrets) if you want W&B logging.

# %% [markdown]
# ## 1. Config

# %%
GITHUB_USER = "ShahnawazKakarh"
GITHUB_REPO = "retinal-selective-prediction"
GITHUB_BRANCH = "master"  # change to feature branch when iterating

APTOS_INPUT_DIR = "/kaggle/input/competitions/aptos2019-blindness-detection"
WORKDIR = "/kaggle/working/retinal-selective-prediction"
SPLITS_DIR = f"{WORKDIR}/data/splits/aptos2019"
OUTPUT_DIR = "/kaggle/working/outputs"

# %% [markdown]
# ## 2. Clone repo and install deps

# %%
import os
import subprocess
import sys


def run(cmd: str, check: bool = True) -> None:
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, check=check)
    if result.returncode != 0 and check:
        raise SystemExit(f"Command failed: {cmd}")


if not os.path.exists(WORKDIR):
    run(f"git clone --branch {GITHUB_BRANCH} --depth 1 "
        f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}.git {WORKDIR}")
else:
    run(f"cd {WORKDIR} && git fetch --depth 1 origin {GITHUB_BRANCH} && "
        f"git reset --hard origin/{GITHUB_BRANCH}")

# Record the exact SHA we're running for the paper
sha = subprocess.check_output(
    f"cd {WORKDIR} && git rev-parse HEAD", shell=True, text=True
).strip()
print(f"Running git SHA: {sha}")

# %%
# Install Python deps. Kaggle images already have torch + torchvision, so we
# skip those to avoid version churn.
run(f"pip install --quiet -q "
    f"'timm>=1.0.11' 'albumentations>=1.4.18' 'omegaconf>=2.3.0' "
    f"'wandb>=0.18.0' 'python-dotenv>=1.0.1' 'netcal>=1.3.5'")

# Make src importable
sys.path.insert(0, WORKDIR)

# %% [markdown]
# ## 3. Wire up secrets (W&B optional)

# %%
try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")
    print("W&B key loaded from Kaggle Secrets.")
except Exception as e:
    print(f"No WANDB_API_KEY secret found ({e}). Training will run without W&B logging.")

os.environ["WANDB_PROJECT"] = "retinal-selective-prediction"

# %% [markdown]
# ## 4. Prepare stratified splits

# %%
run(f"cd {WORKDIR} && python scripts/prepare_aptos_splits.py "
    f"--train-csv {APTOS_INPUT_DIR}/train.csv "
    f"--output-dir {SPLITS_DIR}")

# %% [markdown]
# ## 5. Train

# %%
no_wandb_flag = "" if os.environ.get("WANDB_API_KEY") else "--no-wandb"

run(f"cd {WORKDIR} && python scripts/train.py "
    f"--config configs/baseline.yaml "
    f"--splits-dir {SPLITS_DIR} "
    f"--images-dir {APTOS_INPUT_DIR}/train_images "
    f"{no_wandb_flag}")

# %% [markdown]
# ## 6. Persist the checkpoint as a Kaggle output

# %%
import shutil

best_ckpt = f"{WORKDIR}/experiments/runs/baseline_efficientnet_b0_aptos/best.pt"
config_snapshot = f"{WORKDIR}/experiments/runs/baseline_efficientnet_b0_aptos/config.yaml"

os.makedirs(OUTPUT_DIR, exist_ok=True)
if os.path.exists(best_ckpt):
    shutil.copy(best_ckpt, f"{OUTPUT_DIR}/best.pt")
    shutil.copy(config_snapshot, f"{OUTPUT_DIR}/config.yaml")
    # Write the SHA alongside the checkpoint for traceability
    with open(f"{OUTPUT_DIR}/git_sha.txt", "w") as f:
        f.write(sha + "\n")
    print(f"Saved checkpoint + config + SHA to {OUTPUT_DIR}")
else:
    print(f"WARNING: no checkpoint at {best_ckpt}")

# %% [markdown]
# ## 7. Uncertainty methods on the trained checkpoint

# %%
RUN_DIR = f"{WORKDIR}/experiments/runs/baseline_efficientnet_b0_aptos"
IMAGES_DIR = f"{APTOS_INPUT_DIR}/train_images"

# Deterministic evaluation: kappa, ECE, NLL, Brier, reliability diagram
run(f"cd {WORKDIR} && python scripts/evaluate.py "
    f"--run-dir {RUN_DIR} "
    f"--splits-dir {SPLITS_DIR} "
    f"--images-dir {IMAGES_DIR}")

# Temperature scaling: post-hoc calibration on val, applied to test
run(f"cd {WORKDIR} && python scripts/run_temperature_scaling.py "
    f"--run-dir {RUN_DIR} "
    f"--splits-dir {SPLITS_DIR} "
    f"--images-dir {IMAGES_DIR}")

# MC Dropout: T=30 stochastic forward passes
run(f"cd {WORKDIR} && python scripts/run_mc_dropout.py "
    f"--run-dir {RUN_DIR} "
    f"--splits-dir {SPLITS_DIR} "
    f"--images-dir {IMAGES_DIR} "
    f"--n-samples 30")

# Conformal prediction: split + APS, alpha=0.10 (target 90% coverage)
run(f"cd {WORKDIR} && python scripts/run_conformal.py "
    f"--run-dir {RUN_DIR} "
    f"--splits-dir {SPLITS_DIR} "
    f"--images-dir {IMAGES_DIR} "
    f"--alpha 0.10")

# %% [markdown]
# ## 8. Copy all uncertainty outputs to /kaggle/working/outputs/

# %%
for fname in [
    "metrics.json", "predictions.csv",
    "confusion_matrix.png", "reliability_diagram.png",
    "temperature_scaling.json", "temperature_scaling_predictions.csv",
    "temperature_scaling_selective.json",
    "mc_dropout_predictions.csv", "mc_dropout_selective.json",
    "conformal_predictions.csv", "conformal_summary.json",
]:
    src = f"{RUN_DIR}/{fname}"
    if os.path.exists(src):
        shutil.copy(src, f"{OUTPUT_DIR}/{fname}")
        print(f"saved {fname}")
print("All uncertainty outputs copied to /kaggle/working/outputs/")
