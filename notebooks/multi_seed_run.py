# %% [markdown]
# # Multi-Seed Variance Run (v1.2.0)
#
# This notebook runs the full APTOS pipeline three times with seeds 7, 137, 42
# and aggregates the metrics into a mean ± SD table. This addresses the
# "single seed" limitation called out in `report/report.html` §8.
#
# **Total wall time on Kaggle T4: ~3.5 hours** (3 × 75 minutes each).
# You can run this in three sessions on different days if you don't want a
# single long session.
#
# ## Before running
# 1. Same setup as `notebooks/kaggle_train_baseline.py`: APTOS dataset attached,
#    T4 GPU, Internet ON.
# 2. (Optional) `WANDB_API_KEY` in Kaggle Secrets.
#
# ## Output
# Each seed produces its own run directory:
#   experiments/runs/baseline_efficientnet_b0_aptos_seed_42/
#   experiments/runs/baseline_efficientnet_b0_aptos_seed_7/
#   experiments/runs/baseline_efficientnet_b0_aptos_seed_137/
#
# The final cell aggregates the metrics across seeds and prints mean ± SD.

# %% [markdown]
# ## 1. Configuration

# %%
GITHUB_USER = "ShahnawazKakarh"
GITHUB_REPO = "retinal-selective-prediction"
GITHUB_BRANCH = "v1.2.0-prep"

SEEDS = [42, 7, 137]

APTOS_INPUT_DIR = "/kaggle/input/competitions/aptos2019-blindness-detection"
WORKDIR = f"/kaggle/working/{GITHUB_REPO}"
SPLITS_DIR_TEMPLATE = f"{WORKDIR}/data/splits/aptos2019_seed_{{seed}}"
RUN_DIR_TEMPLATE = f"{WORKDIR}/experiments/runs/baseline_efficientnet_b0_aptos_seed_{{seed}}"


# %% [markdown]
# ## 2. Clone repo + install dependencies

# %%
import os
import subprocess
import sys


def run(cmd: str, check: bool = True) -> None:
    print(f"$ {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True, check=False)
    if check and result.returncode != 0:
        raise SystemExit(f"Command failed (exit {result.returncode}): {cmd}")


os.chdir("/kaggle/working")
run(f"rm -rf {GITHUB_REPO}", check=False)
run(
    f"git clone --depth 1 --branch {GITHUB_BRANCH} "
    f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}.git"
)
os.chdir(WORKDIR)
sys.path.insert(0, WORKDIR)

git_sha = subprocess.check_output("git rev-parse HEAD", shell=True, text=True).strip()
print(f"\nRunning git SHA: {git_sha[:7]}")

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
except Exception:
    os.environ["WANDB_MODE"] = "offline"

run(
    "pip install --quiet -q "
    "'timm>=1.0.11' 'albumentations>=1.4.18' 'omegaconf>=2.3.0' "
    "'wandb>=0.18.0' 'python-dotenv>=1.0.1' 'netcal>=1.3.5'"
)


# %% [markdown]
# ## 3. Run pipeline for each seed

# %%
import yaml
from pathlib import Path

# Read the baseline config so we can write a per-seed variant
with open(f"{WORKDIR}/configs/baseline.yaml") as fh:
    base_cfg = yaml.safe_load(fh)


for seed in SEEDS:
    print("\n" + "=" * 78)
    print(f"SEED {seed}")
    print("=" * 78 + "\n")

    splits_dir = SPLITS_DIR_TEMPLATE.format(seed=seed)
    run_dir = RUN_DIR_TEMPLATE.format(seed=seed)

    # Write per-seed config (just changes experiment.seed and the run_name)
    cfg = dict(base_cfg)
    cfg["experiment"] = {**cfg.get("experiment", {}), "seed": seed}
    cfg["experiment"]["name"] = f"baseline_efficientnet_b0_aptos_seed_{seed}"
    cfg_path = f"{WORKDIR}/configs/baseline_seed_{seed}.yaml"
    with open(cfg_path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    # Prepare splits (different seed -> different stratified shuffle)
    run(
        f"python scripts/prepare_aptos_splits.py "
        f"--train-csv {APTOS_INPUT_DIR}/train.csv "
        f"--output-dir {splits_dir} "
        f"--seed {seed}"
    )

    # Train
    run(
        f"python scripts/train.py "
        f"--config {cfg_path} "
        f"--splits-dir {splits_dir} "
        f"--images-dir {APTOS_INPUT_DIR}/train_images"
    )

    # Evaluate + uncertainty methods + OACSP, all sharing the same run_dir
    for s in [
        f"python scripts/evaluate.py --run-dir {run_dir} --splits-dir {splits_dir} --images-dir {APTOS_INPUT_DIR}/train_images",
        f"python scripts/run_temperature_scaling.py --run-dir {run_dir} --splits-dir {splits_dir} --images-dir {APTOS_INPUT_DIR}/train_images",
        f"python scripts/run_mc_dropout.py --run-dir {run_dir} --splits-dir {splits_dir} --images-dir {APTOS_INPUT_DIR}/train_images --n-samples 30",
        f"python scripts/run_conformal.py --run-dir {run_dir} --splits-dir {splits_dir} --images-dir {APTOS_INPUT_DIR}/train_images --alpha 0.10",
        f"python scripts/run_oacsp_analysis.py "
        f"  --val-predictions  {run_dir}/temperature_scaling_val_predictions.csv "
        f"  --test-predictions {run_dir}/temperature_scaling_predictions.csv "
        f"  --target-coverage  0.80 "
        f"  --output-dir       {run_dir}/oacsp_v1.1.0",
    ]:
        run(s)


# %% [markdown]
# ## 4. Aggregate metrics across seeds

# %%
import json
import numpy as np
import pandas as pd

rows = []
for seed in SEEDS:
    run_dir = Path(RUN_DIR_TEMPLATE.format(seed=seed))
    m = json.loads((run_dir / "metrics.json").read_text())
    ts = json.loads((run_dir / "temperature_scaling.json").read_text())
    mc = json.loads((run_dir / "mc_dropout_selective.json").read_text())

    # Best MC dropout signal is BALD (mutual_information)
    bald = mc["signals"]["mutual_information"] if "signals" in mc else mc.get("mutual_information", {})
    rows.append({
        "seed": seed,
        "test_accuracy": m["accuracy"],
        "test_qwk": m["kappa_quadratic"],
        "ece_baseline": m["ece_15bins"],
        "ece_after_ts": ts.get("ece_after", float("nan")),
        "temperature_T": ts.get("temperature", float("nan")),
        "aurc_bald": bald.get("aurc", float("nan")),
        "excess_aurc_bald": bald.get("excess", float("nan")),
    })

df = pd.DataFrame(rows)
print("\nPer-seed metrics:")
print(df.to_string(index=False))

print("\nMean +/- SD across seeds:")
summary_cols = [c for c in df.columns if c != "seed"]
for c in summary_cols:
    mean = df[c].mean()
    std = df[c].std()
    print(f"  {c:<22} {mean:.4f}  +/- {std:.4f}")

# Save aggregated summary
out_path = Path(WORKDIR) / "experiments" / "runs" / "multi_seed_summary.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)
print(f"\nWrote {out_path}")

summary = {
    "git_sha": git_sha,
    "seeds": SEEDS,
    "per_seed": rows,
    "mean_std": {c: {"mean": float(df[c].mean()), "std": float(df[c].std())} for c in summary_cols},
}
(out_path.parent / "multi_seed_summary.json").write_text(json.dumps(summary, indent=2, default=float))
print(f"Wrote {out_path.parent / 'multi_seed_summary.json'}")
