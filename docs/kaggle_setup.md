# Kaggle Setup & Runbook

This is the end-to-end recipe for running this project's full pipeline on Kaggle's free T4 GPU. Save this somewhere safe; it lets you reproduce every published number from scratch in about 75 minutes.

---

## 1. Why Kaggle

We use Kaggle because it's free, has T4 GPUs, and ships with the APTOS 2019 dataset pre-mounted at `/kaggle/input/competitions/aptos2019-blindness-detection/`. Nothing about the project depends on Kaggle specifically — any machine with a CUDA GPU and the APTOS images can run the same scripts. The Kaggle notebook is just the most convenient runner.

The notebook **does not store anything that matters long-term**. All real state lives on GitHub:
- Source code: branches `master` and `v1.1.0-oacsp`
- Configuration: `configs/baseline.yaml`
- Numbers: `results/results.md`

If the Kaggle session dies, you just clone the repo again and re-run. The trained checkpoint and CSVs are **disposable** — they regenerate deterministically from the same seed.

---

## 2. One-time setup

### 2.1 Make a Kaggle account
Free, at [kaggle.com](https://www.kaggle.com). Verify your phone number — required for free GPU access.

### 2.2 Attach the APTOS 2019 competition
Accept the rules at [https://www.kaggle.com/competitions/aptos2019-blindness-detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection). After this, any notebook you create can attach the dataset with one click.

### 2.3 (Optional) Add W&B API key to Kaggle Secrets
- Go to **Add-ons → Secrets** in any Kaggle notebook
- Name: `WANDB_API_KEY`
- Value: from [wandb.ai/authorize](https://wandb.ai/authorize)

The pipeline runs fine without W&B (falls back to offline mode), but having the key means training runs get logged to your W&B account for later inspection.

### 2.4 Create a new notebook
[New Notebook](https://www.kaggle.com/code), then:
- Right sidebar → **Add Data** → search "aptos2019" → attach **APTOS 2019 Blindness Detection**
- Right sidebar → **Settings** → **Accelerator** → **GPU T4 x2** (or just T4 x1 — single T4 is fine, ~10% slower)
- Right sidebar → **Settings** → **Internet** → **On** (required for cloning the GitHub repo and pip install)

---

## 3. The pipeline (single cell, ~75 minutes)

Paste this as a single cell and run. It clones the repo at the branch you want, trains, evaluates, runs all uncertainty methods, and runs OACSP.

```python
import os, subprocess

# CHOOSE the branch to run:
#   master         — v1.0.0 frozen release
#   v1.1.0-oacsp   — current working branch with OACSP and val/test fix
BRANCH = "v1.1.0-oacsp"

os.chdir("/kaggle/working")
subprocess.run("rm -rf retinal-selective-prediction", shell=True, check=False)
subprocess.run(
    f"git clone --depth 1 --branch {BRANCH} "
    "https://github.com/ShahnawazKakarh/retinal-selective-prediction.git",
    shell=True, check=True
)
os.chdir("/kaggle/working/retinal-selective-prediction")
print("Git SHA:", subprocess.run("git rev-parse HEAD", shell=True, capture_output=True, text=True).stdout.strip())

# W&B key if available
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
    print("W&B key loaded")
except Exception:
    os.environ["WANDB_MODE"] = "offline"
    print("W&B offline mode")

# Pinned dependencies
!pip install --quiet -q 'timm>=1.0.11' 'albumentations>=1.4.18' \
    'omegaconf>=2.3.0' 'wandb>=0.18.0' 'python-dotenv>=1.0.1' 'netcal>=1.3.5'

# Pipeline
SPLITS = "/kaggle/working/retinal-selective-prediction/data/splits/aptos2019"
IMAGES = "/kaggle/input/competitions/aptos2019-blindness-detection/train_images"
TRAIN_CSV = "/kaggle/input/competitions/aptos2019-blindness-detection/train.csv"
RUN_DIR = "/kaggle/working/retinal-selective-prediction/experiments/runs/baseline_efficientnet_b0_aptos"

!python scripts/prepare_aptos_splits.py --train-csv {TRAIN_CSV} --output-dir {SPLITS}
!python scripts/train.py                --config configs/baseline.yaml --splits-dir {SPLITS} --images-dir {IMAGES}
!python scripts/evaluate.py             --run-dir {RUN_DIR} --splits-dir {SPLITS} --images-dir {IMAGES}
!python scripts/run_temperature_scaling.py --run-dir {RUN_DIR} --splits-dir {SPLITS} --images-dir {IMAGES}
!python scripts/run_mc_dropout.py         --run-dir {RUN_DIR} --splits-dir {SPLITS} --images-dir {IMAGES} --n-samples 30
!python scripts/run_conformal.py          --run-dir {RUN_DIR} --splits-dir {SPLITS} --images-dir {IMAGES} --alpha 0.10

# v1.1.0 novel piece: OACSP analysis (calibrated on the proper VAL set, evaluated on TEST)
!python scripts/run_oacsp_analysis.py \
    --val-predictions  {RUN_DIR}/temperature_scaling_val_predictions.csv \
    --test-predictions {RUN_DIR}/temperature_scaling_predictions.csv \
    --target-coverage  0.80 \
    --output-dir       {RUN_DIR}/oacsp_v1.1.0
```

### Timing breakdown (single T4)
| Step | Wall time |
|---|---|
| Clone + pip install | ~30 s |
| Prepare splits | ~3 s |
| Train | ~50 min (early-stops around epoch 15-20) |
| Evaluate (test + val) | ~30 s |
| Temperature Scaling | ~30 s |
| MC Dropout T=30 | ~12 min |
| Conformal split + APS | ~30 s |
| OACSP analysis | ~10 s |
| **Total** | **~75 min** |

After the cell finishes, the **HEADLINE CHECK** section at the bottom of the OACSP output is what gets pasted into `results/results.md`.

---

## 4. Re-running just the OACSP step (no retrain)

If you already have the trained checkpoint at `experiments/runs/baseline_efficientnet_b0_aptos/best.pt` and just want to re-run OACSP (e.g., to try different cost multipliers), use this much shorter cell:

```python
import os, subprocess
BRANCH = "v1.1.0-oacsp"
os.chdir("/kaggle/working/retinal-selective-prediction")
subprocess.run(f"git fetch origin {BRANCH} && git reset --hard origin/{BRANCH}",
               shell=True, check=True)

RUN_DIR = "/kaggle/working/retinal-selective-prediction/experiments/runs/baseline_efficientnet_b0_aptos"

# Optional: different cost multiplier
COST = '{"0": 1.0, "1": 1.0, "2": 1.0, "3": 5.0, "4": 5.0}'  # heavier penalty on severe
!python scripts/run_oacsp_analysis.py \
    --val-predictions  {RUN_DIR}/temperature_scaling_val_predictions.csv \
    --test-predictions {RUN_DIR}/temperature_scaling_predictions.csv \
    --target-coverage  0.80 \
    --cost-multiplier  '{COST}' \
    --output-dir       {RUN_DIR}/oacsp_severe_heavy
```

Runs in 10 seconds.

---

## 5. Multi-seed runs (for v1.1.0)

To estimate run-to-run variance, train 2 more times with different seeds. Edit `configs/baseline.yaml`, change `experiment.seed: 42` to `7` and re-run the training cell above. Repeat for seed `137`.

Each run takes ~50 minutes for training + ~15 minutes for uncertainty methods. Plan to spread it across a Saturday: run seed 7 in the morning, seed 137 in the afternoon. Save the outputs to differently-named run dirs (e.g., `baseline_efficientnet_b0_aptos_seed_7/`).

Better approach: a single cell that loops three seeds (provided in `notebooks/multi_seed_run.py`, planned).

---

## 6. Troubleshooting

### "ConcurrencyViolation: Sequence number must match" when saving the notebook
You have the notebook open in two tabs (sometimes hidden in a Pin tab or a duplicate window from yesterday's session). Close *all* notebook tabs, then re-open the notebook fresh. If it persists, **File → Make a Copy** to clone into a clean draft notebook.

### Session is wiped, `/kaggle/working/` is empty
Normal — Kaggle clears the working directory between sessions. Just re-run the single cell above; it clones the repo fresh and starts over. **The actual valuable artifacts (numbers, code) are on GitHub.** The Kaggle session is disposable.

### Training crashes early with "CUDA out of memory"
- Switch the accelerator to T4 x2 (more VRAM, slightly faster too)
- OR reduce batch size: edit `configs/baseline.yaml`, `data.batch_size: 32` → `16`

### W&B API key not found
Either add it to Kaggle Secrets (section 2.3) or just ignore — the pipeline falls back to `WANDB_MODE=offline` automatically.

### pip dependency conflicts during install
The warnings about `dask-cuda`, `cuml-cu12`, `numba` etc are harmless — they refer to RAPIDS packages that Kaggle pre-installs and that this project doesn't use. The pip install still succeeds for our actual deps.

### Numbers slightly different between runs even with same seed
Expected. CUDA non-determinism + library version drift between Kaggle session snapshots cause ~1-2% variation in metrics. Qualitative findings (Temperature Scaling halves ECE, MC Dropout BALD lowest AURC, OACSP severe-class protection) are stable.

---

## 7. Downloading outputs to your Mac

Generally you don't need to — the headline numbers are saved to `results/results.md` (committed to GitHub) and that's the single source of truth.

If you do need the raw CSVs (e.g. for offline analysis), Kaggle has a clunky download UI. Easier path: have the notebook upload them to a Kaggle Dataset, then download the dataset. Easiest path: print the CSV head to the notebook output and copy-paste.

```python
# Inspect a CSV from inside the notebook
import pandas as pd
df = pd.read_csv(f"{RUN_DIR}/oacsp_v1.1.0/oacsp_comparison.csv")
print(df.to_string())
```

---

## 8. Saving notebook versions

Kaggle's "Save Version" snapshots the notebook + its outputs. **This is not a backup** — the source of truth is GitHub.

Save Version is useful for: capturing the full output log of a training run so you can refer back to the metrics without re-running. We treat each Save Version as a *receipt* — handy for debugging, not a deliverable.

Recommended: hit Save Version once after each pipeline run. Don't worry if you forget — the metrics get persisted to `results/results.md` via the GitHub commit, which is the real receipt.

---

## 9. What goes back into GitHub

After each full pipeline run, the only file that needs to be updated in GitHub is `results/results.md` with the new headline numbers. The CSVs and PNG plots are gitignored by design (they're regeneratable artifacts, not source). The notebook itself (`notebooks/kaggle_train_baseline.py`) only changes when you change the *pipeline*, not when you re-run it.

Workflow:
1. Run the pipeline on Kaggle
2. Copy the "HEADLINE CHECK" output back to Claude (or paste into results.md directly)
3. Commit + push from your Mac

That's it. The Kaggle notebook is a stateless runner.
