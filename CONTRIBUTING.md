# Contributing

Thanks for your interest. This is an active research project, so the contribution surface is broader than a typical library — fixes, new uncertainty methods, new datasets, and reproduction reports are all valuable.

## Development setup

```bash
git clone https://github.com/ShahnawazKakarh/retinal-selective-prediction.git
cd retinal-selective-prediction

python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

pre-commit install            # gitleaks + ruff + standard hygiene hooks
pre-commit run --all-files    # verify everything is clean before first commit
```

## Code style

- **Formatter / linter:** [ruff](https://github.com/astral-sh/ruff). Configured in `pyproject.toml`. Run `ruff format .` and `ruff check . --fix` before committing — or just let pre-commit do it.
- **Line length:** 100. Strict on logic, lenient on string literals.
- **Type hints:** required on every public function. We use `from __future__ import annotations` so unions like `int | None` work on 3.10+.
- **Docstrings:** Google or NumPy style. The first sentence is what reviewers will read — make it count.

## Tests

```bash
pytest -q
```

Current test suite covers the OACSP method (`tests/test_class_conditional.py`, 9 tests) plus any other unit tests under `tests/`. All tests are designed to run on CPU without the trained checkpoint — they use synthetic data with known structure.

## Project versions

This project uses semantic versioning aligned with Zenodo releases.

| Version | Status | Scope |
|---|---|---|
| v1.0.0 | ✅ Released | Reproducible four-method benchmark on APTOS 2019 |
| v1.1.0 | ✅ Released | + OACSP novel post-hoc abstention rule |
| v1.2.0 | 🟡 In progress | + multi-seed variance + IDRiD external validation + paper figures |
| v2.0.0 | 🔵 Planned | + Deep Ensembles + Evidential DL + Messidor-2 + IEEE J-BHI submission |

If you are contributing a new feature, please target the next minor version's branch (currently `v1.2.0-*`). For bug fixes against a released version, target `master` and we'll cherry-pick.

## Regenerating paper figures

All publication figures are produced from saved JSON/CSV outputs — no GPU needed:

```bash
python scripts/generate_paper_figures.py \
    --run-dir experiments/runs/baseline_efficientnet_b0_aptos \
    --output-dir report/figures
```

Figures are saved at 300 DPI as PNG and committed under `report/figures/`. The matplotlib style (fonts, palette, axes) is defined in `setup_style()` at the top of the script.

Tests must pass before a PR is merged. New code should ship with at least a smoke test under `tests/`. We do **not** require unit-test coverage of the training loop itself; we do require that data loaders, splits, and metrics are tested deterministically.

## Adding a new uncertainty method

Drop a file under `src/uncertainty/` exposing one public function:

```python
def my_method_predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    **method_specific_kwargs,
) -> dict[str, np.ndarray]:
    """Return:
       probs:     (N, C)
       labels:    (N,)
       id_codes:  list[str]
       <method-specific uncertainty arrays>
    """
```

Then add a thin entry point under `scripts/run_<method>.py` that:

1. Loads a checkpoint from a baseline run directory.
2. Runs your method on the internal test set.
3. Writes `<method>_predictions.csv` and `<method>_selective.json` next to the checkpoint, using `src/selective/risk_coverage.py` for the AURC / selective-accuracy summary.

The selective-prediction module is method-agnostic on purpose — any uncertainty signal that's "higher = more uncertain" plugs straight in.

## Adding a new dataset

1. Loader under `src/data/<dataset>.py` exposing a `Dataset` class with the same interface as `APTOS2019Dataset` (returns `{"image", "label", "id_code"}`).
2. If the dataset has its own splits (Messidor-2, IDRiD do not have official splits for our 5-class setup), document the split rule in `docs/datasets.md`.
3. External validation only — do **not** retrain on external datasets. The whole point of external validation is to freeze the model and ask "does it generalize?"

## Secrets

The pre-commit hook runs `gitleaks` on every commit. If you stage a file containing what looks like an API key, the commit will fail. **This is intentional.** If you genuinely need to commit a string that triggers the scanner (e.g., an example with the literal word "key="), add a `# gitleaks:allow` comment on that line.

Never commit:
- `.env` (gitignored)
- `kaggle.json` (gitignored by filename)
- `*.pt` / `*.ckpt` model weights (gitignored)
- `data/raw/` or `data/processed/` (gitignored)

If a credential ever lands in a commit, **rotate it immediately**, then rewrite history with `git-filter-repo` or BFG. Don't just delete the file in a follow-up commit — the credential stays in the git history.

## Commit messages

Short, lowercase, present tense, prefix-style:

```
data: add Messidor-2 loader
fix: handle empty MC dropout batch
docs: clarify external-validation policy
```

No need for Conventional Commits formality, but the prefix helps when scanning history.

## Pull requests

For anything beyond a small fix:

1. Open an issue first describing the change.
2. Branch from `master`. Branch names: `feat/<thing>`, `fix/<thing>`, `docs/<thing>`.
3. PR title mirrors the commit-message style.
4. PR description: what changed, why, and how it was tested. If results changed, attach the relevant rows of `metrics.json` or `*_selective.json`.

## Reproducing results

If you reproduce a result in this repo on different hardware, please open an issue with:

- Hardware (GPU model + count, CUDA version)
- Software (`pip freeze | grep -E "torch|timm|albumentations"`)
- Git SHA of the run
- The `metrics.json` and/or `*_selective.json` you obtained

Reproductions — even ones that *don't* match exactly — are scientifically valuable. They're how we find out whether a result is robust or an artifact.
