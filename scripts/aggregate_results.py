"""Aggregate all uncertainty-method results into one paper-ready table.

Scans a directory for `*_selective.json` files written by the various
scripts/run_*.py entry points, plus the baseline `metrics.json`, and emits:

  results/aggregate.csv         — long-form table (one row per method × signal)
  results/aggregate.md          — pretty Markdown for paper / README
  results/aggregate.json        — machine-readable bundle

Usage:
    python scripts/aggregate_results.py \
        --run-dir experiments/runs/baseline_efficientnet_b0_aptos \
        --extra-dirs experiments/runs/ensemble_aggregate \
                     experiments/runs/evidential_efficientnet_b0_aptos \
        --out-dir results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# Maps a JSON filename → method label for the results table.
METHOD_FILES: dict[str, str] = {
    "metrics.json": "Softmax (deterministic)",
    "temperature_scaling_selective.json": "Temperature Scaling",
    "mc_dropout_selective.json": "MC Dropout",
    "ensemble_selective.json": "Deep Ensembles",
    "evidential_selective.json": "Evidential DL",
    "conformal_summary.json": "Conformal",
}


def _row(method: str, signal: str, payload: dict) -> dict:
    """Pull the standard fields out of any selective.json entry."""
    sel_acc = payload.get("selective_accuracy", {})
    row = {
        "method": method,
        "signal": signal,
        "aurc": payload.get("aurc"),
        "excess_aurc": payload.get("excess_aurc"),
    }
    for c in (70, 80, 90, 95):
        key = f"coverage_{c}"
        if key in sel_acc:
            row[f"sel_acc_{c}"] = sel_acc[key].get("selective_accuracy")
    return row


def _read_selective_json(path: Path, method_label: str) -> list[dict]:
    """Read *_selective.json files (MC dropout, ensembles, evidential)."""
    with open(path) as f:
        data = json.load(f)
    rows = []
    overall_acc = data.get("overall_accuracy")
    for signal, payload in data.get("signals", {}).items():
        row = _row(method_label, signal, payload)
        row["accuracy"] = overall_acc
        rows.append(row)
    return rows


def _read_conformal_summary(path: Path) -> list[dict]:
    """Conformal has a slightly different shape — flatten it."""
    with open(path) as f:
        data = json.load(f)
    rows = []
    acc = data.get("top1_accuracy")
    for signal, payload in data.get("signals", {}).items():
        row = _row("Conformal", signal, payload)
        row["accuracy"] = acc
        rows.append(row)
    # Also surface the conformal-specific quantities as their own pseudo-rows
    for kind in ("split_conformal", "aps"):
        if kind in data:
            d = data[kind]
            rows.append({
                "method": f"Conformal — {kind}",
                "signal": "(coverage guarantee)",
                "aurc": None,
                "excess_aurc": None,
                "accuracy": acc,
                "empirical_coverage": d.get("empirical_coverage"),
                "avg_set_size": d.get("avg_set_size"),
                "singleton_fraction": d.get("singleton_fraction"),
            })
    return rows


def _read_baseline_metrics(path: Path) -> list[dict]:
    """The deterministic baseline's metrics.json has a flat shape."""
    with open(path) as f:
        data = json.load(f)
    return [{
        "method": "Softmax (deterministic)",
        "signal": "(test set)",
        "accuracy": data.get("accuracy"),
        "kappa_quadratic": data.get("kappa_quadratic"),
        "ece_15bins": data.get("ece_15bins"),
        "nll": data.get("nll"),
        "brier": data.get("brier"),
    }]


def collect(run_dirs: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for d in run_dirs:
        if not d.exists():
            print(f"  skip (missing): {d}")
            continue
        for fname, label in METHOD_FILES.items():
            path = d / fname
            if not path.exists():
                continue
            print(f"  read: {path}")
            if fname == "metrics.json":
                rows.extend(_read_baseline_metrics(path))
            elif fname == "conformal_summary.json":
                rows.extend(_read_conformal_summary(path))
            else:
                rows.extend(_read_selective_json(path, label))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path,
                    help="Primary baseline run directory.")
    ap.add_argument("--extra-dirs", nargs="*", default=[], type=Path,
                    help="Additional run directories (ensembles, evidential, etc.).")
    ap.add_argument("--out-dir", default=Path("results"), type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect([args.run_dir, *args.extra_dirs])
    if not rows:
        raise SystemExit("No results found. Run experiments first.")

    df = pd.DataFrame(rows)

    # Column order for the human-facing tables
    preferred = [
        "method", "signal", "accuracy", "kappa_quadratic",
        "ece_15bins", "nll", "brier",
        "aurc", "excess_aurc",
        "sel_acc_70", "sel_acc_80", "sel_acc_90", "sel_acc_95",
        "empirical_coverage", "avg_set_size", "singleton_fraction",
    ]
    ordered = [c for c in preferred if c in df.columns] + \
              [c for c in df.columns if c not in preferred]
    df = df[ordered]

    df.to_csv(args.out_dir / "aggregate.csv", index=False)
    with open(args.out_dir / "aggregate.json", "w") as f:
        json.dump(rows, f, indent=2)

    # Markdown — round floats to 4dp, blank out NaN
    df_md = df.copy()
    for col in df_md.select_dtypes(include="float").columns:
        df_md[col] = df_md[col].apply(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    md = df_md.to_markdown(index=False)
    (args.out_dir / "aggregate.md").write_text(
        "# Aggregate results\n\n"
        "Auto-generated by `scripts/aggregate_results.py`. Do not hand-edit.\n\n"
        + md + "\n"
    )

    print(f"\nWrote: {args.out_dir / 'aggregate.csv'}")
    print(f"       {args.out_dir / 'aggregate.md'}")
    print(f"       {args.out_dir / 'aggregate.json'}")
    print(f"\n{len(rows)} rows across {df['method'].nunique()} methods.")


if __name__ == "__main__":
    main()
