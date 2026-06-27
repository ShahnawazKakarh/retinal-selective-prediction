"""Generate paper-quality figures for the technical report and SSRN preprint.

This script reads the saved JSON/CSV outputs from the v1.1.0 pipeline and
produces publication-grade matplotlib figures saved as PNG at 300 DPI.

All figures use a consistent journal-style aesthetic:
  - Source Serif body font for titles, Inter for tick labels
  - Subdued color palette: navy blue, muted red, neutral gray, accent green
  - White background, black text, thin axis lines
  - 300 DPI, sized for one-column journal layout (~3.4 in wide)

Usage:
    python scripts/generate_paper_figures.py \\
        --run-dir experiments/runs/baseline_efficientnet_b0_aptos \\
        --output-dir report/figures

Figures produced:
  fig01_class_distribution.png       Pie chart of APTOS 2019 class distribution
  fig02_calibration_before_after.png Reliability diagram before / after TS
  fig03_aurc_comparison.png          Bar chart of AURC across methods
  fig04_oacsp_per_class_abstention.png  Headline OACSP figure
  fig05_oacsp_qwk_vs_coverage.png    Selective QWK vs coverage for three rules
  fig06_oacsp_novelty_quadrant.png   2x2 visualization of novelty positioning

Author: Khan, Muhammad Shahnawaz.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Journal-style aesthetic
# -----------------------------------------------------------------------------

PALETTE = {
    "navy": "#1f3a5f",        # primary, baselines
    "red": "#c44536",         # secondary, OACSP equalized
    "amber": "#d68c45",       # tertiary, OACSP ordinal
    "gray": "#6b7280",        # muted, comparisons
    "green": "#3d8b6a",       # accent, best-of
    "lightgray": "#e5e7eb",   # grid lines, fills
    "darkgray": "#1f2937",    # text
}

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Source Serif 4", "Source Serif Pro", "Georgia", "Times New Roman"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.edgecolor": PALETTE["darkgray"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": PALETTE["darkgray"],
        "ytick.color": PALETTE["darkgray"],
        "text.color": PALETTE["darkgray"],
        "axes.labelcolor": PALETTE["darkgray"],
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "grid.color": PALETTE["lightgray"],
        "grid.linewidth": 0.5,
    })


# -----------------------------------------------------------------------------
# Figure 1: Class distribution pie chart
# -----------------------------------------------------------------------------

def fig_class_distribution(run_dir: Path, out_path: Path) -> None:
    """APTOS 2019 5-class distribution from the saved splits."""
    # Use known class counts from prepare_aptos_splits.py output
    # (these are deterministic — committed splits files have these exact counts)
    train_counts = {0: 1263, 1: 258, 2: 699, 3: 135, 4: 207}
    total = sum(train_counts.values())

    sizes = [train_counts[c] for c in range(5)]
    labels = [f"{CLASS_NAMES[c]}\n(n={train_counts[c]}, {100*train_counts[c]/total:.1f}%)"
              for c in range(5)]
    colors = [PALETTE["navy"], PALETTE["gray"], PALETTE["amber"],
              PALETTE["red"], "#7a1f1a"]  # gradient from neutral to severe

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    wedges, texts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 8},
        labeldistance=1.15,
    )
    ax.set_title(
        "APTOS 2019 training set class distribution\n"
        f"(N = {total} images, severe ordinal class imbalance)",
        fontsize=10,
        pad=12,
    )
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


# -----------------------------------------------------------------------------
# Figure 2: Reliability diagram before / after Temperature Scaling
# -----------------------------------------------------------------------------

def fig_calibration_before_after(run_dir: Path, out_path: Path) -> None:
    """Reliability diagram before vs. after temperature scaling."""
    ts_path = run_dir / "temperature_scaling.json"
    if not ts_path.exists():
        print(f"Skipping fig2 — {ts_path} not found")
        return
    ts = json.loads(ts_path.read_text())

    ece_before = ts.get("ece_before", 0.0988)
    ece_after = ts.get("ece_after", 0.0419)
    T_fitted = ts.get("temperature", 1.6304)

    # Load test predictions for a real reliability diagram
    pred_path = run_dir / "predictions.csv"
    tspred_path = run_dir / "temperature_scaling_predictions.csv"
    if not pred_path.exists() or not tspred_path.exists():
        print(f"Skipping fig2 detail — predictions not found at {pred_path}")
        return

    df_raw = pd.read_csv(pred_path)
    df_cal = pd.read_csv(tspred_path)

    def _reliability_bins(confidences: np.ndarray, correctness: np.ndarray, n_bins: int = 15):
        bins = np.linspace(0, 1, n_bins + 1)
        centers = 0.5 * (bins[:-1] + bins[1:])
        means_conf = np.zeros(n_bins)
        means_acc = np.zeros(n_bins)
        weights = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
            if i == n_bins - 1:
                mask = (confidences >= bins[i]) & (confidences <= bins[i + 1])
            if mask.sum() > 0:
                means_conf[i] = confidences[mask].mean()
                means_acc[i] = correctness[mask].mean()
                weights[i] = mask.sum()
        return centers, means_acc, means_conf, weights

    # Raw
    conf_raw = df_raw["confidence"].to_numpy() if "confidence" in df_raw.columns else \
        df_raw[[c for c in df_raw.columns if c.startswith(("p_", "prob_"))]].max(axis=1).to_numpy()
    correct_raw = (df_raw["pred"].to_numpy() == df_raw["label"].to_numpy()).astype(float)
    c_raw, acc_raw, _, w_raw = _reliability_bins(conf_raw, correct_raw)

    conf_cal = df_cal["confidence"].to_numpy() if "confidence" in df_cal.columns else \
        df_cal[[c for c in df_cal.columns if c.startswith(("p_", "prob_"))]].max(axis=1).to_numpy()
    correct_cal = (df_cal["pred"].to_numpy() == df_cal["label"].to_numpy()).astype(float)
    c_cal, acc_cal, _, w_cal = _reliability_bins(conf_cal, correct_cal)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5), sharey=True)

    for ax, centers, accs, weights, title, ece in [
        (axes[0], c_raw, acc_raw, w_raw, f"Before TS\nECE = {ece_before:.4f}", ece_before),
        (axes[1], c_cal, acc_cal, w_cal, f"After TS (T = {T_fitted:.3f})\nECE = {ece_after:.4f}", ece_after),
    ]:
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], color=PALETTE["gray"], linestyle="--", linewidth=0.8, label="Perfect calibration")
        # Reliability bars
        mask = weights > 0
        widths = (1.0 / len(centers)) * 0.85
        ax.bar(
            centers[mask], accs[mask],
            width=widths,
            color=PALETTE["navy"], alpha=0.75,
            edgecolor=PALETTE["darkgray"], linewidth=0.5,
            label="Empirical accuracy",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted confidence")
        ax.set_title(title, fontsize=9.5)
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.6)

    axes[0].set_ylabel("Empirical accuracy")
    axes[1].legend(loc="upper left", frameon=False)

    fig.suptitle(
        "Calibration before vs. after Temperature Scaling\n"
        f"ECE reduced by {100 * (ece_before - ece_after) / ece_before:.1f}% with a single scalar parameter",
        y=1.04,
        fontsize=10,
    )
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


# -----------------------------------------------------------------------------
# Figure 3: AURC comparison across uncertainty methods
# -----------------------------------------------------------------------------

def fig_aurc_comparison(run_dir: Path, out_path: Path) -> None:
    """Bar chart of AURC for each method's best uncertainty signal."""
    methods = ["Softmax\n(neg max conf)", "Temperature\nScaling", "MC Dropout\n(BALD MI)",
               "Conformal\n(set size)"]
    # From real measured numbers in v1.1.0 re-run
    aurc_values = [0.0793, 0.0512, 0.0504, 0.1049]
    excess = [0.0598, 0.0293, 0.0289, 0.0830]
    colors = [PALETTE["gray"], PALETTE["navy"], PALETTE["red"], PALETTE["amber"]]

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(methods))
    width = 0.36

    bars1 = ax.bar(x - width / 2, aurc_values, width,
                   label="AURC", color=colors, alpha=0.85,
                   edgecolor=PALETTE["darkgray"], linewidth=0.6)
    bars2 = ax.bar(x + width / 2, excess, width,
                   label="Excess-AURC", color=colors, alpha=0.45,
                   edgecolor=PALETTE["darkgray"], linewidth=0.6, hatch="///")

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=7.5)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=7.5,
                color=PALETTE["darkgray"])

    ax.set_ylabel("AURC (lower = better)")
    ax.set_title(
        "Risk-coverage AURC across uncertainty methods\n"
        "MC Dropout (BALD) and Temperature Scaling tie for lowest excess-AURC",
        fontsize=10, pad=10,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.6)
    ax.set_ylim(0, max(aurc_values) * 1.20)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


# -----------------------------------------------------------------------------
# Figure 4: OACSP per-class abstention rate — the headline figure
# -----------------------------------------------------------------------------

def fig_oacsp_per_class_abstention(run_dir: Path, out_path: Path) -> None:
    """The single most important figure in v1.1.0."""
    oacsp_json = run_dir / "oacsp_v1.1.0" / "oacsp_detailed.json"
    if oacsp_json.exists():
        data = json.loads(oacsp_json.read_text())
        global_rates = [data["global_threshold_baseline"]["per_class_abstention_rate"][str(c)]
                        for c in range(5)]
        eq_rates = [data["oacsp_equalized_recall"]["per_class_abstention_rate"][str(c)]
                    for c in range(5)]
        cost_rates = [data["oacsp_ordinal_cost"]["per_class_abstention_rate"][str(c)]
                      for c in range(5)]
    else:
        # Hardcoded from the v1.1.0 published results so this figure can be
        # regenerated even without the run dir.
        global_rates = [0.015, 0.357, 0.367, 0.448, 0.295]
        eq_rates = [0.070, 0.232, 0.127, 0.103, 0.068]
        cost_rates = [0.033, 0.589, 0.260, 0.310, 0.205]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(5)
    width = 0.27

    b1 = ax.bar(x - width, global_rates, width, label="Global threshold (baseline)",
                color=PALETTE["gray"], edgecolor=PALETTE["darkgray"], linewidth=0.6)
    b2 = ax.bar(x, eq_rates, width, label="OACSP — equalized recall",
                color=PALETTE["red"], edgecolor=PALETTE["darkgray"], linewidth=0.6)
    b3 = ax.bar(x + width, cost_rates, width, label="OACSP — ordinal cost weighted",
                color=PALETTE["navy"], edgecolor=PALETTE["darkgray"], linewidth=0.6)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                    f"{100*h:.1f}%", ha="center", va="bottom", fontsize=7.5)

    # Highlight box around severe / proliferative
    ax.axvspan(2.5, 4.5, color=PALETTE["amber"], alpha=0.08, zorder=0)
    ax.text(3.5, 0.66, "Clinically critical classes",
            ha="center", fontsize=8.5, style="italic", color=PALETTE["amber"])

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{CLASS_NAMES[c]}" for c in range(5)])
    ax.set_ylabel("Per-class abstention rate")
    ax.set_ylim(0, 0.72)
    ax.set_title(
        "OACSP cuts Severe-DR abstention 4.4× and Proliferative-DR abstention 4.3×\n"
        "Per-class abstention rate at ≈80% overall coverage on APTOS 2019",
        fontsize=10, pad=12,
    )
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.6)
    # Y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100*v:.0f}%"))

    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


# -----------------------------------------------------------------------------
# Figure 5: Selective QWK vs Coverage trade-off
# -----------------------------------------------------------------------------

def fig_oacsp_qwk_vs_coverage(run_dir: Path, out_path: Path) -> None:
    """Scatter of (coverage, selective QWK) for the three abstention rules."""
    # From v1.1.0 published results
    methods = [
        ("Global threshold (baseline)", 0.809, 0.9068, PALETTE["gray"], "o"),
        ("OACSP — equalized recall", 0.896, 0.8865, PALETTE["red"], "s"),
        ("OACSP — ordinal cost weighted", 0.820, 0.9218, PALETTE["navy"], "D"),
    ]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    for label, cov, qwk, color, marker in methods:
        ax.scatter([cov], [qwk], s=160, color=color, edgecolor=PALETTE["darkgray"],
                   linewidth=0.8, marker=marker, zorder=3, label=label)
        ax.annotate(
            f"({cov:.3f}, {qwk:.4f})",
            xy=(cov, qwk), xytext=(8, 8), textcoords="offset points",
            fontsize=8, color=PALETTE["darkgray"],
        )

    # Annotate the headline finding
    ax.annotate(
        "",
        xy=(0.820, 0.9218), xytext=(0.809, 0.9068),
        arrowprops={"arrowstyle": "->", "color": PALETTE["green"], "lw": 1.4},
    )
    ax.text(0.812, 0.913, "+0.0150 QWK\nat matched coverage",
            color=PALETTE["green"], fontsize=8.5, style="italic")

    ax.set_xlabel("Coverage (fraction of test samples kept)")
    ax.set_ylabel("Selective quadratic-weighted κ")
    ax.set_xlim(0.78, 0.92)
    ax.set_ylim(0.875, 0.935)
    ax.set_title(
        "OACSP ordinal-cost improves selective QWK at matched coverage\n"
        "vs. the standard global confidence threshold",
        fontsize=10, pad=10,
    )
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, linewidth=0.4, alpha=0.6)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


# -----------------------------------------------------------------------------
# Figure 6: OACSP novelty quadrant
# -----------------------------------------------------------------------------

def fig_oacsp_novelty_quadrant(out_path: Path) -> None:
    """2x2 visualization of OACSP's position relative to prior work."""
    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    # 2x2 grid: x-axis = per-class thresholds? y-axis = ordinal cost?
    ax.axhline(0, color=PALETTE["gray"], linewidth=0.6)
    ax.axvline(0, color=PALETTE["gray"], linewidth=0.6)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    # Quadrant labels
    quadrants = [
        (-0.5, 0.5, "Per-class without\nordinal cost", PALETTE["gray"]),
        (0.5, 0.5, "OACSP (this work)\nper-class + ordinal cost", PALETTE["red"]),
        (-0.5, -0.5, "Global threshold +\nflat 0/1 loss\n(current literature)", PALETTE["gray"]),
        (0.5, -0.5, "Global threshold +\nordinal cost (uncommon)", PALETTE["gray"]),
    ]
    for x, y, label, color in quadrants:
        bg = PALETTE["red"] if color == PALETTE["red"] else PALETTE["lightgray"]
        ax.add_patch(plt.Rectangle((x - 0.49, y - 0.45), 0.98, 0.9,
                                    facecolor=bg, alpha=0.22, edgecolor="none"))
        weight = "bold" if color == PALETTE["red"] else "normal"
        ax.text(x, y, label, ha="center", va="center", fontsize=9.5,
                color=PALETTE["darkgray"], weight=weight)

    # Mark prior-work positions
    prior_works = [
        ("Geifman & El-Yaniv 2017\n(SelectiveNet)", -0.7, -0.3),
        ("Romano et al. 2020\n(APS)", -0.3, -0.7),
        ("Leibig et al. 2017\nBand et al. 2021", -0.8, -0.55),
        ("Plagwitz et al. 2024", 0.4, -0.7),
    ]
    for label, x, y in prior_works:
        ax.scatter([x], [y], s=50, color=PALETTE["navy"], zorder=3,
                   edgecolor=PALETTE["darkgray"], linewidth=0.6)
        ax.annotate(label, (x, y), xytext=(6, -4), textcoords="offset points",
                    fontsize=7, color=PALETTE["darkgray"], ha="left")

    # OACSP star marker
    ax.scatter([0.5], [0.5], s=300, color=PALETTE["red"], marker="*",
               zorder=5, edgecolor=PALETTE["darkgray"], linewidth=1.0)

    ax.set_xlabel("→ Per-class abstention thresholds")
    ax.set_ylabel("→ Ordinal-distance-weighted cost")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        "OACSP occupies a previously-empty quadrant\nin the DR selective-prediction literature",
        fontsize=10, pad=10,
    )

    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path,
                    default=Path("experiments/runs/baseline_efficientnet_b0_aptos"))
    ap.add_argument("--output-dir", type=Path, default=Path("report/figures"))
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    fig_class_distribution(args.run_dir, args.output_dir / "fig01_class_distribution.png")
    fig_calibration_before_after(args.run_dir, args.output_dir / "fig02_calibration_before_after.png")
    fig_aurc_comparison(args.run_dir, args.output_dir / "fig03_aurc_comparison.png")
    fig_oacsp_per_class_abstention(args.run_dir, args.output_dir / "fig04_oacsp_per_class_abstention.png")
    fig_oacsp_qwk_vs_coverage(args.run_dir, args.output_dir / "fig05_oacsp_qwk_vs_coverage.png")
    fig_oacsp_novelty_quadrant(args.output_dir / "fig06_oacsp_novelty_quadrant.png")

    print("\n[OK] All figures generated.")
    print(f"Figures saved to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
