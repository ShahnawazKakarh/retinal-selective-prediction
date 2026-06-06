"""Reliability diagram (calibration plot) — saved as PNG."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.metrics import reliability_curve


def plot_reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
    *,
    n_bins: int = 15,
    title: str = "Reliability diagram",
) -> None:
    """Standard reliability diagram: confidence vs. accuracy per bin.

    Bars show accuracy; the dashed y=x line is perfect calibration.
    """
    curve = reliability_curve(probs, labels, n_bins=n_bins)
    centers = curve["bin_centers"]
    acc = curve["bin_accuracy"]
    counts = curve["bin_count"]

    width = 1.0 / n_bins
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(5, 6),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )

    ax1.bar(centers, np.nan_to_num(acc, nan=0.0), width=width * 0.95,
            edgecolor="black", alpha=0.7, label="Accuracy")
    ax1.plot([0, 1], [0, 1], linestyle="--", color="black",
             linewidth=1, label="Perfect")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Accuracy")
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2.bar(centers, counts, width=width * 0.95, edgecolor="black", alpha=0.7)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Count")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
