"""Conformal prediction — distribution-free coverage guarantees.

Two methods implemented:
  1. SplitConformal — uses softmax-confidence as the conformity score
                      (classical Romano-Patterson-Candès style).
  2. APS (Adaptive Prediction Sets, Romano-Sesia-Candès 2020) —
     conformity score = cumulative softmax mass until the true class is
     covered. Gives smaller sets on average and conditional coverage.

Workflow:
  cp = APS(alpha=0.10).calibrate(val_probs, val_labels)
  pred_sets = cp.predict_sets(test_probs)    # list of np.ndarray of class indices
  set_sizes = cp.set_sizes(test_probs)       # (N,) integers
  coverage  = cp.empirical_coverage(test_probs, test_labels)  # ~ 0.90

For selective prediction, larger sets ⇒ more uncertainty — we use set size
as the abstention signal: abstain when |C(x)| > 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SplitConformal:
    """Split conformal with score s(x, y) = 1 - p_y(x).

    Calibration: q_hat = quantile of {s_i} at level ceil((n+1)(1-alpha))/n.
    Prediction set at test x: { y : 1 - p_y(x) <= q_hat }
                            = { y : p_y(x) >= 1 - q_hat }.
    """

    alpha: float = 0.10
    q_hat: float = field(default=np.nan, init=False)

    def calibrate(self, probs: np.ndarray, labels: np.ndarray) -> "SplitConformal":
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        n = len(labels)
        scores = 1.0 - probs[np.arange(n), labels]
        # Finite-sample correction
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = float(min(level, 1.0))
        self.q_hat = float(np.quantile(scores, level, method="higher"))
        return self

    def predict_sets(self, probs: np.ndarray) -> list[np.ndarray]:
        if np.isnan(self.q_hat):
            raise RuntimeError("Call calibrate() first.")
        threshold = 1.0 - self.q_hat
        return [np.where(p >= threshold)[0] for p in probs]

    def set_sizes(self, probs: np.ndarray) -> np.ndarray:
        return np.array([len(s) for s in self.predict_sets(probs)], dtype=np.int32)

    def empirical_coverage(self, probs: np.ndarray, labels: np.ndarray) -> float:
        sets = self.predict_sets(probs)
        hits = sum(int(y in s) for s, y in zip(sets, labels))
        return hits / len(labels)


@dataclass
class APS:
    """Adaptive Prediction Sets (Romano, Sesia, Candès 2020).

    Conformity score: for true class y, the score is the sum of the largest
    softmax masses until y is reached (with a uniform-randomization tweak for
    exchangeability — implemented here in the simpler deterministic form;
    randomization not yet implemented).
    """

    alpha: float = 0.10
    q_hat: float = field(default=np.nan, init=False)

    @staticmethod
    def _score(probs_row: np.ndarray, label: int) -> float:
        order = np.argsort(-probs_row)            # high → low
        cum = np.cumsum(probs_row[order])
        # Position of the true label in the sorted order
        rank = int(np.where(order == label)[0][0])
        return float(cum[rank])

    def calibrate(self, probs: np.ndarray, labels: np.ndarray) -> "APS":
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        n = len(labels)
        scores = np.array([self._score(probs[i], int(labels[i])) for i in range(n)])
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = float(min(level, 1.0))
        self.q_hat = float(np.quantile(scores, level, method="higher"))
        return self

    def predict_sets(self, probs: np.ndarray) -> list[np.ndarray]:
        if np.isnan(self.q_hat):
            raise RuntimeError("Call calibrate() first.")
        sets: list[np.ndarray] = []
        for p in probs:
            order = np.argsort(-p)
            cum = np.cumsum(p[order])
            # Include classes until cumulative mass exceeds q_hat
            k = int(np.searchsorted(cum, self.q_hat) + 1)
            k = min(k, len(p))
            sets.append(order[:k])
        return sets

    def set_sizes(self, probs: np.ndarray) -> np.ndarray:
        return np.array([len(s) for s in self.predict_sets(probs)], dtype=np.int32)

    def empirical_coverage(self, probs: np.ndarray, labels: np.ndarray) -> float:
        sets = self.predict_sets(probs)
        hits = sum(int(y in s) for s, y in zip(sets, labels))
        return hits / len(labels)
