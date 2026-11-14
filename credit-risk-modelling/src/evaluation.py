"""Evaluation metrics for binary credit-default models."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)


def ks_statistic(y_true, p) -> float:
    """Kolmogorov-Smirnov statistic: max gap between cumulative score distributions of defaulters and non-defaulters."""
    fpr, tpr, _ = roc_curve(y_true, p)
    return float(np.max(tpr - fpr))


def evaluate(y_true, p, threshold: float = 0.5) -> dict:
    """Threshold-free and threshold-based metrics for predicted default probabilities p."""
    y_hat = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_hat),
        "precision": precision_score(y_true, y_hat, zero_division=0),
        "recall": recall_score(y_true, y_hat),
        "f1": f1_score(y_true, y_hat),
        "roc_auc": roc_auc_score(y_true, p),
        "pr_auc": average_precision_score(y_true, p),
        "brier": brier_score_loss(y_true, p),
        "ks": ks_statistic(y_true, p),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def metrics_table(models: dict, X, y, threshold: float = 0.5) -> pd.DataFrame:
    """Evaluate several fitted models on one dataset; rows = models."""
    return pd.DataFrame({n: evaluate(y, m.predict_proba(X)[:, 1], threshold) for n, m in models.items()}).T


# ---------------------------------------------------------------- calibration
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def expected_calibration_error(y_true, p, n_bins: int = 10) -> float:
    """ECE with equal-frequency bins: weighted mean |observed rate - mean predicted PD|."""
    d = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(p)})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), n_bins, labels=False)
    g = d.groupby("bin").agg(obs=("y", "mean"), pred=("p", "mean"), n=("y", "size"))
    return float((g["n"] / g["n"].sum() * (g["obs"] - g["pred"]).abs()).sum())


class Recalibrator:
    """Monotone recalibration of scores: 'sigmoid' (Platt, on logit of score) or 'isotonic'."""

    def __init__(self, method: str = "sigmoid"):
        self.method = method

    def fit(self, p, y):
        if self.method == "sigmoid":
            self.m_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p), y)
        elif self.method == "isotonic":
            self.m_ = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(np.asarray(p), y)
        else:
            raise ValueError(self.method)
        return self

    def predict(self, p):
        if self.method == "sigmoid":
            return self.m_.predict_proba(_logit(p))[:, 1]
        return self.m_.predict(np.asarray(p))


def calibrate_oof(p, y, method: str, seed: int = 42, n_splits: int = 5) -> np.ndarray:
    """Out-of-fold recalibrated probabilities on a dataset (each point is calibrated by a model that never saw it)."""
    p, y = np.asarray(p), np.asarray(y)
    out = np.zeros_like(p, dtype=float)
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(p, y):
        out[te] = Recalibrator(method).fit(p[tr], y[tr]).predict(p[te])
    return out


# ---------------------------------------------------------------- thresholds and costs
def threshold_sweep(y_true, p, thresholds=None, cost_ratios=(1, 2, 3.5, 5, 10)) -> pd.DataFrame:
    """Precision/recall/F1 and normalised misclassification cost across thresholds.

    Cost is in units of one false-positive cost: cost = FP + ratio * FN, divided by n.
    """
    y, p = np.asarray(y_true), np.asarray(p)
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        yh = p >= t
        tp, fp = int((yh & (y == 1)).sum()), int((yh & (y == 0)).sum())
        fn, tn = int((~yh & (y == 1)).sum()), int((~yh & (y == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
        row = {"threshold": t, "precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
               "flagged_share": yh.mean(), "tpr": rec, "fpr": fp / (fp + tn)}
        for r in cost_ratios:
            row[f"cost_r{r}"] = (fp + r * fn) / len(y)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- hypothetical expected credit loss
def hypothetical_ecl(pd_hat, bill, limit, lgd: float, ccf: float):
    """HYPOTHETICAL ECL per borrower = PD x LGD x EAD, with EAD = drawn + CCF x undrawn.

    Drawn exposure is proxied by the latest bill (floored at 0), undrawn = max(limit - drawn, 0).
    LGD and CCF are assumptions, not estimated from the data (the dataset has no recoveries or exposure at default).
    """
    drawn = np.maximum(np.asarray(bill, dtype=float), 0.0)
    ead = drawn + ccf * np.maximum(np.asarray(limit, dtype=float) - drawn, 0.0)
    return np.asarray(pd_hat) * lgd * ead, ead
