"""Plotting helpers with a consistent house style. All figures save to results/figures/."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FIG_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
TARGET = "default payment next month"
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
C_NO, C_DEF = "#2b6cb0", "#c53030"


def set_style() -> None:
    """Apply a common matplotlib/seaborn style."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 200, "axes.titleweight": "bold",
                         "axes.titlesize": 12, "axes.labelsize": 10})


def save(fig, name: str) -> str:
    """Save a figure as a high-resolution PNG in results/figures/."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight")
    return f"results/figures/{name}"  # repo-relative, so notebook outputs carry no local paths


def plot_target(df: pd.DataFrame, name="01_target_distribution.png"):
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(["No default (0)", "Default (1)"], counts.values, color=[C_NO, C_DEF])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v / len(df):.1%})", ha="center", va="bottom")
    ax.set_ylabel("Number of clients"); ax.set_ylim(0, counts.max() * 1.18)
    ax.set_title("Default vs non-default (October 2005)")
    return save(fig, name)


def plot_corr(df: pd.DataFrame, name="02_correlation_heatmap.png"):
    corr = df.drop(columns="ID", errors="ignore").corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax,
                cbar_kws={"label": "Pearson correlation"}, linewidths=0.3)
    ax.set_title("Correlation matrix of all variables (lower triangle)")
    return save(fig, name)


def plot_limit(df: pd.DataFrame, name="03_credit_limit_distribution.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df["LIMIT_BAL"] / 1000, bins=50, color=C_NO, ax=axes[0])
    axes[0].set_xlabel("Credit limit (NT$ thousands)"); axes[0].set_title("Credit limit distribution")
    sns.histplot(np.log10(df["LIMIT_BAL"]), bins=40, color=C_NO, ax=axes[1])
    axes[1].set_xlabel("log10(credit limit, NT$)"); axes[1].set_title("Credit limit, log scale")
    return save(fig, name)


def plot_age(df: pd.DataFrame, name="04_age_distribution.png"):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x="AGE", hue=TARGET, bins=range(20, 82, 2), multiple="stack",
                 palette={0: C_NO, 1: C_DEF}, ax=ax)
    ax.set_xlabel("Age (years)"); ax.set_ylabel("Clients"); ax.set_title("Age distribution by default status")
    return save(fig, name)


def plot_pay_status(df: pd.DataFrame, name="05_repayment_status_distributions.png"):
    labels = ["Sep (PAY_0)", "Aug", "Jul", "Jun", "May", "Apr"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for ax, col, lab in zip(axes.ravel(), PAY_COLS, labels):
        vc = df[col].value_counts().sort_index()
        ax.bar(vc.index.astype(str), vc.values, color=C_NO)
        ax.set_title(f"Repayment status: {lab}"); ax.set_xlabel("Status code")
    axes[0, 0].set_ylabel("Clients"); axes[1, 0].set_ylabel("Clients")
    fig.suptitle("Repayment status codes by month (-2, -1, 0 undocumented/paid; 1+ = months late)", y=1.0)
    fig.tight_layout()
    return save(fig, name)


def default_rate_plot(df, group, title, xlabel, name, rot=0):
    g = df.groupby(group, observed=True)[TARGET].agg(["mean", "size"])
    se = np.sqrt(g["mean"] * (1 - g["mean"]) / g["size"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(g.index.astype(str), g["mean"] * 100, yerr=1.96 * se * 100, color=C_DEF, capsize=3, alpha=0.9)
    ax.axhline(df[TARGET].mean() * 100, color="black", ls="--", lw=1, label="Portfolio average")
    for i, (m, n) in enumerate(zip(g["mean"], g["size"])):
        ax.text(i, 1, f"n={n:,}", ha="center", va="bottom", color="white", fontsize=8, rotation=90)
    ax.set_ylabel("Default rate (%)"); ax.set_xlabel(xlabel); ax.set_title(title); ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=rot)
    return save(fig, name), g


def plot_pay0_default(df: pd.DataFrame, name="08_repayment_history_vs_default.png"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    g = df.groupby("PAY_0")[TARGET].agg(["mean", "size"])
    axes[0].bar(g.index.astype(str), g["mean"] * 100, color=C_DEF)
    axes[0].axhline(df[TARGET].mean() * 100, color="black", ls="--", lw=1)
    axes[0].set_xlabel("PAY_0 (September repayment status)"); axes[0].set_ylabel("Default rate (%)")
    axes[0].set_title("Default rate by most recent repayment status")
    d = df.copy(); d["max_delay"] = d[PAY_COLS].max(axis=1).clip(lower=0).clip(upper=4)
    d["max_delay"] = d["max_delay"].map({0: "0", 1: "1", 2: "2", 3: "3", 4: "4+"})
    h = d.groupby("max_delay")[TARGET].mean().reindex(["0", "1", "2", "3", "4+"]) * 100
    axes[1].bar(h.index, h.values, color=C_DEF)
    axes[1].axhline(df[TARGET].mean() * 100, color="black", ls="--", lw=1)
    axes[1].set_xlabel("Worst delay across Apr-Sep (months)"); axes[1].set_ylabel("Default rate (%)")
    axes[1].set_title("Default rate by worst delinquency, six months")
    fig.tight_layout()
    return save(fig, name)


def plot_transform_effect(raw: pd.DataFrame, transformed: pd.DataFrame, cols, name="09_amount_transform_effect.png"):
    """Raw vs signed-log-scaled distributions for selected amount columns."""
    fig, axes = plt.subplots(2, len(cols), figsize=(4 * len(cols), 6))
    for j, c in enumerate(cols):
        sns.histplot(raw[c], bins=60, color=C_NO, ax=axes[0, j]); axes[0, j].set_title(f"{c}: raw (NT$)")
        sns.histplot(transformed[c], bins=60, color=C_DEF, ax=axes[1, j]); axes[1, j].set_title(f"{c}: signed-log, standardised")
        axes[0, j].set_xlabel(""); axes[1, j].set_xlabel("")
    fig.tight_layout()
    return save(fig, name)


def plot_engineered_vs_default(df: pd.DataFrame, name="10_engineered_features_vs_default.png"):
    """Default rate by decile (or by value) of the engineered features."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (col, title) in zip(axes, [("util_avg", "Average utilisation (6-month bill / limit)"),
                                       ("pay_ratio_avg", "Repayment ratio (paid / billed)"),
                                       ("n_late_months", "Months in delay (out of 6)")]):
        if col == "n_late_months":
            g = df.groupby(col)[TARGET].mean() * 100; x = g.index.astype(str)
        else:
            g = df.groupby(pd.qcut(df[col], 10, duplicates="drop"), observed=True)[TARGET].mean() * 100
            x = [str(i + 1) for i in range(len(g))]
        ax.bar(x, g.values, color=C_DEF)
        ax.axhline(df[TARGET].mean() * 100, color="black", ls="--", lw=1)
        ax.set_title(title, fontsize=10); ax.set_ylabel("Default rate (%)")
        ax.set_xlabel("Decile" if col != "n_late_months" else "Months")
    fig.tight_layout()
    return save(fig, name)


def plot_vif(vif: pd.Series, name="11_vif.png"):
    v = vif.sort_values()
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(v.index, v.values, color=[C_DEF if x >= 5 else C_NO for x in v.values])
    ax.axvline(5, color="black", ls="--", lw=1, label="VIF = 5"); ax.axvline(10, color="black", ls=":", lw=1, label="VIF = 10")
    ax.set_xlabel("Variance inflation factor"); ax.set_title("Multicollinearity of preprocessed training features")
    ax.legend()
    return save(fig, name)


def plot_feature_set_comparison(tab: pd.DataFrame, name="12_feature_set_comparison.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(tab.index, tab["val_roc_auc"], color=C_NO)
    for i, v in enumerate(tab["val_roc_auc"]):
        ax.text(v + 0.0005, i, f"{v:.4f}", va="center")
    ax.set_xlim(tab["val_roc_auc"].min() - 0.02, tab["val_roc_auc"].max() + 0.01)
    ax.set_xlabel("Validation ROC-AUC (L2 logistic regression, C = 1)"); ax.set_title("Feature-set comparison")
    return save(fig, name)


MODEL_COLORS = {"baseline": "#718096", "l1": "#2b6cb0", "l2": "#38a169", "weighted": "#c53030"}
MODEL_LABELS = {"baseline": "Baseline (unregularised)", "l1": "L1 (lasso)", "l2": "L2 (ridge)",
                "weighted": "L2 + class weights"}


def plot_cv_curves(grids: dict, c_grid, l1_nonzero=None, name="13_regularisation_curves.png"):
    """Cross-validated ROC-AUC against C, with the optimum marked (and L1 sparsity if given)."""
    ncols = 2 if l1_nonzero is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 4.5), squeeze=False)
    ax = axes[0, 0]
    for k, g in grids.items():
        m, s = g.cv_results_["mean_test_score"], g.cv_results_["std_test_score"]
        ax.plot(c_grid, m, marker="o", ms=4, color=MODEL_COLORS[k], label=MODEL_LABELS[k])
        ax.fill_between(c_grid, m - s, m + s, color=MODEL_COLORS[k], alpha=0.12)
        ax.plot(g.best_params_["clf__C"], g.best_score_, "*", ms=14, color=MODEL_COLORS[k], mec="black")
    ax.set_xscale("log"); ax.set_xlabel("C (inverse regularisation strength)"); ax.set_ylabel("5-fold CV ROC-AUC (training set)")
    ax.set_title("Regularisation path: CV ROC-AUC (band = ±1 SD across folds)"); ax.legend()
    if l1_nonzero is not None:
        a2 = axes[0, 1]
        a2.plot(c_grid, l1_nonzero, marker="o", color=MODEL_COLORS["l1"]); a2.set_xscale("log")
        a2.set_xlabel("C"); a2.set_ylabel("Non-zero coefficients"); a2.set_title("L1 sparsity along the path")
    fig.tight_layout()
    return save(fig, name)


def plot_roc_pr(models: dict, X, y, name="14_roc_pr_validation.png"):
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for k, m in models.items():
        p = m.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, p); pr, rc, _ = precision_recall_curve(y, p)
        axes[0].plot(fpr, tpr, color=MODEL_COLORS[k], label=f"{MODEL_LABELS[k]} (AUC {roc_auc_score(y, p):.3f})")
        axes[1].plot(rc, pr, color=MODEL_COLORS[k], label=f"{MODEL_LABELS[k]} (AP {average_precision_score(y, p):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1); axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve (validation set)"); axes[0].legend(loc="lower right")
    axes[1].axhline(y.mean(), color="black", ls="--", lw=1, label=f"No-skill ({y.mean():.3f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].set_title("Precision-recall curve (validation set)")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    return save(fig, name)


def plot_model_comparison(tab: pd.DataFrame, name="15_model_comparison.png"):
    """Grouped bars of validation metrics; tab has models as rows."""
    metrics = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("ks", "KS"), ("brier", "Brier (lower is better)")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, (m, title) in zip(axes, metrics):
        vals = tab[m].astype(float)
        ax.bar([MODEL_LABELS[i].replace(" (", "\n(") for i in vals.index], vals.values,
               color=[MODEL_COLORS[i] for i in vals.index])
        for j, v in enumerate(vals.values):
            ax.text(j, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title); ax.tick_params(axis="x", labelsize=7)
        lo, hi = vals.min(), vals.max(); pad = (hi - lo) * 0.8 + 0.01
        ax.set_ylim(max(0, lo - pad), hi + pad * 0.6)
    fig.suptitle("Validation-set model comparison (y-axes truncated: differences among the first three models are small)", y=1.02, fontweight="bold")
    fig.tight_layout()
    return save(fig, name)


def plot_confusion_matrices(models: dict, X, y, threshold=0.5, name="16_confusion_matrices_validation.png"):
    from sklearn.metrics import confusion_matrix
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4))
    for ax, (k, m) in zip(np.atleast_1d(axes), models.items()):
        cm = confusion_matrix(y, (m.predict_proba(X)[:, 1] >= threshold).astype(int))
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred. no default", "Pred. default"], yticklabels=["Actual no default", "Actual default"])
        ax.set_title(f"{MODEL_LABELS[k]}\nthreshold = {threshold:.2f}", fontsize=10)
    fig.tight_layout()
    return save(fig, name)


def plot_coef_comparison(coefs: pd.DataFrame, top=14, name="17_coefficient_shrinkage.png"):
    """Standardised coefficients of the four models for the features with the largest baseline |coef|."""
    order = coefs["baseline"].abs().sort_values(ascending=False).index[:top][::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    h = 0.2
    for j, k in enumerate(coefs.columns):
        ax.barh(np.arange(len(order)) + (j - 1.5) * h, coefs.loc[order, k], height=h, color=MODEL_COLORS[k], label=MODEL_LABELS[k])
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order); ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Coefficient on standardised feature (log-odds per 1 SD)")
    ax.set_title("Effect of regularisation and class weighting on coefficients"); ax.legend(fontsize=8)
    fig.tight_layout()
    return save(fig, name)


def plot_prob_distribution(y, p, name="18_predicted_probability_distribution.png", title="Predicted PD distribution by outcome (validation)"):
    y, p = np.asarray(y), np.asarray(p)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 1, 41)
    ax.hist(p[y == 0], bins=bins, alpha=0.65, color=C_NO, label=f"No default (n={int((y == 0).sum()):,})")
    ax.hist(p[y == 1], bins=bins, alpha=0.65, color=C_DEF, label=f"Default (n={int((y == 1).sum()):,})")
    ax.axvline(p.mean(), color="black", ls="--", lw=1, label=f"Mean predicted PD ({p.mean():.3f})")
    ax.set_xlabel("Predicted probability of default"); ax.set_ylabel("Clients"); ax.set_title(title); ax.legend()
    return save(fig, name)


def plot_risk_bands(bands: pd.DataFrame, name="19_risk_bands.png"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(bands))
    se = np.sqrt(bands["observed_default_rate"] * (1 - bands["observed_default_rate"]) / bands["n"])
    ax.bar(x, bands["observed_default_rate"] * 100, yerr=1.96 * se * 100, capsize=3, color=C_DEF, alpha=0.85, label="Observed default rate (95% CI)")
    ax.plot(x, bands["mean_predicted_pd"] * 100, "ko", label="Mean predicted PD")
    for i, n in enumerate(bands["n"]):
        ax.text(i, 1, f"n={n:,}", ha="center", color="white", fontsize=8, rotation=90, va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(bands.index, fontsize=8)
    ax.set_ylabel("Default rate / PD (%)"); ax.set_title("Illustrative PD risk bands: predicted vs observed (validation, out-of-fold)")
    ax.legend()
    return save(fig, name)


def plot_odds_ratios(tab: pd.DataFrame, name="20_odds_ratios.png"):
    """tab columns: or, lo, hi, p (odds ratio per +1 SD of the standardised feature, 95% CI)."""
    t = tab.sort_values("or")
    fig, ax = plt.subplots(figsize=(8, 9))
    colors = [C_DEF if p < 0.05 else "#a0aec0" for p in t["p"]]
    ax.errorbar(t["or"], range(len(t)), xerr=[t["or"] - t["lo"], t["hi"] - t["or"]], fmt="none", ecolor="#718096", lw=1)
    ax.scatter(t["or"], range(len(t)), c=colors, zorder=3, s=25)
    ax.axvline(1, color="black", lw=1); ax.set_xscale("log")
    ax.set_yticks(range(len(t))); ax.set_yticklabels(t.index, fontsize=8)
    ax.set_xlabel("Odds ratio per +1 SD of the feature (log scale, 95% CI)")
    ax.set_title("Odds ratios from unregularised statsmodels logit (red: p < 0.05)")
    return save(fig, name)


def plot_calibration(panels: dict, y, n_bins=10, name="21_calibration_curves.png"):
    """panels: {panel title: {label: probabilities}}; equal-frequency reliability diagrams."""
    y = np.asarray(y)
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5.2), squeeze=False)
    for ax, (title, series) in zip(axes[0], panels.items()):
        for lab, p in series.items():
            p = np.asarray(p)
            b = pd.qcut(pd.Series(p).rank(method="first"), n_bins, labels=False)
            g = pd.DataFrame({"y": y, "p": p, "b": b}).groupby("b").agg(o=("y", "mean"), m=("p", "mean"))
            ax.plot(g["m"], g["o"], marker="o", ms=4, label=lab)
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
        ax.set_xlim(0, max(0.8, ax.get_xlim()[1])); ax.set_ylim(0, 0.9)
        ax.set_xlabel("Mean predicted PD (decile bins)"); ax.set_ylabel("Observed default rate"); ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return save(fig, name)


def plot_threshold_analysis(sweep: pd.DataFrame, marks: dict, cost_ratios, name="22_threshold_analysis.png"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    ax.plot(sweep["threshold"], sweep["precision"], label="Precision", color=C_NO)
    ax.plot(sweep["threshold"], sweep["recall"], label="Recall", color=C_DEF)
    ax.plot(sweep["threshold"], sweep["f1"], label="F1", color="#38a169")
    ax.plot(sweep["threshold"], sweep["flagged_share"], label="Share of clients flagged", color="#718096", ls=":")
    for lab, t in marks.items():
        ax.axvline(t, color="black", lw=0.8, ls="--"); ax.text(t, 1.01, lab, fontsize=7, ha="center")
    ax.set_xlabel("Classification threshold on PD"); ax.set_ylabel("Metric value"); ax.set_ylim(0, 1.08)
    ax.set_title("Precision-recall trade-off by threshold (validation, calibrated)"); ax.legend(fontsize=8, loc="center right")
    ax = axes[1]
    for r in cost_ratios:
        c = sweep[f"cost_r{r}"]; ax.plot(sweep["threshold"], c, label=f"FN cost = {r} x FP cost")
        ax.plot(sweep["threshold"][c.idxmin()], c.min(), "k*", ms=9)
    ax.set_xlabel("Classification threshold on PD"); ax.set_ylabel("Misclassification cost per client (units of one FP)")
    ax.set_title("Cost curves for illustrative cost ratios (star = minimum)"); ax.legend(fontsize=8)
    fig.tight_layout()
    return save(fig, name)


def plot_scores_roc_pr(y, p, label, prefix, name="23_roc_pr_test.png"):
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
    y, p = np.asarray(y), np.asarray(p)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fpr, tpr, _ = roc_curve(y, p); pr, rc, _ = precision_recall_curve(y, p)
    axes[0].plot(fpr, tpr, color=C_NO, label=f"{label} (AUC {roc_auc_score(y, p):.3f})"); axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate"); axes[0].set_title(f"ROC curve ({prefix})"); axes[0].legend(loc="lower right")
    axes[1].plot(rc, pr, color=C_NO, label=f"{label} (AP {average_precision_score(y, p):.3f})")
    axes[1].axhline(y.mean(), color="black", ls="--", lw=1, label=f"No-skill ({y.mean():.3f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].set_title(f"Precision-recall curve ({prefix})"); axes[1].legend()
    fig.tight_layout()
    return save(fig, name)


def plot_confusion_one(y, p, threshold, title, name="24_confusion_matrix_test.png"):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, (np.asarray(p) >= threshold).astype(int))
    fig, ax = plt.subplots(figsize=(5, 4.2))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Pred. no default", "Pred. default"], yticklabels=["Actual no default", "Actual default"])
    ax.set_title(f"{title}\nthreshold = {threshold:.2f}", fontsize=10)
    return save(fig, name)


def plot_ecl_heatmap(grid: pd.DataFrame, name="25_hypothetical_ecl_sensitivity.png"):
    """grid: rows = LGD, columns = CCF, values = hypothetical ECL as % of total credit limit."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.heatmap(grid * 100, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "ECL, % of total credit limit"})
    ax.set_xlabel("Assumed credit conversion factor (undrawn share expected to be drawn)"); ax.set_ylabel("Assumed LGD")
    ax.set_title("HYPOTHETICAL expected loss (assumed LGD/CCF; 1-month PD)")
    return save(fig, name)
