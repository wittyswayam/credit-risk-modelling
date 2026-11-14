"""Logistic regression model definitions and cross-validated tuning (no other algorithms by design)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from .preprocessing import SEED, build_preprocessor

C_GRID = np.logspace(-4, 2, 13)


def make_pipeline(kind: str, feature_set: str = "engineered", include_sex: bool = False, C: float = 1.0) -> Pipeline:
    """Build one of: 'baseline' (unregularised), 'l1', 'l2', 'weighted' (L2 + balanced class weights)."""
    common = dict(max_iter=10000, random_state=SEED)
    if kind == "baseline":
        clf = LogisticRegression(C=np.inf, **common)
    elif kind == "l2":
        clf = LogisticRegression(C=C, l1_ratio=0, **common)
    elif kind == "l1":
        clf = LogisticRegression(C=C, l1_ratio=1, solver="saga", **common)
    elif kind == "weighted":
        clf = LogisticRegression(C=C, l1_ratio=0, class_weight="balanced", **common)
    else:
        raise ValueError(kind)
    return Pipeline([("prep", build_preprocessor(feature_set, include_sex)), ("clf", clf)])


def tune(kind: str, X, y, scoring: str = "roc_auc", feature_set: str = "engineered", include_sex: bool = False):
    """Tune C by stratified 5-fold CV on training data only. Returns the fitted GridSearchCV."""
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    gs = GridSearchCV(make_pipeline(kind, feature_set, include_sex), {"clf__C": C_GRID}, scoring=scoring,
                      cv=cv, n_jobs=-1, refit=True, return_train_score=False)
    return gs.fit(X, y)


def coefficient_table(pipe: Pipeline) -> pd.DataFrame:
    """Coefficients (on standardised features) and odds ratios of a fitted pipeline."""
    names = pipe.named_steps["prep"].get_feature_names_out()
    coef = pipe.named_steps["clf"].coef_.ravel()
    return pd.DataFrame({"coef": coef, "odds_ratio": np.exp(coef)}, index=names)
