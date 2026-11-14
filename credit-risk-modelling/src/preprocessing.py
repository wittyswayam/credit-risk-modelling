"""Stateless cleaning, feature engineering, splitting and the scikit-learn preprocessing pipeline.

Anything that *learns* from data (scaling, encoding categories) lives inside the
ColumnTransformer and is fitted on training data only. The functions that are
deterministic row-wise recodes (clean_categories, add_features) learn nothing and
are safe to apply before the split.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "default payment next month"
SEED = 42
PAY = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT = [f"PAY_AMT{i}" for i in range(1, 7)]
CATEGORICAL = ["EDUCATION", "MARRIAGE"]
AGE_CENTER = 35  # fixed constant (approx. sample mean) so age_sq is stateless and less collinear with AGE
UTIL_CAP = 2.0   # caps utilisation; affects ~0.1% of rows
ENGINEERED = ["util_sep", "util_avg", "pay_ratio_avg", "max_delay", "n_late_months", "age_sq"]


def clean_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Merge undocumented codes into the documented 'others' category."""
    out = df.copy()
    out["EDUCATION"] = out["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    out["MARRIAGE"] = out["MARRIAGE"].replace({0: 3})
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add candidate engineered features (row-wise, no information from other rows)."""
    out = df.copy()
    limit = out["LIMIT_BAL"]
    out["util_sep"] = (out["BILL_AMT1"] / limit).clip(upper=UTIL_CAP)
    out["util_avg"] = (out[BILL].mean(axis=1) / limit).clip(upper=UTIL_CAP)
    paid = out[PAYAMT].sum(axis=1)
    billed = out[BILL].clip(lower=0).sum(axis=1)
    out["pay_ratio_avg"] = (paid / billed.where(billed > 0)).fillna(1.0).clip(upper=1.0)
    out["max_delay"] = out[PAY].max(axis=1).clip(lower=0)
    out["n_late_months"] = (out[PAY] > 0).sum(axis=1)
    out["age_sq"] = (out["AGE"] - AGE_CENTER) ** 2
    return out


def split_data(df: pd.DataFrame, seed: int = SEED):
    """Stratified 60/20/20 train/validation/test split. Returns (X_tr, X_va, X_te, y_tr, y_va, y_te)."""
    X, y = df.drop(columns=[TARGET, "ID"]), df[TARGET]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def _signed_log(x):
    """sign(x) * log1p(|x|): compresses heavy tails and keeps negative balances."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


def feature_columns(feature_set: str = "base", include_sex: bool = False):
    """Return (log_cols, plain_numeric_cols, categorical_cols) for a named feature set."""
    plain = ["AGE"] + PAY
    log_cols = ["LIMIT_BAL"] + BILL + PAYAMT
    cats = list(CATEGORICAL)
    if include_sex:
        cats = ["SEX"] + cats
    if feature_set == "engineered":
        plain = plain + ENGINEERED
    elif feature_set != "base":
        raise ValueError("feature_set must be 'base' or 'engineered'")
    return log_cols, plain, cats


def build_preprocessor(feature_set: str = "base", include_sex: bool = False) -> ColumnTransformer:
    """ColumnTransformer: signed-log + scale amounts, scale numerics, one-hot categoricals."""
    log_cols, plain, cats = feature_columns(feature_set, include_sex)
    log_pipe = Pipeline([("log", FunctionTransformer(_signed_log, feature_names_out="one-to-one")),
                         ("scale", StandardScaler())])
    return ColumnTransformer(
        [("log_amounts", log_pipe, log_cols),
         ("numeric", StandardScaler(), plain),
         ("categorical", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cats)],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
