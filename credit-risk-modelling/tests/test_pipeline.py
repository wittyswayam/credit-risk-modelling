import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_loader import DATA_DIR, XLS_NAME, load_raw, TARGET
from src.evaluation import ks_statistic, calibrate_oof, hypothetical_ecl, expected_calibration_error
from src.preprocessing import (add_features, build_preprocessor, clean_categories, feature_columns,
                               split_data, _signed_log)

needs_data = pytest.mark.skipif(not (DATA_DIR / XLS_NAME).exists(), reason="dataset not downloaded")


@pytest.fixture(scope="module")
def df():
    return add_features(clean_categories(load_raw()))


@needs_data
def test_raw_shape_and_target():
    raw = load_raw()
    assert raw.shape == (30000, 25)
    assert abs(raw[TARGET].mean() - 0.2212) < 1e-4


@needs_data
def test_categories_cleaned(df):
    assert set(df["EDUCATION"].unique()) == {1, 2, 3, 4}
    assert set(df["MARRIAGE"].unique()) == {1, 2, 3}


@needs_data
def test_split_disjoint_stratified_sizes(df):
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(df)
    assert (len(X_tr), len(X_va), len(X_te)) == (18000, 6000, 6000)
    assert not (set(X_tr.index) & set(X_va.index)) and not (set(X_tr.index) & set(X_te.index)) and not (set(X_va.index) & set(X_te.index))
    for y in (y_tr, y_va, y_te):
        assert abs(y.mean() - df[TARGET].mean()) < 1e-3


@needs_data
def test_scaler_fitted_on_training_rows_only(df):
    X_tr, X_va, *_ = split_data(df)
    prep = build_preprocessor("engineered").fit(X_tr)
    scaler = prep.named_transformers_["log_amounts"].named_steps["scale"]
    assert np.isclose(scaler.mean_[0], _signed_log(X_tr["LIMIT_BAL"]).mean())
    assert not np.isclose(scaler.mean_[0], _signed_log(df["LIMIT_BAL"]).mean())
    Z = prep.transform(X_va)
    assert Z.shape[1] == 31 and np.isfinite(Z.values).all()


@needs_data
def test_engineered_features_in_range(df):
    assert df["util_sep"].max() <= 2.0 and df["pay_ratio_avg"].between(0, 1).all()
    assert df["n_late_months"].between(0, 6).all()


def test_sex_excluded_by_default():
    assert "SEX" not in feature_columns("engineered")[2]
    assert "SEX" in feature_columns("engineered", include_sex=True)[2]


def test_ks_statistic_known_cases():
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert ks_statistic(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.0


def test_hypothetical_ecl_arithmetic():
    ecl, ead = hypothetical_ecl([0.1], [1000], [3000], lgd=0.5, ccf=0.5)
    assert np.isclose(ead[0], 1000 + 0.5 * 2000) and np.isclose(ecl[0], 0.1 * 0.5 * 2000)
    ecl0, ead0 = hypothetical_ecl([0.1], [-200], [3000], lgd=0.5, ccf=0.0)
    assert ead0[0] == 0 and ecl0[0] == 0  # negative balances floored at zero


def test_calibration_oof_bounds_and_ece():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.9, 2000); y = (rng.uniform(size=2000) < p).astype(int)
    for m in ("sigmoid", "isotonic"):
        out = calibrate_oof(p, y, m)
        assert out.shape == p.shape and out.min() >= 0 and out.max() <= 1
    assert expected_calibration_error(y, p) < 0.06
