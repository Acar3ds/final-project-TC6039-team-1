# -*- coding: utf-8 -*-
"""
test_ml_models.py
Unit tests for C4 — Machine Learning Classical Models Module

HOW TO RUN IN GOOGLE COLAB:
    1. Upload ml_models.py, test_ml_models.py, and df.csv to Colab.
    2. Run:  %run test_ml_models.py
    3. Done. pytest installs and runs automatically.

Test plan (from project specification §8.3):
    Normal case : verify all three models train and produce valid metrics.
    Edge case   : verify that invalid inputs raise the expected exceptions.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Auto-setup — installs pytest and imports ml_models from any location
# ─────────────────────────────────────────────────────────────────────────────

import subprocess
import sys
import os

# Install pytest if missing
try:
    import pytest
except ImportError:
    print("[C4 Tests] Installing pytest...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest", "-q"])
    except Exception:
        pass
    import pytest

# Make ml_models importable whether it lives in src/ or the same folder
sys.path.insert(0, ".")
sys.path.insert(0, "src")

import numpy as np
from ml_models import train_models


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper — synthetic traffic data for tests
# ─────────────────────────────────────────────────────────────────────────────

def _make_traffic_data(n_train: int = 80, n_test: int = 20, seed: int = 0):
    """
    Generate a small synthetic traffic dataset for testing.

    Features: [hour, minute, weekday_number]
    Target:   vehicle counts with morning/evening peaks + noise.

    Args:
        n_train (int): Number of training samples.
        n_test  (int): Number of test samples.
        seed    (int): Random seed for reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    rng   = np.random.default_rng(seed)
    n     = n_train + n_test
    hours = rng.integers(0, 24, n).astype(float)
    mins  = rng.integers(0, 60, n).astype(float)
    wday  = rng.integers(1, 6,  n).astype(float)

    y = (
        20.0
        + 80.0 * np.exp(-0.5 * ((hours - 8)  / 1.5) ** 2)
        + 70.0 * np.exp(-0.5 * ((hours - 17) / 1.5) ** 2)
        + rng.normal(0, 5, n)
    )
    y = np.clip(y, 0, None)

    X = np.column_stack([hours, mins, wday])
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


# ─────────────────────────────────────────────────────────────────────────────
# Normal case tests
# ─────────────────────────────────────────────────────────────────────────────

def test_normal_case_returns_dataframe():
    """
    Normal case: train_models() must return a DataFrame with one row per model
    and the expected metric columns.
    """
    X_train, X_test, y_train, y_test = _make_traffic_data()

    metrics = train_models(
        X_train, X_test, y_train, y_test,
        feature_names=["hour", "minute", "weekday_number"],
        save_figures=False, show_figures=False,
    )

    assert len(metrics) == 3, f"Expected 3 model rows, got {len(metrics)}"

    expected_cols = {"Model", "RMSE_train", "RMSE_test", "R2_train", "R2_test", "CV_RMSE_mean"}
    assert expected_cols.issubset(set(metrics.columns)), (
        f"Missing columns: {expected_cols - set(metrics.columns)}"
    )


def test_normal_case_metrics_are_valid():
    """
    Normal case: RMSE values must be non-negative and R² must be <= 1.
    """
    X_train, X_test, y_train, y_test = _make_traffic_data()

    metrics = train_models(
        X_train, X_test, y_train, y_test,
        save_figures=False, show_figures=False,
    )

    for _, row in metrics.iterrows():
        assert row["RMSE_train"]   >= 0,   f"{row['Model']}: RMSE_train is negative"
        assert row["RMSE_test"]    >= 0,   f"{row['Model']}: RMSE_test is negative"
        assert row["R2_test"]      <= 1.0, f"{row['Model']}: R2_test > 1"
        assert row["CV_RMSE_mean"] >= 0,   f"{row['Model']}: CV_RMSE_mean is negative"


def test_normal_case_model_names():
    """
    Normal case: the three expected model names must appear in the metrics table.
    """
    X_train, X_test, y_train, y_test = _make_traffic_data()

    metrics = train_models(
        X_train, X_test, y_train, y_test,
        save_figures=False, show_figures=False,
    )

    model_names = metrics["Model"].tolist()
    assert "Linear Regression" in model_names
    assert "Decision Tree"     in model_names
    assert "Random Forest"     in model_names


# ─────────────────────────────────────────────────────────────────────────────
# Edge case tests
# ─────────────────────────────────────────────────────────────────────────────

def test_edge_case_empty_X_train_raises_assertion():
    """
    Edge case: empty X_train must raise AssertionError.
    Pseudocode assertion: assert size(X_train) > 0
    """
    X_train = np.array([]).reshape(0, 3)
    X_test  = np.array([[8.0, 30.0, 1.0]])
    y_train = np.array([])
    y_test  = np.array([50.0])

    with pytest.raises(AssertionError):
        train_models(X_train, X_test, y_train, y_test,
                     save_figures=False, show_figures=False)


def test_edge_case_dimension_mismatch_raises_value_error():
    """
    Edge case: mismatched X_train/y_train lengths must raise ValueError.
    Pseudocode exception: IF len(X_train) != len(y_train) RAISE DimensionError.
    """
    X_train = np.array([[1.0, 0.0, 1.0], [2.0, 0.0, 1.0]])  # 2 rows
    y_train = np.array([10.0])                                 # 1 row
    X_test  = np.array([[3.0, 0.0, 1.0]])
    y_test  = np.array([15.0])

    with pytest.raises(ValueError):
        train_models(X_train, X_test, y_train, y_test,
                     save_figures=False, show_figures=False)


def test_edge_case_negative_y_train_raises_value_error():
    """
    Edge case: negative vehicle counts in y_train must raise ValueError.
    Pseudocode exception: IF negative values exist in y_train RAISE InvalidTrafficDataError.
    """
    X_train = np.array([[8.0, 0.0, 1.0], [9.0, 0.0, 1.0], [10.0, 0.0, 1.0]])
    y_train = np.array([50.0, -5.0, 30.0])
    X_test  = np.array([[11.0, 0.0, 1.0]])
    y_test  = np.array([20.0])

    with pytest.raises(ValueError):
        train_models(X_train, X_test, y_train, y_test,
                     save_figures=False, show_figures=False)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — runs pytest automatically when executed directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n[C4 Tests] Running all unit tests...\n")
    result = pytest.main([__file__, "-v"])
    if result == 0:
        print("\n[C4 Tests] All tests passed ✅")
    else:
        print("\n[C4 Tests] Some tests failed ❌")
