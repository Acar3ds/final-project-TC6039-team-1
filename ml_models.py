# -*- coding: utf-8 -*-
"""
ml_models.py
C4 — Machine Learning Classical Models Module
Author: [Your Name] | TC6039.1 Applied Computing — Final Project

Module: train_models(X_train, X_test, y_train, y_test) → metrics_table

Purpose:
    Compare three classical supervised learning models on the traffic flow
    regression problem produced by C1 (data_loader.py):
        1. Linear Regression   — sklearn baseline
        2. Decision Tree       — non-linear, interpretable
        3. Random Forest       — ensemble, best generalization expected

Execution order:
    C1 (data_loader.py) → produces data/processed/df.csv
    C2 (regression.py)  → baseline regression metrics
    C3 (optimizer.py)   → optimized coefficients
    C4 (ml_models.py)   ← YOU ARE HERE

Complexity:
    Linear Regression  : O(n m²)
    Decision Tree      : O(n m log n)
    Random Forest      : O(t · n · m · log n)   where t = number of trees

HOW TO RUN IN GOOGLE COLAB:
    1. Upload ml_models.py, test_ml_models.py, and df.csv to Colab.
    2. Run:  %run ml_models.py
    3. Done. Setup, data loading, training, and plots all happen automatically.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Colab setup — runs automatically when the script starts
# ─────────────────────────────────────────────────────────────────────────────

import subprocess
import sys
import os
from pathlib import Path

def _setup():
    """
    Install missing dependencies and create required folder structure.
    Runs automatically at import time so no manual setup is needed in Colab.
    """
    # Install pytest if not available
    try:
        import pytest
    except ImportError:
        print("[C4] Installing pytest...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest", "-q"])
        except Exception:
            pass  # Will be caught later if pytest is truly needed

    # Create folder structure expected by the project
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    # If df.csv is in the root (uploaded directly to Colab) copy it to the right place
    if Path("df.csv").exists() and not Path("data/processed/df.csv").exists():
        import shutil
        shutil.copy("df.csv", "data/processed/df.csv")
        print("[C4] df.csv copied to data/processed/df.csv")

_setup()

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────

import csv
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

FIGURES_DIR  = Path("reports/figures")
RANDOM_STATE = 42

# ─────────────────────────────────────────────────────────────────────────────
# Helper — metrics computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute RMSE and R² for a set of predictions.

    Args:
        y_true (np.ndarray): Ground-truth target values (vehicles/hour).
        y_pred (np.ndarray): Model predictions (vehicles/hour).

    Returns:
        dict: {"rmse": float, "r2": float}
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "r2": r2}


# ─────────────────────────────────────────────────────────────────────────────
# Core public function — C4 entry point
# ─────────────────────────────────────────────────────────────────────────────

def train_models(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
    feature_names: list = None,
    save_figures: bool = True,
    show_figures: bool = True,
) -> pd.DataFrame:
    """
    Train and compare three ML models on the traffic flow regression task.

    Precondition:
        len(X_train) == len(y_train)
        len(X_test)  == len(y_test)
        X_train must not be empty

    Args:
        X_train (np.ndarray): Training feature matrix [n_samples, n_features].
                              Features: hour, minute, weekday_number.
        X_test  (np.ndarray): Test feature matrix [p_samples, n_features].
        y_train (np.ndarray): Training target — vehicle counts (vehicles/hour).
        y_test  (np.ndarray): Test target — vehicle counts (vehicles/hour).
        feature_names (list | None): Names of features for plots.
        save_figures (bool): Save generated plots to disk. Default True.
        show_figures (bool): Display plots interactively. Default True.

    Returns:
        pd.DataFrame: Comparison table with columns
                      [Model, RMSE_train, RMSE_test, R2_train, R2_test, CV_RMSE_mean].

    Raises:
        AssertionError : If X_train is empty (C4 pseudocode assertion).
        ValueError     : If train/test array lengths are incompatible.
        ValueError     : If negative target values are found in y_train.
    """

    # ── Precondition checks ───────────────────────────────────────────────────
    assert X_train.size > 0, (
        "[C4] AssertionError: X_train is empty. "
        "Ensure data_loader.py (C1) ran successfully."
    )
    if len(X_train) != len(y_train):
        raise ValueError(
            f"[C4] DimensionError: len(X_train)={len(X_train)} "
            f"!= len(y_train)={len(y_train)}."
        )
    if len(X_test) != len(y_test):
        raise ValueError(
            f"[C4] DimensionError: len(X_test)={len(X_test)} "
            f"!= len(y_test)={len(y_test)}."
        )
    if np.any(y_train < 0):
        raise ValueError(
            "[C4] InvalidTrafficDataError: y_train contains negative vehicle counts."
        )

    # ── Feature name defaults ─────────────────────────────────────────────────
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
        X_test  = X_test.reshape(-1, 1)

    n_features = X_train.shape[1]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    print(f"[C4] Training set  : {X_train.shape[0]} samples, {n_features} features")
    print(f"[C4] Test set      : {X_test.shape[0]} samples")
    print(f"[C4] Features      : {feature_names}")

    # ── Step 1 — Define models inside sklearn Pipelines ───────────────────────
    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LinearRegression()),
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestRegressor(
                n_estimators=100, max_depth=7,
                random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ]),
    }

    # ── Steps 2-6 — Train, predict, compute metrics ───────────────────────────
    rows        = []
    predictions = {}

    for model_name, pipeline in models.items():
        print(f"\n[C4] Training: {model_name} ...")

        pipeline.fit(X_train, y_train)

        y_pred_train = pipeline.predict(X_train)
        y_pred_test  = pipeline.predict(X_test)
        predictions[model_name] = y_pred_test

        train_metrics = _compute_metrics(y_train, y_pred_train)
        test_metrics  = _compute_metrics(y_test,  y_pred_test)

        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=5,
            scoring="neg_mean_squared_error",
        )
        cv_rmse = float(np.sqrt(-cv_scores.mean()))

        print(f"  Train  → RMSE: {train_metrics['rmse']:.4f}  R²: {train_metrics['r2']:.4f}")
        print(f"  Test   → RMSE: {test_metrics['rmse']:.4f}  R²: {test_metrics['r2']:.4f}")
        print(f"  CV-5   → RMSE: {cv_rmse:.4f}")

        rows.append({
            "Model":        model_name,
            "RMSE_train":   round(train_metrics["rmse"], 4),
            "RMSE_test":    round(test_metrics["rmse"],  4),
            "R2_train":     round(train_metrics["r2"],   4),
            "R2_test":      round(test_metrics["r2"],    4),
            "CV_RMSE_mean": round(cv_rmse,              4),
        })

    metrics_table = pd.DataFrame(rows)
    print("\n[C4] ── Metrics Comparison Table ──────────────────────────────────")
    print(metrics_table.to_string(index=False))

    importance_data = _extract_feature_importance(models, feature_names)

    _plot_predictions(y_test, predictions, save_figures, show_figures)
    _plot_metrics_comparison(metrics_table, save_figures, show_figures)
    _plot_feature_importance(importance_data, feature_names, save_figures, show_figures)

    return metrics_table


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_feature_importance(models: dict, feature_names: list) -> dict:
    """
    Extract feature importance scores from each trained model where available.

    Args:
        models (dict): Trained sklearn Pipeline objects keyed by model name.
        feature_names (list): Names of input features.

    Returns:
        dict: {model_name: np.ndarray of importance scores}
    """
    importance_data = {}
    for model_name, pipeline in models.items():
        inner = pipeline.named_steps["model"]
        if hasattr(inner, "coef_"):
            importance_data[model_name] = np.abs(inner.coef_)
        elif hasattr(inner, "feature_importances_"):
            importance_data[model_name] = inner.feature_importances_
    return importance_data


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_predictions(y_test, predictions, save_figures, show_figures):
    """
    Plot actual vs predicted traffic flow for all three models side by side.

    Args:
        y_test (np.ndarray): Actual test values (vehicles/hour).
        predictions (dict): {model_name: predicted values}.
        save_figures (bool): Save figure if True.
        show_figures (bool): Display interactively if True.
    """
    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    colors = ["#0077B6", "#E63946", "#2A9D8F"]
    x_axis = np.arange(len(y_test))

    for ax, (model_name, y_pred), color in zip(axes, predictions.items(), colors):
        ax.plot(x_axis, y_test,  label="Actual",    color="black", linewidth=1.5, alpha=0.7)
        ax.plot(x_axis, y_pred,  label="Predicted", color=color,  linewidth=1.5, linestyle="--")
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2   = float(r2_score(y_test, y_pred))
        ax.set_title(f"{model_name}\nRMSE={rmse:.2f} | R²={r2:.3f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Test Sample Index", fontsize=10)
        ax.set_ylabel("Vehicle Count (vehicles/hour)", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("C4 — Actual vs Predicted Traffic Flow\nby Classical ML Model",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_and_show(fig, "c4_predictions_comparison.png", save_figures, show_figures)


def _plot_metrics_comparison(metrics_table, save_figures, show_figures):
    """
    Bar chart comparing RMSE and R² across the three models.

    Args:
        metrics_table (pd.DataFrame): Output of train_models().
        save_figures (bool): Save figure if True.
        show_figures (bool): Display interactively if True.
    """
    fig, (ax_rmse, ax_r2) = plt.subplots(1, 2, figsize=(12, 5))
    model_names = metrics_table["Model"].tolist()
    rmse_values = metrics_table["RMSE_test"].tolist()
    r2_values   = metrics_table["R2_test"].tolist()
    colors      = ["#0077B6", "#E63946", "#2A9D8F"]

    bars = ax_rmse.bar(model_names, rmse_values, color=colors, edgecolor="black", alpha=0.85)
    ax_rmse.set_title("Test RMSE by Model\n(lower is better)", fontsize=12, fontweight="bold")
    ax_rmse.set_ylabel("RMSE (vehicles/hour)", fontsize=11)
    ax_rmse.set_xlabel("Model", fontsize=11)
    ax_rmse.grid(axis="y", linestyle="--", alpha=0.5)
    for bar, val in zip(bars, rmse_values):
        ax_rmse.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.01 * max(rmse_values),
                     f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    bars = ax_r2.bar(model_names, r2_values, color=colors, edgecolor="black", alpha=0.85)
    ax_r2.set_title("Test R² by Model\n(higher is better)", fontsize=12, fontweight="bold")
    ax_r2.set_ylabel("R² (coefficient of determination)", fontsize=11)
    ax_r2.set_xlabel("Model", fontsize=11)
    ax_r2.set_ylim(0, 1.05)
    ax_r2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar, val in zip(bars, r2_values):
        ax_r2.text(bar.get_x() + bar.get_width() / 2,
                   bar.get_height() + 0.01,
                   f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    fig.suptitle("C4 — Model Performance Comparison (Test Set)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_and_show(fig, "c4_metrics_comparison.png", save_figures, show_figures)


def _plot_feature_importance(importance_data, feature_names, save_figures, show_figures):
    """
    Horizontal bar chart of feature importance for each model.

    Args:
        importance_data (dict): {model_name: np.ndarray of importance scores}.
        feature_names (list): Names of input features.
        save_figures (bool): Save figure if True.
        show_figures (bool): Display interactively if True.
    """
    if not importance_data:
        return

    n_plots = len(importance_data)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, max(3, len(feature_names))))
    if n_plots == 1:
        axes = [axes]

    colors = ["#0077B6", "#E63946", "#2A9D8F"]
    for ax, (model_name, importances), color in zip(axes, importance_data.items(), colors):
        sorted_idx   = np.argsort(importances)
        sorted_names = [feature_names[i] for i in sorted_idx]
        sorted_vals  = importances[sorted_idx]
        ax.barh(sorted_names, sorted_vals, color=color, edgecolor="black", alpha=0.85)
        ax.set_title(f"{model_name}\nFeature Importance", fontsize=11, fontweight="bold")
        ax.set_xlabel("Importance Score", fontsize=10)
        ax.grid(axis="x", linestyle="--", alpha=0.4)

    fig.suptitle("C4 — Feature Importance Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_and_show(fig, "c4_feature_importance.png", save_figures, show_figures)


def _save_and_show(fig, filename, save_figures, show_figures):
    """
    Save a figure to FIGURES_DIR and/or display it.

    Args:
        fig (plt.Figure): Matplotlib figure object.
        filename (str): Output file name.
        save_figures (bool): Write to disk if True.
        show_figures (bool): Call plt.show() if True.
    """
    if save_figures:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = FIGURES_DIR / filename
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[C4] Figure saved → {out_path}")
    if show_figures:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading — reads C1 processed CSV
# ─────────────────────────────────────────────────────────────────────────────

def _load_processed_data(csv_path: Path):
    """
    Load processed dataset from C1 and build feature matrix X and target y.

    Features: hour [0-23], minute [0-59], weekday_number [1-5].
    Target: vehicle detections per (hour, minute, weekday) bucket.

    Args:
        csv_path (Path): Path to data/processed/df.csv.

    Returns:
        tuple: (X, y, feature_names)

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If no weekday data is found.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"[C4] File not found: {csv_path}")

    buckets: dict = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("day_type") != "weekday":
                continue
            key = (int(row["hour"]), int(row["minute"]), int(row["weekday_number"]))
            buckets[key] = buckets.get(key, 0) + 1

    if not buckets:
        raise ValueError("[C4] No weekday data found in the CSV.")

    keys          = sorted(buckets.keys())
    X             = np.array(keys, dtype=float)
    y             = np.array([buckets[k] for k in keys], dtype=float)
    feature_names = ["hour", "minute", "weekday_number"]

    print(f"[C4] Loaded {len(y)} samples from {csv_path}")
    return X, y, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic fallback
# ─────────────────────────────────────────────────────────────────────────────

def _generate_synthetic_data(n_samples: int = 300):
    """
    Generate synthetic traffic data for demo/testing when no CSV is available.

    Simulates morning peak ~8h, evening peak ~17h, midday bump ~13h.

    Args:
        n_samples (int): Number of samples to generate.

    Returns:
        tuple: (X, y, feature_names)
    """
    rng     = np.random.default_rng(RANDOM_STATE)
    hours   = rng.integers(0, 24, n_samples).astype(float)
    minutes = rng.integers(0, 60, n_samples).astype(float)
    weekday = rng.integers(1, 6,  n_samples).astype(float)

    y = (
        20.0
        + 80.0 * np.exp(-0.5 * ((hours - 8)  / 1.5) ** 2)
        + 70.0 * np.exp(-0.5 * ((hours - 17) / 1.5) ** 2)
        + 30.0 * np.exp(-0.5 * ((hours - 13) / 1.0) ** 2)
        + rng.normal(0, 5, n_samples)
    )
    y = np.clip(y, 0, None)

    X             = np.column_stack([hours, minutes, weekday])
    feature_names = ["hour", "minute", "weekday_number"]

    print(f"[C4] Generated {n_samples} synthetic traffic samples (fallback mode).")
    return X, y, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    CSV_PATH = Path("data/processed/df.csv")

    try:
        X, y, feature_names = _load_processed_data(CSV_PATH)
    except (FileNotFoundError, ValueError) as err:
        print(f"[C4] WARNING: {err}")
        print("[C4] Falling back to synthetic traffic data.\n")
        X, y, feature_names = _generate_synthetic_data(n_samples=300)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE,
    )

    print(f"[C4] Train size : {len(X_train)} samples")
    print(f"[C4] Test size  : {len(X_test)} samples\n")

    metrics_table = train_models(
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        feature_names=feature_names,
        save_figures=True, show_figures=True,
    )

    print("\n" + "=" * 60)
    print("C4 — FINAL METRICS SUMMARY")
    print("=" * 60)
    print(metrics_table.to_string(index=False))
    print("=" * 60)

    best_row = metrics_table.loc[metrics_table["RMSE_test"].idxmin()]
    print(f"\n[C4] Best model : {best_row['Model']}")
    print(f"       RMSE     = {best_row['RMSE_test']:.4f} vehicles/hour")
    print(f"       R²       = {best_row['R2_test']:.4f}")
    print("\n[C4] Module complete.")
