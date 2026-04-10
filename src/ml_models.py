# -*- coding: utf-8 -*-
"""
ml_models.py
C4 — Machine Learning Classical Models Module
Author: [Your Name] | TC6039.1 Applied Computing — Final Project
"""

# ─────────────────────────────────────────────────────────────────────────────
# Colab auto-setup
# ─────────────────────────────────────────────────────────────────────────────

import subprocess
import sys
import os
from pathlib import Path

def _setup():
    try:
        import pytest
    except ImportError:
        print("[C4] Installing pytest...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest", "-q"])
        except Exception:
            pass

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    if Path("df.csv").exists() and not Path("data/processed/df.csv").exists():
        import shutil
        shutil.copy("df.csv", "data/processed/df.csv")
        print("[C4] df.csv copied to data/processed/df.csv")

_setup()

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

FIGURES_DIR  = Path("reports/figures")
RANDOM_STATE = 42

# ─────────────────────────────────────────────────────────────────────────────
# Helper — metrics
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
# Core public function
# ─────────────────────────────────────────────────────────────────────────────

def train_models(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
    feature_names: list = None,
    timestamps_test: pd.DatetimeIndex = None,
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
        X_test  (np.ndarray): Test feature matrix [p_samples, n_features].
        y_train (np.ndarray): Training target — vehicle counts (vehicles/hour).
        y_test  (np.ndarray): Test target — vehicle counts (vehicles/hour).
        feature_names (list | None): Names of features for plots.
        timestamps_test (pd.DatetimeIndex | None): Real datetime for each test sample.
        save_figures (bool): Save generated plots to disk. Default True.
        show_figures (bool): Display plots interactively. Default True.

    Returns:
        pd.DataFrame: Comparison table —
                      [Model, RMSE_train, RMSE_test, R2_train, R2_test, CV_RMSE_mean].

    Raises:
        AssertionError : If X_train is empty.
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

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
        X_test  = X_test.reshape(-1, 1)

    n_features = X_train.shape[1]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    print(f"[C4] Training set  : {X_train.shape[0]} samples, {n_features} features")
    print(f"[C4] Test set      : {X_test.shape[0]} samples")
    print(f"[C4] Features      : {feature_names}")

    # ── Define models ─────────────────────────────────────────────────────────
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

    # ── Train, predict, compute metrics ───────────────────────────────────────
    rows        = []
    predictions = {}

    for model_name, pipeline in models.items():
        print(f"\n[C4] Training: {model_name} ...")

        pipeline.fit(X_train, y_train)

        y_pred_train = pipeline.predict(X_train)
        y_pred_test  = pipeline.predict(X_test)

        if model_name == "Linear Regression":
            y_pred_train = np.clip(y_pred_train, 0, None)
            y_pred_test  = np.clip(y_pred_test,  0, None)
        predictions[model_name] = y_pred_test

        train_metrics = _compute_metrics(y_train, y_pred_train)
        test_metrics  = _compute_metrics(y_test,  y_pred_test)

        n_folds = min(5, len(X_train))
        if n_folds >= 2:
            cv_scores = cross_val_score(
                pipeline, X_train, y_train, cv=n_folds,
                scoring="neg_mean_squared_error",
            )
            cv_rmse = float(np.sqrt(-cv_scores.mean()))
        else:
            cv_rmse = train_metrics["rmse"]
            print(f"  [C4] Warning: only {len(X_train)} training sample(s) — "
                  f"CV skipped, using train RMSE as proxy.")

        print(f"  Train  → RMSE: {train_metrics['rmse']:.4f}  R²: {train_metrics['r2']:.4f}")
        print(f"  Test   → RMSE: {test_metrics['rmse']:.4f}  R²: {test_metrics['r2']:.4f}")
        print(f"  CV-{n_folds}   → RMSE: {cv_rmse:.4f}")

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

    # ── Standard plots ────────────────────────────────────────────────────────
    _plot_predictions(y_test, predictions, timestamps_test, save_figures, show_figures)
    _plot_metrics_comparison(metrics_table, save_figures, show_figures)
    _plot_feature_importance(importance_data, feature_names, save_figures, show_figures)

    # ── NEW: Model structure plots ────────────────────────────────────────────
    _plot_linear_regression_coefficients(models["Linear Regression"], feature_names, save_figures, show_figures)
    _plot_decision_tree(models["Decision Tree"], feature_names, save_figures, show_figures)
    _plot_random_forest_sample_trees(models["Random Forest"], feature_names, save_figures, show_figures)

    return metrics_table


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Model structure visualizations
# ─────────────────────────────────────────────────────────────────────────────

def _plot_linear_regression_coefficients(pipeline, feature_names, save_figures, show_figures):
    """
    Plot Linear Regression coefficients after inverse-scaling.

    The pipeline applies StandardScaler before fitting, so the raw coef_
    values correspond to scaled features.  We multiply each coefficient by
    the feature's standard deviation to express it in the original feature
    units (i.e. the change in vehicles/hour per unit change in the raw
    cyclic feature), making the bar chart directly interpretable.

    Args:
        pipeline: Trained sklearn Pipeline with 'scaler' and 'model' steps.
        feature_names (list): Names of input features.
        save_figures (bool): Save figure if True.
        show_figures (bool): Display interactively if True.
    """
    model  = pipeline.named_steps["model"]
    scaler = pipeline.named_steps["scaler"]

    # Rescale coefficients to original feature space
    coefs = model.coef_ * scaler.scale_

    sorted_idx   = np.argsort(np.abs(coefs))
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_coefs = coefs[sorted_idx]
    colors       = ["#E63946" if c < 0 else "#0077B6" for c in sorted_coefs]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(sorted_names, sorted_coefs, color=colors, edgecolor="black", alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(
        "Linear Regression — Coefficients\n"
        "(rescaled to original feature units · blue = positive, red = negative)",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Coefficient value (vehicles/hour per unit of feature)", fontsize=10)
    ax.set_ylabel("Feature", fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, sorted_coefs):
        ax.text(
            val + (0.002 * np.abs(sorted_coefs).max() * np.sign(val) if val != 0 else 0.002),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left" if val >= 0 else "right", fontsize=8
        )

    intercept = model.intercept_
    ax.text(
        0.98, 0.02,
        f"Intercept = {intercept:.2f} veh/h",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, color="gray",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray")
    )

    fig.tight_layout()
    _save_and_show(fig, "c4_linear_regression_coefficients.png", save_figures, show_figures)


def _plot_decision_tree(pipeline, feature_names, save_figures, show_figures):
    """
    Visualize the full Decision Tree structure using sklearn's plot_tree.

    The tree is trained with max_depth=5 (set in train_models), so the
    diagram is readable.  Each node shows the split condition, number of
    samples, and predicted value (mean vehicles/hour of samples in that node).

    Args:
        pipeline: Trained sklearn Pipeline with 'scaler' and 'model' steps.
        feature_names (list): Names of input features.
        save_figures (bool): Save figure if True.
        show_figures (bool): Display interactively if True.
    """
    dt_model = pipeline.named_steps["model"]

    fig, ax = plt.subplots(figsize=(28, 12))
    plot_tree(
        dt_model,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        fontsize=7,
        ax=ax,
        precision=2,
        impurity=False,          # hide MSE — keeps the diagram clean
    )
    ax.set_title(
        f"Decision Tree Structure — max_depth={dt_model.max_depth} | "
        f"{dt_model.tree_.node_count} nodes | "
        f"{dt_model.tree_.n_leaves} leaves",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    _save_and_show(fig, "c4_decision_tree_structure.png", save_figures, show_figures)

    # Also print a compact text version for reference
    print("\n[C4] Decision Tree — text summary (depth ≤ 3):")
    print(export_text(dt_model, feature_names=feature_names, max_depth=3))


def _plot_random_forest_sample_trees(pipeline, feature_names, save_figures, show_figures,
                                     n_sample_trees: int = 3):
    """
    Plot a sample of individual trees from the Random Forest side by side.

    Drawing all 100 trees is impractical, so we show the first n_sample_trees
    estimators (depth capped at 3 for readability).  A note in the title
    reminds the reader that the final prediction is the average of all trees.

    Args:
        pipeline: Trained sklearn Pipeline with 'scaler' and 'model' steps.
        feature_names (list): Names of input features.
        save_figures (bool): Save figure if True.
        show_figures (bool): Display interactively if True.
        n_sample_trees (int): Number of individual trees to display. Default 3.
    """
    rf_model    = pipeline.named_steps["model"]
    total_trees = len(rf_model.estimators_)
    n_show      = min(n_sample_trees, total_trees)

    fig, axes = plt.subplots(1, n_show, figsize=(18 * n_show // 3, 10))
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        plot_tree(
            rf_model.estimators_[i],
            feature_names=feature_names,
            filled=True,
            rounded=True,
            fontsize=7,
            ax=ax,
            max_depth=3,          # cap depth for readability
            precision=2,
            impurity=False,
        )
        ax.set_title(
            f"Tree #{i + 1} of {total_trees}\n(displayed depth ≤ 3)",
            fontsize=11, fontweight="bold"
        )

    fig.suptitle(
        f"Random Forest — Sample of {n_show} Individual Trees\n"
        f"(final prediction = average of all {total_trees} trees)",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    _save_and_show(fig, "c4_random_forest_sample_trees.png", save_figures, show_figures)


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance
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
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _plot_predictions(y_test, predictions, timestamps_test, save_figures, show_figures):
    """
    Plot actual vs predicted traffic flow using a categorical x-axis.

    Args:
        y_test (np.ndarray): Actual test values (vehicles/hour).
        predictions (dict): {model_name: predicted values array}.
        timestamps_test (pd.DatetimeIndex | None): Datetime for each test sample.
        save_figures (bool): Save figure to FIGURES_DIR if True.
        show_figures (bool): Display interactively if True.
    """
    use_time_axis = timestamps_test is not None and len(timestamps_test) == len(y_test)

    if use_time_axis:
        sort_idx     = np.argsort(timestamps_test)
        ts_sorted    = pd.DatetimeIndex(np.array(timestamps_test)[sort_idx])
        y_real       = y_test[sort_idx]
        preds_sorted = {name: arr[sort_idx] for name, arr in predictions.items()}

        x_axis = np.arange(len(y_real))

        dates      = ts_sorted.date
        tick_pos   = []
        tick_label = []
        prev_date  = None
        for i, d in enumerate(dates):
            if d != prev_date:
                day_name  = ts_sorted[i].strftime("%a")
                date_str  = ts_sorted[i].strftime("%d %b")
                tick_pos.append(i)
                tick_label.append(f"{day_name}\n{date_str}")
                prev_date = d
    else:
        x_axis       = np.arange(len(y_test))
        y_real       = y_test
        preds_sorted = predictions
        tick_pos     = []
        tick_label   = []

    n_models = len(predictions)
    fig, axes = plt.subplots(n_models, 1, figsize=(14, 4 * n_models), sharex=True)
    if n_models == 1:
        axes = [axes]

    colors = ["#0077B6", "#E63946", "#2A9D8F"]

    for ax, (model_name, y_pred), color in zip(axes, preds_sorted.items(), colors):
        ax.plot(x_axis, y_real, label="Real Traffic",
                color="black", linewidth=1.8, alpha=0.8, marker="o", markersize=3)
        ax.plot(x_axis, y_pred, label=f"{model_name} Prediction",
                color=color, linewidth=1.8, linestyle="--", marker="x", markersize=3)

        for tp in tick_pos[1:]:
            ax.axvline(tp, color="gray", linewidth=0.6, linestyle=":", alpha=0.5)

        rmse = float(np.sqrt(mean_squared_error(y_real, y_pred)))
        r2   = float(r2_score(y_real, y_pred))

        ax.set_title(f"{model_name} — RMSE={rmse:.2f} veh/h | R²={r2:.3f}",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("Vehicles per Hour", fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.4)

        if use_time_axis and tick_pos:
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_label, fontsize=8, ha="center")

    axes[-1].set_xlabel("Date and Hour (Weekday Timeline — weekends excluded)", fontsize=11)
    fig.suptitle("C4 — Real vs Predicted Traffic Flow by Classical ML Model",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_and_show(fig, "c4_predictions_timeline.png", save_figures, show_figures)


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
        try:
            from IPython.display import display
            display(fig)
        except ImportError:
            plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_processed_data(csv_path: Path):
    """
    Load processed dataset from C1 and aggregate by (date, hour).

    Args:
        csv_path (Path): Path to data/processed/df.csv.

    Returns:
        tuple: (X, y, feature_names, timestamps)

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If no weekday data is found.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"[C4] File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df[df["day_type"] == "weekday"].copy()

    if df.empty:
        raise ValueError("[C4] No weekday data found in the CSV.")

    df_hourly = df.groupby(["date", "hour", "weekday_number"]).size().reset_index(
        name="vehicles_per_hour"
    )

    df_hourly["date"]       = pd.to_datetime(df_hourly["date"])
    df_hourly["datetime"]   = df_hourly["date"] + pd.to_timedelta(df_hourly["hour"], unit="h")
    df_hourly["is_weekday"] = 1.0

    month = df_hourly["date"].dt.month
    df_hourly["month_sin"] = np.sin(2 * np.pi * month / 12)
    df_hourly["month_cos"] = np.cos(2 * np.pi * month / 12)

    h = df_hourly["hour"]
    df_hourly["hour_sin"]    = np.sin(2 * np.pi * h / 24)
    df_hourly["hour_cos"]    = np.cos(2 * np.pi * h / 24)
    df_hourly["hour_sin2"]   = np.sin(4 * np.pi * h / 24)
    df_hourly["hour_cos2"]   = np.cos(4 * np.pi * h / 24)

    wd = df_hourly["weekday_number"]
    df_hourly["weekday_sin"] = np.sin(2 * np.pi * (wd - 1) / 5)
    df_hourly["weekday_cos"] = np.cos(2 * np.pi * (wd - 1) / 5)

    feature_names = [
        "hour_sin", "hour_cos",
        "hour_sin2", "hour_cos2",
        "weekday_sin", "weekday_cos",
        "month_sin", "month_cos",
    ]

    X          = df_hourly[feature_names].values.astype(float)
    y          = df_hourly["vehicles_per_hour"].values.astype(float)
    timestamps = pd.DatetimeIndex(df_hourly["datetime"])

    print(f"[C4] Loaded {len(y)} hourly samples from {csv_path}")
    return X, y, feature_names, timestamps


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic fallback
# ─────────────────────────────────────────────────────────────────────────────

def _generate_synthetic_data(n_days: int = 10):
    """
    Generate synthetic hourly traffic data for demo/testing.

    Args:
        n_days (int): Number of weekdays to simulate.

    Returns:
        tuple: (X, y, feature_names, timestamps)
    """
    rng = np.random.default_rng(RANDOM_STATE)

    rows = []
    start = pd.Timestamp("2026-03-02")
    day   = start
    days_added = 0

    while days_added < n_days:
        if day.weekday() < 5:
            for hour in range(24):
                vehicles = (
                    20.0
                    + 80.0 * np.exp(-0.5 * ((hour - 8)  / 1.5) ** 2)
                    + 70.0 * np.exp(-0.5 * ((hour - 17) / 1.5) ** 2)
                    + 30.0 * np.exp(-0.5 * ((hour - 13) / 1.0) ** 2)
                    + rng.normal(0, 5)
                )
                vehicles = max(0.0, vehicles)
                rows.append({
                    "datetime":       day + pd.Timedelta(hours=hour),
                    "hour":           hour,
                    "weekday_number": day.weekday() + 1,
                    "is_weekday":     1.0,
                    "month":          day.month,
                    "vehicles":       vehicles,
                })
            days_added += 1
        day += pd.Timedelta(days=1)

    df_syn = pd.DataFrame(rows)

    df_syn["hour_sin"]    = np.sin(2 * np.pi * df_syn["hour"] / 24)
    df_syn["hour_cos"]    = np.cos(2 * np.pi * df_syn["hour"] / 24)
    df_syn["hour_sin2"]   = np.sin(4 * np.pi * df_syn["hour"] / 24)
    df_syn["hour_cos2"]   = np.cos(4 * np.pi * df_syn["hour"] / 24)
    df_syn["month_sin"]   = np.sin(2 * np.pi * df_syn["month"] / 12)
    df_syn["month_cos"]   = np.cos(2 * np.pi * df_syn["month"] / 12)
    df_syn["weekday_sin"] = np.sin(2 * np.pi * (df_syn["weekday_number"] - 1) / 5)
    df_syn["weekday_cos"] = np.cos(2 * np.pi * (df_syn["weekday_number"] - 1) / 5)

    feature_names = [
        "hour_sin", "hour_cos",
        "hour_sin2", "hour_cos2",
        "weekday_sin", "weekday_cos",
        "month_sin", "month_cos",
    ]

    X          = df_syn[feature_names].values.astype(float)
    y          = df_syn["vehicles"].values.astype(float)
    timestamps = pd.DatetimeIndex(df_syn["datetime"])

    print(f"[C4] Generated {len(y)} synthetic hourly samples ({n_days} weekdays).")
    return X, y, feature_names, timestamps


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    GITHUB_URL = (
        "https://raw.githubusercontent.com/Acar3ds/final-project-TC6039-team-1"
        "/refs/heads/C1-Data-Loader/data/processed/df.csv"
    )
    CSV_PATH = Path("data/processed/df.csv")

    X, y, feature_names, timestamps = None, None, None, None

    # ── Tier 1: GitHub ────────────────────────────────────────────────────────
    try:
        print("[C4] Loading data from GitHub...")
        df_raw = pd.read_csv(GITHUB_URL)
        df_wk  = df_raw[df_raw["day_type"] == "weekday"].copy()
        if df_wk.empty:
            raise ValueError("No weekday data found in GitHub CSV.")
        df_hourly = df_wk.groupby(["date", "hour", "weekday_number"]).size().reset_index(
            name="vehicles_per_hour"
        )
        df_hourly["date"]       = pd.to_datetime(df_hourly["date"])
        df_hourly["datetime"]   = df_hourly["date"] + pd.to_timedelta(df_hourly["hour"], unit="h")
        df_hourly["is_weekday"] = 1.0

        month = df_hourly["date"].dt.month
        df_hourly["month_sin"] = np.sin(2 * np.pi * month / 12)
        df_hourly["month_cos"] = np.cos(2 * np.pi * month / 12)

        h = df_hourly["hour"]
        df_hourly["hour_sin"]    = np.sin(2 * np.pi * h / 24)
        df_hourly["hour_cos"]    = np.cos(2 * np.pi * h / 24)
        df_hourly["hour_sin2"]   = np.sin(4 * np.pi * h / 24)
        df_hourly["hour_cos2"]   = np.cos(4 * np.pi * h / 24)

        wd = df_hourly["weekday_number"]
        df_hourly["weekday_sin"] = np.sin(2 * np.pi * (wd - 1) / 5)
        df_hourly["weekday_cos"] = np.cos(2 * np.pi * (wd - 1) / 5)

        feature_names = [
            "hour_sin", "hour_cos",
            "hour_sin2", "hour_cos2",
            "weekday_sin", "weekday_cos",
            "month_sin", "month_cos",
        ]
        X          = df_hourly[feature_names].values.astype(float)
        y          = df_hourly["vehicles_per_hour"].values.astype(float)
        timestamps = pd.DatetimeIndex(df_hourly["datetime"])
        print(f"[C4] Loaded {len(y)} hourly samples from GitHub.")
    except Exception as e:
        print(f"[C4] GitHub load failed: {e}")

    # ── Tier 2: local CSV ─────────────────────────────────────────────────────
    if X is None:
        try:
            X, y, feature_names, timestamps = _load_processed_data(CSV_PATH)
        except (FileNotFoundError, ValueError) as err:
            print(f"[C4] Local CSV failed: {err}")

    # ── Tier 3: synthetic ─────────────────────────────────────────────────────
    if X is None:
        print("[C4] Falling back to synthetic traffic data.\n")
        X, y, feature_names, timestamps = _generate_synthetic_data(n_days=10)

    # ── Chronological 80/20 split ─────────────────────────────────────────────
    split   = int(0.8 * len(X))
    X_train, X_test = X[:split],  X[split:]
    y_train, y_test = y[:split],  y[split:]
    ts_test         = timestamps[split:]

    print(f"[C4] Train size : {len(X_train)} hourly samples")
    print(f"[C4] Test size  : {len(X_test)}  hourly samples\n")

    metrics_table = train_models(
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        feature_names=feature_names,
        timestamps_test=ts_test,
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
