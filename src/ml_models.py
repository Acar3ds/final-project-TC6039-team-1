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

    Data is aggregated by (date, hour) — one row per hour per day —
    so predictions are plotted against real timestamps (Mon–Fri timeline),
    not arbitrary sample indices.

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

FIX APPLIED (v2):
    FIX — Linear Regression extrapolation (negative predictions):
        The 'month' integer feature caused Linear Regression to extrapolate
        a downward trend when the test set fell in a later month than the
        training set (e.g. train = Jan, test = Feb/Mar), producing predictions
        as low as -50 veh/h.  The feature has been replaced with cyclic
        sin/cos encoding of the month, which is bounded in [-1, 1] and never
        extrapolates outside that range.  Combined with the already-present
        hour_sin / hour_cos features, the model now tracks seasonal patterns
        without going negative.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Colab auto-setup
# ─────────────────────────────────────────────────────────────────────────────

import subprocess
import sys
import os
from pathlib import Path

def _setup():
    """
    Install missing dependencies and create required folder structure.
    Runs automatically so no manual setup is needed in Colab.
    """
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
from sklearn.tree import DecisionTreeRegressor
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
        timestamps_test (pd.DatetimeIndex | None): Real datetime for each test
            sample. When provided, the prediction plot uses a real time axis
            (Mon–Fri timeline) instead of sample indices.
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

        # Vehicle counts cannot be negative — clip LR predictions to 0.
        # Tree-based models already predict non-negative values naturally
        # (they average training targets which are ≥ 0), so clipping is
        # only needed for the unconstrained linear model.
        if model_name == "Linear Regression":
            y_pred_train = np.clip(y_pred_train, 0, None)
            y_pred_test  = np.clip(y_pred_test,  0, None)
        predictions[model_name] = y_pred_test

        train_metrics = _compute_metrics(y_train, y_pred_train)
        test_metrics  = _compute_metrics(y_test,  y_pred_test)

        # Adjust number of CV folds to the available training samples
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

    _plot_predictions(y_test, predictions, timestamps_test, save_figures, show_figures)
    _plot_metrics_comparison(metrics_table, save_figures, show_figures)
    _plot_feature_importance(importance_data, feature_names, save_figures, show_figures)

    return metrics_table


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

def _insert_weekend_gaps(x_axis, y_real, preds_sorted, gap_threshold_hours=12):
    """
    FIX 1 — Insert NaN where consecutive timestamps are more than
    gap_threshold_hours apart (i.e. weekend gaps).

    Matplotlib skips NaN values, so the line is broken instead of drawing a
    misleading diagonal across Saturday and Sunday.

    Args:
        x_axis (np.ndarray): Sorted datetime array.
        y_real (np.ndarray): Actual traffic values.
        preds_sorted (dict): {model_name: predicted values array}.
        gap_threshold_hours (int): Minimum gap (hours) to be considered a break.

    Returns:
        tuple: (x_with_nans, y_with_nans, preds_with_nans)
            All arrays have NaN rows inserted at each detected gap.
    """
    if len(x_axis) == 0:
        return x_axis, y_real, preds_sorted

    # Detect gap positions (index i means a gap between i and i+1)
    time_diffs = np.diff(x_axis.astype("datetime64[h]").astype(float))
    gap_positions = np.where(time_diffs > gap_threshold_hours)[0]

    if len(gap_positions) == 0:
        return x_axis, y_real, preds_sorted

    # Build new arrays by inserting NaN rows after each gap
    x_out    = list(x_axis)
    y_out    = list(y_real.astype(float))
    preds_out = {name: list(arr.astype(float)) for name, arr in preds_sorted.items()}

    # Insert from the end so indices stay valid
    for pos in reversed(gap_positions):
        insert_at = pos + 1
        # Midpoint timestamp for the NaN marker (purely cosmetic)
        mid_ts = x_axis[pos] + (x_axis[pos + 1] - x_axis[pos]) / 2
        x_out.insert(insert_at, mid_ts)
        y_out.insert(insert_at, np.nan)
        for name in preds_out:
            preds_out[name].insert(insert_at, np.nan)

    x_out_arr = np.array(x_out)
    y_out_arr = np.array(y_out)
    preds_out_arr = {name: np.array(arr) for name, arr in preds_out.items()}

    print(f"[C4] Inserted {len(gap_positions)} weekend gap(s) in prediction plot.")
    return x_out_arr, y_out_arr, preds_out_arr


def _plot_predictions(y_test, predictions, timestamps_test, save_figures, show_figures):
    """
    Plot actual vs predicted traffic flow using a categorical x-axis so that
    weekend days (Sat/Sun) never appear and leave no blank space.

    Strategy: sort all test samples chronologically, then use their integer
    position (0, 1, 2, …) as the x coordinate.  Day-change tick marks are
    inserted where the date changes, and labeled with "Weekday DD Mon".
    This guarantees that only the data that actually exists occupies space on
    the axis — weekends are completely absent.

    Args:
        y_test (np.ndarray): Actual test values (vehicles/hour).
        predictions (dict): {model_name: predicted values array}.
        timestamps_test (pd.DatetimeIndex | None): Datetime for each test sample.
        save_figures (bool): Save figure to FIGURES_DIR if True.
        show_figures (bool): Display interactively if True.
    """
    use_time_axis = timestamps_test is not None and len(timestamps_test) == len(y_test)

    if use_time_axis:
        # ── Sort chronologically ──────────────────────────────────────────────
        sort_idx     = np.argsort(timestamps_test)
        ts_sorted    = pd.DatetimeIndex(np.array(timestamps_test)[sort_idx])
        y_real       = y_test[sort_idx]
        preds_sorted = {name: arr[sort_idx] for name, arr in predictions.items()}

        # ── Categorical x: just integer positions 0..N-1 ─────────────────────
        x_axis = np.arange(len(y_real))

        # ── Build day-boundary ticks ──────────────────────────────────────────
        # Find the index of the first sample of each new calendar date.
        dates      = ts_sorted.date
        tick_pos   = []
        tick_label = []
        prev_date  = None
        for i, d in enumerate(dates):
            if d != prev_date:
                day_name  = ts_sorted[i].strftime("%a")   # Mon, Tue …
                date_str  = ts_sorted[i].strftime("%d %b") # 24 Feb
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

        # Draw a subtle vertical line at each day boundary
        for tp in tick_pos[1:]:   # skip index 0 — no line at the very start
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

    axes[-1].set_xlabel("Date and Hour (Weekday Timeline — weekends excluded)",
                        fontsize=11)

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
# Data loading — same grouping as C5 (date + hour) to keep timestamps
# ─────────────────────────────────────────────────────────────────────────────

def _load_processed_data(csv_path: Path):
    """
    Load processed dataset from C1 and aggregate by (date, hour).

    FIX 2 — 'month' replaced with cyclic (month_sin, month_cos):
        Using month as a plain integer caused Linear Regression to learn a
        monotone trend and extrapolate out of range on the test set (negative
        predictions).  Cyclic encoding maps month onto the unit circle so the
        model sees January and December as adjacent and never extrapolates
        outside [-1, 1].

    Features: hour, weekday_number, is_weekday, hour_sin, hour_cos,
              month_sin, month_cos.
    Target: vehicles_per_hour — total detections in that hour.

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

    df_hourly["date"]      = pd.to_datetime(df_hourly["date"])
    df_hourly["datetime"]  = df_hourly["date"] + pd.to_timedelta(df_hourly["hour"], unit="h")
    df_hourly["is_weekday"] = 1.0

    # ── Cyclic month encoding (bounded, no extrapolation) ─────────────────────
    month = df_hourly["date"].dt.month
    df_hourly["month_sin"] = np.sin(2 * np.pi * month / 12)
    df_hourly["month_cos"] = np.cos(2 * np.pi * month / 12)

    # ── Cyclic hour encoding — 1st and 2nd harmonic ───────────────────────────
    # 1st harmonic: one cycle per day — captures the broad day/night rhythm.
    # 2nd harmonic: two cycles per day — captures the BIMODAL traffic pattern
    #   (morning peak ~8h + evening peak ~17h). Without the 2nd harmonic, Linear
    #   Regression can only fit a single sinusoid and is forced into large
    #   negative values to simultaneously reach two separate peaks.
    h = df_hourly["hour"]
    df_hourly["hour_sin"]  = np.sin(2 * np.pi * h / 24)
    df_hourly["hour_cos"]  = np.cos(2 * np.pi * h / 24)
    df_hourly["hour_sin2"] = np.sin(4 * np.pi * h / 24)   # 2nd harmonic
    df_hourly["hour_cos2"] = np.cos(4 * np.pi * h / 24)
    # ── Cyclic weekday encoding ───────────────────────────────────────────────
    # weekday_number as a raw integer (1=Mon … 5=Fri) misleads Linear Regression
    # into a spurious linear trend across the week. Mapping onto a 5-point circle
    # keeps it bounded: Mon and Fri are now adjacent, no artificial ordering.
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

    # Cyclic encodings — 1st and 2nd harmonic of hour
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

    # Try GitHub first
    try:
        print("[C4] Loading data from GitHub...")
        df_raw = pd.read_csv(GITHUB_URL)
        df_wk  = df_raw[df_raw["day_type"] == "weekday"].copy()
        if df_wk.empty:
            raise ValueError("No weekday data found in GitHub CSV.")
        df_hourly = df_wk.groupby(["date", "hour", "weekday_number"]).size().reset_index(
            name="vehicles_per_hour"
        )
        df_hourly["date"]      = pd.to_datetime(df_hourly["date"])
        df_hourly["datetime"]  = df_hourly["date"] + pd.to_timedelta(df_hourly["hour"], unit="h")
        df_hourly["is_weekday"] = 1.0

        # ── Cyclic month encoding ──────────────────────────────────────────
        month = df_hourly["date"].dt.month
        df_hourly["month_sin"] = np.sin(2 * np.pi * month / 12)
        df_hourly["month_cos"] = np.cos(2 * np.pi * month / 12)

        # ── Cyclic hour encoding — 1st and 2nd harmonic ───────────────────
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

    # Try local CSV if GitHub failed
    if X is None:
        try:
            X, y, feature_names, timestamps = _load_processed_data(CSV_PATH)
        except (FileNotFoundError, ValueError) as err:
            print(f"[C4] Local CSV failed: {err}")

    # Fallback to synthetic
    if X is None:
        print("[C4] Falling back to synthetic traffic data.\n")
        X, y, feature_names, timestamps = _generate_synthetic_data(n_days=10)

    # ── Chronological train/test split (80% train / 20% test) ───────────────
    split   = int(0.8 * len(X))
    X_train, X_test = X[:split],     X[split:]
    y_train, y_test = y[:split],     y[split:]
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
