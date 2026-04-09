# -*- coding: utf-8 -*-
"""
optimizer.py
C3 — Optimization Module
Author: Kenia Gabriela Hermida Núñez
Module: optimize_parameters(model_coeffs, X, y, degree, ...) → optimal_params, convergence_history, sensitivity_results
Purpose: Take the regression model produced by C2 (regression.py) and optimize its
         coefficients to further minimize RMSE, using:
             1. Custom gradient descent (step-by-step, convergence tracked)
             2. scipy BFGS (quasi-Newton, for comparison)
         Automatically plots the convergence curve and runs a sensitivity
         analysis on one coefficient — both required by the C3 pseudocode spec.
Depends on:
    data_loader.py (C1) — must be run first to produce data/processed/df.csv.
                           df.csv must contain columns: hour, minute, day_type.
    regression.py  (C2) — fit_regression() returns a model dict; optimizer.py
                           consumes model["coeffs"] as the starting point.
Execution order:
    C1 (data_loader.py) → produces df.csv
    C2 (regression.py)  → consumes df.csv, produces model dict
    C3 (optimizer.py)   → consumes C2 model dict + df.csv, produces optimal params
Complexity: O(k · n)  where k = number of GD iterations, n = number of samples

Interface (Wilson et al. §7a — document purpose of every public function):
    optimize_parameters(model_coeffs, X, y, degree, ...) → (np.ndarray, list[float], dict)
    plot_convergence_curve(convergence_history, ...)      → None
    sensitivity_analysis(model_coeffs, X, y, degree, ...) → dict
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe in Colab / CI
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize

# C2 interface: fit_regression returns a dict with keys
# 'coeffs', 'degree', 'r2', 'rmse', 'mse', 'y_hat', 'residuals'
from regression import fit_regression


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(X: np.ndarray) -> "tuple[np.ndarray, float, float]":
    """
    Z-score normalise a 1-D array to zero mean and unit standard deviation.

    Args:
        X: 1-D feature array (n,).

    Returns:
        Tuple (X_norm, mean, std) — caller may invert with X * std + mean.

    Notes:
        Applied internally so that large raw values (e.g. minutes since midnight
        ~720–839 or their squares ~518 000) don't overshoot the gradient step.
        All reported RMSE values remain on the original y scale.
    """
    mu  = float(X.mean())
    std = float(X.std())
    if std < 1e-12:         # avoid division-by-zero on constant features
        std = 1.0
    return (X - mu) / std, mu, std


def _build_design_matrix(X: np.ndarray, degree: int) -> np.ndarray:
    """
    Build a polynomial design matrix [1 | x | x² | x³] consistent with regression.py.

    Args:
        X:      Feature array of shape (n,) or (n, 1).
        degree: Polynomial degree — one of {1, 2, 3}.

    Returns:
        Design matrix of shape (n, degree + 1) with intercept column first.

    Raises:
        AssertionError: If degree is not in {1, 2, 3}.
    """
    assert degree in (1, 2, 3), "degree must be 1, 2, or 3"

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    X_poly = X.copy()
    if degree >= 2:
        X_poly = np.hstack([X_poly, X ** 2])
    if degree >= 3:
        X_poly = np.hstack([X_poly, X ** 3])

    return np.hstack([np.ones((X_poly.shape[0], 1)), X_poly])


def _rmse_objective(coeffs: np.ndarray, X_design: np.ndarray, y: np.ndarray) -> float:
    """
    Objective function  f(theta) = RMSE(y, X_design @ theta).

    Args:
        coeffs:   Coefficient vector theta of shape (degree + 1,).
        X_design: Design matrix of shape (n, degree + 1).
        y:        Target vector of shape (n,).

    Returns:
        float: Root mean squared error.
    """
    y_hat = X_design @ coeffs
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def _rmse_gradient(
    coeffs: np.ndarray, X_design: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """
    Analytical gradient of RMSE with respect to theta:
        d(RMSE)/d(theta) = -(1 / (n * RMSE)) * X^T (y - X*theta)

    Args:
        coeffs:   Coefficient vector theta of shape (degree + 1,).
        X_design: Design matrix of shape (n, degree + 1).
        y:        Target vector of shape (n,).

    Returns:
        Gradient vector of shape (degree + 1,).
    """
    n         = len(y)
    y_hat     = X_design @ coeffs
    residuals = y - y_hat
    rmse_val  = float(np.sqrt(np.mean(residuals ** 2)))

    if rmse_val < 1e-12:            # already at (near) zero — gradient is zero
        return np.zeros_like(coeffs)

    return -(1.0 / (n * rmse_val)) * (X_design.T @ residuals)


# ─────────────────────────────────────────────────────────────────────────────
# C3 Step 4 — convergence curve plot (called inside optimize_parameters)
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence_curve(
    convergence_history: "list[float]",
    save_path: "str | None" = "reports/figures/convergence_curve.png",
    show: bool = False,
) -> None:
    """
    Plot objective value (RMSE) versus gradient-descent iteration.

    Called automatically by optimize_parameters() — satisfies the C3 pseudocode
    requirement "Track convergence of the objective function."

    Args:
        convergence_history: RMSE recorded at each GD iteration.
        save_path:           File path to save the PNG (None = skip saving).
        show:                Display the figure interactively (default False for CI).

    Returns:
        None
    """
    if not convergence_history:
        print("[C3] convergence_history is empty — nothing to plot.")
        return

    iterations = list(range(1, len(convergence_history) + 1))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(iterations, convergence_history, color="#0077B6", linewidth=2,
            label="RMSE per iteration")
    ax.axhline(
        convergence_history[-1], color="#E63946", linewidth=1.2, linestyle="--",
        label=f"Final RMSE = {convergence_history[-1]:.4f}",
    )
    ax.set_xlabel("Iteration (vehicles / hour [count])", fontsize=12)
    ax.set_ylabel("Objective  f(θ) = RMSE  [vehicles]", fontsize=12)
    ax.set_title(
        "C3 — Optimization Convergence Curve\n"
        "Gradient Descent: RMSE vs Iteration",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[C3] Convergence curve saved → {out}")

    if show:
        plt.show()

    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# C3 Step 5 — sensitivity analysis (called inside optimize_parameters)
# ─────────────────────────────────────────────────────────────────────────────

def sensitivity_analysis(
    model_coeffs: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    degree: int = 1,
    param_index: int = 1,
    perturbation_range: "np.ndarray | None" = None,
    save_path: "str | None" = "reports/figures/sensitivity_analysis.png",
    show: bool = False,
) -> dict:
    """
    Vary one coefficient of the optimized model and observe how RMSE changes.

    Satisfies the C3 pseudocode requirement: "Include analysis of sensitivity —
    how does the solution change when a constraint is varied?"

    Args:
        model_coeffs:      Optimized coefficient vector from optimize_parameters().
        X:                 Feature array (n,) — should be the NORMALISED X used
                           during optimization.
        y:                 Target vector of shape (n,) — vehicle counts [vehicles].
        degree:            Polynomial degree used during fitting.
        param_index:       Index of the coefficient to perturb
                           (0 = intercept, 1 = θ₁, 2 = θ₂, …).
        perturbation_range: Multipliers applied to the chosen coefficient.
                           Defaults to np.linspace(0.5, 1.5, 21) — ±50 % range.
        save_path:         Where to save the sensitivity plot (None = skip).
        show:              Display interactively (default False for CI).

    Returns:
        dict with keys:
            "param_values"  — actual coefficient values tested (list[float])
            "rmse_values"   — RMSE at each perturbed value    (list[float])
            "optimal_value" — coefficient value with lowest RMSE (float)

    Raises:
        AssertionError: If len(X) <= 10 or param_index out of range.
    """
    assert len(X) > 10, (
        f"[C3] sensitivity_analysis requires > 10 samples (got {len(X)})."
    )
    assert 0 <= param_index < len(model_coeffs), (
        f"[C3] param_index {param_index} out of range "
        f"for coeffs of length {len(model_coeffs)}."
    )

    if perturbation_range is None:
        perturbation_range = np.linspace(0.5, 1.5, 21)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    X_design   = _build_design_matrix(X, degree)
    base_value = model_coeffs[param_index]

    param_values: list = []
    rmse_values:  list = []

    for multiplier in perturbation_range:
        theta_perturbed            = model_coeffs.copy()
        perturbed_val              = base_value * multiplier
        theta_perturbed[param_index] = perturbed_val
        param_values.append(perturbed_val)
        rmse_values.append(_rmse_objective(theta_perturbed, X_design, y))

    best_idx = int(np.argmin(rmse_values))
    print(
        f"[C3] Sensitivity on θ[{param_index}]: "
        f"best value = {param_values[best_idx]:.4f}  "
        f"→ RMSE = {rmse_values[best_idx]:.4f} [vehicles]"
    )

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(param_values, rmse_values, "o-", color="#2D6A4F", linewidth=2,
            markersize=5)
    ax.axvline(param_values[best_idx], color="#E63946", linestyle="--",
               label=f"Best θ[{param_index}] = {param_values[best_idx]:.4f}")
    ax.set_xlabel(f"Value of θ[{param_index}]  [dimensionless]", fontsize=12)
    ax.set_ylabel("RMSE  [vehicles]", fontsize=12)
    ax.set_title(
        f"C3 — Sensitivity Analysis: RMSE vs θ[{param_index}]",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[C3] Sensitivity plot saved → {out}")

    if show:
        plt.show()

    plt.close(fig)

    return {
        "param_values":  param_values,
        "rmse_values":   rmse_values,
        "optimal_value": param_values[best_idx],
    }


# ─────────────────────────────────────────────────────────────────────────────
# C3 Public API — optimize_parameters
# ─────────────────────────────────────────────────────────────────────────────

def optimize_parameters(
    model_coeffs: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    degree: int = 1,
    learning_rate: float = 1e-2,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    sensitivity_param_index: int = 1,
    plot_save_path: "str | None" = "reports/figures/convergence_curve.png",
    show_plots: bool = False,
) -> "tuple[np.ndarray, list[float], dict]":
    """
    Optimize regression coefficients to minimize RMSE (C3 pseudocode).

    Implements the full MODULE optimize_parameters(X, y) → optimal_parameters
    specified in the project pseudocode:
        1. Define objective function as RMSE of the regression model.
        2. Define parameter search space (gradient descent from zeros; BFGS comparison).
        3. Apply optimization algorithm (custom GD + scipy BFGS).
        4. Track convergence of the objective function (plot saved automatically).
        5. Return parameters that minimize prediction error.
    Additionally runs sensitivity_analysis() automatically.

    Args:
        model_coeffs:            Coefficient vector from fit_regression() — used as
                                 reference; GD starts from zeros for a meaningful curve.
                                 Shape: (degree + 1,).
        X:                       Feature array (n,) or (n, 1) — raw (un-normalised) values.
                                 Units: hours [h] or minutes since midnight [min].
        y:                       Target vector (n,) — vehicle counts [vehicles].
        degree:                  Polynomial degree matching the C2 model (1, 2, or 3).
        learning_rate:           GD step size α (default 1e-2, safe for normalised X).
        max_iterations:          Maximum GD iterations (default 1 000).
        tolerance:               Convergence threshold on RMSE improvement (default 1e-6).
        sensitivity_param_index: Coefficient index to perturb in sensitivity analysis.
        plot_save_path:          Where to save the convergence figure (None = skip).
        show_plots:              Display figures interactively (default False for CI).

    Returns:
        optimal_params:       Refined coefficients with lowest achieved RMSE.
                              Shape: (degree + 1,). Units: dimensionless (normalised X).
        convergence_history:  RMSE at every GD iteration [vehicles].
        sensitivity_results:  Dict from sensitivity_analysis().

    Raises:
        AssertionError: If len(X) <= 10 (dataset too small).
        ValueError:     If len(X) != len(y).
    """

    # ── Preconditions (Wilson §5a — explicit assertions) ──────────────────────
    assert len(X) > 10, (
        f"[C3] Dataset must have > 10 samples (got {len(X)})."
    )
    if len(X) != len(y):
        raise ValueError(
            f"[C3] X and y must have equal length (X={len(X)}, y={len(y)})."
        )
    assert degree in (1, 2, 3), "degree must be 1, 2, or 3"

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # ── Step 1 — Define objective: f(theta) = RMSE ───────────────────────────
    # Normalise X so large feature magnitudes don't overshoot the GD step.
    # All RMSE values are computed on the original y scale → interpretable.
    X_norm, _X_mu, _X_sigma = _normalise(X.ravel())
    X_norm   = X_norm.reshape(-1, 1)
    X_design = _build_design_matrix(X_norm, degree)

    # Reference RMSE from C2 (evaluate C2 coeffs on normalised design matrix)
    rmse_c2_ref = _rmse_objective(model_coeffs, X_design, y)
    print(f"[C3] C2 model RMSE (reference)      : {rmse_c2_ref:.4f} [vehicles]")

    # ── Step 2 — Define parameter search space ────────────────────────────────
    # GD starts from zeros so the convergence curve shows real progress.
    theta_init = np.zeros(degree + 1)
    print(f"[C3] GD initialised from zeros      : {theta_init}")

    # ── Step 3 — Apply optimization algorithm (custom GD) ────────────────────
    convergence_history: list = []
    theta_gd = theta_init.copy()

    for _ in range(max_iterations):
        current_rmse = _rmse_objective(theta_gd, X_design, y)

        # Step 4 — track objective per iteration (satisfies pseudocode §4)
        convergence_history.append(current_rmse)

        grad     = _rmse_gradient(theta_gd, X_design, y)
        theta_gd = theta_gd - learning_rate * grad

        # Convergence check
        if len(convergence_history) > 1:
            if abs(convergence_history[-2] - convergence_history[-1]) < tolerance:
                break

    rmse_gd = _rmse_objective(theta_gd, X_design, y)
    print(f"[C3] Gradient Descent final RMSE    : {rmse_gd:.4f} [vehicles]")

    # scipy BFGS — quasi-Newton comparison (also required by C3 spec)
    result_bfgs = minimize(
        fun=_rmse_objective,
        x0=theta_init,
        args=(X_design, y),
        method="BFGS",
        jac=_rmse_gradient,
        options={"maxiter": max_iterations, "gtol": tolerance},
    )
    theta_bfgs = result_bfgs.x
    rmse_bfgs  = _rmse_objective(theta_bfgs, X_design, y)
    print(f"[C3] BFGS final RMSE                : {rmse_bfgs:.4f} [vehicles]")

    # ── Step 5 — Return parameters that minimise RMSE ────────────────────────
    if rmse_bfgs <= rmse_gd:
        optimal_params = theta_bfgs
        print(f"[C3] ✓ BFGS selected              → RMSE = {rmse_bfgs:.4f}")
    else:
        optimal_params = theta_gd
        print(f"[C3] ✓ Gradient Descent selected  → RMSE = {rmse_gd:.4f}")

    rmse_after       = min(rmse_gd, rmse_bfgs)
    improvement_pct  = (
        100.0 * (rmse_c2_ref - rmse_after) / rmse_c2_ref
        if rmse_c2_ref > 0 else 0.0
    )
    print(f"[C3] RMSE improvement over C2       : {improvement_pct:.2f} %")
    print(f"[C3] GD convergence steps           : {len(convergence_history)}")

    # ── Step 4 (plot) — convergence curve (called automatically) ─────────────
    print("[C3] Plotting convergence curve ...")
    plot_convergence_curve(
        convergence_history,
        save_path=plot_save_path,
        show=show_plots,
    )

    # ── Sensitivity analysis (called automatically after optimization) ─────────
    print(f"[C3] Running sensitivity analysis on θ[{sensitivity_param_index}] ...")
    sensitivity_results = sensitivity_analysis(
        model_coeffs    = optimal_params,
        X               = X_norm.ravel(),   # same normalised space as GD/BFGS
        y               = y,
        degree          = degree,
        param_index     = sensitivity_param_index,
        save_path       = (
            plot_save_path.replace("convergence_curve", "sensitivity_analysis")
            if plot_save_path else None
        ),
        show            = show_plots,
    )

    # ── Postcondition ─────────────────────────────────────────────────────────
    assert len(optimal_params) == degree + 1, (
        "[C3] Postcondition: optimal_params length must equal degree + 1."
    )

    return optimal_params, convergence_history, sensitivity_results


# ─────────────────────────────────────────────────────────────────────────────
# CLI — demo: runs C2 first, then passes its model dict into C3
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import csv

    CSV_PATH = Path("data/processed/df.csv")
    X_data, y_data = None, None

    # Load real data produced by data_loader.py (C1).
    if CSV_PATH.exists():
        print(f"[C3] Loading processed data from {CSV_PATH} ...")
        minutely: dict = {}

        with open(CSV_PATH, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("day_type") == "weekday":
                    h   = int(row["hour"])
                    m   = int(row["minute"])
                    key = h * 60 + m       # minutes since midnight
                    minutely[key] = minutely.get(key, 0) + 1

        if len(minutely) > 10:
            minutes = sorted(minutely.keys())
            X_data  = np.array(minutes, dtype=float)
            y_data  = np.array([minutely[k] for k in minutes], dtype=float)
            print(f"[C3] {len(X_data)} minute-buckets loaded.")
        else:
            print("[C3] Insufficient CSV data — falling back to synthetic.")

    # Fallback: synthetic bimodal traffic curve (morning + evening peaks)
    if X_data is None:
        print("[C3] Using synthetic traffic data for demo.")
        rng    = np.random.default_rng(42)
        X_data = np.arange(24, dtype=float)
        y_data = (
            50
            + 80 * np.exp(-0.5 * ((X_data - 8) / 2) ** 2)
            + 60 * np.exp(-0.5 * ((X_data - 17) / 2) ** 2)
            + rng.normal(0, 5, 24)
        )

    DEGREE = 2

    # ── Step A: C2 fit — fit_regression returns a dict (not a tuple) ──────────
    print("\n[C2] Fitting initial regression model ...")
    model_c2 = fit_regression(X_data, y_data, degree=DEGREE)   # returns dict
    coeffs_c2 = model_c2["coeffs"]
    r2_c2     = model_c2["r2"]
    rmse_c2   = model_c2["rmse"]
    print(f"[C2] Coefficients : {coeffs_c2}")
    print(f"[C2] R²           : {r2_c2:.4f}")
    print(f"[C2] RMSE         : {rmse_c2:.4f} [vehicles]")

    # ── Step B: C3 optimization ───────────────────────────────────────────────
    print("\n[C3] Optimizing C2 model parameters ...")
    optimal_coeffs, history, sensitivity = optimize_parameters(
        model_coeffs            = coeffs_c2,
        X                       = X_data,
        y                       = y_data,
        degree                  = DEGREE,
        learning_rate           = 1e-2,
        tolerance               = 1e-7,
        sensitivity_param_index = 1,    # perturb the linear coefficient (θ₁)
        show_plots              = False,
    )

    print("\n[C3] ── Final Results ──────────────────────────────────────")
    print(f"[C3] Optimal coefficients  : {optimal_coeffs}")
    print(f"[C3] Sensitivity best θ₁  : {sensitivity['optimal_value']:.4f}")
    print("[C3] Done.")
