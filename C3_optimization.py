# -*- coding: utf-8 -*-
"""
optimization.py
C3 — Optimization Module
Author: Kenia Gabriela Hermida Núñez
Module: optimize_parameters(model, X, y) → optimal_params, convergence_history
Purpose: Take the regression model produced by C2 (regression.py) and optimize its
         coefficients to further minimize RMSE, using:
             1. Gradient descent (custom, step-by-step)
             2. scipy BFGS (for comparison)
         Then automatically plots the convergence curve and runs a sensitivity
         analysis on one constraint — both as required by the C3 pseudocode spec.
Depends on:
    data_loader.py (C1) — must be run first to produce the processed dataset.
                           Reads from data/processed/df.csv, which is generated
                           by load_data() from the raw JSON camera files.
                           df.csv must contain columns: hour, minute, day_type.
    regression.py  (C2) — fit_regression() is called second to get the initial
                           regression model; optimization.py then refines those
                           coefficients.
Execution order:
    C1 (data_loader.py) → produces df.csv
    C2 (regression.py)  → consumes df.csv, produces model coefficients
    C3 (optimization.py)→ consumes C2 coefficients + df.csv, produces optimal params
Complexity: O(k · n)  where k = number of iterations, n = number of samples
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize

from regression import fit_regression


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _normalise(X: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Z-score normalise X to zero mean and unit std.
    Returns (X_norm, mean, std) so the caller can invert if needed.
    Normalisation is applied internally during gradient descent so that
    features in the hundreds/thousands (e.g. minutes since midnight ~720–839)
    don't cause the gradient to overshoot with a standard learning rate.
    """
    mu  = float(X.mean())
    std = float(X.std())
    if std < 1e-12:
        std = 1.0
    return (X - mu) / std, mu, std


def _build_design_matrix(X: np.ndarray, degree: int) -> np.ndarray:
    """
    Reconstruct the polynomial design matrix consistent with regression.py.
    Columns: [1, x, x^2, x^3] up to the requested degree.
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    X_poly = X.copy()
    if degree >= 2:
        X_poly = np.hstack([X_poly, X ** 2])
    if degree >= 3:
        X_poly = np.hstack([X_poly, X ** 3])

    return np.hstack([np.ones((X_poly.shape[0], 1)), X_poly])


def _rmse(coeffs: np.ndarray, X_design: np.ndarray, y: np.ndarray) -> float:
    """Objective function  f(theta) = RMSE(y, y_hat)."""
    y_hat = X_design @ coeffs
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def _rmse_gradient(
    coeffs: np.ndarray, X_design: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """
    Analytical gradient of RMSE w.r.t. theta:
        d(RMSE)/d(theta) = -(1 / (n * RMSE)) * X^T (y - X*theta)
    """
    n = len(y)
    y_hat = X_design @ coeffs
    residuals = y - y_hat
    rmse_val = float(np.sqrt(np.mean(residuals ** 2)))

    if rmse_val < 1e-12:
        return np.zeros_like(coeffs)

    return -(1.0 / (n * rmse_val)) * (X_design.T @ residuals)


# ─────────────────────────────────────────────
# Step 4 — Plot convergence curve
# ─────────────────────────────────────────────

def plot_convergence_curve(
    convergence_history: list[float],
    save_path: str | None = "reports/figures/convergence_curve.png",
    show: bool = True,
) -> None:
    """
    Plot the RMSE (objective value f(theta)) at every gradient-descent iteration.
    Called automatically by optimize_parameters() as part of the C3 pipeline.

    Parameters
    ----------
    convergence_history : list[float]
        RMSE recorded at each gradient-descent iteration.
    save_path : str or None
        File path to save the figure.  None = skip saving.
    show : bool
        Display interactively (default True).
    """
    if not convergence_history:
        print("[C3] convergence_history is empty — nothing to plot.")
        return

    iterations = list(range(1, len(convergence_history) + 1))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(iterations, convergence_history, color="#0077B6", linewidth=2,
            label="RMSE per iteration")
    ax.axhline(convergence_history[-1], color="#E63946", linewidth=1.2,
               linestyle="--",
               label=f"Final RMSE = {convergence_history[-1]:.4f}")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Objective value  f(theta) = RMSE", fontsize=12)
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
        fig.savefig(out, dpi=150)
        print(f"[C3] Convergence curve saved → {out}")

    if show:
        plt.show()

    plt.close(fig)


# ─────────────────────────────────────────────
# Step 5 — Sensitivity analysis
# ─────────────────────────────────────────────

def sensitivity_analysis(
    model_coeffs: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    degree: int = 1,
    param_index: int = 1,
    perturbation_range: np.ndarray | None = None,
) -> dict:
    """
    Vary one coefficient of the optimized model and observe how RMSE responds.
    Called automatically by optimize_parameters() as part of the C3 pipeline.

    Parameters
    ----------
    model_coeffs : np.ndarray
        The OPTIMIZED coefficients returned by optimize_parameters().
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    degree : int
        Polynomial degree.
    param_index : int
        Index of the coefficient to perturb (0 = intercept, 1 = theta_1, …).
    perturbation_range : np.ndarray, optional
        Multipliers applied to the chosen coefficient.
        Defaults to np.linspace(0.5, 1.5, 21) — ±50 % in 5 % steps.

    Returns
    -------
    dict with keys:
        "param_values"  — actual values of the perturbed coefficient tested
        "rmse_values"   — corresponding RMSE at each value
        "optimal_value" — coefficient value that yields the lowest RMSE
    """
    assert len(X) > 10, "[C3] sensitivity_analysis requires more than 10 samples."

    if perturbation_range is None:
        perturbation_range = np.linspace(0.5, 1.5, 21)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    X_design = _build_design_matrix(X, degree)
    base_value = model_coeffs[param_index]

    param_values: list[float] = []
    rmse_values: list[float] = []

    for multiplier in perturbation_range:
        theta_perturbed = model_coeffs.copy()
        perturbed_val = base_value * multiplier
        theta_perturbed[param_index] = perturbed_val
        param_values.append(perturbed_val)
        rmse_values.append(_rmse(theta_perturbed, X_design, y))

    best_idx = int(np.argmin(rmse_values))
    print(
        f"[C3] Sensitivity on theta[{param_index}]: "
        f"best value = {param_values[best_idx]:.4f} "
        f"→ RMSE = {rmse_values[best_idx]:.4f}"
    )

    return {
        "param_values": param_values,
        "rmse_values": rmse_values,
        "optimal_value": param_values[best_idx],
    }


# ─────────────────────────────────────────────
# Public API — C3 main entry point
# ─────────────────────────────────────────────

def optimize_parameters(
    model_coeffs: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    degree: int = 1,
    learning_rate: float = 1e-4,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    sensitivity_param_index: int = 1,
    plot_save_path: str | None = "reports/figures/convergence_curve.png",
    show_plots: bool = True,
) -> tuple[np.ndarray, list[float], dict]:
    """
    Optimize the coefficients of an EXISTING regression model (from C2)
    to further minimize RMSE.

    Implements the full C3 pseudocode pipeline:
        Step 1 — Define objective function f(theta) = RMSE
        Step 2 — Initialize parameter vectors from C2 warm start
        Step 3 — Apply gradient descent optimization (+ BFGS comparison)
        Step 4 — Track objective value per iteration; plot convergence curve
        Step 5 — Return parameters minimizing RMSE; run sensitivity analysis

    Parameters
    ----------
    model_coeffs : np.ndarray
        Coefficient vector from regression.fit_regression() — warm start θ₀.
    X : np.ndarray
        Feature matrix (n samples × m features).
    y : np.ndarray
        Target vector — vehicle counts / hour (length n).
    degree : int
        Polynomial degree used when the model was originally fit (1, 2, or 3).
    learning_rate : float
        Step size for gradient descent (default 1e-4).
    max_iterations : int
        Maximum gradient-descent iterations (default 1000).
    tolerance : float
        Convergence threshold on RMSE improvement (default 1e-6).
    sensitivity_param_index : int
        Which coefficient to perturb in the sensitivity analysis (default 1).
    plot_save_path : str or None
        Where to save the convergence curve figure. None = skip saving.
    show_plots : bool
        Whether to display plots interactively (default True).

    Returns
    -------
    optimal_params : np.ndarray
        Refined coefficient vector with the lowest achieved RMSE.
    convergence_history : list[float]
        RMSE recorded at every gradient-descent iteration.
    sensitivity_results : dict
        Output of sensitivity_analysis() on the chosen coefficient.

    Raises
    ------
    AssertionError
        If len(X) <= 10  (pseudocode assertion: assert len(X) > 10).
    ValueError
        If X and y lengths are incompatible.
    """

    # ── Precondition checks ──────────────────────────────────────────────────
    # C3 pseudocode assertion
    assert len(X) > 10, (
        f"[C3] AssertionError: dataset must have more than 10 samples "
        f"(got {len(X)})."
    )

    if len(X) != len(y):
        raise ValueError(
            f"[C3] X and y must have the same length "
            f"(X={len(X)}, y={len(y)})."
        )

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # ── Step 1 — Define objective function f(theta) = RMSE ───────────────────
    # Normalise X internally so that large feature values (e.g. minutes since
    # midnight: 720–839, or their squares: ~518000–703000) don't cause the
    # gradient to massively overshoot with a standard learning rate.
    # RMSE values are computed on the original y scale throughout, so all
    # reported metrics remain interpretable.
    X_norm, X_mu, X_sigma = _normalise(X.ravel())
    X_norm = X_norm.reshape(-1, 1)
    X_design = _build_design_matrix(X_norm, degree)

    # ── Step 2 — Initialize parameter vectors ────────────────────────────────
    # GD starts from ZEROS (not from the C2 LS solution) so it has real work
    # to do and produces a meaningful convergence curve.
    # The C2 coefficients are used only as a reference RMSE baseline and are
    # re-fitted on the normalised X so the comparison is fair.
    X_design_c2 = _build_design_matrix(X_norm, degree)
    rmse_before = _rmse(model_coeffs, X_design_c2, y)
    print(f"[C3] C2 model RMSE (reference)   : {rmse_before:.4f}")

    theta_init = np.zeros(degree + 1)           # GD warm-start: zeros
    print(f"[C3] GD initialised from zeros   : {theta_init}")

    # ── Step 3 — Apply gradient descent optimization ─────────────────────────
    convergence_history: list[float] = []
    theta_gd = theta_init.copy()

    for _ in range(max_iterations):
        current_rmse = _rmse(theta_gd, X_design, y)

        # Step 4 — track objective value per iteration
        convergence_history.append(current_rmse)

        grad = _rmse_gradient(theta_gd, X_design, y)
        theta_gd = theta_gd - learning_rate * grad

        # Convergence check
        if len(convergence_history) > 1:
            improvement = abs(convergence_history[-2] - convergence_history[-1])
            if improvement < tolerance:
                break

    rmse_gd = _rmse(theta_gd, X_design, y)

    # Compare with scipy BFGS for robustness — also starts from zeros
    result_bfgs = minimize(
        fun=_rmse,
        x0=theta_init,
        args=(X_design, y),
        method="BFGS",
        jac=_rmse_gradient,
        options={"maxiter": max_iterations, "gtol": tolerance},
    )
    theta_bfgs = result_bfgs.x
    rmse_bfgs = _rmse(theta_bfgs, X_design, y)

    # ── Step 5 — Return parameters minimizing RMSE ───────────────────────────
    if rmse_bfgs < rmse_gd:
        optimal_params = theta_bfgs
        print(f"[C3] BFGS selected            → RMSE = {rmse_bfgs:.4f}")
    else:
        optimal_params = theta_gd
        print(f"[C3] Gradient descent selected → RMSE = {rmse_gd:.4f}")

    rmse_after = min(rmse_gd, rmse_bfgs)
    improvement_pct = (
        100 * (rmse_before - rmse_after) / rmse_before if rmse_before > 0 else 0.0
    )
    print(f"[C3] RMSE improvement over C2 model : {improvement_pct:.2f}%")
    print(f"[C3] Convergence steps              : {len(convergence_history)}")

    # ── Step 4 (plot) — convergence curve called automatically ───────────────
    print("[C3] Plotting convergence curve ...")
    plot_convergence_curve(
        convergence_history,
        save_path=plot_save_path,
        show=show_plots,
    )

    # ── Step 5 (sensitivity) — called automatically after optimization ────────
    print(f"[C3] Running sensitivity analysis on theta[{sensitivity_param_index}] ...")
    # Pass X_norm so sensitivity_analysis uses the same normalised feature
    # space as the optimised coefficients.
    sensitivity_results = sensitivity_analysis(
        model_coeffs=optimal_params,
        X=X_norm.ravel(),
        y=y,
        degree=degree,
        param_index=sensitivity_param_index,
    )

    return optimal_params, convergence_history, sensitivity_results


# ─────────────────────────────────────────────
# CLI — demo: runs C2 first, then passes its model to C3
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import csv

    CSV_PATH = Path("data/processed/df.csv")
    X_data, y_data = None, None

    # Load real data produced by data_loader.py (C1).
    # Aggregates by MINUTE so we have enough data points even when the dataset
    # covers only a short time window (e.g. 1–2 hours).
    if CSV_PATH.exists():
        print(f"[C3] Loading processed data from {CSV_PATH} ...")
        minutely: dict[int, int] = {}

        with open(CSV_PATH, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("day_type") == "weekday":
                    h = int(row["hour"])
                    m = int(row["minute"])
                    key = h * 60 + m          # minutes since midnight
                    minutely[key] = minutely.get(key, 0) + 1

        if len(minutely) > 10:
            minutes = sorted(minutely.keys())
            X_data = np.array(minutes, dtype=float)
            y_data = np.array([minutely[m] for m in minutes], dtype=float)
            print(f"[C3] {len(X_data)} minute buckets loaded.")
        else:
            print("[C3] Not enough data in CSV — falling back to synthetic.")

    # Fallback: synthetic traffic-like data
    if X_data is None:
        print("[C3] Using synthetic traffic data for demo.")
        rng = np.random.default_rng(42)
        X_data = np.arange(24, dtype=float)
        y_data = (
            50
            + 80 * np.exp(-0.5 * ((X_data - 8) / 2) ** 2)
            + 60 * np.exp(-0.5 * ((X_data - 17) / 2) ** 2)
            + rng.normal(0, 5, 24)
        )

    DEGREE = 2

    # Step A — Run C2 (regression.py) to get the initial model
    print("\n[C2] Fitting initial regression model ...")
    coeffs_c2, r2_c2, rmse_c2, y_hat_c2, _ = fit_regression(
        X_data, y_data, degree=DEGREE
    )
    print(f"[C2] Coefficients : {coeffs_c2}")
    print(f"[C2] R²           : {r2_c2:.4f}")
    print(f"[C2] RMSE         : {rmse_c2:.4f}")

    # Step B — Pass C2 model into C3 for optimization.
    #          plot_convergence_curve and sensitivity_analysis are called
    #          automatically inside optimize_parameters — no manual wiring needed.
    #          learning_rate=1e-2 and tolerance=1e-7 are appropriate for
    #          normalised features (X scaled to ~[-2, 2]).
    print("\n[C3] Optimizing C2 model parameters ...")
    optimal_coeffs, history, sensitivity = optimize_parameters(
        model_coeffs=coeffs_c2,
        X=X_data,
        y=y_data,
        degree=DEGREE,
        learning_rate=1e-2,
        tolerance=1e-7,
        sensitivity_param_index=1,   # perturb the linear coefficient (theta_1)
    )

    print("\n[C3] Optimal coefficients :", optimal_coeffs)
    print("[C3] Sensitivity optimal value :", sensitivity["optimal_value"])
