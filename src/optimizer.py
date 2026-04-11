"""
C3 — Optimization Module
Module: optimize_parameters(model_coeffs, X, y, degree, ...) → optimal_params, convergence_history, sensitivity_results
Purpose: Take the regression model produced by C2 (regression.py) and optimize its
         coefficients to further minimize RMSE, using:
             1. Custom gradient_descent() — step-by-step, convergence tracked.
             2. scipy BFGS (quasi-Newton, for comparison via scipy.optimize.minimize).
         Automatically plots the convergence curve and runs a sensitivity analysis on one coefficient 
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")           
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize

# C2 interface: fit_regression(features, target, degree) → dict
from src.regression import fit_regression

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────
def _build_design_matrix(X: np.ndarray, degree: int) -> np.ndarray:
    """
    Build a polynomial design matrix [1 | x | x² | x³] consistent with regression.py.

    Args:
        X:      Feature array of shape (n,) or (n, p).
        degree: Polynomial degree — one of {1, 2, 3}.

    Returns:
        Design matrix of shape (n, 1 + p*degree) with intercept column first.

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
    Objective function  f(θ) = RMSE(y, X_design @ θ).

    Args:
        coeffs:   Coefficient vector θ of shape (p,).
        X_design: Design matrix of shape (n, p).
        y:        Target vector of shape (n,).

    Returns:
        float: Root mean squared error [vehicles/hour].
    """
    y_hat = X_design @ coeffs
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def _rmse_gradient(coeffs: np.ndarray, X_design: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Analytical gradient of RMSE w.r.t. θ:
        ∇_θ RMSE = -(1 / (n · RMSE)) · Xᵀ(y − Xθ)

    Args:
        coeffs:   Coefficient vector θ of shape (p,).
        X_design: Design matrix of shape (n, p).
        y:        Target vector of shape (n,).

    Returns:
        Gradient vector of shape (p,).
    """
    n         = len(y)
    y_hat     = X_design @ coeffs
    residuals = y - y_hat
    rmse_val  = float(np.sqrt(np.mean(residuals ** 2)))

    if rmse_val < 1e-12:
        return np.zeros_like(coeffs)

    return -(1.0 / (n * rmse_val)) * (X_design.T @ residuals)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: gradient_descent()  
# ─────────────────────────────────────────────────────────────────────────────
def gradient_descent(
    f,
    grad_f,
    x0: np.ndarray,
    lr: float,
    max_iter: int,
    tol: float = 1e-7,
) -> "tuple[np.ndarray, list[float]]":
    """
    Generic gradient descent optimizer.

    Implements the update rule:
        x_{k+1} = x_k − lr · ∇f(x_k)

    and stops early when |f(x_{k}) − f(x_{k-1})| < tol.

    Args:
        f:        Callable  f(x) → float  — objective function to minimize.
        grad_f:   Callable  grad_f(x) → ndarray  — gradient of f.
        x0:       Initial parameter vector, shape (p,).
        lr:       Learning rate α > 0  (step size).
        max_iter: Maximum number of iterations.
        tol:      Convergence tolerance on absolute change in f(x).
                  Iteration stops when |f_{k} − f_{k-1}| < tol.

    Returns:
        x_opt:   ndarray — parameter vector at the last (or converged) iteration.
        history: list[float] — value of f(x) recorded at each iteration.

    Raises:
        AssertionError: If lr <= 0 or max_iter < 1.
    """
    assert lr > 0,       "lr (learning rate) must be positive"
    assert max_iter >= 1, "max_iter must be at least 1"

    x       = np.asarray(x0, dtype=float).copy()
    history: list = []

    for _ in range(max_iter):
        fval = float(f(x))
        history.append(fval)

        # Early stopping
        if len(history) > 1 and abs(history[-2] - history[-1]) < tol:
            break

        x = x - lr * np.asarray(grad_f(x), dtype=float)

    return x, history


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: plot_convergence_curve()
# ─────────────────────────────────────────────────────────────────────────────
def plot_convergence_curve(
    convergence_history: "list[float]",
    save_path: "str | None" = "reports/figures/convergence_curve.png",
    show: bool = False,
) -> None:
    """
    Plot objective value (RMSE) versus gradient-descent iteration.

    Args:
        convergence_history: RMSE recorded at each GD iteration [vehicles/hour].
        save_path:           File path to save the PNG.  None = skip saving.
        show:                Display the figure interactively (default False).
    """
    assert len(convergence_history) > 0, "[C3] convergence_history must not be empty."

    iterations = list(range(1, len(convergence_history) + 1))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(iterations, convergence_history, color="#0077B6", linewidth=2,
            label="RMSE per iteration")
    ax.axhline(
        convergence_history[-1], color="#E63946", linewidth=1.2, linestyle="--",
        label=f"Final RMSE = {convergence_history[-1]:.4f} [vehicles/hour]",
    )
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Objective  f(θ) = RMSE  [vehicles/hour]", fontsize=12)
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
# PUBLIC: sensitivity_analysis()
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

    Args:
        model_coeffs:      Optimized coefficient vector from optimize_parameters().
        X:                 Feature array (n,) or (n, p) — normalised.
        y:                 Target vector (n,) — vehicle counts [vehicles/hour].
        degree:            Polynomial degree used during fitting {1, 2, 3}.
        param_index:       Index of the coefficient to perturb
                           (0 = intercept, 1 = θ₁, 2 = θ₂, …).
        perturbation_range: Multipliers applied to the chosen coefficient.
                           Defaults to np.linspace(0.5, 1.5, 21) — ±50% range.
        save_path:         Where to save the sensitivity plot.  None = skip.
        show:              Display interactively (default False).
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
        theta_perturbed              = model_coeffs.copy()
        perturbed_val                = base_value * multiplier
        theta_perturbed[param_index] = perturbed_val
        param_values.append(float(perturbed_val))
        rmse_values.append(_rmse_objective(theta_perturbed, X_design, y))

    best_idx = int(np.argmin(rmse_values))
    print(
        f"[C3] Sensitivity on θ[{param_index}]: "
        f"best value = {param_values[best_idx]:.4f}  "
        f"→ RMSE = {rmse_values[best_idx]:.4f} [vehicles/hour]"
    )

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(param_values, rmse_values, "o-", color="#2D6A4F", linewidth=2, markersize=5)
    ax.axvline(param_values[best_idx], color="#E63946", linestyle="--",
               label=f"Best θ[{param_index}] = {param_values[best_idx]:.4f}")
    ax.axvline(float(base_value), color="#0077B6", linestyle=":",
               label=f"Optimized value = {base_value:.4f}")
    ax.set_xlabel(f"Value of θ[{param_index}]  [dimensionless]", fontsize=12)
    ax.set_ylabel("RMSE  [vehicles/hour]", fontsize=12)
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

    assert len(param_values) == len(rmse_values), (
        "[C3] Postcondition: param_values and rmse_values must have equal length."
    )

    return {
        "param_values":  param_values,
        "rmse_values":   rmse_values,
        "optimal_value": param_values[best_idx],
    }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: optimize_parameters() 
# ─────────────────────────────────────────────────────────────────────────────
def optimize_parameters(
    model_coeffs: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    degree: int = 1,
    learning_rate: float = 1e-2,
    max_iterations: int = 100_000,
    tolerance: float = 1e-7,
    sensitivity_param_index: int = 1,
    plot_save_path: "str | None" = "report/figures/convergence_curve.png",
    show_plots: bool = False,
    pre_normalized: bool = False,
) -> "tuple[np.ndarray, list[float], dict]":
    """
    Optimize regression coefficients to minimize RMSE — C3 pseudocode.

    Implements MODULE optimize_parameters(model, X, y) → optimal_params:
        1. Define objective function = RMSE.
        2. Initialize parameter vectors (GD from zeros; BFGS warm-started from C2).
        3. Apply gradient descent optimization.
        4. Track objective value per iteration → convergence curve saved.
        5. Return parameters minimizing RMSE (best of GD vs BFGS).
    Additionally runs sensitivity_analysis() per framework §C3 spec.

    Args:
        model_coeffs:            Coefficient vector from fit_regression() (C2).
                                 Shape: (1 + p·degree,).
        X:                       Feature array (n,) or (n, p) — raw or pre-normalised.
        y:                       Target vector (n,) — vehicle counts [vehicles/hour].
        degree:                  Polynomial degree matching the C2 model {1, 2, 3}.
        learning_rate:           GD step size α ∈ (0, 1].  Default 1e-2.
        max_iterations:          Maximum GD iterations.  Default 100 000.
        tolerance:               Early-stop threshold on |Δ RMSE|.  Default 1e-7.
        sensitivity_param_index: Coefficient index to perturb in sensitivity analysis.
        plot_save_path:          Path to save the convergence figure.  None = skip.
        show_plots:              Display figures interactively.  Default False.
        pre_normalized:          Set True if X is already z-score normalised (avoids
                                 double normalisation in multi-module pipelines).

    Returns:
        optimal_params:       Refined coefficients with lowest achieved RMSE.
        convergence_history:  RMSE at every GD iteration [vehicles/hour].
        sensitivity_results:  Dict from sensitivity_analysis().
    """

    # ── Preconditions ─────────────────────────────────────────────────────────
    assert len(X) > 10, (
        f"[C3] Dataset must have > 10 samples (got {len(X)})."
    )
    if len(X) != len(y):
        raise ValueError(
            f"[C3] X and y must have equal length (X={len(X)}, y={len(y)})."
        )
    assert degree in (1, 2, 3), "degree must be 1, 2, or 3"
    assert learning_rate > 0,   "learning_rate must be positive"
    assert max_iterations >= 1, "max_iterations must be >= 1"

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # ── Step 1 — Objective: f(θ) = RMSE ──────────────────────────────────────
    if pre_normalized:
        X_norm = X.copy()
    else:
        _mu    = X.mean(axis=0)
        _sigma = X.std(axis=0)
        _sigma[_sigma < 1e-12] = 1.0
        X_norm = (X - _mu) / _sigma

    X_design = _build_design_matrix(X_norm, degree)

    rmse_c2_ref = _rmse_objective(model_coeffs, X_design, y)

    # ── Step 2 — Initialize parameter vectors ────────────────────────────────
    n_coeffs   = X_design.shape[1]
    theta_init = np.zeros(n_coeffs)       # GD starts from zero for visible convergence
    print(f"[C3] GD initialized from zeros      : shape {theta_init.shape}")

    # ── Step 3 — Gradient descent (calls public gradient_descent()) ───────────
    def _f(theta):    return _rmse_objective(theta, X_design, y)
    def _gf(theta):   return _rmse_gradient(theta,  X_design, y)

    theta_gd, convergence_history = gradient_descent(
        f        = _f,
        grad_f   = _gf,
        x0       = theta_init,
        lr       = learning_rate,
        max_iter = max_iterations,
        tol      = tolerance,
    )

    rmse_gd = _rmse_objective(theta_gd, X_design, y)
    print(f"[C3] Gradient Descent final RMSE    : {rmse_gd:.4f} [vehicles/hour]")
    print(f"[C3] GD convergence steps           : {len(convergence_history)}")

    # ── scipy BFGS — quasi-Newton comparison (Clase 7, p. 24–25) ─────────────
    result_bfgs = minimize(
        fun     = _f,
        x0      = theta_init,
        jac     = _gf,
        method  = "BFGS",
        options = {"maxiter": max_iterations, "gtol": tolerance},
    )
    theta_bfgs = result_bfgs.x
    rmse_bfgs  = _rmse_objective(theta_bfgs, X_design, y)
    print(f"[C3] BFGS final RMSE                : {rmse_bfgs:.4f} [vehicles/hour]")

    # ── Step 5 — Select best parameters ──────────────────────────────────────
    if rmse_bfgs <= rmse_gd:
        optimal_params = theta_bfgs
        print(f"[C3] ✓ BFGS selected              → RMSE = {rmse_bfgs:.4f} [vehicles/hour]")
    else:
        optimal_params = theta_gd
        print(f"[C3] ✓ Gradient Descent selected  → RMSE = {rmse_gd:.4f} [vehicles/hour]")

    rmse_after = min(rmse_gd, rmse_bfgs)

    # ── Step 4 — Convergence curve plot ──────────────────────────────────────
    print("[C3] Plotting convergence curve ...")
    plot_convergence_curve(convergence_history, save_path=plot_save_path, show=show_plots)

    # ── Sensitivity analysis ──────────────────────────────────────────────────
    sens_path = None
    if plot_save_path:
        sens_path = str(plot_save_path).replace("convergence_curve", "sensitivity_analysis")

    print(f"[C3] Running sensitivity analysis on θ[{sensitivity_param_index}] ...")
    sensitivity_results = sensitivity_analysis(
        model_coeffs = optimal_params,
        X            = X_norm,
        y            = y,
        degree       = degree,
        param_index  = sensitivity_param_index,
        save_path    = sens_path,
        show         = show_plots,
    )

    # ── Postconditions ────────────────────────────────────────────────────────
    assert len(optimal_params) == X_design.shape[1], (
        "[C3] Postcondition: optimal_params length must equal design matrix columns."
    )
    assert len(convergence_history) >= 1, (
        "[C3] Postcondition: convergence_history must contain at least one entry."
    )

    return optimal_params, convergence_history, sensitivity_results


# ─────────────────────────────────────────────────────────────────────────────
# CLI — demo: C1 → C2 → C3 full pipeline
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import csv as _csv

    # ── Locate df.csv ──────────────────────────────────────────────────────────
    _script_dir = Path(__file__).resolve().parent
    CSV_PATH    = _script_dir / "data" / "processed" / "df.csv"

    if not CSV_PATH.exists():
        print("[C3] df.csv not found — running data_loader.py ...")
        _loader = _script_dir / "data_loader.py"
        if _loader.exists():
            import subprocess, sys
            subprocess.run([sys.executable, str(_loader)], check=True)
        else:
            print("[C3] WARNING: data_loader.py not found.")

    X_data, y_data = None, None

    if CSV_PATH.exists():
        print(f"[C3] Loading data from {CSV_PATH} ...")
        import collections as _col
        wday_hour:      dict = _col.Counter()
        dates_per_wday: dict = _col.defaultdict(set)

        with open(CSV_PATH, newline="") as fh:
            reader = _csv.DictReader(fh)
            for row in reader:
                if row.get("day_type") == "weekday":
                    w = int(row["weekday_number"])
                    wday_hour[(w, int(row["hour"]))] += 1
                    dates_per_wday[w].add(row["date"])

        if len(wday_hour) > 10:
            keys   = list(wday_hour.keys())
            X_data = np.array([[w, h] for (w, h) in keys], dtype=float)
            y_data = np.array(
                [wday_hour[k] / len(dates_per_wday[k[0]]) for k in keys],
                dtype=float,
            )
            print(f"[C3] {len(y_data)} (weekday, hour) buckets loaded.")
            print(f"[C3] X shape: {X_data.shape} | y range: {y_data.min():.1f}–{y_data.max():.1f}")
        else:
            print("[C3] Insufficient CSV data — falling back to synthetic.")

    # Fallback: synthetic bimodal traffic curve
    if X_data is None:
        print("[C3] Using synthetic traffic data for demo.")
        rng    = np.random.default_rng(42)
        X_data = np.arange(24, dtype=float).reshape(-1, 1)
        y_data = (
            50
            + 80 * np.exp(-0.5 * ((X_data.ravel() - 8) / 2) ** 2)
            + 60 * np.exp(-0.5 * ((X_data.ravel() - 17) / 2) ** 2)
            + rng.normal(0, 5, 24)
        )

    DEGREE = 2

    # Normalise X column-wise
    if X_data.ndim == 1:
        X_data = X_data.reshape(-1, 1)
    X_mu    = X_data.mean(axis=0)
    X_sigma = X_data.std(axis=0)
    X_sigma[X_sigma < 1e-12] = 1.0
    X_norm  = (X_data - X_mu) / X_sigma
    print(f"[C3] X normalized — μ={X_mu}  σ={X_sigma}")

    # ── C2: fit regression ────────────────────────────────────────────────────
    model_c2  = fit_regression(X_norm, y_data, degree=DEGREE)
    coeffs_c2 = model_c2["coeffs"]

    # ── C3: optimize ─────────────────────────────────────────────────────────
    print("\n[C3] Optimizing C2 model parameters ...")
    optimal_coeffs, history, sensitivity = optimize_parameters(
        model_coeffs            = coeffs_c2,
        X                       = X_norm,
        y                       = y_data,
        degree                  = DEGREE,
        learning_rate           = 0.5,
        tolerance               = 1e-7,
        sensitivity_param_index = 1,
        show_plots              = False,
        pre_normalized          = True,
    )

    print("\n[C3] ── Final Results ─────────────────────────────────────")
    print(f"[C3] Optimal coefficients  : {optimal_coeffs}")
    print(f"[C3] Convergence steps     : {len(history)}")
    print(f"[C3] Sensitivity best θ₁  : {sensitivity['optimal_value']:.4f}")
    print("[C3] Done.")
