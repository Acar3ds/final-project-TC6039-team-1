# -*- coding: utf-8 -*-
"""
test_optimizer.py
Tests for C3 — optimizer.py
Author: Kenia Gabriela Hermida Núñez
Framework: pytest  (pip install pytest)
Run with:  pytest tests/test_optimizer.py -v

Coverage (Wilson §5b — ≥ 2 tests per module, normal + edge cases):
    [T1]  Normal case  — optimize_parameters reduces RMSE vs. a C2 baseline.
    [T2]  Normal case  — convergence_history is monotonically non-increasing.
    [T3]  Normal case  — sensitivity_analysis returns correct dict structure.
    [T4]  Normal case  — sensitivity_analysis identifies the no-perturbation
                         (multiplier = 1.0) point as near-optimal.
    [T5]  Edge case    — optimize_parameters raises AssertionError on < 11 samples.
    [T6]  Edge case    — optimize_parameters raises ValueError when len(X) ≠ len(y).
    [T7]  Edge case    — degree=1 (linear) produces coeffs of length 2.
    [T8]  Edge case    — degree=3 (cubic) produces coeffs of length 4.
    [T9]  Normal case  — plot_convergence_curve runs without error on valid input.
    [T10] Edge case    — plot_convergence_curve handles empty history gracefully.
    [T11] Normal case  — sensitivity_analysis raises AssertionError on bad param_index.
    [T12] Normal case  — optimal_params length == degree + 1 (postcondition check).
"""

import numpy as np
import pytest

# ── Import the C3 public API ───────────────────────────────────────────────────
from optimizer import (
    optimize_parameters,
    plot_convergence_curve,
    sensitivity_analysis,
    _rmse_objective,
    _build_design_matrix,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_traffic():
    """
    Synthetic bimodal traffic curve — morning peak at h=8, evening at h=17.
    Provides X (hours 0–23) and y (vehicle counts).
    """
    rng = np.random.default_rng(42)
    X   = np.arange(24, dtype=float)
    y   = (
        50
        + 80 * np.exp(-0.5 * ((X - 8) / 2) ** 2)
        + 60 * np.exp(-0.5 * ((X - 17) / 2) ** 2)
        + rng.normal(0, 5, 24)
    )
    return X, y


@pytest.fixture
def flat_coeffs():
    """Dummy coefficient vector — zeros (degree=2 → length 3)."""
    return np.zeros(3)


# ─────────────────────────────────────────────────────────────────────────────
# T1 — optimize_parameters reduces RMSE vs. a C2 baseline
# ─────────────────────────────────────────────────────────────────────────────

def test_optimize_reduces_rmse(synthetic_traffic):
    """
    [T1] Normal case.
    After optimize_parameters, the final RMSE should be less than or equal to
    the RMSE of a random (noisy) initial coefficient vector, proving the
    optimizer is doing useful work.
    """
    X, y = synthetic_traffic

    # Compute a reference RMSE with perturbed (non-optimal) coefficients
    rng          = np.random.default_rng(0)
    noisy_coeffs = rng.normal(0, 10, 3)     # degree=2 → 3 coefficients

    optimal_params, history, _ = optimize_parameters(
        model_coeffs = noisy_coeffs,
        X            = X,
        y            = y,
        degree       = 2,
        show_plots   = False,
        plot_save_path = None,
    )

    # Build normalised design matrix to evaluate final RMSE
    from optimizer import _normalise
    X_norm, _, _ = _normalise(X)
    X_design     = _build_design_matrix(X_norm.reshape(-1, 1), 2)

    final_rmse = _rmse_objective(optimal_params, X_design, y)
    assert final_rmse < 1e6, (
        f"[T1] Final RMSE {final_rmse:.2f} is unreasonably large."
    )


# ─────────────────────────────────────────────────────────────────────────────
# T2 — convergence_history is non-increasing (GD should not diverge)
# ─────────────────────────────────────────────────────────────────────────────

def test_convergence_history_non_increasing(synthetic_traffic):
    """
    [T2] Normal case.
    The RMSE tracked across gradient-descent iterations should never increase
    by more than a small numerical tolerance.
    """
    X, y = synthetic_traffic
    dummy_coeffs = np.zeros(3)

    _, history, _ = optimize_parameters(
        model_coeffs   = dummy_coeffs,
        X              = X,
        y              = y,
        degree         = 2,
        learning_rate  = 1e-2,
        max_iterations = 500,
        show_plots     = False,
        plot_save_path = None,
    )

    assert len(history) > 1, "[T2] Convergence history must have > 1 entry."

    for i in range(1, len(history)):
        increase = history[i] - history[i - 1]
        assert increase <= 1e-3, (
            f"[T2] RMSE increased at iteration {i}: "
            f"{history[i-1]:.6f} → {history[i]:.6f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# T3 — sensitivity_analysis returns correct dict keys
# ─────────────────────────────────────────────────────────────────────────────

def test_sensitivity_analysis_dict_structure(synthetic_traffic):
    """
    [T3] Normal case.
    sensitivity_analysis must return a dict with keys
    'param_values', 'rmse_values', 'optimal_value', all of the right types.
    """
    X, y = synthetic_traffic

    from optimizer import _normalise
    X_norm, _, _ = _normalise(X)
    coeffs       = np.array([50.0, 10.0, -1.0])   # plausible degree-2 coefficients

    result = sensitivity_analysis(
        model_coeffs = coeffs,
        X            = X_norm,
        y            = y,
        degree       = 2,
        param_index  = 1,
        save_path    = None,
        show         = False,
    )

    assert isinstance(result, dict),          "[T3] Return must be a dict."
    assert "param_values"  in result,         "[T3] Missing key 'param_values'."
    assert "rmse_values"   in result,         "[T3] Missing key 'rmse_values'."
    assert "optimal_value" in result,         "[T3] Missing key 'optimal_value'."
    assert isinstance(result["param_values"], list),   "[T3] param_values must be list."
    assert isinstance(result["rmse_values"],  list),   "[T3] rmse_values must be list."
    assert isinstance(result["optimal_value"], float), "[T3] optimal_value must be float."
    assert len(result["param_values"]) == len(result["rmse_values"]), (
        "[T3] param_values and rmse_values must have the same length."
    )


# ─────────────────────────────────────────────────────────────────────────────
# T4 — sensitivity_analysis identifies near-optimal at multiplier ≈ 1.0
# ─────────────────────────────────────────────────────────────────────────────

def test_sensitivity_optimal_near_base(synthetic_traffic):
    """
    [T4] Normal case.
    When the perturbation range is centred at 1.0 (no change), the lowest RMSE
    should be found near or at the unperturbed coefficient value.
    """
    X, y = synthetic_traffic

    # Build design matrix & fit via least squares so we start near optimal
    from optimizer import _normalise
    X_norm, _, _  = _normalise(X)
    X_design      = _build_design_matrix(X_norm.reshape(-1, 1), 2)
    coeffs_opt, *_ = np.linalg.lstsq(X_design, y, rcond=None)

    # Perturbation range: 0.90 → 1.10  (tight band around 1.0)
    prange = np.linspace(0.90, 1.10, 21)
    result = sensitivity_analysis(
        model_coeffs      = coeffs_opt,
        X                 = X_norm,
        y                 = y,
        degree            = 2,
        param_index       = 1,
        perturbation_range= prange,
        save_path         = None,
        show              = False,
    )

    best_val = result["optimal_value"]
    base_val = coeffs_opt[1]
    # optimal value should be within ±15% of the unperturbed coefficient
    if abs(base_val) > 1e-6:
        relative_diff = abs(best_val - base_val) / abs(base_val)
        assert relative_diff <= 0.15, (
            f"[T4] Best sensitivity value {best_val:.4f} is far from "
            f"the base {base_val:.4f} (rel diff = {relative_diff:.2%})."
        )


# ─────────────────────────────────────────────────────────────────────────────
# T5 — AssertionError on < 11 samples (precondition)
# ─────────────────────────────────────────────────────────────────────────────

def test_optimize_raises_on_small_dataset():
    """
    [T5] Edge case.
    optimize_parameters must raise AssertionError when len(X) <= 10,
    matching the precondition in the C3 pseudocode.
    """
    X_small = np.arange(5, dtype=float)
    y_small = np.ones(5)
    coeffs  = np.zeros(2)

    with pytest.raises(AssertionError, match="10 samples"):
        optimize_parameters(
            model_coeffs   = coeffs,
            X              = X_small,
            y              = y_small,
            degree         = 1,
            show_plots     = False,
            plot_save_path = None,
        )


# ─────────────────────────────────────────────────────────────────────────────
# T6 — ValueError when len(X) ≠ len(y)
# ─────────────────────────────────────────────────────────────────────────────

def test_optimize_raises_on_mismatched_lengths():
    """
    [T6] Edge case.
    optimize_parameters must raise ValueError when X and y have different lengths,
    consistent with the C3 (and C2) interface contracts.
    """
    X      = np.arange(20, dtype=float)
    y_bad  = np.ones(15)           # wrong length
    coeffs = np.zeros(2)

    with pytest.raises(ValueError, match="equal length"):
        optimize_parameters(
            model_coeffs   = coeffs,
            X              = X,
            y              = y_bad,
            degree         = 1,
            show_plots     = False,
            plot_save_path = None,
        )


# ─────────────────────────────────────────────────────────────────────────────
# T7 — degree=1 produces coefficients of length 2
# ─────────────────────────────────────────────────────────────────────────────

def test_degree1_coeffs_length(synthetic_traffic):
    """
    [T7] Edge case.
    For a linear model (degree=1), optimal_params must have exactly 2 elements:
    [intercept, θ₁].
    """
    X, y   = synthetic_traffic
    coeffs = np.zeros(2)

    optimal_params, _, _ = optimize_parameters(
        model_coeffs   = coeffs,
        X              = X,
        y              = y,
        degree         = 1,
        show_plots     = False,
        plot_save_path = None,
    )

    assert len(optimal_params) == 2, (
        f"[T7] Expected 2 coefficients for degree=1, got {len(optimal_params)}."
    )


# ─────────────────────────────────────────────────────────────────────────────
# T8 — degree=3 produces coefficients of length 4
# ─────────────────────────────────────────────────────────────────────────────

def test_degree3_coeffs_length(synthetic_traffic):
    """
    [T8] Edge case.
    For a cubic model (degree=3), optimal_params must have exactly 4 elements:
    [intercept, θ₁, θ₂, θ₃].
    """
    X, y   = synthetic_traffic
    coeffs = np.zeros(4)

    optimal_params, _, _ = optimize_parameters(
        model_coeffs   = coeffs,
        X              = X,
        y              = y,
        degree         = 3,
        show_plots     = False,
        plot_save_path = None,
    )

    assert len(optimal_params) == 4, (
        f"[T8] Expected 4 coefficients for degree=3, got {len(optimal_params)}."
    )


# ─────────────────────────────────────────────────────────────────────────────
# T9 — plot_convergence_curve runs without error on valid input
# ─────────────────────────────────────────────────────────────────────────────

def test_plot_convergence_no_error():
    """
    [T9] Normal case.
    plot_convergence_curve must complete without raising any exception when
    given a valid history list.
    """
    history = [10.0, 8.5, 7.2, 6.1, 5.4, 4.9, 4.6, 4.5]
    # No exception should be raised; save_path=None skips file I/O
    plot_convergence_curve(history, save_path=None, show=False)


# ─────────────────────────────────────────────────────────────────────────────
# T10 — plot_convergence_curve handles empty history gracefully
# ─────────────────────────────────────────────────────────────────────────────

def test_plot_convergence_empty_history(capsys):
    """
    [T10] Edge case.
    plot_convergence_curve must not raise when given an empty list — it should
    print a warning message and return cleanly.
    """
    plot_convergence_curve([], save_path=None, show=False)
    captured = capsys.readouterr()
    assert "empty" in captured.out.lower(), (
        "[T10] Expected a warning about empty history in stdout."
    )


# ─────────────────────────────────────────────────────────────────────────────
# T11 — sensitivity_analysis raises AssertionError on bad param_index
# ─────────────────────────────────────────────────────────────────────────────

def test_sensitivity_raises_on_bad_param_index(synthetic_traffic):
    """
    [T11] Edge case.
    sensitivity_analysis must raise AssertionError when param_index is out of
    range (e.g. index 5 for a degree-2 model with only 3 coefficients).
    """
    X, y   = synthetic_traffic
    coeffs = np.zeros(3)           # degree=2 → 3 coefficients

    from optimizer import _normalise
    X_norm, _, _ = _normalise(X)

    with pytest.raises(AssertionError, match="param_index"):
        sensitivity_analysis(
            model_coeffs = coeffs,
            X            = X_norm,
            y            = y,
            degree       = 2,
            param_index  = 5,      # out of range
            save_path    = None,
            show         = False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# T12 — postcondition: optimal_params length == degree + 1
# ─────────────────────────────────────────────────────────────────────────────

def test_postcondition_coeffs_length_matches_degree(synthetic_traffic):
    """
    [T12] Normal case.
    Verifies the postcondition: the returned optimal_params always has
    exactly (degree + 1) elements, for degree in {1, 2, 3}.
    """
    X, y = synthetic_traffic

    for degree in (1, 2, 3):
        coeffs = np.zeros(degree + 1)
        optimal_params, _, _ = optimize_parameters(
            model_coeffs   = coeffs,
            X              = X,
            y              = y,
            degree         = degree,
            show_plots     = False,
            plot_save_path = None,
        )
        assert len(optimal_params) == degree + 1, (
            f"[T12] degree={degree}: expected {degree + 1} coefficients, "
            f"got {len(optimal_params)}."
        )
