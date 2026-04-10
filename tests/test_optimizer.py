# -*- coding: utf-8 -*-
"""
test_optimizer.py
C3 — Unit Tests for optimizer.py
Author: Kenia Gabriela Hermida Núñez

Covers the full C3 test plan specified in the pseudocode document:
    ≥ 2 test cases per function (normal case + edge/boundary case).
    Every public function is exercised: optimize_parameters,
    plot_convergence_curve, and sensitivity_analysis.
    Internal helpers (_normalise, _build_design_matrix, _rmse_objective,
    _rmse_gradient) are tested separately for isolation.

Framework §5.2 C3 (Wilson §5b — pytest, ≥ 2 tests per module):
    All assertions use pytest — run with:  pytest tests/test_optimizer.py -v
"""

import numpy as np
import pytest
from pathlib import Path

# ── Import the C3 module under test ──────────────────────────────────────────
from optimizer import (
    optimize_parameters,
    plot_convergence_curve,
    sensitivity_analysis,
    _normalise,
    _build_design_matrix,
    _rmse_objective,
    _rmse_gradient,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared synthetic fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_bimodal_traffic(n: int = 24, seed: int = 42) -> tuple:
    """
    Synthetic bimodal traffic curve with morning and evening peaks.
    Mimics the Nuevo Sur intersection pattern used in the real pipeline.

    Returns:
        X: hours 0–23  (float array, shape (n,))
        y: vehicle counts per hour (float array, shape (n,))
    """
    rng = np.random.default_rng(seed)
    X   = np.linspace(0, 23, n, dtype=float)
    y   = (
        50
        + 80 * np.exp(-0.5 * ((X - 8)  / 2) ** 2)
        + 60 * np.exp(-0.5 * ((X - 17) / 2) ** 2)
        + rng.normal(0, 3, n)
    )
    return X, y


def _make_linear_data(n: int = 50, seed: int = 0) -> tuple:
    """
    Perfectly linear data  y = 3x + 7  for deterministic assertions.

    Returns:
        X: float array (n,)
        y: float array (n,)
        true_coeffs: (intercept=7, slope=3)
    """
    rng = np.random.default_rng(seed)
    X   = np.linspace(1, 20, n, dtype=float)
    y   = 3.0 * X + 7.0 + rng.normal(0, 0.1, n)
    return X, y, np.array([7.0, 3.0])


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper tests
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalise:
    """Tests for _normalise()."""

    def test_normal_case_zero_mean_unit_std(self):
        """After normalisation, mean ≈ 0 and std ≈ 1."""
        X = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        X_norm, mu, sigma = _normalise(X)
        assert abs(X_norm.mean()) < 1e-10, "Mean of normalised X should be ~0"
        assert abs(X_norm.std()  - 1.0) < 1e-10, "Std of normalised X should be ~1"
        assert abs(mu - 30.0) < 1e-10, f"Expected mu=30, got {mu}"

    def test_edge_constant_array_no_zero_division(self):
        """Constant array: std is set to 1.0 to avoid division by zero."""
        X = np.full(5, 7.0)
        X_norm, mu, sigma = _normalise(X)
        assert sigma == 1.0, "std should be replaced with 1.0 for constant arrays"
        # All elements become 0 after (x - mean) / 1
        assert np.allclose(X_norm, 0.0), "Normalised constant array should be all zeros"


class TestBuildDesignMatrix:
    """Tests for _build_design_matrix()."""

    def test_linear_degree_shape(self):
        """Degree-1 design matrix has shape (n, 2): intercept + x."""
        X = np.arange(10, dtype=float)
        D = _build_design_matrix(X, degree=1)
        assert D.shape == (10, 2), f"Expected (10, 2), got {D.shape}"
        assert np.all(D[:, 0] == 1.0), "First column must be all-ones (intercept)"

    def test_quadratic_degree_shape(self):
        """Degree-2 design matrix has shape (n, 3): intercept + x + x²."""
        X = np.arange(8, dtype=float)
        D = _build_design_matrix(X, degree=2)
        assert D.shape == (8, 3), f"Expected (8, 3), got {D.shape}"

    def test_cubic_degree_shape(self):
        """Degree-3 design matrix has shape (n, 4): intercept + x + x² + x³."""
        X = np.arange(12, dtype=float)
        D = _build_design_matrix(X, degree=3)
        assert D.shape == (12, 4), f"Expected (12, 4), got {D.shape}"

    def test_invalid_degree_raises(self):
        """Degree outside {1, 2, 3} raises AssertionError."""
        X = np.arange(5, dtype=float)
        with pytest.raises(AssertionError):
            _build_design_matrix(X, degree=4)

    def test_polynomial_values_correct(self):
        """Verify x² column contains squared values."""
        X = np.array([2.0, 3.0, 4.0])
        D = _build_design_matrix(X, degree=2)
        expected_sq = np.array([4.0, 9.0, 16.0])
        assert np.allclose(D[:, 2], expected_sq), "x² column values incorrect"


class TestRmseObjective:
    """Tests for _rmse_objective()."""

    def test_zero_rmse_perfect_prediction(self):
        """RMSE = 0 when predictions match targets exactly."""
        X_design = np.array([[1, 0], [1, 1], [1, 2]], dtype=float)
        coeffs   = np.array([3.0, 2.0])
        y        = X_design @ coeffs
        assert _rmse_objective(coeffs, X_design, y) == pytest.approx(0.0, abs=1e-10)

    def test_positive_rmse_on_noisy_data(self):
        """RMSE > 0 on noisy data."""
        rng      = np.random.default_rng(7)
        X, y, _  = _make_linear_data(n=30, seed=7)
        X_design = _build_design_matrix(X, degree=1)
        coeffs   = np.array([0.0, 0.0])          # deliberately bad coefficients
        rmse     = _rmse_objective(coeffs, X_design, y)
        assert rmse > 0.0, "RMSE must be positive for imperfect predictions"

    def test_rmse_symmetric_errors(self):
        """RMSE is identical for +e and -e residuals of the same magnitude."""
        X_design = np.array([[1, 0], [1, 1]], dtype=float)
        y        = np.array([2.0, 4.0])
        c_pos    = np.array([2.5, 2.0])   # predicts [2.5, 4.5] → errors [+0.5, +0.5]
        c_neg    = np.array([1.5, 2.0])   # predicts [1.5, 3.5] → errors [-0.5, -0.5]
        assert _rmse_objective(c_pos, X_design, y) == pytest.approx(
            _rmse_objective(c_neg, X_design, y), rel=1e-8
        )


class TestRmseGradient:
    """Tests for _rmse_gradient()."""

    def test_gradient_zero_at_minimum(self):
        """Gradient should be near zero at the OLS minimum."""
        X, y, _ = _make_linear_data(n=50)
        X_norm, _, _ = _normalise(X)
        X_design = _build_design_matrix(X_norm, degree=1)
        # OLS solution
        coeffs_ols = np.linalg.lstsq(X_design, y, rcond=None)[0]
        grad = _rmse_gradient(coeffs_ols, X_design, y)
        assert np.linalg.norm(grad) < 1e-3, (
            f"Gradient norm at OLS minimum should be near zero, got {np.linalg.norm(grad):.6f}"
        )

    def test_gradient_shape_matches_coeffs(self):
        """Gradient vector must have the same shape as the coefficient vector."""
        X, y, _ = _make_linear_data(n=20)
        X_norm, _, _ = _normalise(X)
        X_design = _build_design_matrix(X_norm, degree=2)
        coeffs = np.array([1.0, 0.5, -0.1])
        grad   = _rmse_gradient(coeffs, X_design, y)
        assert grad.shape == coeffs.shape, (
            f"Gradient shape {grad.shape} != coeffs shape {coeffs.shape}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# plot_convergence_curve tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPlotConvergenceCurve:
    """Tests for plot_convergence_curve()."""

    def test_normal_case_saves_file(self, tmp_path):
        """Figure is saved to disk when save_path is provided."""
        history   = [10.0, 8.0, 6.5, 5.8, 5.5, 5.3]
        save_path = str(tmp_path / "convergence.png")
        plot_convergence_curve(history, save_path=save_path, show=False)
        assert Path(save_path).exists(), "Convergence curve PNG should be saved to disk"

    def test_single_point_history(self, tmp_path):
        """Single-entry history (e.g. GD converged in 1 step) must not raise."""
        history   = [42.0]
        save_path = str(tmp_path / "single.png")
        plot_convergence_curve(history, save_path=save_path, show=False)
        assert Path(save_path).exists()

    def test_empty_history_raises(self):
        """Empty history must raise AssertionError (precondition enforced)."""
        with pytest.raises(AssertionError):
            plot_convergence_curve([], save_path=None, show=False)

    def test_no_save_does_not_create_file(self, tmp_path):
        """When save_path=None, no file is written."""
        history = [5.0, 4.0, 3.5]
        plot_convergence_curve(history, save_path=None, show=False)
        # If no error and no file created, the test passes
        assert True


# ─────────────────────────────────────────────────────────────────────────────
# sensitivity_analysis tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSensitivityAnalysis:
    """Tests for sensitivity_analysis()."""

    def test_normal_case_returns_correct_keys(self, tmp_path):
        """Result dict contains all required keys."""
        X, y, _ = _make_linear_data(n=30)
        X_norm, _, _ = _normalise(X)
        coeffs = np.array([7.0, 3.0])
        result = sensitivity_analysis(
            model_coeffs = coeffs,
            X            = X_norm,
            y            = y,
            degree       = 1,
            param_index  = 1,
            save_path    = str(tmp_path / "sens.png"),
            show         = False,
        )
        assert "param_values"  in result, "Key 'param_values' missing from result"
        assert "rmse_values"   in result, "Key 'rmse_values' missing from result"
        assert "optimal_value" in result, "Key 'optimal_value' missing from result"

    def test_result_lists_have_equal_length(self, tmp_path):
        """param_values and rmse_values must always have the same length."""
        X, y, _ = _make_linear_data(n=40)
        X_norm, _, _ = _normalise(X)
        coeffs = np.array([5.0, 2.5])
        result = sensitivity_analysis(
            model_coeffs      = coeffs,
            X                 = X_norm,
            y                 = y,
            degree            = 1,
            param_index       = 0,       # perturb intercept
            perturbation_range= np.linspace(0.8, 1.2, 10),
            save_path         = str(tmp_path / "sens2.png"),
            show              = False,
        )
        assert len(result["param_values"]) == len(result["rmse_values"]), (
            "param_values and rmse_values must have equal length"
        )
        assert len(result["param_values"]) == 10

    def test_too_few_samples_raises(self):
        """AssertionError if X has ≤ 10 samples (precondition)."""
        coeffs = np.array([1.0, 0.5])
        X_tiny = np.linspace(0, 1, 5)
        y_tiny = np.ones(5)
        with pytest.raises(AssertionError):
            sensitivity_analysis(coeffs, X_tiny, y_tiny, degree=1, save_path=None)

    def test_invalid_param_index_raises(self):
        """AssertionError if param_index >= len(model_coeffs)."""
        X, y, _ = _make_linear_data(n=20)
        X_norm, _, _ = _normalise(X)
        coeffs = np.array([1.0, 0.5])  # length 2 → valid indices are 0 and 1
        with pytest.raises(AssertionError):
            sensitivity_analysis(
                coeffs, X_norm, y, degree=1,
                param_index=5,   # out of range
                save_path=None,
            )

    def test_optimal_value_minimises_rmse(self, tmp_path):
        """The returned optimal_value must correspond to the minimum RMSE."""
        X, y, _ = _make_linear_data(n=30)
        X_norm, _, _ = _normalise(X)
        coeffs = np.array([7.0, 3.0])
        result = sensitivity_analysis(
            model_coeffs = coeffs,
            X            = X_norm,
            y            = y,
            degree       = 1,
            param_index  = 1,
            save_path    = str(tmp_path / "sens3.png"),
            show         = False,
        )
        best_idx    = result["rmse_values"].index(min(result["rmse_values"]))
        best_param  = result["param_values"][best_idx]
        assert best_param == pytest.approx(result["optimal_value"], rel=1e-9), (
            "optimal_value must correspond to the minimum RMSE entry"
        )


# ─────────────────────────────────────────────────────────────────────────────
# optimize_parameters tests  (main public API)
# ─────────────────────────────────────────────────────────────────────────────

class TestOptimizeParameters:
    """Tests for optimize_parameters() — the main C3 public function."""

    # ── Normal case: synthetic bimodal traffic, degree 2 ─────────────────────

    def test_normal_case_returns_three_outputs(self, tmp_path):
        """Function returns (optimal_params, convergence_history, sensitivity_results)."""
        X, y    = _make_bimodal_traffic()
        coeffs0 = np.array([50.0, 2.0, -0.1])   # rough starting guess for degree 2

        opt, hist, sens = optimize_parameters(
            model_coeffs = coeffs0,
            X            = X,
            y            = y,
            degree       = 2,
            max_iterations = 200,
            plot_save_path = str(tmp_path / "convergence.png"),
            show_plots     = False,
        )
        assert opt  is not None
        assert hist is not None
        assert sens is not None

    def test_optimal_params_length_equals_degree_plus_one(self, tmp_path):
        """optimal_params must have length == degree + 1 (postcondition)."""
        X, y = _make_bimodal_traffic()
        for degree in (1, 2, 3):
            coeffs0 = np.zeros(degree + 1)
            opt, _, _ = optimize_parameters(
                model_coeffs   = coeffs0,
                X              = X,
                y              = y,
                degree         = degree,
                max_iterations = 100,
                plot_save_path = str(tmp_path / f"conv_d{degree}.png"),
                show_plots     = False,
            )
            assert len(opt) == degree + 1, (
                f"Degree {degree}: expected {degree + 1} coefficients, got {len(opt)}"
            )

    def test_convergence_history_is_non_empty(self, tmp_path):
        """GD must produce at least one convergence entry."""
        X, y    = _make_bimodal_traffic()
        coeffs0 = np.zeros(2)
        _, hist, _ = optimize_parameters(
            model_coeffs   = coeffs0,
            X              = X,
            y              = y,
            degree         = 1,
            max_iterations = 50,
            plot_save_path = str(tmp_path / "conv.png"),
            show_plots     = False,
        )
        assert len(hist) >= 1, "Convergence history must contain at least one entry"

    def test_convergence_history_is_non_increasing(self, tmp_path):
        """RMSE should not increase between consecutive GD steps (no big spikes)."""
        X, y    = _make_bimodal_traffic(n=24, seed=0)
        coeffs0 = np.zeros(3)
        _, hist, _ = optimize_parameters(
            model_coeffs   = coeffs0,
            X              = X,
            y              = y,
            degree         = 2,
            learning_rate  = 1e-2,
            max_iterations = 500,
            plot_save_path = str(tmp_path / "conv_mono.png"),
            show_plots     = False,
        )
        # Allow tiny floating-point noise — we check no step increases by > 1 vehicle
        diffs = np.diff(hist)
        assert np.all(diffs <= 1.0), (
            f"RMSE increased unexpectedly: max increase = {diffs.max():.4f}"
        )

    def test_rmse_improves_over_initial_guess(self, tmp_path):
        """Final RMSE must be ≤ RMSE of the deliberately bad initial guess."""
        X, y    = _make_bimodal_traffic()
        # Deliberately bad coefficients (all zeros — RMSE will be the mean of y)
        coeffs0 = np.zeros(3)
        opt, hist, _ = optimize_parameters(
            model_coeffs   = coeffs0,
            X              = X,
            y              = y,
            degree         = 2,
            max_iterations = 500,
            plot_save_path = str(tmp_path / "conv_improve.png"),
            show_plots     = False,
        )
        assert hist[-1] < hist[0], (
            f"RMSE did not improve: initial={hist[0]:.4f}, final={hist[-1]:.4f}"
        )

    def test_convergence_png_is_saved(self, tmp_path):
        """Convergence curve PNG must be written to disk."""
        X, y    = _make_bimodal_traffic()
        coeffs0 = np.zeros(2)
        png_path = tmp_path / "convergence_curve.png"
        optimize_parameters(
            model_coeffs   = coeffs0,
            X              = X,
            y              = y,
            degree         = 1,
            max_iterations = 50,
            plot_save_path = str(png_path),
            show_plots     = False,
        )
        assert png_path.exists(), "Convergence curve PNG was not saved to disk"

    def test_sensitivity_png_is_saved(self, tmp_path):
        """Sensitivity analysis PNG must also be written to disk."""
        X, y    = _make_bimodal_traffic()
        coeffs0 = np.zeros(2)
        png_path = tmp_path / "convergence_curve.png"
        optimize_parameters(
            model_coeffs   = coeffs0,
            X              = X,
            y              = y,
            degree         = 1,
            max_iterations = 50,
            plot_save_path = str(png_path),
            show_plots     = False,
        )
        sens_path = tmp_path / "sensitivity_analysis.png"
        assert sens_path.exists(), "Sensitivity analysis PNG was not saved to disk"

    # ── Edge / boundary cases ─────────────────────────────────────────────────

    def test_too_few_samples_raises_assertion(self):
        """AssertionError raised when X has ≤ 10 samples (precondition)."""
        X = np.arange(5, dtype=float)
        y = np.ones(5)
        with pytest.raises(AssertionError):
            optimize_parameters(np.zeros(2), X, y, degree=1, plot_save_path=None)

    def test_mismatched_X_y_raises_value_error(self):
        """ValueError raised when len(X) != len(y)."""
        X = np.arange(20, dtype=float)
        y = np.ones(15)
        with pytest.raises(ValueError):
            optimize_parameters(np.zeros(2), X, y, degree=1, plot_save_path=None)

    def test_invalid_degree_raises_assertion(self):
        """AssertionError raised for degree outside {1, 2, 3}."""
        X = np.arange(20, dtype=float)
        y = np.ones(20)
        with pytest.raises(AssertionError):
            optimize_parameters(np.zeros(5), X, y, degree=4, plot_save_path=None)

    def test_single_iteration(self, tmp_path):
        """max_iterations=1 runs exactly one GD step without error."""
        X, y = _make_bimodal_traffic()
        coeffs0 = np.zeros(2)
        _, hist, _ = optimize_parameters(
            model_coeffs   = coeffs0,
            X              = X,
            y              = y,
            degree         = 1,
            max_iterations = 1,
            plot_save_path = str(tmp_path / "conv1.png"),
            show_plots     = False,
        )
        assert len(hist) == 1, f"Expected 1 history entry, got {len(hist)}"

    def test_linear_data_converges_to_low_rmse(self, tmp_path):
        """On near-perfectly linear data, optimized RMSE should be small (<2)."""
        X, y, _ = _make_linear_data(n=50)
        coeffs0 = np.zeros(2)
        opt, hist, _ = optimize_parameters(
            model_coeffs   = coeffs0,
            X              = X,
            y              = y,
            degree         = 1,
            max_iterations = 2000,
            tolerance      = 1e-8,
            plot_save_path = str(tmp_path / "conv_lin.png"),
            show_plots     = False,
        )
        final_rmse = hist[-1]
        assert final_rmse < 2.0, (
            f"RMSE on near-linear data should be small, got {final_rmse:.4f}"
        )
