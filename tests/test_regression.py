
import numpy as np
import pytest
from regression import mse, fit_regression, predict, cross_validate


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def simple_linear_data():
    """Perfect linear relationship: y = 3 + 2x."""
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 50)
    y = 3.0 + 2.0 * x
    return x, y


@pytest.fixture
def noisy_quadratic_data():
    """Quadratic with noise: y = 1 + 0.5x + 3x² + noise."""
    rng = np.random.default_rng(42)
    x = np.linspace(-5, 5, 100)
    y = 1.0 + 0.5 * x + 3.0 * x ** 2 + rng.normal(0, 2, size=100)
    return x, y


@pytest.fixture
def multivariate_data():
    """Two predictors: y = 10 + 2*x1 - 3*x2."""
    rng = np.random.default_rng(7)
    x1 = rng.uniform(0, 10, 80)
    x2 = rng.uniform(0, 10, 80)
    features = np.column_stack([x1, x2])
    target = 10.0 + 2.0 * x1 - 3.0 * x2
    return features, target


# ── Tests for mse() ───────────────────────────────────────────────

class TestMSE:
    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mse(y, y) == pytest.approx(0.0)

    def test_known_mse(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 5.0])
        # error = [0, 0, -2], squared = [0, 0, 4], mean = 4/3
        assert mse(y_true, y_pred) == pytest.approx(4.0 / 3.0)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(AssertionError):
            mse([1, 2], [1, 2, 3])


# ── Tests for fit_regression() ────────────────────────────────────

class TestFitRegression:
    def test_perfect_linear_fit(self, simple_linear_data):
        x, y = simple_linear_data
        model = fit_regression(x, y, degree=1)
        assert model["r2"] == pytest.approx(1.0, abs=1e-10)
        assert model["rmse"] == pytest.approx(0.0, abs=1e-10)
        # coeffs should be [3.0, 2.0]
        assert model["coeffs"][0] == pytest.approx(3.0, abs=1e-8)
        assert model["coeffs"][1] == pytest.approx(2.0, abs=1e-8)

    def test_quadratic_improves_over_linear(self, noisy_quadratic_data):
        x, y = noisy_quadratic_data
        linear_model = fit_regression(x, y, degree=1)
        quadratic_model = fit_regression(x, y, degree=2)
        assert quadratic_model["r2"] > linear_model["r2"]
        assert quadratic_model["rmse"] < linear_model["rmse"]

    def test_multivariate_perfect_fit(self, multivariate_data):
        features, target = multivariate_data
        model = fit_regression(features, target, degree=1)
        assert model["r2"] == pytest.approx(1.0, abs=1e-8)

    def test_residuals_sum_to_zero(self, noisy_quadratic_data):
        x, y = noisy_quadratic_data
        model = fit_regression(x, y, degree=2)
        # OLS residuals sum to ~0 when intercept is included
        assert np.sum(model["residuals"]) == pytest.approx(0.0, abs=1e-8)

    def test_invalid_degree_raises(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        with pytest.raises(AssertionError):
            fit_regression(x, y, degree=5)

    def test_empty_dataset_raises(self):
        with pytest.raises(AssertionError):
            fit_regression(np.array([]), np.array([]), degree=1)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(AssertionError):
            fit_regression(np.array([1, 2, 3]), np.array([1, 2]), degree=1)


# ── Tests for predict() ──────────────────────────────────────────

class TestPredict:
    def test_predict_matches_training(self, simple_linear_data):
        x, y = simple_linear_data
        model = fit_regression(x, y, degree=1)
        y_pred = predict(model, x)
        np.testing.assert_allclose(y_pred, model["y_hat"], atol=1e-10)

    def test_predict_new_points(self, simple_linear_data):
        x, y = simple_linear_data
        model = fit_regression(x, y, degree=1)
        new_x = np.array([20.0, 30.0])
        expected = 3.0 + 2.0 * new_x
        y_pred = predict(model, new_x)
        np.testing.assert_allclose(y_pred, expected, atol=1e-8)


# ── Tests for cross_validate() ───────────────────────────────────

class TestCrossValidate:
    def test_returns_k_folds(self, noisy_quadratic_data):
        x, y = noisy_quadratic_data
        result = cross_validate(2, x, y, k=5)
        assert len(result["fold_metrics"]) == 5

    def test_fold_numbers_sequential(self, noisy_quadratic_data):
        x, y = noisy_quadratic_data
        result = cross_validate(1, x, y, k=3)
        fold_numbers = [f["fold"] for f in result["fold_metrics"]]
        assert fold_numbers == [1, 2, 3]

    def test_cv_rmse_is_positive(self, noisy_quadratic_data):
        x, y = noisy_quadratic_data
        result = cross_validate(2, x, y, k=5)
        assert result["mean_rmse"] > 0
        assert all(f["rmse"] > 0 for f in result["fold_metrics"])

    def test_perfect_data_cv(self, simple_linear_data):
        x, y = simple_linear_data
        result = cross_validate(1, x, y, k=5)
        assert result["mean_r2"] == pytest.approx(1.0, abs=1e-6)
        assert result["mean_rmse"] == pytest.approx(0.0, abs=1e-6)

    def test_invalid_k_raises(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        with pytest.raises(AssertionError):
            cross_validate(1, x, y, k=1)

    def test_reproducibility(self, noisy_quadratic_data):
        x, y = noisy_quadratic_data
        result_a = cross_validate(2, x, y, k=5)
        result_b = cross_validate(2, x, y, k=5)
        assert result_a["mean_rmse"] == pytest.approx(result_b["mean_rmse"])
