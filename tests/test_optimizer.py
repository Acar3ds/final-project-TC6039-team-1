import os
import tempfile
import unittest
import numpy as np

from optimizer import optimize_parameters, _build_design_matrix, _rmse_objective
from regression import fit_regression

def _make_peak_segment(seed=42):
    rng   = np.random.default_rng(seed)
    hours = np.array([7, 8, 9] * 20, dtype=float)
    base  = np.array([40, 90, 120] * 20, dtype=float)
    y     = base + rng.normal(0, 5, len(base))
    X     = hours.reshape(-1, 1)
    mu, sigma = X.mean(), X.std()
    return (X - mu) / sigma, y

def _make_synthetic_traffic(seed=42):
    rng = np.random.default_rng(seed)
    hours = np.arange(24, dtype=float)
    y = (
    30
    + 80 * np.exp(-0.5 * ((hours - 8) / 2) ** 2)
    + rng.normal(0, 3, 24)
    )
    X = hours.reshape(-1, 1)
    mu, sigma = X.mean(), X.std()
    return (X - mu) / sigma, y

# ─────────────────────────────────────────────────────────────
# NORMAL CASE — Peak Hour Trend
# ─────────────────────────────────────────────────────────────

class TestNormalCase(unittest.TestCase):

    def test_peak_hour_trend(self):
        X, y = _make_peak_segment()
        model = fit_regression(X, y, degree=1)

        # R² ≥ 0.70
        self.assertGreaterEqual(model["r2"], 0.70)

        # Positive slope
        self.assertGreater(model["coeffs"][1], 0)

        # Optimization improves RMSE
        with tempfile.TemporaryDirectory() as tmp:
            optimal, _, _ = optimize_parameters(
                model["coeffs"], X, y,
                degree=1,
                learning_rate=0.3,
                show_plots=False,
                plot_save_path=os.path.join(tmp, "conv.png"),
                pre_normalized=True,
            )

        X_des = _build_design_matrix(X, 1)
        rmse_opt = _rmse_objective(optimal, X_des, y)

        self.assertLessEqual(rmse_opt, model["rmse"] + 1e-6)

# ─────────────────────────────────────────────────────────────
# EDGE CASE — Outlier Detection
# ─────────────────────────────────────────────────────────────

class TestEdgeCase(unittest.TestCase):

    def _sigma_filter(self, y, n_sigma=3):
        return np.abs(y - y.mean()) <= n_sigma * y.std()

    def test_outlier_handling(self):
        X, y = _make_synthetic_traffic()

        # Inject outlier
        y_spike = y.copy()
        y_spike[10] = 1000

        # Filter
        mask = self._sigma_filter(y_spike)
        X_clean, y_clean = X[mask], y_spike[mask]

        # Fit models
        model_spike = fit_regression(X, y_spike, degree=1)
        model_clean = fit_regression(X_clean, y_clean, degree=1)

        # RMSE improves after cleaning
        self.assertLess(model_clean["rmse"], model_spike["rmse"])

        # Optimization still works
        with tempfile.TemporaryDirectory() as tmp:
            optimal, history, _ = optimize_parameters(
                model_spike["coeffs"], X, y_spike,
                degree=1,
                learning_rate=0.1,
                show_plots=False,
                plot_save_path=os.path.join(tmp, "conv.png"),
                pre_normalized=True,
            )

            self.assertFalse(np.any(np.isnan(optimal)))
            self.assertGreater(len(history), 0)
    
    
if __name__ == "__main__":
    unittest.main()

