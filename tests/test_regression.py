import numpy as np
from src.regression import fit_regression

def test_normal_case():
    X = np.array([[1], [2], [3], [4]], dtype=float)
    y = np.array([2, 4, 6, 8], dtype=float)

    coeffs, r2, rmse, y_hat, residuals = fit_regression(X, y)

    assert len(y_hat) == len(y)
    assert len(residuals) == len(y)
    assert rmse >= 0

def test_dimension_error():
    X = np.array([[1], [2]], dtype=float)
    y = np.array([2], dtype=float)

    try:
        fit_regression(X, y)
        assert False
    except ValueError:
        assert True
