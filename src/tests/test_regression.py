import numpy as np
import pytest

from src.regression import fit_regression, cross_validate


def test_normal_case():
    X = np.array([[1], [2], [3], [4]], dtype=float)
    y = np.array([2, 4, 6, 8], dtype=float)

    coeffs, r2, rmse, y_hat, residuals = fit_regression(X, y)

    assert len(y_hat) == len(y)
    assert len(residuals) == len(y)
    assert rmse >= 0
    assert r2 <= 1


def test_dimension_error():
    X = np.array([[1], [2]], dtype=float)
    y = np.array([2], dtype=float)

    with pytest.raises(ValueError):
        fit_regression(X, y)


def test_cross_validate_normal_case():
    X = np.array([[1], [2], [3], [4], [5], [6]], dtype=float)
    y = np.array([2, 4, 6, 8, 10, 12], dtype=float)

    avg_rmse, avg_r2 = cross_validate(X, y, degree=1, k=3)

    assert avg_rmse >= 0
    assert avg_r2 <= 1


def test_cross_validate_invalid_k():
    X = np.array([[1], [2]], dtype=float)
    y = np.array([2, 4], dtype=float)

    with pytest.raises(AssertionError):
        cross_validate(X, y, degree=1, k=1)
