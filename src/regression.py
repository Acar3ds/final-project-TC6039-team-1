import numpy as np

def fit_regression(X, y, degree=1):
    assert degree >= 1 and degree <= 3, "degree must be between 1 and 3"

    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows")

    if len(X) == 0:
        raise ValueError("Empty dataset")

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    X_poly = X.copy()

    if degree >= 2:
        X_poly = np.hstack([X_poly, X**2])
    if degree >= 3:
        X_poly = np.hstack([X_poly, X**3])

    X_design = np.hstack([np.ones((X_poly.shape[0], 1)), X_poly])

    coeffs = np.linalg.lstsq(X_design, y, rcond=None)[0]

    y_hat = X_design @ coeffs
    residuals = y - y_hat

    rmse = np.sqrt(np.mean((y - y_hat) ** 2))

    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0

    return coeffs, r2, rmse, y_hat, residuals
