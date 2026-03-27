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

def cross_validate(X, y, degree=1, k=5):
    assert k >= 2, "k must be at least 2"

    if len(X) != len(y):
        raise ValueError("X and y must have same size")

    n = len(X)
    indices = np.arange(n)
    folds = np.array_split(indices, k)

    rmses = []
    r2s = []

    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        coeffs, _, _, _, _ = fit_regression(X_train, y_train, degree)

        # reconstruir diseño
        if X_val.ndim == 1:
            X_val = X_val.reshape(-1, 1)

        X_poly = X_val.copy()
        if degree >= 2:
            X_poly = np.hstack([X_poly, X_val**2])
        if degree >= 3:
            X_poly = np.hstack([X_poly, X_val**3])

        X_design = np.hstack([np.ones((X_poly.shape[0], 1)), X_poly])

        y_hat = X_design @ coeffs

        rmse = np.sqrt(np.mean((y_val - y_hat)**2))

        ss_res = np.sum((y_val - y_hat)**2)
        ss_tot = np.sum((y_val - np.mean(y_val))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0

        rmses.append(rmse)
        r2s.append(r2)

    return np.mean(rmses), np.mean(r2s)

import matplotlib.pyplot as plt

def plot_residuals(y, y_hat):
    residuals = y - y_hat

    plt.scatter(y_hat, residuals)
    plt.axhline(0)
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.show()
