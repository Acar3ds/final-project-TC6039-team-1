
import numpy as np
import matplotlib.pyplot as plt


# ── Base metrics ──────────────────────────────────────────────────

def mse(y_true, y_pred):
    """Compute mean squared error between true and predicted values.

    Args:
        y_true: Array of observed values.
        y_pred: Array of predicted values.

    Returns:
        float: Mean squared error.

    Raises:
        ValueError: If arrays have different lengths.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length"
    return float(np.mean((y_true - y_pred) ** 2))


# ── Design matrix construction ───────────────────────────────────

def _build_design_matrix(features, degree):
    """Build polynomial design matrix with intercept column.

    For features of shape (n, p) and a given degree, produces:
    [1 | X | X² | X³]  (up to the specified degree).

    Args:
        features: ndarray of shape (n,) or (n, p) with predictor values.
        degree: Polynomial degree (1, 2, or 3).

    Returns:
        ndarray: Design matrix of shape (n, 1 + p * degree) with
                 an intercept column prepended.

    Raises:
        AssertionError: If degree is not in {1, 2, 3}.
    """
    assert degree in (1, 2, 3), "degree must be 1, 2, or 3"

    if features.ndim == 1:
        features = features.reshape(-1, 1)

    polynomial_terms = [features]
    if degree >= 2:
        polynomial_terms.append(features ** 2)
    if degree >= 3:
        polynomial_terms.append(features ** 3)

    feature_block = np.hstack(polynomial_terms)
    num_samples = feature_block.shape[0]
    intercept = np.ones((num_samples, 1))

    return np.hstack([intercept, feature_block])


# ── Regression fitting ────────────────────────────────────────────

def fit_regression(features, target, degree=1):
    """Fit polynomial regression via ordinary least squares (OLS).

    Args:
        features: ndarray of shape (n,) or (n, p) — independent variables.
        target: ndarray of shape (n,) — dependent variable.
        degree: Polynomial degree, one of {1, 2, 3}.

    Returns:
        dict: Model dictionary with keys:
            - coeffs (ndarray): Regression coefficients (intercept first).
            - degree (int): Polynomial degree used.
            - r2 (float): Coefficient of determination on training data.
            - rmse (float): Root mean squared error on training data.
            - mse (float): Mean squared error on training data.
            - y_hat (ndarray): Predicted values.
            - residuals (ndarray): Residuals (target − y_hat).

    Raises:
        ValueError: If features and target have mismatched lengths
                    or the dataset is empty.
        AssertionError: If degree is not in {1, 2, 3}.
    """
    features = np.asarray(features, dtype=float)
    target = np.asarray(target, dtype=float)

    assert len(features) > 0, "Dataset must not be empty"
    assert len(features) == len(target), (
        f"features has {len(features)} rows but target has {len(target)}"
    )
    assert degree in (1, 2, 3), "degree must be 1, 2, or 3"

    design_matrix = _build_design_matrix(features, degree)
    coefficients = np.linalg.lstsq(design_matrix, target, rcond=None)[0]

    predicted = design_matrix @ coefficients
    residuals = target - predicted

    train_mse = mse(target, predicted)
    train_rmse = np.sqrt(train_mse)

    sum_sq_residuals = np.sum(residuals ** 2)
    sum_sq_total = np.sum((target - np.mean(target)) ** 2)
    r_squared = 1.0 - sum_sq_residuals / sum_sq_total if sum_sq_total != 0 else 0.0

    model = {
        "coeffs": coefficients,
        "degree": degree,
        "r2": r_squared,
        "rmse": train_rmse,
        "mse": train_mse,
        "y_hat": predicted,
        "residuals": residuals,
    }
    return model


# ── Prediction ────────────────────────────────────────────────────

def predict(model, new_features):
    """Generate predictions using a fitted model.

    Args:
        model: Model dictionary returned by fit_regression.
        new_features: ndarray of shape (n,) or (n, p) — new observations.

    Returns:
        ndarray: Predicted values of shape (n,).

    Raises:
        AssertionError: If new_features is empty.
    """
    new_features = np.asarray(new_features, dtype=float)
    assert len(new_features) > 0, "new_features must not be empty"

    design_matrix = _build_design_matrix(new_features, model["degree"])
    return design_matrix @ model["coeffs"]

# ── Cross-validation ─────────────────────────────────────────────

def cross_validate(model_degree, features, target, k=5):
    """Perform k-fold cross-validation for a given polynomial degree.

    Shuffles indices with a fixed seed (42) for reproducibility,
    splits into k folds, and reports per-fold and aggregate metrics.

    Args:
        model_degree: Polynomial degree to evaluate (1, 2, or 3).
        features: ndarray of shape (n,) or (n, p).
        target: ndarray of shape (n,).
        k: Number of folds (default 5).

    Returns:
        dict: Cross-validation results with keys:
            - fold_metrics (list[dict]): Per-fold dicts with
              'fold', 'rmse', 'r2', 'mse'.
            - mean_rmse (float): Average RMSE across folds.
            - mean_r2 (float): Average R² across folds.
            - std_rmse (float): Std. deviation of RMSE across folds.

    Raises:
        AssertionError: If k < 2, arrays have mismatched lengths,
                        or degree is invalid.
    """
    features = np.asarray(features, dtype=float)
    target = np.asarray(target, dtype=float)

    assert k >= 2, "k must be at least 2"
    assert len(features) == len(target), "features and target must have the same length"
    assert model_degree in (1, 2, 3), "model_degree must be 1, 2, or 3"

    if features.ndim == 1:
        features = features.reshape(-1, 1)

    num_samples = len(features)
    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed=42)
    rng.shuffle(indices)
    folds = np.array_split(indices, k)

    fold_metrics = []

    for fold_index in range(k):
        validation_indices = folds[fold_index]
        training_indices = np.concatenate(
            [folds[j] for j in range(k) if j != fold_index]
        )

        train_features = features[training_indices]
        train_target = target[training_indices]
        val_features = features[validation_indices]
        val_target = target[validation_indices]

        trained_model = fit_regression(train_features, train_target, degree=model_degree)
        val_predicted = predict(trained_model, val_features)

        fold_rmse = np.sqrt(mse(val_target, val_predicted))

        sum_sq_res = np.sum((val_target - val_predicted) ** 2)
        sum_sq_tot = np.sum((val_target - np.mean(val_target)) ** 2)
        fold_r2 = 1.0 - sum_sq_res / sum_sq_tot if sum_sq_tot != 0 else 0.0

        fold_metrics.append({
            "fold": fold_index + 1,
            "rmse": fold_rmse,
            "r2": fold_r2,
            "mse": mse(val_target, val_predicted),
        })

    all_rmses = [fold["rmse"] for fold in fold_metrics]
    all_r2s = [fold["r2"] for fold in fold_metrics]

    return {
        "fold_metrics": fold_metrics,
        "mean_rmse": float(np.mean(all_rmses)),
        "mean_r2": float(np.mean(all_r2s)),
        "std_rmse": float(np.std(all_rmses)),
    }


# ── Residual plots ────────────────────────────────────────────────

def plot_residuals(model, save_path=None):
    """Plot residuals vs predicted values with homoscedasticity analysis.

    Left panel: scatter of residuals with ±1σ and ±2σ bands per
    binned segment to visually assess heteroscedasticity.
    Right panel: histogram of residuals overlaid with theoretical
    normal curve.

    Args:
        model: Model dictionary returned by fit_regression.
        save_path: Optional file path to save the figure (PNG, 150 dpi).

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    predicted = model["y_hat"]
    residuals = model["residuals"]

    fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left panel: residuals vs predicted ---
    ax_scatter.scatter(
        predicted, residuals,
        alpha=0.5, edgecolors="k", linewidths=0.3,
    )
    ax_scatter.axhline(0, color="red", linestyle="--", linewidth=1)

    # Binned standard deviation bands for homoscedasticity check
    num_bins = 8
    bin_edges = np.linspace(predicted.min(), predicted.max(), num_bins + 1)
    bin_centers = []
    bin_std_devs = []

    for bin_idx in range(num_bins):
        mask = (predicted >= bin_edges[bin_idx]) & (predicted < bin_edges[bin_idx + 1])
        if mask.sum() > 2:
            center = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2
            std_dev = np.std(residuals[mask])
            bin_centers.append(center)
            bin_std_devs.append(std_dev)

    if bin_centers:
        lower_1s = [-s for s in bin_std_devs]
        upper_1s = bin_std_devs
        lower_2s = [-2 * s for s in bin_std_devs]
        upper_2s = [2 * s for s in bin_std_devs]

        ax_scatter.fill_between(
            bin_centers, lower_1s, upper_1s,
            alpha=0.15, color="orange", label="±1σ per segment",
        )
        ax_scatter.fill_between(
            bin_centers, lower_2s, upper_2s,
            alpha=0.08, color="orange", label="±2σ per segment",
        )

    ax_scatter.set_xlabel("Predicted values (ŷ)")
    ax_scatter.set_ylabel("Residuals (y − ŷ)")
    ax_scatter.set_title(f"Residuals vs Predicted — degree {model['degree']}")
    ax_scatter.legend(fontsize=8)

    # --- Right panel: residual distribution ---
    ax_hist.hist(residuals, bins=25, edgecolor="k", alpha=0.7, density=True)

    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)
    x_range = np.linspace(residuals.min(), residuals.max(), 100)
    normal_curve = (
        (1 / (residual_std * np.sqrt(2 * np.pi)))
        * np.exp(-0.5 * ((x_range - residual_mean) / residual_std) ** 2)
    )
    ax_hist.plot(x_range, normal_curve, "r-", linewidth=2, label="Theoretical normal")

    ax_hist.set_xlabel("Residuals")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Residual distribution")
    ax_hist.legend(fontsize=8)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path+"residual_plot.png", dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()
    return fig


# ── Model comparison ─────────────────────────────────────────────

def compare_models(features, target, k=5, save_path=None):
    """Compare linear, quadratic, and cubic regression models.

    Prints a summary table with training and cross-validation metrics,
    and generates a side-by-side residual plot for each degree.

    Args:
        features: ndarray of shape (n,) or (n, p).
        target: ndarray of shape (n,).
        k: Number of cross-validation folds (default 5).
        save_path: Optional file path to save the comparison figure.

    Returns:
        list[dict]: One dict per degree with keys:
            'degree', 'label', 'r2_train', 'rmse_train',
            'r2_cv', 'rmse_cv', 'std_rmse_cv', 'model', 'cv'.
    """
    features = np.asarray(features, dtype=float)
    target = np.asarray(target, dtype=float)

    assert len(features) > 0, "Dataset must not be empty"

    degree_labels = {
        1: "Linear (degree 1)",
        2: "Quadratic (degree 2)",
        3: "Cubic (degree 3)",
    }

    header = (
        f"\n{'Model':<25} {'R² train':>10} {'RMSE train':>12} "
        f"{'R² CV':>10} {'RMSE CV':>12} {'σ RMSE CV':>12}"
    )
    print(header)
    print("─" * 83)

    comparison_results = []

    for degree in (1, 2, 3):
        model = fit_regression(features, target, degree=degree)
        cv_results = cross_validate(degree, features, target, k=k)

        row = {
            "degree": degree,
            "label": degree_labels[degree],
            "r2_train": model["r2"],
            "rmse_train": model["rmse"],
            "r2_cv": cv_results["mean_r2"],
            "rmse_cv": cv_results["mean_rmse"],
            "std_rmse_cv": cv_results["std_rmse"],
            "model": model,
            "cv": cv_results,
        }
        comparison_results.append(row)

        print(
            f"{degree_labels[degree]:<25} "
            f"{model['r2']:>10.4f} {model['rmse']:>12.2f} "
            f"{cv_results['mean_r2']:>10.4f} {cv_results['mean_rmse']:>12.2f} "
            f"{cv_results['std_rmse']:>12.2f}"
        )

    # --- Side-by-side residual plots ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, row in enumerate(comparison_results):
        ax = axes[idx]
        trained_model = row["model"]

        ax.scatter(
            trained_model["y_hat"], trained_model["residuals"],
            alpha=0.5, edgecolors="k", linewidths=0.3,
        )
        ax.axhline(0, color="red", linestyle="--")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residuals")
        ax.set_title(
            f"{row['label']}\n"
            f"R²={trained_model['r2']:.3f}  RMSE={trained_model['rmse']:.1f}"
        )

    plt.suptitle("Residual comparison by polynomial degree", fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path+"compare_models_plot.png", dpi=150, bbox_inches="tight")
        print(f"\nFigure saved to {save_path}")

    plt.show()
    return comparison_results
