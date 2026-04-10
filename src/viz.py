# Importar funciones de módulos externos (ajustar según nombres reales de funciones)
from pathlib import Path
from . import data_loader, regression, optimizer, ml_models, dl_model

def generate_report(data, _type=None):
    """
    Genera el reporte correspondiente según el parámetro _type.
    _type puede ser: 'data_loader', 'regression', 'optimizer', 'ml_models', 'dl_model'.
    """

    if _type == 'data_loader':
        plot_eda_summary(data["summary"])
        return

    if _type == 'regression':
        regression.plot_residuals(data["model"], data["save_path"])
        regression.compare_models(data["features"], data["target"], k=data["k"], save_path=data["save_path"])
        return

    if _type == 'optimizer':
        print("Plots from optimizer.py already saved.")
        return

    if _type == 'ml_models':
        print("Plots from ml_models.py already saved.")
        return

    if _type == 'dl_model':
        # Aquí puedes agregar la lógica para el reporte de deep learning si es necesario
        return

    print("Tipo de reporte no reconocido. Usa uno de: data_loader, regression, optimizer, ml_models, dl_model.")


def plot_eda_summary(summary):
    # Save figures and statistics to the analysis directory
    analysis_dir = Path("report/figures")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Save descriptive statistics to JSON
    stats_file = analysis_dir / "descriptive_statistics.json"
    try:
        with open(stats_file, 'w') as f:
            json.dump(summary['describe'], f, indent=2, default=str)
        print(f"Saved descriptive statistics to {stats_file}")
    except Exception as e:
        print(f"Failed to save descriptive statistics: {e}")

    for name, fig in summary['plots'].items():
        try:
            fig.savefig(analysis_dir / f"{name}.png")
            print(f"Saved plot to {analysis_dir / f'{name}.png'}")
        except Exception:
            # If the object is not a Figure, attempt to convert or skip
            try:
                # some seaborn objects may expose .fig
                fig.fig.savefig(analysis_dir / f"{name}.png")
                print(f"Saved plot to {analysis_dir / f'{name}.png'}")
            except Exception:
                print(f"Could not save plot {name}")

def plot_learning_curves(metrics_history: Dict[str, List[float]]) -> None:
    """ Genera las gráficas de pérdida (Loss) y métrica (MAE) vs Época. """
    assert isinstance(metrics_history, dict), "PRECONDICIÓN: El historial debe ser un diccionario."

    epochs_axis = range(1, len(metrics_history['train_loss']) + 1)
    plt.figure(figsize=(14, 5))

    # Gráfica 1: Pérdida (MSE)
    plt.subplot(1, 2, 1)
    plt.plot(epochs_axis, metrics_history['train_loss'], label='Trainning (RMSE)')
    plt.plot(epochs_axis, metrics_history['val_loss'], label='Validation (RMSE)')
    plt.xlabel('Epoc')
    plt.ylabel('Root Mean Squared Error')
    plt.title('Loss-Curve vs Epoc')
    plt.legend()
    plt.grid(True)

    # Gráfica 2: MAE
    plt.subplot(1, 2, 2)
    plt.plot(epochs_axis, metrics_history['train_mae'], label='Trainning (MAE)')
    plt.plot(epochs_axis, metrics_history['val_mae'], label='Validation (MAE)')
    plt.xlabel('Epoc')
    plt.ylabel('Mean Absolute Error (Vehicles)')
    plt.title('MAE vs Epoc')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted(actual_values: np.ndarray, predicted_values: np.ndarray) -> None:
    """ Genera una gráfica de líneas comparando los valores reales vs los predichos en una secuencia temporal. """
    assert len(actual_values) == len(predicted_values), "PRECONDICIÓN: Los arreglos deben tener el mismo tamaño."

    time_sequence = range(len(actual_values))

    plt.figure(figsize=(12, 6))
    plt.plot(time_sequence, actual_values, label='Vehicular Traffic', color='blue', marker='o', markersize=4, linestyle='-')
    plt.plot(time_sequence, predicted_values, label='Predicted Traffic', color='red', marker='x', markersize=4, linestyle='--')

    plt.xlabel('Hourly Evaluated Sequence (Temporal Index)')
    plt.ylabel('Vehicular Volume (vehicles per hour)')
    plt.title('Real vs. Predicted (C5)')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()

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
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()
    return fig

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
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved to {save_path}")

    plt.show()
    return comparison_results

