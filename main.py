# Importaciones necesarias
import matplotlib.pyplot as plt
import pandas as pd

from src import data_loader, regression, optimizer, ml_models, dl_model, viz

figures_path = r"../report/figures/"
print("\n--- Running Data Loader (data_loader.py) ---\n")

if hasattr(data_loader, 'load_data') and hasattr(data_loader, 'clean_data'):
    #df = data_loader.load_data()
    df = data_loader.load_df_csv()
    df = data_loader.clean_data(df)
    print(df.head())

    summary = data_loader.eda_summary(df)

    data = {
        "summary": summary
    }

    viz.generate_report(data, _type="data_loader")
else:
    print("The function load_data is not available in data_loader.py")

print("\n--- Regression analysis with regression.py ---\n")
try:
    # Select features and target columns (adjust as needed)
    features = df[["hour", "weekday_number"]].values
    target = df["counter"].values
    # Fit polynomial regression (degree 1)
    model = regression.fit_regression(features, target, degree=1)
    print(f"Regression R^2: {model['r2']:.4f}, RMSE: {model['rmse']:.2f}")

    data = {
        "model": model,
        "features": features,
        "target": target,
        "k": 5,
        "save_path": figures_path
    }

    viz.generate_report(data, _type="regression")
except Exception as e:
    print(f"Error in regression analysis: {e}")

if hasattr(optimizer, 'pipeline_execution'):
    optimal_coeffs, history, sensitivity = optimizer.optimize_parameters(df)

    data = {
        "optimal_coeffs": optimal_coeffs,
        "history": history,
        "sensitivity": sensitivity
    }

    viz.generate_report(data, _type="optimizer")
else:
    print("The function pipeline_execution is not available in optimizer.py")

# Run the Machine Learning pipeline (C4)
print("\n--- Running Machine Learning pipeline (ml_models.py) ---\n")
if hasattr(ml_models, 'pipeline_execution'):
    metrics_table = ml_models.pipeline_execution(df)
    data = {
        "metrics_table": metrics_table
    }
    viz.generate_report(data, _type="ml_models")
else:
    print("The function pipeline_execution is not available in ml_models.py")

# Run the Deep Learning pipeline (C5)
print("\n--- Running Deep Learning pipeline (dl_model.py) ---\n")
if hasattr(dl_model, 'pipeline_execution'):
    history, actual_traffic_volumes, predicted_traffic_volumes = dl_model.pipeline_execution(df)

    data = {
        "history": history,
        "actual_traffic_volumes": actual_traffic_volumes,
        "predicted_traffic_volumes": predicted_traffic_volumes
    }
    viz.generate_report(data, _type="dl_model")
else:
    print("The function pipeline_execution is not available in dl_model.py")
