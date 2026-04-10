"""
Módulo C5 - Deep Learning (dl_model.py)
Contiene la implementación de la red neuronal (MLP) en PyTorch para datos tabulares,
incluyendo Batch Normalization, Dropout, transformaciones cíclicas temporales,
y gráficas comparativas de resultados.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List

class TrafficPredictionMLP(nn.Module):
    """
    Multilayer Perceptron (MLP) para predecir el flujo vehicular a partir de datos tabulares.
    """
    def __init__(self, input_features: int, hidden_neurons: int = 64, dropout_prob: float = 0.2):
        """
        Inicializa la red neuronal con las capas requeridas.

        Args:
            input_features (int): Número de características (features) de entrada.
            hidden_neurons (int, opcional): Número de neuronas en la capa oculta principal. Por defecto 64.
            dropout_prob (float, opcional): Probabilidad de Dropout para regularización. Por defecto 0.2.
        """
        super(TrafficPredictionMLP, self).__init__()

        assert isinstance(input_features, int) and input_features > 0, "PRECONDICIÓN: input_features debe ser entero > 0"
        assert isinstance(hidden_neurons, int) and hidden_neurons > 0, "PRECONDICIÓN: hidden_neurons debe ser entero > 0"
        assert 0.0 <= dropout_prob < 1.0, "PRECONDICIÓN: dropout_prob debe estar entre 0 y 1"

        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_neurons),
            nn.BatchNorm1d(hidden_neurons),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_neurons, hidden_neurons // 2),
            nn.BatchNorm1d(hidden_neurons // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_neurons // 2, 1)
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """ Paso hacia adelante de la red neuronal. """
        assert isinstance(input_tensor, torch.Tensor), "PRECONDICIÓN: La entrada debe ser un tensor de PyTorch."
        return self.network(input_tensor)


def train_deep_learning_model(
    neural_model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    total_epochs: int = 50,
    learning_rate: float = 0.001
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """ Ciclo de entrenamiento del modelo de Deep Learning. """
    assert isinstance(total_epochs, int) and total_epochs > 0, "PRECONDICIÓN: total_epochs debe ser mayor a 0."
    assert isinstance(learning_rate, float) and learning_rate > 0.0, "PRECONDICIÓN: learning_rate debe ser positivo."
    assert train_loader is not None and validation_loader is not None, "PRECONDICIÓN: Los cargadores no pueden ser nulos."

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(neural_model.parameters(), lr=learning_rate)

    training_history = {
        'train_loss': [], 'val_loss': [],
        'train_mae': [], 'val_mae': []
    }

    for epoch in range(total_epochs):
        # --- Fase de Entrenamiento ---
        neural_model.train()
        cumulative_train_loss = 0.0
        cumulative_train_mae = 0.0

        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            predictions = neural_model(batch_features)
            loss = loss_function(predictions, batch_targets)
            loss.backward()
            optimizer.step()

            cumulative_train_loss += loss.item() * batch_features.size(0)
            cumulative_train_mae += torch.abs(predictions - batch_targets).sum().item()

        epoch_train_loss = np.sqrt(cumulative_train_loss / len(train_loader.dataset))
        epoch_train_mae = cumulative_train_mae / len(train_loader.dataset)

        # --- Fase de Validación ---
        neural_model.eval()
        cumulative_val_loss = 0.0
        cumulative_val_mae = 0.0

        with torch.no_grad():
            for val_features, val_targets in validation_loader:
                val_predictions = neural_model(val_features)
                val_loss = loss_function(val_predictions, val_targets)

                cumulative_val_loss += val_loss.item() * val_features.size(0)
                cumulative_val_mae += torch.abs(val_predictions - val_targets).sum().item()

        epoch_val_loss = np.sqrt(cumulative_val_loss / len(validation_loader.dataset))
        epoch_val_mae = cumulative_val_mae / len(validation_loader.dataset)

        # Registrar métricas
        training_history['train_loss'].append(epoch_train_loss)
        training_history['val_loss'].append(epoch_val_loss)
        training_history['train_mae'].append(epoch_train_mae)
        training_history['val_mae'].append(epoch_val_mae)

    return neural_model, training_history


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


# ==============================================================================
# EJECUCIÓN DEL PIPELINE C5
# ==============================================================================
def pipeline_execution(df):
    print("Cargando y preparando el set de datos...")

    # 1. Cargar datos
    dataframe = df.copy()

    # Agrupar vehículos por fecha, hora y tipo de día para saber cuántos pasaron por hora
    traffic_summary = dataframe.groupby(['date', 'hour', 'weekday_number']).size().reset_index(name='traffic_volume')

    # Ordenar cronológicamente para que la gráfica comparativa tenga sentido temporal
    traffic_summary = traffic_summary.sort_values(by=['date', 'hour']).reset_index(drop=True)

    # 2. Preparar Tensores para PyTorch e Ingeniería de Características (Transformaciones Cíclicas)
    # Para la hora (0 a 23 -> divisor 24)
    traffic_summary['hour_sin'] = np.sin(2 * np.pi * traffic_summary['hour'] / 24)
    traffic_summary['hour_cos'] = np.cos(2 * np.pi * traffic_summary['hour'] / 24)

    # Para el día de la semana (0 a 6 -> divisor 7)
    traffic_summary['weekday_sin'] = np.sin(2 * np.pi * traffic_summary['weekday_number'] / 7)
    traffic_summary['weekday_cos'] = np.cos(2 * np.pi * traffic_summary['weekday_number'] / 7)

    # Actualizamos el vector de características con las 4 variables cíclicas
    features = traffic_summary[['hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos']].values.astype(np.float32)
    targets = traffic_summary['traffic_volume'].values.astype(np.float32).reshape(-1, 1)

    features_tensor = torch.tensor(features)
    targets_tensor = torch.tensor(targets)

    # 3. Creación de DataLoaders (Partición 80% Entreno / 20% Validación)
    full_dataset = TensorDataset(features_tensor, targets_tensor)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 4. Inicializar y Entrenar la Red
    print("Iniciando entrenamiento del MLP...")
    # input_features ajustado a 4 para recibir hour_sin, hour_cos, weekday_sin, weekday_cos
    model_c5 = TrafficPredictionMLP(input_features=4, hidden_neurons=64, dropout_prob=0.1)
    trained_model, history = train_deep_learning_model(model_c5, train_dl, val_dl, total_epochs=100, learning_rate=0.01)

    # 5. Visualizar Curvas de Aprendizaje
    print("Generando curvas de validación (Cierra la ventana de la gráfica para continuar)...")
    plot_learning_curves(history)

    # 6. Generar Gráfica Comparativa (Real vs Predicho)
    print("\nGenerando gráfica comparativa temporal...")
    trained_model.eval()

    # Tomar las últimas 100 secuencias para visualización
    sequence_length = min(100, len(features_tensor))
    recent_features = features_tensor[-sequence_length:]
    actual_traffic_volumes = targets_tensor[-sequence_length:].numpy().flatten()

    with torch.no_grad():
        predicted_traffic_volumes = trained_model(recent_features).numpy().flatten()
        # Evitar predicciones físicas imposibles (negativas)
        predicted_traffic_volumes = np.maximum(0, predicted_traffic_volumes)

    # Llamar a la función que grafica la comparativa
    plot_actual_vs_predicted(actual_traffic_volumes, predicted_traffic_volumes)

    print("Ejecución del módulo C5 completada exitosamente.")

    return history, actual_traffic_volumes, predicted_traffic_volumes

if __name__ == '__main__':
    pipeline_execution()
