"""
Módulo C5 - Deep Learning (dl_model.py)
Contiene la implementación de la red neuronal (MLP) en PyTorch para datos tabulares,
incluyendo Batch Normalization, Dropout, y el ciclo de entrenamiento.
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
    """Multilayer Perceptron (MLP) para predecir el flujo vehicular."""
    
    def __init__(self, input_features: int, hidden_neurons: int = 64, dropout_prob: float = 0.2):
        """
        Inicializa la red neuronal.

        Args:
            input_features (int): Número de características (features) de entrada.
            hidden_neurons (int): Número de neuronas en la capa oculta principal.
            dropout_prob (float): Probabilidad de Dropout.
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
        """Paso hacia adelante de la red."""
        assert isinstance(input_tensor, torch.Tensor), "PRECONDICIÓN: La entrada debe ser un tensor de PyTorch."
        return self.network(input_tensor)


def train_deep_learning_model(
    neural_model: nn.Module, 
    train_loader: DataLoader, 
    validation_loader: DataLoader, 
    total_epochs: int = 50, 
    learning_rate: float = 0.001
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Ciclo de entrenamiento del modelo de Deep Learning."""
    
    assert isinstance(total_epochs, int) and total_epochs > 0, "PRECONDICIÓN: total_epochs debe ser mayor a 0."
    assert isinstance(learning_rate, float) and learning_rate > 0.0, "PRECONDICIÓN: learning_rate debe ser positivo."

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(neural_model.parameters(), lr=learning_rate)
    
    training_history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}

    for epoch in range(total_epochs):
        # Fase Entrenamiento
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
            
        epoch_train_loss = cumulative_train_loss / len(train_loader.dataset)
        epoch_train_mae = cumulative_train_mae / len(train_loader.dataset)
        
        # Fase Validación
        neural_model.eval()
        cumulative_val_loss = 0.0
        cumulative_val_mae = 0.0
        
        with torch.no_grad():
            for val_features, val_targets in validation_loader:
                val_predictions = neural_model(val_features)
                val_loss = loss_function(val_predictions, val_targets)
                
                cumulative_val_loss += val_loss.item() * val_features.size(0)
                cumulative_val_mae += torch.abs(val_predictions - val_targets).sum().item()
                
        epoch_val_loss = cumulative_val_loss / len(validation_loader.dataset)
        epoch_val_mae = cumulative_val_mae / len(validation_loader.dataset)
        
        training_history['train_loss'].append(epoch_train_loss)
        training_history['val_loss'].append(epoch_val_loss)
        training_history['train_mae'].append(epoch_train_mae)
        training_history['val_mae'].append(epoch_val_mae)

    return neural_model, training_history


def plot_learning_curves(metrics_history: Dict[str, List[float]]) -> None:
    """Genera gráficas de pérdida y métrica vs Época."""
    assert isinstance(metrics_history, dict), "PRECONDICIÓN: El historial debe ser un diccionario."
    
    epochs_axis = range(1, len(metrics_history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs_axis, metrics_history['train_loss'], label='Entrenamiento (MSE)')
    ax1.plot(epochs_axis, metrics_history['val_loss'], label='Validación (MSE)')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Error Cuadrático Medio')
    ax1.set_title('Curva de Pérdida vs Época')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs_axis, metrics_history['train_mae'], label='Entrenamiento (MAE)')
    ax2.plot(epochs_axis, metrics_history['val_mae'], label='Validación (MAE)')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Error Absoluto Medio (Vehículos)')
    ax2.set_title('Métrica de Evaluación vs Época')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def plot_actual_vs_predicted(actual_values: np.ndarray, predicted_values: np.ndarray) -> None:
    """Genera gráfica comparando valores reales vs predichos."""
    assert len(actual_values) == len(predicted_values), "PRECONDICIÓN: Los arreglos deben tener el mismo tamaño."
    
    time_sequence = range(len(actual_values))
    plt.figure(figsize=(12, 6))
    plt.plot(time_sequence, actual_values, label='Tráfico Real', color='blue', marker='o', markersize=4)
    plt.plot(time_sequence, predicted_values, label='Tráfico Predicho', color='red', linestyle='--')
    plt.xlabel('Secuencia Temporal')
    plt.ylabel('Volumen Vehicular')
    plt.title('Comparativa: Real vs Predicho (C5)')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Si alguien corre este archivo directamente, muestra un mensaje de advertencia
    print("Este módulo contiene las funciones base de C5. Ejecuta main.py para correr el pipeline completo.")
