import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.dl_model import TrafficPredictionMLP, train_deep_learning_model

# ── Fixtures (Datos de Prueba Simulados) ───────────────────────────

@pytest.fixture
def dummy_dataloaders():
    """Genera DataLoaders de entrenamiento y validación sintéticos para las pruebas."""
    # 100 muestras, 2 características (ej. hora y día)
    features = torch.rand((100, 2), dtype=torch.float32) * 10.0
    # Objetivo (volumen): Relación lineal simple (feature1 * 2 + feature2) + ruido
    targets = (features[:, 0] * 2 + features[:, 1]).view(-1, 1) + torch.randn((100, 1))

    dataset = TensorDataset(features, targets)
    
    # 80 entreno / 20 validación
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [80, 20])
    
    train_dl = DataLoader(train_dataset, batch_size=16)
    val_dl = DataLoader(val_dataset, batch_size=16)
    
    return train_dl, val_dl


# ── Tests para TrafficPredictionMLP ────────────────────────────────

class TestTrafficPredictionMLP:
    
    def test_mlp_initialization(self):
        """Verifica que el modelo se inicialice correctamente con las capas adecuadas."""
        model = TrafficPredictionMLP(input_features=2, hidden_neurons=32, dropout_prob=0.1)
        assert isinstance(model, nn.Module)
        assert isinstance(model.network, nn.Sequential)

    def test_forward_pass_output_shape(self):
        """Verifica que un tensor de entrada genere una predicción del tamaño correcto."""
        model = TrafficPredictionMLP(input_features=2)
        # Batch simulado de 5 registros con 2 features
        dummy_input = torch.randn(5, 2)
        
        model.eval() # Modo evaluación para apagar el Dropout
        with torch.no_grad():
            output = model(dummy_input)
            
        # La salida debe ser de tamaño (5 predicciones, 1 valor)
        assert output.shape == (5, 1)

    def test_invalid_input_features_raises(self):
        """Verifica que se lance AssertionError si se piden <=0 features (Wilson §5a)."""
        with pytest.raises(AssertionError, match="entero > 0"):
            TrafficPredictionMLP(input_features=0)

    def test_invalid_dropout_raises(self):
        """Verifica que se lance AssertionError si el dropout no es un % válido."""
        with pytest.raises(AssertionError, match="entre 0 y 1"):
            TrafficPredictionMLP(input_features=2, dropout_prob=1.5)


# ── Tests para train_deep_learning_model ──────────────────────────

class TestTrainDeepLearningModel:

    def test_training_loop_executes_and_returns_history(self, dummy_dataloaders):
        """Comprueba que el ciclo de entrenamiento no falle y retorne las métricas."""
        train_dl, val_dl = dummy_dataloaders
        
        model = TrafficPredictionMLP(input_features=2)
        trained_model, history = train_deep_learning_model(
            model, train_dl, val_dl, total_epochs=2, learning_rate=0.01
        )
        
        # Verificar tipos de retorno
        assert isinstance(trained_model, nn.Module)
        assert isinstance(history, dict)
        
        # Verificar que el diccionario contenga las 4 listas
        expected_keys = ['train_loss', 'val_loss', 'train_mae', 'val_mae']
        for key in expected_keys:
            assert key in history
            assert len(history[key]) == 2 # 2 épocas = 2 registros

    def test_invalid_epochs_raises(self, dummy_dataloaders):
        """Verifica aserción si se pide entrenar con épocas negativas."""
        train_dl, val_dl = dummy_dataloaders
        model = TrafficPredictionMLP(input_features=2)
        
        with pytest.raises(AssertionError, match="mayor a 0"):
            train_deep_learning_model(model, train_dl, val_dl, total_epochs=0)

    def test_invalid_learning_rate_raises(self, dummy_dataloaders):
        """Verifica aserción si la tasa de aprendizaje es negativa."""
        train_dl, val_dl = dummy_dataloaders
        model = TrafficPredictionMLP(input_features=2)
        
        with pytest.raises(AssertionError, match="positivo"):
            train_deep_learning_model(model, train_dl, val_dl, total_epochs=1, learning_rate=-0.05)
