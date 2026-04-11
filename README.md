
# Vehicular Traffic Prediction System

Este sistema implementa un pipeline completo para la predicción y análisis de tráfico vehicular, incluyendo carga de datos, análisis exploratorio, regresión, optimización, modelos de machine learning y deep learning, así como visualización y generación de reportes.

## Instalación

1. Clona este repositorio y navega a la carpeta principal del proyecto.
2. (Recomendado) Crea y activa un entorno virtual (venv) para aislar las dependencias:
	```bash
	python -m venv venv
	# En Windows:
	venv\Scripts\activate
	# En Mac/Linux:
	source venv/bin/activate
	```
3. Instala las dependencias necesarias:
	```bash
	pip install -r requirements.txt
	```

## Ejecución por componentes

Puedes ejecutar el sistema completo o cada componente de forma independiente:

- **main.py**: Ejecuta el pipeline completo, generando reportes y visualizaciones.
	```bash
	python main.py
	```
- **src/data_loader.py**: Funciones para cargar y limpiar datos desde archivos JSON o CSV.
- **src/regression.py**: Ajuste y evaluación de modelos de regresión polinomial.
- **src/optimizer.py**: Optimización de parámetros del modelo usando descenso de gradiente y BFGS.
- **src/ml_models.py**: Implementación y evaluación de modelos de machine learning clásicos.
- **src/dl_model.py**: Implementación y entrenamiento de modelos de deep learning (MLP).
- **src/viz.py**: Funciones de visualización y generación de reportes automáticos.

## Estructura de carpetas

- **data/raw/**: Archivos de datos originales en formato JSON.
- **data/processed/**: Datos procesados y listos para análisis (ej. df.csv).
- **data/analysis/**: Estadísticas descriptivas y resultados intermedios.
- **report/figures/**: Gráficas y figuras generadas automáticamente.
- **notebooks/**: Notebooks de exploración y análisis inicial.
- **src/**: Código fuente de todos los módulos principales.
- **tests/**: Pruebas unitarias para cada componente.

## Descripción breve de archivos principales

- **main.py**: Orquestador principal del pipeline. Ejecuta la carga de datos, análisis, modelos y reportes.
- **src/data_loader.py**: Funciones para cargar, limpiar y resumir datos.
- **src/regression.py**: Ajuste de modelos de regresión y análisis de residuos.
- **src/optimizer.py**: Optimización de modelos mediante descenso de gradiente y BFGS.
- **src/ml_models.py**: Modelos de machine learning (Random Forest, SVM, etc.).
- **src/dl_model.py**: Modelos de deep learning (MLP) para predicción de tráfico.
- **src/viz.py**: Visualización y generación de reportes automáticos.
- **tests/test_*.py**: Pruebas unitarias para cada módulo.

## Pruebas

Para ejecutar todas las pruebas unitarias:
```bash
pytest
```

---
Este sistema está diseñado para ser modular y extensible, permitiendo analizar y predecir tráfico vehicular usando diferentes enfoques de modelado y visualización.

