import pandas as pd
import pytest
import sys
import os

@pytest.fixture
def dummy_df():
	# DataFrame con columnas mínimas para pruebas
	data = {
		'hour': [1, 2, 3],
		'weekday_number': [0, 1, 2],
		'counter': [10, 20, 30]
	}
	return pd.DataFrame(data)

# Asegura que src esté en el path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import viz

def test_generate_report_exists():
	assert hasattr(viz, 'generate_report'), "viz debe tener la función generate_report"

def test_generate_report_callable():
	assert callable(viz.generate_report), "generate_report debe ser callable"

def test_generate_report_runs():
	# Debe ejecutarse sin lanzar excepción con un DataFrame dummy
	try:
		data = {'model': None, 'features': None, 'target': None, 'k': 2, 'save_path': '.', 'summary': None}
		viz.generate_report(data, _type=None)
	except Exception as e:
		pytest.fail(f"generate_report lanzó una excepción: {e}")

def test_generate_report_with_dummy_df(dummy_df):
	# Prueba que generate_report acepte un DataFrame dummy para 'regression'
	try:
		data = {
			'model': {'y_hat': [1,2,3], 'residuals': [0.1, -0.2, 0.3], 'degree': 1, 'r2': 0.9, 'rmse': 1.0},
			'features': dummy_df[['hour', 'weekday_number']].values,
			'target': dummy_df['counter'].values,
			'k': 2,
			'save_path': '.'
		}
		viz.generate_report(data, _type='regression')
	except Exception as e:
		pytest.fail(f"generate_report con dummy_df lanzó una excepción: {e}")
