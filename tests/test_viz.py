import pytest
import sys
import os

# Asegura que src esté en el path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import viz

def test_generate_report_exists():
	assert hasattr(viz, 'generate_report'), "viz debe tener la función generate_report"

def test_generate_report_callable():
	assert callable(viz.generate_report), "generate_report debe ser callable"

def test_generate_report_runs():
	# Debe ejecutarse sin lanzar excepción (aunque no haga nada)
	try:
		viz.generate_report()
	except Exception as e:
		pytest.fail(f"generate_report lanzó una excepción: {e}")
