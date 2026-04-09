# ─────────────────────────────────────────────────────────────────────────────
# C4 — Orquestador Google Colab
# ─────────────────────────────────────────────────────────────────────────────

# 1. Limpiar y clonar
!rm -rf final-project-TC6039-team-1
!git clone -b C4-Machine-Learning https://github.com/Acar3ds/final-project-TC6039-team-1.git

# 2. Agregar src al path del kernel
import sys
sys.path.insert(0, "final-project-TC6039-team-1/src")

# 3. Activar matplotlib para Colab
%matplotlib inline

# 4. Correr pipeline C4 (las gráficas aparecen aquí)
%cd final-project-TC6039-team-1
%run src/ml_models.py

# 5. Correr tests
!PYTHONPATH=src pytest tests/test_ml_models.py -v
