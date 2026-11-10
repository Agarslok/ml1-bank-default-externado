"""Genera codigo/tercera_entrega_modelamiento_final.ipynb"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "tercera_entrega_modelamiento_final.ipynb"
CELLS = []


def md(t):
    CELLS.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in t.strip().split("\n")],
        }
    )


def code(t):
    CELLS.append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [line + "\n" for line in t.strip().split("\n")],
        }
    )


md(
    """# Entrega final — Modelo congelado (regresión logística, escenario B)

**Machine Learning I · Universidad Externado de Colombia**

Notebook de la Fase 3. Las decisiones fijas están en `documentacion/tercera_entrega/contrato_decisiones.md`.

**Qué hace este cuaderno (y qué no):**
- Reproduce el pipeline de la 2.ª entrega, entrena el **modelo final** y guarda `joblib`.
- Calcula **umbrales por costo simbólico** (FN:FP = 2:1 y 5:1) solo con predicciones **OOF en entrenamiento**.
- Exporta tablas CV y figuras para el informe.
- El conjunto de **prueba** aparece solo como **lectura** (métricas de la 2.ª entrega y reporte con umbrales ya fijados). **No** se usa test para elegir modelo ni umbral.

**Lectura de scores:** `predict_proba` es un **índice de riesgo relativo** para ordenar clientes en una muestra similar. No es la probabilidad real de mora en el mercado peruano (prevalencia ~71 % en esta base vs 4–11 % SBS)."""
)

code(
    """from pathlib import Path
import json
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd()
if (ROOT / "tercera_entrega_lib.py").exists():
    COD = ROOT
elif (ROOT / "codigo" / "tercera_entrega_lib.py").exists():
    COD = ROOT / "codigo"
else:
    COD = Path(".")

import sys
sys.path.insert(0, str(COD))
from tercera_entrega_lib import (
    ESCENARIOS, ESC_PRINCIPAL_KEY, SEED, run_entrega_final, cargar_datos_y_particion,
)

OUT_DIR = COD / "salidas_tercera"
INFORME_DIR = ROOT.parent / "documentacion" / "tercera_entrega" / "entregables_para_informe" if (ROOT.parent / "documentacion").exists() else COD / "salidas_tercera"
print("Código:", COD.resolve())
print("Contrato: documentacion/tercera_entrega/contrato_decisiones.md")"""
)

md("## 1. Datos y partición (igual que 2.ª entrega)")
code(
    """_, df, X_tr, X_te, y_tr, y_te = cargar_datos_y_particion()
cols_b = ESCENARIOS[ESC_PRINCIPAL_KEY]
print("Clientes únicos:", len(df), "| Train:", len(y_tr), "| Test:", len(y_te))
print("Escenario B:", len(cols_b), "variables")
print(cols_b)"""
)

md(
    """## 2. Validación cruzada (selección histórica)

Comparación congelada de logística, LDA, KNN y Naive Bayes. La decisión de modelo final se documenta con esta CV, no con nuevos experimentos en test."""
)
code(
    """info = run_entrega_final(OUT_DIR)
tabla_cv = pd.read_csv(OUT_DIR / "tabla_cv_modelo_final.csv")
tabla_esc = pd.read_csv(OUT_DIR / "tabla_cv_escenarios_abc.csv")
display(tabla_cv)
display(tabla_esc)
print("k KNN (referencia):", info)"""
)

md(
    """## 3. Umbrales: Youden y costos FN/FP (solo OOF en train)

Costo simbólico: \\(C = r \\cdot FN + FP\\) con \\(r =\\) costo(FN)/costo(FP). Los umbrales **no** se optimizan en test."""
)
code(
    """umb = json.loads((OUT_DIR / "umbrales_recomendados.json").read_text(encoding="utf-8"))
dest = pd.read_csv(OUT_DIR / "tabla_umbrales_destacados_oof.csv")
display(dest.round(3))
print(json.dumps(umb, indent=2, ensure_ascii=False))
display(Image(filename=str(OUT_DIR / "fig_costo_vs_umbral.png")))"""
)

md(
    """## 4. Modelo final serializado

Pipeline completo ajustado en **todo el entrenamiento** (escenario B, sin balanceo). Mismo objeto usará Streamlit."""
)
code(
    """print("Archivo:", OUT_DIR / "pipeline_logistica_B.joblib")
pd.read_csv(OUT_DIR / "tabla_configuracion_definitiva.csv")"""
)

md(
    """## 5. Conjunto de prueba — solo lectura

- `tabla_test_cierre_segunda_entrega.csv` copia la evaluación de la 2.ª entrega.
- `tabla_test_verificacion_solo_lectura.csv` comprueba que, con el mismo split y semilla, el AUC coincide.
- `tabla_test_umbrales_solo_lectura.csv` aplica umbrales **ya elegidos en OOF** (reporte, no optimización)."""
)
code(
    """display(pd.read_csv(OUT_DIR / "tabla_test_cierre_segunda_entrega.csv"))
display(pd.read_csv(OUT_DIR / "tabla_test_verificacion_solo_lectura.csv"))
display(pd.read_csv(OUT_DIR / "tabla_test_umbrales_solo_lectura.csv").round(3))"""
)

md(
    """## 6. Copia de entregables para informe y presentación (Isabella)

Se replica la carpeta `entregables_para_informe` al ejecutar `python tercera_entrega_lib.py` desde terminal."""
)
code(
    """if INFORME_DIR.exists():
    run_entrega_final(INFORME_DIR)
    print("Listo:", INFORME_DIR.resolve())"""
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": CELLS,
}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Escrito", OUT)
