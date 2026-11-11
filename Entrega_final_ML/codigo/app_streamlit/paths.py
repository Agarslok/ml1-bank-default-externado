"""Rutas a artefactos del modelo congelado (escenario B)."""
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
CODIGO_DIR = APP_DIR.parent
ROOT = CODIGO_DIR.parent

CANDIDATOS_PIPELINE = [
    CODIGO_DIR / "salidas_tercera" / "pipeline_logistica_B.joblib",
    ROOT / "documentacion" / "tercera_entrega" / "entregables_para_informe" / "pipeline_logistica_B.joblib",
]

CANDIDATOS_UMBRALES = [
    CODIGO_DIR / "salidas_tercera" / "umbrales_recomendados.json",
    ROOT / "documentacion" / "tercera_entrega" / "entregables_para_informe" / "umbrales_recomendados.json",
]

CANDIDATOS_DATOS = [
    ROOT / "datos" / "database.csv",
    CODIGO_DIR / "datos" / "database.csv",
]

COLS_B = [
    "edad",
    "dias_lab",
    "exp_sf",
    "nivel_ahorro",
    "ingreso",
    "linea_sf",
    "deuda_sf",
    "score",
    "vivienda",
    "zona",
    "nivel_educ",
]


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError("No se encontró ninguno de: " + ", ".join(str(p) for p in paths))
