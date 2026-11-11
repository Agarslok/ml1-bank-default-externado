"""Genera tercera_entrega_modelamiento_final.ipynb autocontenido (sin importar .py)."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "tercera_entrega_modelamiento_final.ipynb"
CELLS: list[dict] = []


def md(text: str) -> None:
    CELLS.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")],
        }
    )


def code(text: str) -> None:
    CELLS.append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")],
        }
    )


md(
    """# Entrega final — Modelo congelado (regresión logística, escenario B)

Machine Learning I · Universidad Externado de Colombia

Este cuaderno corre solo, sin necesitar otros archivos.

**Qué hace este cuaderno**
1. Carga `database.csv`, deduplica y parte train/prueba (80/20, semilla 42).
2. Repite la CV de la 2.ª entrega (logística, LDA, KNN, Naive Bayes; escenarios A/B/C).
3. Calcula probabilidades OOF (validación cruzada en entrenamiento) y fija umbrales (0,5, Youden, costos con r = 1, 2, 5).
4. Entrena el pipeline final escenario B y guarda `pipeline_logistica_B.joblib`.
5. Reporta la prueba en modo solo lectura (sin optimizar umbrales ahí).

**Qué no hace**
- No usa el conjunto de prueba para elegir modelo ni umbral.
- No reabre QDA, Ridge ni Lasso.

Colab: suba `database.csv` a `/content/` o `/content/datos/`. En local, déjelo en `datos/database.csv` respecto a la raíz del paquete (`Entrega_final_ML/`).

Ojo: `predict_proba[:, 1]` aquí solo sirve para ordenar clientes, no es la probabilidad real de mora en Perú."""
)

code(
    """# Imports
from pathlib import Path
import json
import shutil

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler"""
)

code(
    """# Rutas

def find_project_dirs():
    candidatos_base = [Path.cwd(), *Path.cwd().parents]
    for base in candidatos_base:
        if (base / "datos" / "database.csv").exists():
            codigo = base / "codigo" if (base / "codigo").is_dir() else base
            return base, codigo
    for p in (
        Path("/content"),
        Path.cwd(),
    ):
        for rel in ("database.csv", "datos/database.csv"):
            if (p / rel).exists():
                proyecto = p if rel == "datos/database.csv" else p.parent
                if not (proyecto / "datos" / "database.csv").exists() and (p / "database.csv").exists():
                    proyecto = p
                codigo = proyecto / "codigo" if (proyecto / "codigo").is_dir() else proyecto
                return proyecto, codigo
    raise FileNotFoundError(
        "No encontré database.csv. En Colab: Files → subir a /content. "
        "En local: datos/database.csv en la raíz del proyecto."
    )


PROYECTO_DIR, CODIGO_DIR = find_project_dirs()
OUT_DIR = CODIGO_DIR / "salidas_tercera"
_candidatos_informe = [
    PROYECTO_DIR / "documentacion" / "entregables_para_informe",
    PROYECTO_DIR / "documentacion" / "tercera_entrega" / "entregables_para_informe",
]
INFORME_DIR = next((c for c in _candidatos_informe if c.parent.exists()), _candidatos_informe[0])
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Proyecto:", PROYECTO_DIR.resolve())
print("Salidas:", OUT_DIR.resolve())
print("Informe (figuras):", INFORME_DIR.resolve() if INFORME_DIR.parent.exists() else "(opcional)")"""
)

code(
    """# Constantes
SEED = 42
NUM_ALL = [
    "edad", "dias_lab", "exp_sf", "nivel_ahorro", "ingreso", "linea_sf",
    "deuda_sf", "score", "atraso", "clasif_sbs",
]
CAT_ALL = ["vivienda", "zona", "nivel_educ"]
ESC_PRINCIPAL_KEY = "B. Principal (sin atraso ni clasif_sbs)"
ESCENARIOS = {
    "A. Completo (13 variables)": [c for c in NUM_ALL + CAT_ALL],
    ESC_PRINCIPAL_KEY: [c for c in NUM_ALL + CAT_ALL if c not in ("atraso", "clasif_sbs")],
    "C. Conservador (sin atraso, clasif_sbs ni score)": [
        c for c in NUM_ALL + CAT_ALL if c not in ("atraso", "clasif_sbs", "score")
    ],
}
K_LISTA = [5, 11, 21, 31, 51, 101, 151, 201]
MODELO_FINAL = LogisticRegression(C=1e6, max_iter=5000)
cols_b = ESCENARIOS[ESC_PRINCIPAL_KEY]
print("Modelo final: logística | Escenario B:", cols_b)"""
)

md(
    """## 2. Funciones auxiliares (pipeline, métricas, umbrales)

Mismo diseño que la 2.ª entrega: imputación mediana + indicador, escalado, one-hot con categorías infrecuentes, sin balanceo en el modelo final."""
)

code(
    """def resolve_data_path():
    for base in [PROYECTO_DIR, Path.cwd(), *Path.cwd().parents]:
        p = base / "datos" / "database.csv"
        if p.exists():
            return p
    for p in (
        Path("datos/database.csv"),
        Path("../datos/database.csv"),
        Path("../../datos/database.csv"),
        Path("/content/database.csv"),
        Path("/content/datos/database.csv"),
    ):
        if p.exists():
            return p
    raise FileNotFoundError("database.csv no encontrado")


def construir_pipeline(cols, modelo, muestreo=None):
    num = [c for c in cols if c in NUM_ALL]
    cat = [c for c in cols if c in CAT_ALL]
    prep = ColumnTransformer(
        [
            (
                "num",
                Pipeline([
                    ("imp", SimpleImputer(strategy="median", add_indicator=True)),
                    ("esc", StandardScaler()),
                ]),
                num,
            ),
            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    min_frequency=50,
                    handle_unknown="infrequent_if_exist",
                    sparse_output=False,
                ),
                cat,
            ),
        ]
    )
    pasos = [("prep", prep)]
    if muestreo is not None:
        pasos.append(("muestreo", muestreo))
    pasos.append(("modelo", modelo))
    return ImbPipeline(pasos)


def metricas_en_umbral(y_true, proba, t):
    pred = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "umbral": t,
        "sensibilidad_TPR": recall_score(y_true, pred, pos_label=1, zero_division=0),
        "especificidad_TNR": recall_score(y_true, pred, pos_label=0, zero_division=0),
        "precision_mora1": precision_score(y_true, pred, pos_label=1, zero_division=0),
        "FN": int(fn),
        "FP": int(fp),
        "VP": int(tp),
        "VN": int(tn),
    }


def umbral_youden(y_true, proba):
    ts = np.linspace(0.05, 0.95, 181)
    mejor_t, mejor_j = 0.5, -1.0
    for t in ts:
        m = metricas_en_umbral(y_true, proba, t)
        j = m["sensibilidad_TPR"] + m["especificidad_TNR"] - 1
        if j > mejor_j:
            mejor_j, mejor_t = j, t
    return float(mejor_t), float(mejor_j)


def umbral_costo(y_true, proba, ratio_fn_fp):
    ts = np.linspace(0.05, 0.95, 181)
    mejor_t, mejor_c = 0.5, np.inf
    for t in ts:
        m = metricas_en_umbral(y_true, proba, t)
        costo = ratio_fn_fp * m["FN"] + m["FP"]
        if costo < mejor_c:
            mejor_c, mejor_t = costo, t
    return float(mejor_t), float(mejor_c)


def cv_cuatro_modelos(X_tr, y_tr, cols):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    auc_k = {
        k: cross_validate(
            construir_pipeline(cols, KNeighborsClassifier(k)),
            X_tr[cols], y_tr, cv=cv, scoring={"roc": "roc_auc"},
        )["test_roc"]
        for k in K_LISTA
    }
    k_mejor = max(auc_k, key=lambda k: auc_k[k].mean())
    modelos = {
        "Regresión logística": MODELO_FINAL,
        "LDA": LinearDiscriminantAnalysis(),
        "KNN": KNeighborsClassifier(k_mejor),
        "Naive Bayes": GaussianNB(),
    }
    filas = []
    for nombre, m in modelos.items():
        r = cross_validate(
            construir_pipeline(cols, m),
            X_tr[cols], y_tr, cv=cv,
            scoring={"roc": "roc_auc", "pr": "average_precision"},
        )
        filas.append({
            "Modelo": nombre,
            "ROC-AUC CV (media)": round(r["test_roc"].mean(), 3),
            "ROC-AUC CV (desv.)": round(r["test_roc"].std(), 3),
            "PR-AUC CV (media)": r["test_pr"].mean(),
            "PR-AUC CV (desv.)": r["test_pr"].std(),
            "Hiperparámetro elegido": f"k = {k_mejor}" if nombre == "KNN" else "—",
        })
    return pd.DataFrame(filas), k_mejor


def cv_escenarios_abc(X_tr, y_tr):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    filas = []
    for nom, cols in ESCENARIOS.items():
        r = cross_validate(
            construir_pipeline(cols, MODELO_FINAL),
            X_tr[cols], y_tr, cv=cv, scoring={"roc": "roc_auc"},
        )
        filas.append({"Escenario": nom, "AUC CV (logística)": round(r["test_roc"].mean(), 3)})
    return pd.DataFrame(filas)"""
)

md("## 3. Datos y partición (igual que 2.ª entrega)")

code(
    """np.random.seed(SEED)
data_path = resolve_data_path()
df_raw = pd.read_csv(data_path)
df = df_raw.drop_duplicates().reset_index(drop=True)
X_full = df.drop(columns="mora")
y = df["mora"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X_full, y, test_size=0.20, stratify=y, random_state=SEED,
)
print("Archivo:", data_path)
print("Clientes únicos:", len(df), "| Train:", len(y_tr), "| Test:", len(y_te))
print("Prevalencia mora (train):", round(100 * y_tr.mean(), 1), "%")"""
)

md(
    """## 4. Validación cruzada — comparación de modelos (histórica)

Documenta por qué el modelo final sigue siendo **regresión logística** en escenario B. No es una nueva competición en test."""
)

code(
    """tabla_cv, k_mejor = cv_cuatro_modelos(X_tr, y_tr, cols_b)
tabla_cv_esc = cv_escenarios_abc(X_tr, y_tr)
tabla_cv.to_csv(OUT_DIR / "tabla_cv_modelo_final.csv", index=False)
tabla_cv_esc.to_csv(OUT_DIR / "tabla_cv_escenarios_abc.csv", index=False)
display(tabla_cv)
display(tabla_cv_esc)
print("KNN — k elegido en CV:", k_mejor)"""
)

md(
    """## 5. Umbrales con predicciones OOF (solo entrenamiento)

$C = r \\cdot FN + FP$. Los cortes se buscan en la grilla de umbrales sobre probabilidades OOF (`cross_val_predict`, 5 folds). **Línea base:** predecir siempre mora (FN=0, FP = buenos pagadores en train)."""
)

code(
    """cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
pipe_oof = construir_pipeline(cols_b, MODELO_FINAL)
proba_oof = cross_val_predict(pipe_oof, X_tr[cols_b], y_tr, cv=cv, method="predict_proba")[:, 1]

t_youden, j_youden = umbral_youden(y_tr, proba_oof)
t_c1, cost_c1 = umbral_costo(y_tr, proba_oof, 1.0)
t_c2, cost_c2 = umbral_costo(y_tr, proba_oof, 2.0)
t_c5, cost_c5 = umbral_costo(y_tr, proba_oof, 5.0)

n_buenos = int((y_tr == 0).sum())
baseline_fp = n_buenos
baseline_cost = {1: baseline_fp, 2: baseline_fp, 5: baseline_fp}

print("Youden t =", round(t_youden, 3), "| J =", round(j_youden, 3))
print("Costo mín. r=1 → t =", round(t_c1, 3), "C =", round(cost_c1, 0))
print("Costo mín. r=2 → t =", round(t_c2, 3), "C =", round(cost_c2, 0))
print("Costo mín. r=5 → t =", round(t_c5, 3), "C =", round(cost_c5, 0))
print("Línea base FP (OOF):", baseline_fp)"""
)

code(
    """ts = np.linspace(0.05, 0.95, 181)
filas_umbral = []
for t in ts:
    m = metricas_en_umbral(y_tr, proba_oof, t)
    m["costo_ratio1"] = m["FN"] + m["FP"]
    m["costo_ratio2"] = 2 * m["FN"] + m["FP"]
    m["costo_ratio5"] = 5 * m["FN"] + m["FP"]
    m["J_youden"] = m["sensibilidad_TPR"] + m["especificidad_TNR"] - 1
    filas_umbral.append(m)
df_umbral = pd.DataFrame(filas_umbral)
df_umbral.to_csv(OUT_DIR / "tabla_umbral_costos_oof.csv", index=False)


def fila_costo(etiqueta, t, m):
    out = {**m, "escenario": etiqueta, "umbral": t}
    out["C_r1"] = m["FN"] + m["FP"]
    out["C_r2"] = 2 * m["FN"] + m["FP"]
    out["C_r5"] = 5 * m["FN"] + m["FP"]
    return out


baseline_m = metricas_en_umbral(y_tr, np.ones_like(proba_oof), 0.5)
baseline_m["FN"], baseline_m["FP"] = 0, baseline_fp

destacados = [
    fila_costo("linea_base_siempre_mora", np.nan, baseline_m),
    fila_costo("convencion_0.5", 0.5, metricas_en_umbral(y_tr, proba_oof, 0.5)),
    fila_costo("youden_oof", t_youden, metricas_en_umbral(y_tr, proba_oof, t_youden)),
    fila_costo("costo_FN_1x_FP", t_c1, metricas_en_umbral(y_tr, proba_oof, t_c1)),
    fila_costo("costo_FN_2x_FP", t_c2, metricas_en_umbral(y_tr, proba_oof, t_c2)),
    fila_costo("costo_FN_5x_FP", t_c5, metricas_en_umbral(y_tr, proba_oof, t_c5)),
]
df_dest = pd.DataFrame(destacados)
df_dest.to_csv(OUT_DIR / "tabla_umbrales_destacados_oof.csv", index=False)

comite = df_dest[
    ["escenario", "umbral", "FN", "FP", "sensibilidad_TPR", "especificidad_TNR", "C_r1", "C_r2", "C_r5"]
].copy()
for r in (1, 2, 5):
    comite[f"linea_base_C_r{r}"] = baseline_cost[r]
comite.to_csv(OUT_DIR / "tabla_umbral_costos_comite_oof.csv", index=False)

display(df_dest.round(3))"""
)

code(
    """fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.plot(df_umbral["umbral"], df_umbral["costo_ratio1"], label="r=1: FN+FP")
ax.plot(df_umbral["umbral"], df_umbral["costo_ratio2"], label="r=2: 2·FN+FP")
ax.plot(df_umbral["umbral"], df_umbral["costo_ratio5"], label="r=5: 5·FN+FP")
ax.axhline(baseline_cost[2], color="gray", ls="--", lw=1, label=f"Línea base (r=2), C={baseline_cost[2]}")
ax.axvline(t_youden, color="black", ls=":", lw=1)
ax.set_xlabel("Umbral t (OOF entrenamiento)")
ax.set_ylabel("Costo simbólico")
ax.set_title("Costo vs umbral — logística B")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8)
fig.subplots_adjust(bottom=0.22)
fig.savefig(OUT_DIR / "fig_costo_vs_umbral.png", dpi=140)
plt.show()

umbrales = {
    "t_convencion": 0.5,
    "t_youden_oof": round(t_youden, 4),
    "t_costo_fn_1x_fp_oof": round(t_c1, 4),
    "t_costo_fn_2x_fp_oof": round(t_c2, 4),
    "t_costo_fn_5x_fp_oof": round(t_c5, 4),
    "j_youden_oof": round(j_youden, 4),
    "costo_simbolico_oof_ratio1": round(cost_c1, 1),
    "costo_simbolico_oof_ratio2": round(cost_c2, 1),
    "costo_simbolico_oof_ratio5": round(cost_c5, 1),
    "linea_base_FP_oof": baseline_fp,
    "linea_base_costo_r1": baseline_cost[1],
    "nota": "Umbrales elegidos solo con OOF en entrenamiento. No optimizar en test.",
}
(OUT_DIR / "umbrales_recomendados.json").write_text(
    json.dumps(umbrales, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(json.dumps(umbrales, indent=2, ensure_ascii=False))"""
)

md(
    """## 6. Modelo final — ajuste en todo el train y serialización

Objeto que consume **Streamlit** (`pipeline_logistica_B.joblib`). Mismas 11 variables, sin balanceo."""
)

code(
    """pipe_final = construir_pipeline(cols_b, MODELO_FINAL)
pipe_final.fit(X_tr[cols_b], y_tr)
joblib.dump(pipe_final, OUT_DIR / "pipeline_logistica_B.joblib")

import sklearn
import imblearn

config = {
    "modelo": "LogisticRegression(C=1e6, max_iter=5000)",
    "escenario": ESC_PRINCIPAL_KEY,
    "variables": cols_b,
    "balanceo": "ninguno",
    "semilla": SEED,
    "particion": "80/20 estratificada",
    "k_knn_referencia_cv": int(k_mejor),
    "scikit_learn_version": sklearn.__version__,
    "imbalanced_learn_version": imblearn.__version__,
}
(OUT_DIR / "configuracion_definitiva.json").write_text(
    json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
)
pd.DataFrame([config]).to_csv(OUT_DIR / "tabla_configuracion_definitiva.csv", index=False)
display(pd.DataFrame([config]))
print("Guardado:", OUT_DIR / "pipeline_logistica_B.joblib")"""
)

md(
    """## 7. Conjunto de prueba — **solo lectura**

Copiamos la tabla de cierre de la 2.ª entrega, verificamos AUC reentrenado y aplicamos umbrales **ya fijados en OOF**."""
)

code(
    """ref_test = CODIGO_DIR / "salidas_segunda" / "tabla_test_principal.csv"
if ref_test.exists():
    pd.read_csv(ref_test).to_csv(OUT_DIR / "tabla_test_cierre_segunda_entrega.csv", index=False)
else:
    print("No se encontró", ref_test.name, ", se omite la copia de la 2.ª entrega.")

proba_te = pipe_final.predict_proba(X_te[cols_b])[:, 1]
auc_verify = roc_auc_score(y_te, proba_te)
verif = pd.DataFrame([
    {
        "AUC_reentrenado_mismo_train_test": round(auc_verify, 3),
        "AUC_informe_segunda_entrega": 0.722,
        "uso": "Verificación. No usar test para elegir modelo ni umbral.",
    }
])
verif.to_csv(OUT_DIR / "tabla_test_verificacion_solo_lectura.csv", index=False)

filas_te = []
for etiqueta, tval in [
    ("convencion_0.5", 0.5),
    ("youden_oof", umbrales["t_youden_oof"]),
    ("costo_FN_1x_FP_oof", umbrales["t_costo_fn_1x_fp_oof"]),
    ("costo_FN_2x_FP_oof", umbrales["t_costo_fn_2x_fp_oof"]),
    ("costo_FN_5x_FP_oof", umbrales["t_costo_fn_5x_fp_oof"]),
]:
    m = metricas_en_umbral(y_te, proba_te, tval)
    m["escenario_umbral"] = etiqueta
    filas_te.append(m)
pd.DataFrame(filas_te).to_csv(OUT_DIR / "tabla_test_umbrales_solo_lectura.csv", index=False)

resumen = pd.DataFrame([
    {"clave": "n_train", "valor": len(y_tr)},
    {"clave": "n_test", "valor": len(y_te)},
    {"clave": "pct_mora", "valor": round(100 * y_tr.mean(), 1)},
    {"clave": "auc_cv_logistica", "valor": tabla_cv.loc[0, "ROC-AUC CV (media)"]},
])
resumen.to_csv(OUT_DIR / "resumen_ejecucion.csv", index=False)

display(verif)
display(pd.read_csv(OUT_DIR / "tabla_test_umbrales_solo_lectura.csv").round(3))"""
)

md(
    """## 8. Copia de archivos para el informe

Si existe la carpeta `documentacion/entregables_para_informe/`, replica tablas y figuras para compilar el PDF. En Colab puede omitirse."""
)

code(
    """ARTEFACTOS = [
    "tabla_cv_modelo_final.csv",
    "tabla_cv_escenarios_abc.csv",
    "tabla_umbral_costos_oof.csv",
    "tabla_umbrales_destacados_oof.csv",
    "tabla_umbral_costos_comite_oof.csv",
    "fig_costo_vs_umbral.png",
    "umbrales_recomendados.json",
    "pipeline_logistica_B.joblib",
    "configuracion_definitiva.json",
    "tabla_configuracion_definitiva.csv",
    "tabla_test_cierre_segunda_entrega.csv",
    "tabla_test_verificacion_solo_lectura.csv",
    "tabla_test_umbrales_solo_lectura.csv",
    "resumen_ejecucion.csv",
]

if INFORME_DIR.parent.exists():
    INFORME_DIR.mkdir(parents=True, exist_ok=True)
    for nombre in ARTEFACTOS:
        src = OUT_DIR / nombre
        if src.exists():
            shutil.copy2(src, INFORME_DIR / nombre)
    print("Copiados →", INFORME_DIR.resolve())
else:
    print("No hay carpeta para el informe; salidas en", OUT_DIR)"""
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
print("Escrito:", OUT)
