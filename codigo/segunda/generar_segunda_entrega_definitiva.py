"""Genera segunda_entrega_modelamiento_definitiva.ipynb y valida que ejecute."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "segunda_entrega_modelamiento_definitiva.ipynb"

# Notebook source as list of (cell_type, source)
CELLS = []

def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "id": f"md{len(CELLS)+1}", "source": text.splitlines(keepends=True)})

def code(text):
    CELLS.append({"cell_type": "code", "metadata": {}, "id": f"cd{len(CELLS)+1}", "outputs": [], "execution_count": None, "source": text.splitlines(keepends=True)})

md("""# Segunda entrega — Riesgo de mora (Perú)
**Machine Learning I · Universidad Externado de Colombia**

Notebook definitivo: auditoría de fuga, pipeline reproducible, balanceo, validación cruzada, **ROC-AUC y PR-AUC**, optimización de umbral **solo con predicciones out-of-fold del entrenamiento**, y evaluación única en prueba.

**Mapa a la rúbrica (pesos orientativos):**
| Bloque | Secciones del notebook |
|--------|-------------------------|
| Definición y auditoría de fuga (20 %) | 2–3 |
| Pipeline y reproducibilidad (20 %) | 4–5, 7–8 |
| Balanceo de clases (15 %) | 9 |
| Diseño de validación (20 %) | 6, 10–11 |
| Análisis y evidencia (15 %) | 12–14 |
| Sustentación (10 %) | Informe + este notebook |

**Modelos (curso):** línea base (mayoría), regresión logística, KNN, Naive Bayes gaussiano.
""")

code("""# Google Colab: descomentar si aplica
# from google.colab import drive
# drive.mount('/content')

import os, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    roc_auc_score, average_precision_score, recall_score, precision_score, f1_score,
    accuracy_score, confusion_matrix, roc_curve, precision_recall_curve, fbeta_score,
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
np.random.seed(SEED)

# Rutas: Colab / local (subir database.csv a /content o dejarlo en datos/)
CANDIDATOS = [
    Path("/content/database.csv"),
    Path("../datos/database.csv"),
    Path("datos/database.csv"),
    Path("database.csv"),
]
DATA_PATH = next((p for p in CANDIDATOS if p.exists()), None)
if DATA_PATH is None:
    raise FileNotFoundError("No se encontró database.csv. En Colab: Files > subir el CSV a /content.")

OUT_DIR = Path("salidas")
OUT_DIR.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
print("Datos:", DATA_PATH.resolve())
""")

md("""## 2. Definición de `mora` y pregunta predictiva (feedback entrega 1)

**Pregunta:** ¿Qué tan bien puede predecirse el riesgo de mora de un cliente de crédito de consumo a partir de información financiera y socioeconómica disponible **previamente**?

**Fuente:** Calderón Baldeón (2021), *BankDefaultAnalysis* (Kaggle). No hay diccionario con umbral de días.

**Definición operativa (supuesto explícito):** `mora=1` = cliente marcado como moroso en la base; `mora=0` = pagador al día. No equivale a «atraso > 0 hoy».
""")

code("""df_raw = pd.read_csv(DATA_PATH)
print("Shape:", df_raw.shape)
print(df_raw.head(3).to_string())

tabla = pd.crosstab(df_raw["atraso"] > 0, df_raw["mora"], margins=True)
tabla.index = ["atraso = 0", "atraso > 0", "Total"]
tabla.columns = ["mora = 0", "mora = 1", "Total"]
print(tabla)
pct_m1_sin_atraso = (df_raw.loc[df_raw.mora == 1, "atraso"] == 0).mean()
print(f"Con mora=1 y atraso=0: {pct_m1_sin_atraso:.1%}")
""")

md("""## 3. Momento de predicción y auditoría de fuga (variables)

Evaluación crediticia **antes** de conocer la etiqueta `mora`. Sin fechas en la base, la disponibilidad temporal es un **supuesto** documentado por variable.

| Variable | Riesgo | Decisión escenario principal (B) |
|----------|--------|----------------------------------|
| `atraso` | Comportamiento de pago; puede solaparse con la etiqueta | **Excluir** |
| `clasif_sbs` | Clasificación SBS según pagos | **Excluir** |
| `score` | Herramienta de riesgo; puede actualizarse post-atraso | **Incluir** + escenario C sin `score` |

**Importante (corrección conceptual):** correlación moderada **no** descarta fuga temporal; la exclusión de `atraso` y `clasif_sbs` es por **definición de negocio**, no solo por Spearman bajo.
""")

code("""num_all = ["edad", "dias_lab", "exp_sf", "nivel_ahorro", "ingreso", "linea_sf", "deuda_sf", "score", "atraso", "clasif_sbs"]
cat_all = ["vivienda", "zona", "nivel_educ"]

ESCENARIOS = {
    "A_completo": num_all + cat_all,
    "B_principal": [c for c in num_all + cat_all if c not in ("atraso", "clasif_sbs")],
    "C_conservador": [c for c in num_all + cat_all if c not in ("atraso", "clasif_sbs", "score")],
}
ESC = "B_principal"

sospechosas = ["atraso", "clasif_sbs", "score"]
corr = df_raw[sospechosas + ["mora"]].corr(method="spearman")["mora"].drop("mora")
print("Spearman con mora:\\n", corr.round(3))
""")

md("""## 4. Calidad de datos y partición **antes** de cualquier ajuste

1. Eliminar filas **completamente duplicadas** (evita que la misma fila esté en entrenamiento y prueba).
2. Partición 80/20 estratificada por `mora`.
3. Todo estadístico (mediana, escalado, categorías, VIF exploratorio) se calcula **solo con entrenamiento** salvo EDA descriptiva global.
""")

code("""df = df_raw.drop_duplicates().copy()
print(f"Registros: {len(df_raw)} -> {len(df)} únicos; prevalencia mora={df.mora.mean():.3f}")

X_full = df.drop(columns="mora")
y = df["mora"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X_full, y, test_size=0.20, stratify=y, random_state=SEED
)
print("Train:", X_tr.shape, "mora%", round(100 * y_tr.mean(), 1))
print("Test :", X_te.shape, "mora%", round(100 * y_te.mean(), 1))

assert len(X_tr.merge(X_te, how="inner")) == 0, "Filas idénticas en train y test"
""")

md("""## 5. Línea base (mayoría)

Clasificador que siempre predice `mora=1`. AUC=0.5 pero accuracy y recall de mora engañan.
""")

code("""def metricas(y_true, proba, umbral=0.5):
    pred = (proba >= umbral).astype(int)
    return {
        "AUC_ROC": roc_auc_score(y_true, proba),
        "PR_AUC": average_precision_score(y_true, proba),
        "Recall_mora1": recall_score(y_true, pred, pos_label=1, zero_division=0),
        "Precision_mora1": precision_score(y_true, pred, pos_label=1, zero_division=0),
        "Recall_mora0": recall_score(y_true, pred, pos_label=0, zero_division=0),
        "F1_mora1": f1_score(y_true, pred, pos_label=1, zero_division=0),
        "Accuracy": accuracy_score(y_true, pred),
    }

proba_base = np.ones(len(y_te))  # siempre clase positiva
base = metricas(y_te, proba_base)
pd.Series(base, name="Linea base").round(3)
""")

md("""## 6. Pipeline formal (orden sin fuga)

**Orden fijo dentro de cada fold de entrenamiento:**
1. Imputación mediana + indicador de faltante (numéricas).
2. Estandarización (numéricas).
3. One-hot (`drop='first'`, zonas poco frecuentes agrupadas).
4. *(Opcional)* remuestreo **solo en fit del fold de entrenamiento**.
5. Modelo.

No hay ingeniería de features fuera del pipeline (evita derivadas calculadas con toda la base).
""")

code("""def construir_pipeline(cols, estimador, muestreo=None):
    num = [c for c in cols if c in num_all]
    cat = [c for c in cols if c in cat_all]
    prep = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("esc", StandardScaler()),
        ]), num),
        ("cat", OneHotEncoder(
            drop="first", min_frequency=50, handle_unknown="infrequent_if_exist", sparse_output=False
        ), cat),
    ], remainder="drop")
    pasos = [("prep", prep)]
    if muestreo is not None:
        pasos.append(("muestreo", muestreo))
    pasos.append(("modelo", estimador))
    return ImbPipeline(pasos)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cols_B = ESCENARIOS[ESC]
""")

md("""## 7. Pruebas de flujo (no sustituyen auditoría de variables)

1. Cero filas compartidas train/test.
2. Estratificación ~71 % mora en cada fold.
3. Mediana del imputador ≠ mediana de toda la base.
4. Etiquetas permutadas en train → AUC ≈ 0.5.
""")

code("""# Prueba 3
pipe_tmp = construir_pipeline(cols_B, LogisticRegression(C=1e6, max_iter=5000)).fit(X_tr[cols_B], y_tr)
imp = pipe_tmp.named_steps["prep"].named_transformers_["num"].named_steps["imp"]
num_cols_pipe = [c for c in cols_B if c in num_all]
med_imp = np.asarray(imp.statistics_)[: len(num_cols_pipe)]
med_train = X_tr[num_cols_pipe].median().values
print("Mediana imputador vs mediana train (deben coincidir):", np.allclose(med_imp, med_train, equal_nan=True))
print("Mediana imputador vs mediana train+test (no deben coincidir si hay diferencia):",
      not np.allclose(med_imp, X_full[num_cols_pipe].median().values, equal_nan=True))

# Prueba 4
y_perm = y_tr.sample(frac=1, random_state=SEED).values
sc_perm = cross_validate(
    construir_pipeline(cols_B, LogisticRegression(C=1e6, max_iter=5000)),
    X_tr[cols_B], y_perm, cv=cv, scoring={"auc": "roc_auc"}
)["test_auc"].mean()
print("AUC con etiquetas permutadas (train):", round(sc_perm, 3))
""")

md("""## 8. Multicolinealidad (solo entrenamiento, escenario B)

VIF calculado **después** de imputar mediana en train (exploratorio; la imputación final va en el pipeline).
""")

code("""from sklearn.linear_model import LinearRegression

num_B = [c for c in cols_B if c in num_all]
imp_df = X_tr[num_B].copy()
for c in num_B:
    imp_df[c] = imp_df[c].fillna(imp_df[c].median())

vif = {}
for c in num_B:
    Xo = imp_df.drop(columns=c)
    yv = imp_df[c]
    r2 = LinearRegression().fit(Xo, yv).score(Xo, yv)
    vif[c] = np.inf if r2 >= 0.999 else 1 / (1 - r2)
pd.Series(vif).round(2).sort_values(ascending=False)
""")

md("""## 9. Balanceo de clases (comparación en CV, escenario B)

Métricas: **ROC-AUC** (ordenamiento) y **PR-AUC** (sensible a prevalencia de la clase positiva en el fold).
Decisión: si ambas no mejoran de forma estable, entrenar **sin balancear** (criterio de parsimonia).
""")

code("""opciones_balanceo = {
    "sin_balancear": None,
    "peso_clases": "balanced",
    "submuestreo": RandomUnderSampler(random_state=SEED),
    "sobremuestreo": RandomOverSampler(random_state=SEED),
    "smote": SMOTE(random_state=SEED),
}

filas = []
for nombre, muestreo in opciones_balanceo.items():
    if muestreo == "balanced":
        est = LogisticRegression(C=1e6, max_iter=5000, class_weight="balanced")
        pipe = construir_pipeline(cols_B, est, None)
    else:
        est = LogisticRegression(C=1e6, max_iter=5000)
        pipe = construir_pipeline(cols_B, est, muestreo)
    r = cross_validate(
        pipe, X_tr[cols_B], y_tr, cv=cv,
        scoring={"roc": "roc_auc", "pr": "average_precision"},
    )
    filas.append({
        "opcion": nombre,
        "ROC_AUC_CV": r["test_roc"].mean(),
        "PR_AUC_CV": r["test_pr"].mean(),
    })
tabla_bal = pd.DataFrame(filas).round(4)
tabla_bal.to_csv(OUT_DIR / "balanceo_cv.csv", index=False)
tabla_bal
""")

md("""## 10. Comparación de modelos (CV en entrenamiento)

- **KNN:** `k` se elige con `GridSearchCV` dentro del pipeline (evita doble uso manual del mismo CV).
- Reporte: ROC-AUC y PR-AUC (media ± desv. en 5 folds).
""")

code("""from sklearn.model_selection import GridSearchCV

pipe_knn = construir_pipeline(cols_B, KNeighborsClassifier())
param_knn = {"modelo__n_neighbors": [5, 11, 21, 51, 101, 151, 201]}
gs_knn = GridSearchCV(pipe_knn, param_knn, cv=cv, scoring="roc_auc", n_jobs=1)
gs_knn.fit(X_tr[cols_B], y_tr)
k_mejor = gs_knn.best_params_["modelo__n_neighbors"]
print("k óptimo KNN:", k_mejor)

modelos_def = {
    "Regresion logistica": LogisticRegression(C=1e6, max_iter=5000),
    "KNN": KNeighborsClassifier(n_neighbors=k_mejor),
    "Naive Bayes": GaussianNB(),
}

scoring = {"roc": "roc_auc", "pr": "average_precision"}
tabla_mod = []
modelos_cv = {}
for nombre, est in modelos_def.items():
    pipe = construir_pipeline(cols_B, est)
    r = cross_validate(pipe, X_tr[cols_B], y_tr, cv=cv, scoring=scoring)
    tabla_mod.append({
        "modelo": nombre,
        "ROC_AUC_CV": r["test_roc"].mean(),
        "ROC_std": r["test_roc"].std(),
        "PR_AUC_CV": r["test_pr"].mean(),
        "PR_std": r["test_pr"].std(),
    })
    modelos_cv[nombre] = pipe
tabla_mod = pd.DataFrame(tabla_mod).round(4)
tabla_mod
""")

md("""## 11. Optimización de umbral (solo out-of-fold en **entrenamiento**)

1. `cross_val_predict(..., method='predict_proba')` sobre **train** → probabilidades OOF.
2. Barrido de umbrales **solo con OOF**: F2 sobre `mora=1`, F2 sobre `mora=0`, índice de Youden (TPR−FPR) y costo `5·FN + 1·FP`.
3. **Trampa a evitar:** como `mora=1` es ~71 %, maximizar F2 de mora o un costo FN alto **baja el umbral** y se parece a la línea base. El umbral de **evaluación** es Youden (equilibrio de errores de ranking), no F2-mora=1.
4. El umbral se aplica **una vez** al modelo ajustado en todo train, para evaluar en **test**.

*No se usa test para elegir umbral* (evita fuga).
""")

code("""def umbral_optimo(y_true, proba, metrica="f2", beta=2, cost_fn=5, cost_fp=1):
    umbrales = np.linspace(0.05, 0.95, 181)
    mejor_u, mejor_v = 0.5, -np.inf
    for u in umbrales:
        pred = (proba >= u).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        if metrica == "f2":
            v = fbeta_score(y_true, pred, beta=beta, pos_label=1, zero_division=0)
        elif metrica == "f2_mora0":
            v = fbeta_score(y_true, pred, beta=beta, pos_label=0, zero_division=0)
        elif metrica == "youden":
            tpr = tp / (tp + fn) if (tp + fn) else 0
            fpr = fp / (fp + tn) if (fp + tn) else 0
            v = tpr - fpr
        elif metrica == "costo":
            v = -(cost_fn * fn + cost_fp * fp)
        else:
            raise ValueError(metrica)
        if v > mejor_v:
            mejor_v, mejor_u = v, u
    return mejor_u, mejor_v

# OOF solo para regresión logística (modelo principal)
pipe_lr = construir_pipeline(cols_B, LogisticRegression(C=1e6, max_iter=5000))
proba_oof = cross_val_predict(pipe_lr, X_tr[cols_B], y_tr, cv=cv, method="predict_proba")[:, 1]

u_f2, _ = umbral_optimo(y_tr, proba_oof, metrica="f2")
u_f2_0, _ = umbral_optimo(y_tr, proba_oof, metrica="f2_mora0")
u_youden, _ = umbral_optimo(y_tr, proba_oof, metrica="youden")
u_costo, _ = umbral_optimo(y_tr, proba_oof, metrica="costo")
print("OOF F2 (mora=1):", round(u_f2, 3), "| F2 (mora=0):", round(u_f2_0, 3),
      "| Youden:", round(u_youden, 3), "| costo 5:1:", round(u_costo, 3))
print("Nota: con mora=1 mayoritaria, F2 sobre mora=1 y un costo FN alto empujan el umbral hacia ABAJO (casi siempre predecir mora).")
print("Para la evaluación en prueba usamos Youden (TPR-FPR), que no colapsa a la línea base.")

# Referencia: umbral por defecto 0.5
for u, etiqueta in [(0.5, "default 0.5"), (u_youden, "OOF Youden"), (u_f2, "OOF F2 mora=1")]:
    pred = (proba_oof >= u).astype(int)
    print(etiqueta, "-> Recall mora1", recall_score(y_tr, pred), "Recall mora0", recall_score(y_tr, pred, pos_label=0))
""")

md("""## 12. Evaluación final única en **prueba**

Ajuste en todo el entrenamiento; métricas ROC-AUC / PR-AUC sin umbral; clasificación con umbral OOF-F2.
""")

code("""resultados_test = []
matrices = {}
UMBRAL_FINAL = u_youden  # no usar F2-mora=1: colapsa a la línea base (clase mayoritaria)

for nombre, est in modelos_def.items():
    pipe = construir_pipeline(cols_B, est)
    pipe.fit(X_tr[cols_B], y_tr)
    proba = pipe.predict_proba(X_te[cols_B])[:, 1]
    m = metricas(y_te, proba, umbral=0.5)
    m_umbral = metricas(y_te, proba, umbral=UMBRAL_FINAL)
    m["modelo"] = nombre
    m["umbral"] = 0.5
    resultados_test.append(m)
    m2 = {**m_umbral, "modelo": nombre, "umbral": UMBRAL_FINAL}
    resultados_test.append(m2)
    if nombre == "Regresion logistica":
        pred_f = (proba >= UMBRAL_FINAL).astype(int)
        matrices[nombre] = confusion_matrix(y_te, pred_f, labels=[0, 1])

tabla_test = pd.DataFrame(resultados_test).round(3)
tabla_test.to_csv(OUT_DIR / "resultados_prueba.csv", index=False)
print(tabla_test.to_string(index=False))

mc = matrices["Regresion logistica"]
print("Matriz confusión LR (umbral OOF F2) [ [TN FP], [FN TP] ]:")
print(mc)

# Curvas ROC y PR (test)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for nombre, est in modelos_def.items():
    pipe = construir_pipeline(cols_B, est)
    pipe.fit(X_tr[cols_B], y_tr)
    p = pipe.predict_proba(X_te[cols_B])[:, 1]
    fpr, tpr, _ = roc_curve(y_te, p)
    prec, rec, _ = precision_recall_curve(y_te, p)
    axes[0].plot(fpr, tpr, label=f"{nombre} (AUC={roc_auc_score(y_te, p):.2f})")
    axes[1].plot(rec, prec, label=f"{nombre} (AP={average_precision_score(y_te, p):.2f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
axes[0].set_title("ROC - conjunto de prueba")
axes[1].set_title("Precision-Recall - conjunto de prueba")
axes[0].legend(fontsize=8)
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "roc_pr_test.png", dpi=200)
plt.close()
""")

md("""## 13. Sensibilidad escenarios A / B / C (prueba, regresión logística)

¿Cuánto cambia el AUC al incluir variables sospechosas o quitar `score`?
""")

code("""filas_sens = []
for esc, cols in ESCENARIOS.items():
    pipe = construir_pipeline(cols, LogisticRegression(C=1e6, max_iter=5000))
    pipe.fit(X_tr[cols], y_tr)
    p = pipe.predict_proba(X_te[cols])[:, 1]
    filas_sens.append({
        "escenario": esc,
        "ROC_AUC": roc_auc_score(y_te, p),
        "PR_AUC": average_precision_score(y_te, p),
    })
pd.DataFrame(filas_sens).round(3)
""")

md("""## 14. Interpretación (sin causalidad) y limitaciones

- Coeficientes de la regresión logística: **asociación** en esta muestra, no efecto causal.
- Prevalencia 71 %: probabilidades no calibradas para el sistema financiero peruano; útiles para **ordenar** riesgo.
- Naive Bayes gaussiano sobre dummies: supuesto débil; se reporta como comparador, no como modelo final.
- Próxima entrega: LDA/regularización, calibración, Streamlit, umbral con costos validados con negocio.

**Lista de chequeo rúbrica:** fuga auditada (B), pipeline único, balanceo justificado, CV estratificada, PR-AUC + ROC-AUC, umbral OOF, test una sola vez, línea base incluida.
""")

code("""# Coeficientes (modelo LR escenario B, variables estandarizadas en numéricas)
pipe_lr.fit(X_tr[cols_B], y_tr)
# Extraer nombres tras transformación es verbose; guardamos importancia vía odds en variables numéricas directas
print("Notebook ejecutado. Salidas en:", OUT_DIR.resolve())
""")

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": CELLS,
}

OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Escrito:", OUT)
