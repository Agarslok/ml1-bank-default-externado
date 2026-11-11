"""Lógica reproducible — entrega final (modelo congelado según contrato_decisiones.md).

Usado por: `python tercera_entrega_lib.py` y la app Streamlit.
El notebook `tercera_entrega_modelamiento_final.ipynb` es autocontenido (no importa este archivo).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
NUM_ALL = [
    "edad",
    "dias_lab",
    "exp_sf",
    "nivel_ahorro",
    "ingreso",
    "linea_sf",
    "deuda_sf",
    "score",
    "atraso",
    "clasif_sbs",
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


def resolve_data_path() -> Path:
    codigo = Path(__file__).resolve().parent.parent
    proyecto = codigo.parent
    for p in (
        proyecto / "datos" / "database.csv",
        codigo / "datos" / "database.csv",
        Path(__file__).resolve().parent / "datos" / "database.csv",
        Path("datos/database.csv"),
        Path("../datos/database.csv"),
        Path("../../datos/database.csv"),
    ):
        if p.exists():
            return p
    raise FileNotFoundError("No se encontró database.csv en datos/")


def construir_pipeline(cols, modelo, muestreo=None):
    num = [c for c in cols if c in NUM_ALL]
    cat = [c for c in cols if c in CAT_ALL]
    prep = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median", add_indicator=True)),
                        ("esc", StandardScaler()),
                    ]
                ),
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


def cargar_datos_y_particion(data_path: Path | None = None):
    data_path = data_path or resolve_data_path()
    df_raw = pd.read_csv(data_path)
    df = df_raw.drop_duplicates().reset_index(drop=True)
    X_full = df.drop(columns="mora")
    y = df["mora"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_full, y, test_size=0.20, stratify=y, random_state=SEED
    )
    return df_raw, df, X_tr, X_te, y_tr, y_te


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


def umbral_costo(y_true, proba, ratio_fn_fp: float):
    """Minimiza c_FN * FN + c_FP * FP con c_FN / c_FP = ratio_fn_fp."""
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
            X_tr[cols],
            y_tr,
            cv=cv,
            scoring={"roc": "roc_auc"},
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
            X_tr[cols],
            y_tr,
            cv=cv,
            scoring={"roc": "roc_auc", "pr": "average_precision"},
        )
        filas.append(
            {
                "Modelo": nombre,
                "ROC-AUC CV (media)": round(r["test_roc"].mean(), 3),
                "ROC-AUC CV (desv.)": round(r["test_roc"].std(), 3),
                "PR-AUC CV (media)": r["test_pr"].mean(),
                "PR-AUC CV (desv.)": r["test_pr"].std(),
                "Hiperparámetro elegido": f"k = {k_mejor}" if nombre == "KNN" else "—",
            }
        )
    return pd.DataFrame(filas), k_mejor


def cv_escenarios_abC(X_tr, y_tr):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    filas = []
    for nom, cols in ESCENARIOS.items():
        r = cross_validate(
            construir_pipeline(cols, MODELO_FINAL),
            X_tr[cols],
            y_tr,
            cv=cv,
            scoring={"roc": "roc_auc"},
        )
        filas.append({"Escenario": nom, "AUC CV (logística)": round(r["test_roc"].mean(), 3)})
    return pd.DataFrame(filas)


def run_entrega_final(out_dir: Path) -> dict:
    np.random.seed(SEED)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, df, X_tr, X_te, y_tr, y_te = cargar_datos_y_particion()
    cols_b = ESCENARIOS[ESC_PRINCIPAL_KEY]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    tabla_cv, k_mejor = cv_cuatro_modelos(X_tr, y_tr, cols_b)
    tabla_cv.to_csv(out_dir / "tabla_cv_modelo_final.csv", index=False)

    tabla_cv_esc = cv_escenarios_abC(X_tr, y_tr)
    tabla_cv_esc.to_csv(out_dir / "tabla_cv_escenarios_abc.csv", index=False)

    pipe_oof = construir_pipeline(cols_b, MODELO_FINAL)
    proba_oof = cross_val_predict(pipe_oof, X_tr[cols_b], y_tr, cv=cv, method="predict_proba")[:, 1]

    t_youden, j_youden = umbral_youden(y_tr, proba_oof)
    t_c1, cost_c1 = umbral_costo(y_tr, proba_oof, 1.0)
    t_c2, cost_c2 = umbral_costo(y_tr, proba_oof, 2.0)
    t_c5, cost_c5 = umbral_costo(y_tr, proba_oof, 5.0)

    n_buenos = int((y_tr == 0).sum())
    baseline_fn, baseline_fp = 0, n_buenos
    baseline_cost = {1: baseline_fp, 2: baseline_fp, 5: baseline_fp}

    ts = np.linspace(0.05, 0.95, 181)
    filas_umbral = []
    for t in ts:
        m = metricas_en_umbral(y_tr, proba_oof, t)
        m["costo_ratio1"] = m["FN"] + m["FP"]
        m["costo_ratio2"] = 2 * m["FN"] + m["FP"]
        m["costo_ratio5"] = 5 * m["FN"] + m["FP"]
        m["J_youden"] = m["sensibilidad_TPR"] + m["especificidad_TNR"] - 1
        filas_umbral.append(m)
    df_umbral = pd.DataFrame(filas_umbral)
    df_umbral.to_csv(out_dir / "tabla_umbral_costos_oof.csv", index=False)

    def fila_costo(etiqueta, t, m):
        out = {**m, "escenario": etiqueta, "umbral": t}
        out["C_r1"] = m["FN"] + m["FP"]
        out["C_r2"] = 2 * m["FN"] + m["FP"]
        out["C_r5"] = 5 * m["FN"] + m["FP"]
        return out

    baseline_m = metricas_en_umbral(
        y_tr,
        np.ones_like(proba_oof),
        0.5,
    )
    baseline_m["FN"], baseline_m["FP"] = baseline_fn, baseline_fp

    destacados = [
        fila_costo("linea_base_siempre_mora", np.nan, baseline_m),
        fila_costo("convencion_0.5", 0.5, metricas_en_umbral(y_tr, proba_oof, 0.5)),
        fila_costo("youden_oof", t_youden, metricas_en_umbral(y_tr, proba_oof, t_youden)),
        fila_costo("costo_FN_1x_FP", t_c1, metricas_en_umbral(y_tr, proba_oof, t_c1)),
        fila_costo("costo_FN_2x_FP", t_c2, metricas_en_umbral(y_tr, proba_oof, t_c2)),
        fila_costo("costo_FN_5x_FP", t_c5, metricas_en_umbral(y_tr, proba_oof, t_c5)),
    ]
    df_dest = pd.DataFrame(destacados)
    df_dest.to_csv(out_dir / "tabla_umbrales_destacados_oof.csv", index=False)

    comite = df_dest[
        [
            "escenario",
            "umbral",
            "FN",
            "FP",
            "sensibilidad_TPR",
            "especificidad_TNR",
            "C_r1",
            "C_r2",
            "C_r5",
        ]
    ].copy()
    comite["linea_base_C_r1"] = baseline_cost[1]
    comite["linea_base_C_r2"] = baseline_cost[2]
    comite["linea_base_C_r5"] = baseline_cost[5]
    comite.to_csv(out_dir / "tabla_umbral_costos_comite_oof.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
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
    fig.savefig(out_dir / "fig_costo_vs_umbral.png", dpi=140)
    plt.close(fig)

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
    (out_dir / "umbrales_recomendados.json").write_text(
        json.dumps(umbrales, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    pipe_final = construir_pipeline(cols_b, MODELO_FINAL)
    pipe_final.fit(X_tr[cols_b], y_tr)
    joblib.dump(pipe_final, out_dir / "pipeline_logistica_B.joblib")

    import sklearn
    import imblearn

    config = {
        "modelo": "LogisticRegression(C=1e6, max_iter=5000)",
        "escenario": ESC_PRINCIPAL_KEY,
        "variables": cols_b,
        "balanceo": "ninguno",
        "semilla": SEED,
        "particion": "80/20 estratificada",
        "k_knn_referencia_cv": k_mejor,
        "scikit_learn_version": sklearn.__version__,
        "imbalanced_learn_version": imblearn.__version__,
    }
    (out_dir / "configuracion_definitiva.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame([config]).to_csv(out_dir / "tabla_configuracion_definitiva.csv", index=False)

    # Test solo lectura: CSV de 2.ª entrega + verificación opcional de AUC logística
    root_codigo = Path(__file__).resolve().parent.parent
    ref_test = None
    for candidato in (
        root_codigo / "salidas_segunda" / "tabla_test_principal.csv",
        root_codigo / "salidas" / "tabla_test_principal.csv",
    ):
        if candidato.exists():
            ref_test = candidato
            break
    if ref_test is not None:
        pd.read_csv(ref_test).to_csv(out_dir / "tabla_test_cierre_segunda_entrega.csv", index=False)
    proba_te = pipe_final.predict_proba(X_te[cols_b])[:, 1]
    auc_verify = roc_auc_score(y_te, proba_te)
    verif = pd.DataFrame(
        [
            {
                "AUC_reentrenado_mismo_train_test": round(auc_verify, 3),
                "AUC_informe_segunda_entrega": 0.722,
                "uso": "Verificación. No usar test para elegir modelo ni umbral.",
            }
        ]
    )
    verif.to_csv(out_dir / "tabla_test_verificacion_solo_lectura.csv", index=False)

  # Métricas en test con umbrales fijos (solo reporte, no optimización)
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
    pd.DataFrame(filas_te).to_csv(out_dir / "tabla_test_umbrales_solo_lectura.csv", index=False)

    resumen = pd.DataFrame(
        [
            {"clave": "n_train", "valor": len(y_tr)},
            {"clave": "n_test", "valor": len(y_te)},
            {"clave": "pct_mora", "valor": round(100 * y_tr.mean(), 1)},
            {"clave": "auc_cv_logistica", "valor": tabla_cv.loc[0, "ROC-AUC CV (media)"]},
        ]
    )
    resumen.to_csv(out_dir / "resumen_ejecucion.csv", index=False)

    return {
        "out_dir": str(out_dir),
        "t_youden": t_youden,
        "t_c1": t_c1,
        "t_c2": t_c2,
        "t_c5": t_c5,
        "auc_test_verify": auc_verify,
        "baseline_fp": baseline_fp,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    dirs = [
        root.parent / "salidas_tercera",
        root.parent.parent / "documentacion" / "tercera_entrega" / "entregables_para_informe",
    ]
    for d in dirs:
        info = run_entrega_final(d)
        print("Salidas en", info["out_dir"])
        print(
            "Youden", round(info["t_youden"], 3),
            "costo 2x", round(info["t_c2"], 3),
            "costo 5x", round(info["t_c5"], 3),
            "AUC test", round(info["auc_test_verify"], 3),
        )
