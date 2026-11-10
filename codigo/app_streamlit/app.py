"""
Priorización de riesgo de mora — modelo final (regresión logística, escenario B).
No sustituye decisión humana del comité de riesgo.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from paths import CANDIDATOS_DATOS, CANDIDATOS_PIPELINE, CANDIDATOS_UMBRALES, COLS_B, first_existing

st.set_page_config(
    page_title="Riesgo de mora — demo",
    page_icon="📊",
    layout="wide",
)

DISCLAIMER = """
**Limitaciones (leer antes de usar)**

- El score es un **índice de riesgo relativo** del modelo para **ordenar** solicitudes en una muestra
  con características similares a la del proyecto. **No** es la probabilidad real de mora en el Perú.
- En los datos de entrenamiento ~**71 %** de clientes están en mora. La prevalencia reportada por la SBS
  en el país es mucho menor (aprox. 4–11 %).
- No contamos con fechas ni diccionario oficial. No demostramos predicción de un evento **futuro**.
- El `score` del solicitante puede no estar verificado en el momento exacto de la mora.
- Umbrales elegidos con validación **out-of-fold en entrenamiento**, no optimizados en la muestra de prueba.
"""


@st.cache_resource
def load_artifacts():
    pipe_path = first_existing(CANDIDATOS_PIPELINE)
    umb_path = first_existing(CANDIDATOS_UMBRALES)
    pipeline = joblib.load(pipe_path)
    umbrales = json.loads(umb_path.read_text(encoding="utf-8"))
    return pipeline, umbrales, pipe_path, umb_path


@st.cache_data
def load_reference_data():
    data_path = first_existing(CANDIDATOS_DATOS)
    df = pd.read_csv(data_path).drop_duplicates().reset_index(drop=True)
    num = [c for c in COLS_B if c not in ("vivienda", "zona", "nivel_educ")]
    medians = df[num].apply(pd.to_numeric, errors="coerce").median().to_dict()
    cats = {
        "vivienda": sorted(df["vivienda"].dropna().astype(str).unique()),
        "zona": sorted(df["zona"].dropna().astype(str).unique()),
        "nivel_educ": sorted(df["nivel_educ"].dropna().astype(str).unique()),
    }
    return df, medians, cats, data_path


def umbrales_opciones(umb: dict) -> dict[str, tuple[float, str]]:
    return {
        "Convención (0,50)": (
            umb["t_convencion"],
            "Corte habitual. En esta base prioriza detectar morosos y rechaza muchos buenos pagadores.",
        ),
        f"Youden OOF (≈ {umb['t_youden_oof']:.2f})": (
            umb["t_youden_oof"],
            "Equilibrio sensibilidad–especificidad estimado solo en entrenamiento (OOF).",
        ),
        f"Costo FN = 1× FP (≈ {umb.get('t_costo_fn_1x_fp_oof', 0.57):.2f})": (
            umb.get("t_costo_fn_1x_fp_oof", 0.57),
            "Pesos iguales a FN y FP en OOF (escenario simbólico).",
        ),
        f"Costo FN = 2× FP (≈ {umb['t_costo_fn_2x_fp_oof']:.2f})": (
            umb["t_costo_fn_2x_fp_oof"],
            "Si dejar pasar un moroso cuesta el doble que rechazar un buen pagador (escenario simbólico).",
        ),
        f"Costo FN = 5× FP (≈ {umb['t_costo_fn_5x_fp_oof']:.2f})": (
            umb["t_costo_fn_5x_fp_oof"],
            "Si el costo de un falso negativo es muy alto (escenario simbólico).",
        ),
    }


def row_to_frame(row: dict) -> pd.DataFrame:
    return pd.DataFrame([{c: row[c] for c in COLS_B}])


def main():
    st.title("Priorización de riesgo de mora")
    st.caption("Universidad Externado · Machine Learning I · Modelo final: regresión logística, escenario B")

    try:
        pipeline, umbrales, pipe_path, umb_path = load_artifacts()
        df_ref, medians, cats, data_path = load_reference_data()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info(
            "Ejecute primero `python codigo/tercera/tercera_entrega_lib.py` "
            "o `python codigo/tercera_entrega_lib.py` para generar el pipeline."
        )
        sys.exit(1)

    with st.sidebar:
        st.header("Criterio de corte")
        opciones = umbrales_opciones(umbrales)
        etiqueta = st.selectbox("Umbral operativo", list(opciones.keys()))
        t, ayuda_umbral = opciones[etiqueta]
        st.caption(ayuda_umbral)
        st.divider()
        st.markdown(DISCLAIMER)
        with st.expander("Artefactos cargados"):
            st.write("Pipeline:", pipe_path.name)
            st.write("Umbrales:", umb_path.name)
            st.write("Referencia datos:", data_path.name)

    tab_manual, tab_ejemplo = st.tabs(["Ingreso manual", "Ejemplo desde la base"])

    with tab_ejemplo:
        st.write(
            "Elija un cliente de la muestra (sin duplicados) para ver el score del modelo con los mismos "
            "preprocesamientos que en el entrenamiento."
        )
        idx = st.number_input("Índice de fila", min_value=0, max_value=len(df_ref) - 1, value=0, step=1)
        fila = df_ref.iloc[int(idx)]
        st.dataframe(fila[COLS_B], use_container_width=True)
        if st.button("Predecir este cliente", type="primary", key="pred_ejemplo"):
            X = row_to_frame(fila[COLS_B].to_dict())
            proba = float(pipeline.predict_proba(X)[0, 1])
            pred = int(proba >= t)
            mostrar_resultado(proba, pred, t, etiqueta)

    with tab_manual:
        st.subheader("Perfil del solicitante (escenario B)")
        c1, c2, c3 = st.columns(3)
        row = {}
        with c1:
            row["edad"] = st.number_input("Edad", min_value=18, max_value=100, value=int(medians["edad"]))
            row["dias_lab"] = st.number_input(
                "Días laborando", min_value=0, value=int(medians["dias_lab"]), step=1
            )
            row["exp_sf"] = st.number_input("Experiencia SF", min_value=0, value=int(medians["exp_sf"]))
            row["nivel_ahorro"] = st.number_input(
                "Nivel de ahorro", min_value=0, value=int(medians["nivel_ahorro"])
            )
        with c2:
            row["ingreso"] = st.number_input("Ingreso", min_value=0.0, value=float(medians["ingreso"]))
            row["linea_sf"] = st.number_input(
                "Línea SF", min_value=0.0, value=float(medians["linea_sf"])
            )
            row["deuda_sf"] = st.number_input(
                "Deuda SF", min_value=0.0, value=float(medians["deuda_sf"])
            )
            row["score"] = st.number_input("Score crediticio", min_value=0.0, value=float(medians["score"]))
        with c3:
            row["vivienda"] = st.selectbox("Vivienda", cats["vivienda"], index=0)
            row["zona"] = st.selectbox("Zona", cats["zona"], index=cats["zona"].index("Lima") if "Lima" in cats["zona"] else 0)
            row["nivel_educ"] = st.selectbox("Nivel educativo", cats["nivel_educ"], index=0)

        if st.button("Calcular riesgo", type="primary", key="pred_manual"):
            X = row_to_frame(row)
            proba = float(pipeline.predict_proba(X)[0, 1])
            pred = int(proba >= t)
            mostrar_resultado(proba, pred, t, etiqueta)


def mostrar_resultado(proba: float, pred: int, t: float, etiqueta: str):
    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Índice de riesgo (modelo)", f"{proba:.3f}")
        st.progress(min(max(proba, 0.0), 1.0))
        st.caption("Valor de `predict_proba` para la clase mora=1. Sirve para **ordenar** clientes.")
    with col_b:
        etiqueta_clase = "Mayor riesgo relativo (mora predicha)" if pred == 1 else "Menor riesgo relativo en este corte"
        st.metric("Decisión con umbral", etiqueta_clase)
        st.write(f"Umbral aplicado: **{t:.3f}** ({etiqueta})")

    if pred == 1:
        st.warning(
            "Con este umbral el modelo ubicaría al cliente en el grupo de **mayor riesgo relativo**. "
            "Revise costos de falsos positivos y negativos con el comité."
        )
    else:
        st.success(
            "Con este umbral el modelo no marcaría mora elevada. "
            "Un umbral más bajo (p. ej. costo FN 5×) sería más conservador frente a morosos."
        )

    st.info(
        "Comparar umbrales en el panel izquierdo (0,50, Youden, 2× y 5×) para ver cómo cambia la decisión "
        "sin reinterpretar el score como probabilidad del mercado."
    )


if __name__ == "__main__":
    main()
