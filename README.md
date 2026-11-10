# BankDefaultAnalysis — Machine Learning I (U. Externado)

**Equipo:** Juan Angel Triana, Isabella Pensamiento · **Profesora:** Juliana De Mier  
**Tema:** Riesgo de mora (Perú, Kaggle) · **Modelo final:** regresión logística, escenario B (11 variables), `SEED=42`.

Repositorio **privado** para compartir código, informe LaTeX, salidas y demo Streamlit de la **entrega final (Fase 3)**.

**Remoto:** https://github.com/Agarslok/ml1-bank-default-externado (invitar a colaboradores desde *Settings → Collaborators*).

## Mapa del proyecto

| Qué | Dónde |
|-----|--------|
| **Contrato de decisiones (modelo congelado)** | [`documentacion/tercera_entrega/contrato_decisiones.md`](documentacion/tercera_entrega/contrato_decisiones.md) |
| **Informe LaTeX final** | [`documentacion/tercera_entrega/main.tex`](documentacion/tercera_entrega/main.tex) + [`entregables_para_informe/`](documentacion/tercera_entrega/entregables_para_informe/) |
| **Paquete para presentación (Isabella)** | [`documentacion/tercera_entrega/PAQUETE_PARA_ISABELLA.md`](documentacion/tercera_entrega/PAQUETE_PARA_ISABELLA.md) |
| **Pipeline y umbrales** | `python codigo/tercera/tercera_entrega_lib.py` → `codigo/salidas_tercera/` |
| **Notebook Fase 3 (autocontenido, estilo Colab)** | [`codigo/tercera/tercera_entrega_modelamiento_final.ipynb`](codigo/tercera/tercera_entrega_modelamiento_final.ipynb) — no importa el `.py` |
| **Notebook con salidas** | [`codigo/tercera/tercera_entrega_modelamiento_final_ejecutado.ipynb`](codigo/tercera/tercera_entrega_modelamiento_final_ejecutado.ipynb) |
| **App Streamlit** | [`codigo/app_streamlit/`](codigo/app_streamlit/) — ver README allí |
| **2.ª entrega (referencia)** | [`segunda_entrega/`](segunda_entrega/) |
| **Datos** | [`datos/database.csv`](datos/database.csv) |

## Streamlit (demo oral)

```powershell
pip install -r codigo/app_streamlit/requirements.txt
streamlit run codigo/app_streamlit/app.py
```

Requisito: existir `pipeline_logistica_B.joblib` y `umbrales_recomendados.json` (generar con `tercera_entrega_lib.py`).

## Reproducir números Fase 3

```powershell
python codigo/tercera/tercera_entrega_lib.py
```

Métricas de referencia: CV AUC logística ≈ **0.698**, test AUC ≈ **0.722**; Youden OOF **t ≈ 0.71**; costos OOF **t ≈ 0.57 (r=1)**, **0.27 (r=2)**, **0.135 (r=5)**.

## Checklist de auditoría (entrega)

- [x] Modelo congelado documentado en contrato
- [x] Umbrales solo con OOF; test en tablas *solo lectura*
- [x] Tabla de costos con línea base y **r ∈ {1, 2, 5}**
- [x] Streamlit: mismo pipeline, 11 vars, umbrales JSON, README
- [x] Informe: limitaciones, interpretación banco, **por qué Streamlit** (§ despliegue)
- [x] Notebook ejecutado con salidas
- [ ] PDF compilado en Overleaf (local sin `pdflatex` completo)
- [ ] Firma Isabella en contrato §8
- [ ] Captura Streamlit en vivo (opcional, sustituye PNG estático)

## Overleaf

Subir `documentacion/tercera_entrega/main.tex` y la carpeta `entregables_para_informe/` (figuras, joblib no hace falta en PDF; sí las PNG).
