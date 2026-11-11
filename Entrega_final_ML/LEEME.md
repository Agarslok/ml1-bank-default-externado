# Entrega final — Machine Learning I (Fase 3)

Paquete de entrega del proyecto **Riesgo de mora (Perú)**. Mantén esta estructura de carpetas: la aplicación Streamlit (`paths.py`) busca los artefactos en las rutas indicadas abajo.

## Estructura

```
Entrega_final_ML/
├── LEEME.md                          ← este archivo
├── requirements.txt                  ← versiones para app y notebook
├── Entrega_final_ML_informe.pdf      ← compilar desde main.tex (Overleaf)
├── Diapositivas_entrega_final.pdf    ← Isabella
├── datos/
│   └── database.csv
├── codigo/
│   ├── tercera_entrega_modelamiento_final_ejecutado.ipynb
│   ├── tercera_entrega_lib.py
│   ├── app_streamlit/
│   │   ├── app.py
│   │   ├── paths.py
│   │   └── README.md
│   └── salidas_tercera/
│       ├── pipeline_logistica_B.joblib
│       ├── umbrales_recomendados.json
│       └── (tablas .csv y figuras .png)
└── documentacion/
    ├── contrato_decisiones.md        ← §8 firmado por ambos
    ├── resumen_para_presentacion.md
    └── guion_comite_riesgo.md        ← Isabella
```

Raíz del proyecto = carpeta que contiene `datos/` y `codigo/` (en el zip puede ser `Entrega_final_ML/`).

## Versiones de librerías

| Librería | Versión |
|----------|---------|
| scikit-learn | **1.9.1** |
| imbalanced-learn | **0.14.2** |
| Python | 3.12 (recomendado) |

Instalación:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Notebook (reproducir o verificar)

1. Coloca `database.csv` en `datos/`.
2. Abre `codigo/tercera_entrega_modelamiento_final_ejecutado.ipynb` (ya incluye salidas).
3. Para regenerar artefactos (opcional): `python codigo/tercera_entrega_lib.py` desde la raíz del proyecto.

## Aplicación Streamlit

**Importante:** ejecutar desde `codigo/app_streamlit/`:

```powershell
cd codigo\app_streamlit
streamlit run app.py
```

`paths.py` busca, en este orden:

- Pipeline: `codigo/salidas_tercera/pipeline_logistica_B.joblib`
- Umbrales: `codigo/salidas_tercera/umbrales_recomendados.json`
- Datos de ejemplo: `datos/database.csv`

Si falta el pipeline, ejecuta `python codigo/tercera_entrega_lib.py` desde la raíz.

## Contenido mínimo del zip

- [ ] PDF informe (`Entrega_final_ML_informe.pdf`)
- [ ] Diapositivas PDF
- [ ] Este `LEEME.md` y `requirements.txt`
- [ ] `datos/database.csv`
- [ ] Notebook ejecutado + `tercera_entrega_lib.py`
- [ ] `app_streamlit/` + `salidas_tercera/` completas
- [ ] `contrato_decisiones.md` firmado §8
- [ ] `resumen_para_presentacion.md`
