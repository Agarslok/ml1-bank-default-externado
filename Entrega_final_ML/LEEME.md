# Entrega final — Machine Learning I (Fase 3)

Integrantes: Isabella Pensamiento, Juan Angel Triana  
Proyecto: riesgo de mora (Perú), regresión logística escenario B.

Mantén esta estructura: `paths.py` de Streamlit busca archivos en las rutas de abajo.

## Estructura

```
Entrega_final_ML/
├── LEEME.md
├── requirements.txt
├── Entrega_final_ML_informe.pdf
├── Diapositivas_entrega_final.pdf
├── datos/
│   └── database.csv
├── codigo/
│   ├── tercera_entrega_modelamiento_final_ejecutado.ipynb
│   ├── app_streamlit/
│   │   ├── app.py
│   │   └── paths.py
│   └── salidas_tercera/
│       ├── pipeline_logistica_B.joblib
│       └── umbrales_recomendados.json
```

Raíz del paquete = carpeta que contiene `datos/` y `codigo/`.

## Dependencias

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Versiones importantes al cargar el `.joblib`: **scikit-learn 1.9.1**, **imbalanced-learn 0.14.2**.

## Notebook

1. Coloca `database.csv` en `datos/`.
2. Abre `codigo/tercera_entrega_modelamiento_final_ejecutado.ipynb` (incluye salidas).
3. El cuaderno es autocontenido: al ejecutarlo de punta a punta vuelve a generar `codigo/salidas_tercera/` (incluido el pipeline de la demo).

## Streamlit

Desde la raíz del paquete:

```powershell
pip install -r requirements.txt
cd codigo\app_streamlit
streamlit run app.py
```

La app lee `codigo/salidas_tercera/pipeline_logistica_B.joblib`, `umbrales_recomendados.json` y `datos/database.csv`.

Si falta el pipeline, ejecuta el notebook hasta el final o copia de nuevo los archivos de `salidas_tercera/`.

## Publicar en internet (enlace para cualquiera)

Opción recomendada: **Streamlit Community Cloud** (gratis).

1. Sube el proyecto a **GitHub** (repositorio público). No subas `.venv/`. Deben ir `datos/database.csv`, `codigo/salidas_tercera/*` y `codigo/app_streamlit/`.
2. Entra en [share.streamlit.io](https://share.streamlit.io), inicia sesión con GitHub y **Create app**.
3. Repo, rama `main`, archivo principal: `codigo/app_streamlit/app.py`.
4. Deploy. Obtendrás un enlace del tipo `https://nombre-app.streamlit.app`.

**Si el repo en GitHub tiene la carpeta `Entrega_final_ML/` dentro** (no es la raíz del repo), Streamlit **no** lee
`Entrega_final_ML/requirements.txt` y verás errores como `ModuleNotFoundError: joblib`.
Solución: crea `requirements.txt` en la **raíz del repositorio** (véase `requirements-raiz-github.txt`) y vuelve a desplegar.

Para Jupyter en local: `pip install -r requirements-dev.txt` además de `requirements.txt`.
