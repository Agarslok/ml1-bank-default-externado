# Demo Streamlit — riesgo de mora (entrega final)

Aplicación mínima del **modelo final** (regresión logística, escenario B). Carga el mismo `pipeline` exportado en el notebook de tercera entrega.

## Requisitos previos

Generar el pipeline (si aún no existe):

```powershell
python codigo\tercera\tercera_entrega_lib.py
```

o, si el script sigue en `codigo/`:

```powershell
python codigo\tercera_entrega_lib.py
```

Debe existir al menos uno de:

- `codigo/salidas_tercera/pipeline_logistica_B.joblib`
- `documentacion/tercera_entrega/entregables_para_informe/pipeline_logistica_B.joblib`

## Versiones de librerías (compatibilidad con `joblib`)

El `pipeline_logistica_B.joblib` se guardó y se probó con:

| Librería | Versión |
|----------|---------|
| **scikit-learn** | **1.9.1** |
| **imbalanced-learn** | **0.14.2** |

Si al ejecutar la app aparece un aviso de incompatibilidad de versiones, instale las mismas versiones antes de cargar el modelo (también en Colab):

```powershell
pip install scikit-learn==1.9.1 imbalanced-learn==0.14.2
```

La versión exacta queda registrada en `configuracion_definitiva.json` (`scikit_learn_version`, `imbalanced_learn_version`).

Si el `joblib` lo generaste en **Colab**, ejecuta allí `import sklearn; print(sklearn.__version__)` y usa **esa** misma versión al correr la app (puede diferir de 1.9.1 local).

## Instalación

Desde la carpeta del proyecto (recomendado usar un entorno virtual):

```powershell
pip install -r codigo\app_streamlit\requirements.txt
```

## Ejecutar

```powershell
streamlit run codigo\app_streamlit\app.py
```

El navegador abrirá la interfaz local (por defecto `http://localhost:8501`).

## Uso

1. Elija el **umbral operativo** en la barra lateral (0,50, Youden OOF, costo FN 1×/2×/5× FP).
2. **Ingreso manual:** complete el perfil del solicitante (11 variables del escenario B).
3. **Ejemplo desde la base:** seleccione un índice de `datos/database.csv` (muestra sin duplicados).

La salida muestra el **índice de riesgo relativo** del modelo y la clase según el umbral. Los textos de limitación están en el panel lateral.

## Coherencia con el entrenamiento

- Mismas columnas que escenario B (sin `atraso` ni `clasif_sbs`).
- Umbrales leídos de `umbrales_recomendados.json` (calculados en OOF, no en test).
- `database.csv` en `datos/` solo sirve como referencia de categorías y medias por defecto.

## Ejecutar la app (manual)

Desde la raíz del proyecto o desde `codigo/app_streamlit/`:

```powershell
pip install -r codigo\app_streamlit\requirements.txt
streamlit run codigo\app_streamlit\app.py
```

Se abre el navegador en `http://localhost:8501`. Use el panel lateral para cambiar el umbral.

## Captura para el informe (sin Print Screen)

**Opción 1 — Captura estática con Python** (rápida, sirve para el PDF si no quieres levantar el servidor):

```powershell
python codigo\app_streamlit\generar_captura_streamlit.py --estatica
```

Escribe `documentacion/tercera_entrega/entregables_para_informe/captura_streamlit.png`.

**Opción 2 — Captura real de la interfaz Streamlit** (requiere Playwright):

```powershell
pip install playwright streamlit
playwright install chromium
python codigo\app_streamlit\generar_captura_streamlit.py
```

El script arranca Streamlit en segundo plano, abre la página en Chromium headless y guarda la misma ruta `captura_streamlit.png`.

**Opción 3 — Manual:** con la app abierta, captura de pantalla y guardar en `entregables_para_informe/captura_streamlit.png`.
