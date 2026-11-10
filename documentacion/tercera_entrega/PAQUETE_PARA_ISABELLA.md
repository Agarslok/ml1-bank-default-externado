# Paquete para Isabella (entrega final)

Juan — listado de archivos actualizados tras correcciones del review.

## Código y datos

| Archivo | Ruta |
|---------|------|
| Streamlit | `codigo/app_streamlit/app.py` (+ `paths.py`, `README.md`) |
| Librería Fase 3 | `codigo/tercera/tercera_entrega_lib.py` |
| Notebook (**autocontenido**, sin depender del `.py`) | `codigo/tercera/tercera_entrega_modelamiento_final.ipynb` |
| Notebook **con salidas** | `codigo/tercera/tercera_entrega_modelamiento_final_ejecutado.ipynb` (copia en `tercera_entrega/v1/`) |
| Salidas | `codigo/salidas_tercera/` (incluye `tabla_test_umbrales_solo_lectura.csv`) |
| Contrato | `documentacion/tercera_entrega/contrato_decisiones.md` |

## Informe y presentación

| Archivo | Ruta |
|---------|------|
| LaTeX informe final | `documentacion/tercera_entrega/main.tex` |
| Figuras + tablas + joblib | `documentacion/tercera_entrega/entregables_para_informe/` |
| Resumen oral | `documentacion/tercera_entrega/entregables_para_informe/resumen_para_presentacion.md` |

## Overleaf

Subir `main.tex` + carpeta `entregables_para_informe/` + `logo_uexternado.png` (opcional).

## Notebook ejecutado

Desde `codigo/`:

```powershell
jupyter nbconvert --to notebook --execute tercera/tercera_entrega_modelamiento_final.ipynb --output tercera/tercera_entrega_modelamiento_final_ejecutado.ipynb
```

## Captura Streamlit (informe)

Reemplazar `entregables_para_informe/captura_streamlit.png` con captura **clara** en vivo (pestaña Ingreso manual, Youden, botón Calcular riesgo), o:

```powershell
python codigo/app_streamlit/generar_captura_streamlit.py --estatica
```
