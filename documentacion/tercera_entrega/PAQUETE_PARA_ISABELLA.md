# Paquete para Isabella (entrega final)

Juan — listado para compartir (repartición Fase 3). Todo está en el repo privado; rutas locales abajo.

**Checklist de envío**
- [x] `codigo/app_streamlit/app.py` (+ `paths.py`, `requirements.txt`, `README.md`)
- [x] `codigo/tercera/tercera_entrega_lib.py` (Streamlit / regenerar salidas; el notebook no lo importa)
- [x] `documentacion/tercera_entrega/contrato_decisiones.md` (§5 con **r ∈ {1, 2, 5}**)
- [x] `documentacion/tercera_entrega/entregables_para_informe/resumen_para_presentacion.md`
- [x] `codigo/tercera/tercera_entrega_modelamiento_final_ejecutado.ipynb` (con salidas)
- [x] `codigo/salidas_tercera/` completa → `tabla_test_umbrales_solo_lectura.csv`
- [ ] PDF nuevo desde Overleaf (`main.tex` actualizado) — pendiente compilar

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
