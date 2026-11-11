# Resumen para presentación (Isabella)

**Modelo final:** regresión logística, escenario B (11 variables), sin balanceo, semilla 42.

## Apertura comité

¿Podemos priorizar solicitantes por **riesgo relativo** sin rechazar de más a buenos pagadores? El **umbral** importa más que el AUC.

## Números clave

- AUC CV logística: **0,698** | AUC prueba (cierre 2.ª entrega): **0,722** (discriminación moderada, no 72 % aciertos).
- Prevalencia en la muestra **71 %** → el score **ordena**, no es probabilidad real en Perú.

## Tabla 6 — costos OOF (para slide; igual que el informe)

$C = r \cdot FN + FP$. Línea base: predecir siempre mora (FN = 0, FP = 1 409 buenos pagadores en OOF).

| Criterio | $t$ | FN | FP | $C$ (r=1) | $C$ (r=2) | $C$ (r=5) | Esp. |
|----------|-----|----|----|-----------|-----------|-----------|------|
| Línea base | — | 0 | 1409 | 1409 | 1409 | 1409 | 0,00 |
| Convención 0,50 | 0,50 | 244 | 1076 | 1320 | 1564 | 2296 | 0,24 |
| Costo **r=1** (mín. OOF) | 0,57 | 381 | 921 | **1302** | 1683 | 2826 | 0,35 |
| Youden OOF | 0,71 | 1090 | 544 | 1634 | 2724 | 5994 | 0,61 |
| Costo **r=2** (mín. OOF) | 0,27 | 13 | 1366 | 1379 | **1392** | 1431 | 0,03 |
| Costo **r=5** (mín. OOF) | 0,14 | 0 | 1404 | 1404 | 1404 | **1404** | 0,00 |

**Conclusión oral (comité):** El modelo solo **mejora** frente a rechazar a todos cuando los dos errores cuestan **parecido** (con $r=1$ el costo baja de 1409 a 1302, una reducción de **~7,6 %**). Si un moroso cuesta el **doble o más** que un falso rechazo ($r=2$, $r=5$), el corte óptimo en OOF **no aporta** frente a la línea base en costo simbólico.

**Youden:** En costo ($C$) Youden es **peor** que la línea base para $r=1$, $r=2$ y $r=5$. Sirve para **repartir** FN y FP (más buenos pagadores reconocidos, más morosos pasan), no para **ahorrar** costo frente a “rechazar a todos”. El valor estable del proyecto sigue siendo **ordenar por score** y discutir umbrales con el comité.

## Umbrales en prueba (solo lectura)

No se optimizó test en Fase 3. Detalle en `tabla_test_umbrales_solo_lectura.csv` dentro de `codigo/salidas_tercera/`.

## Limitaciones (siempre visibles)

Temporalidad no demostrada | 71 % mora | score sin diccionario | test ya usado en 2.ª entrega.

## Streamlit (demo local)

Carpeta `codigo/app_streamlit/`. Desde ahí: `streamlit run app.py` (ver `LEEME.md` del paquete de entrega).
