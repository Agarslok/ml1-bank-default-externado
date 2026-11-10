# Resumen para presentación (Isabella)

**Modelo final:** regresión logística, escenario B (11 variables), sin balanceo, semilla 42.

## Apertura comité

¿Podemos priorizar solicitantes por **riesgo relativo** sin rechazar de más a buenos pagadores? El **umbral** importa más que el AUC.

## Números clave

- AUC CV logística: **0,698** | AUC prueba (cierre 2.ª entrega): **0,722** (discriminación moderada, no 72 % aciertos).
- Prevalencia en la muestra **71 %** → el score **ordena**, no es probabilidad real en Perú.

## Tabla de costos OOF (mensaje para slide)

| Criterio | t | C (r=2) | ¿Mejor que línea base (1409)? |
|----------|---|---------|-------------------------------|
| Línea base siempre mora | — | 1409 | — |
| Costo r=1 (mín. OOF) | 0,57 | **1302** (C con r=1) | Sí (1302 vs 1409) |
| Youden | 0,71 | 2724 | No (pero equilibra recalls) |
| Costo r=2 | 0,27 | **1392** | Apenas sí |
| Costo r=5 | 0,14 | **1404** | Casi igual → como línea base |

**Conclusión oral:** con FN muy caro el corte bajo marca a casi todos y el binario parece la línea base. El valor está en **ordenar por score** y en umbrales intermedios (r=1, Youden).

## Umbrales en prueba (solo lectura)

Ver `tabla_test_umbrales_solo_lectura.csv`. No usaron test para elegir umbrales en Fase 3.

## Limitaciones (siempre visibles)

Temporalidad no demostrada | 71 % mora | score sin diccionario | test ya usado en 2.ª entrega.

## Streamlit

**Por qué (1 frase oral):** la guía §8.3 pide despliegue básico; Streamlit encaja con Python/sklearn, muestra el mismo pipeline en vivo y permite cambiar el umbral en la sustentación sin otro servidor.

Demo: mismo `pipeline_logistica_B.joblib`, 11 variables, umbrales en JSON. Código: `codigo/app_streamlit/`.

## Archivos en esta carpeta

Figuras ROC, PR, coeficientes, costo vs umbral, captura app, CSVs de umbrales y CV.
