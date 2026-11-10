# Rúbrica de evaluación — Segunda entrega (orientación del equipo)

Pesos para autoevaluación y alineación del informe/notebook (complementa la guía oficial del proyecto integrador, sección 7.5).

| Criterio | Peso | Evidencia esperada |
|----------|------|-------------------|
| **Definición y auditoría de fuga** | 20 % | Definición operativa de `mora`; momento de predicción; tabla de variables; escenarios A/B/C; exclusión justificada de `atraso` y `clasif_sbs`. |
| **Pipeline y reproducibilidad** | 20 % | `Pipeline`/`ImbPipeline`, semilla fija, orden imputar → escalar → OHE → (balanceo) → modelo; sin ajustar con datos de prueba. |
| **Balanceo de clases** | 15 % | Comparación empírica (sin balancear, pesos, sub/sobremuestreo, SMOTE); decisión argumentada con ROC-AUC y PR-AUC. |
| **Diseño de validación** | 20 % | 80/20 estratificado; CV estratificada k=5; hiperparámetros (k KNN) dentro del pipeline; **umbral solo con OOF en train**; test una vez. |
| **Análisis y evidencia** | 15 % | Línea base; tablas y curvas ROC/PR; matrices de confusión; sensibilidad; limitaciones (muestra 71 %, sin diccionario). |
| **Sustentación** | 10 % | Informe 3–5 páginas + oral; todo el equipo puede explicar decisiones. |

**Nota:** El archivo anterior en esta ruta contenía por error la plantilla de bitácora; usar `bitacora_decisiones.md` para decisiones y este archivo para la rúbrica ponderada.
