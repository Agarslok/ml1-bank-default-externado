# Bitácora de decisiones — Segunda entrega

## 1. Problema
- Clasificación binaria: `mora` (1 = cliente marcado como moroso; 0 = pagador al día). Definición **operativa** porque la fuente no tiene diccionario ni umbral de días.
- Pregunta **predictiva:** capacidad de predecir riesgo con información previa (no causalidad).

## 2. Fuga y variables
- Momento de predicción: evaluación crediticia **antes** de conocer `mora`.
- Escenario principal **B:** se excluyen `atraso` y `clasif_sbs` por definición de comportamiento de pago (no solo por correlación).
- `score` se mantiene en B; escenario **C** mide sensibilidad al quitarlo.

## 3. Datos
- Duplicados exactos eliminados **antes** de partir (8.399 → 6.074).
- Faltantes: mediana + indicador, **dentro del pipeline**, solo train de cada fold.
- Partición 80/20 estratificada, semilla 42.

## 4. Modelos y métricas
- Línea base: mayoría (`mora=1`).
- Candidatos: regresión logística, KNN (`k` por GridSearchCV en train), Naive Bayes.
- Métricas: **ROC-AUC** (principal, alineada al artículo) y **PR-AUC** (sensible a prevalencia).
- No se usa “Precision > 70 %” como regla de calidad.

## 5. Pipeline
Imputar → escalar → one-hot → (balanceo opcional, solo fit) → modelo. `ImbPipeline` + `StratifiedKFold`.

## 6. Balanceo
Comparar sin balancear, peso de clases, sub/sobremuestreo y SMOTE en CV. Si ROC/PR no mejoran de forma estable, modelo principal **sin balancear**.

## 7. Umbral
Elegido con probabilidades **out-of-fold en entrenamiento**. F2 sobre `mora=1` y costo FN alto **colapsan hacia la línea base** (umbral ~0.14). Umbral de evaluación: **Youden (TPR−FPR)**. El conjunto de **prueba no se usa** para elegir umbral ni hiperparámetros.

## 8. Interpretación
Coeficientes = asociaciones en esta muestra (71 % mora ≠ prevalencia SBS 4–11 %). LDA/regularización quedan para la entrega final si el curso las cubre.
