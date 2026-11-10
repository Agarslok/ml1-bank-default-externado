# Contrato de decisiones — Entrega final (Fase 3)

**Proyecto:** Riesgo de incumplimiento crediticio (Perú) — Machine Learning I  
**Integrantes:** Isabella Pensamiento, Juan Angel Triana  
**Profesora:** Juliana De Mier  
**Fecha de congelamiento:** 9 de noviembre de 2026 (técnica — Juan). Pendiente firma Isabella en §8.  
**Referencias entregadas (2.ª entrega):** `segunda_entrega/Ml_segunda_entrega_final.pdf`, `segunda_entrega/Diapositivas Riesgo de mora Segunda entrega.pdf`, `segunda_entrega/segunda_entrega_modelamiento_final.ipynb`

Este documento fija las decisiones que **no cambiamos** en la entrega final salvo acuerdo escrito de los dos integrantes y alineación con la retroalimentación de la tutora (`feed_back.txt`) y la guía §8 (`tercera_entrega_limeamientos.txt`).

---

## 1. Pregunta y decisión de negocio

**Pregunta técnica (predictiva):** ¿Qué tan bien puede predecirse el riesgo de mora de un cliente de crédito de consumo a partir de información financiera y socioeconómica disponible **previamente**?

**Decisión de negocio (comité de riesgo):** ¿Podemos usar la información disponible de un solicitante para **priorizar clientes por riesgo** sin rechazar innecesariamente demasiados buenos pagadores? El resultado operativo depende del **umbral**, no solo del AUC.

**Definición operativa de `mora` (sin cambio):** `mora = 1` significa cliente marcado como moroso en la base. `mora = 0` significa pagador al día. No afirmamos mora futura ni un periodo exacto. Sin diccionario ni fechas esto sigue siendo un supuesto explícito.

---

## 2. Modelo final propuesto (congelado)

| Ítem | Decisión |
|------|----------|
| **Algoritmo final** | Regresión logística |
| **Escenario de variables** | **B — Principal** (11 predictores, sin `atraso` ni `clasif_sbs`) |
| **Balanceo en entrenamiento** | **Sin balancear** (cinco opciones probadas en CV, ROC-AUC y PR-AUC casi iguales) |
| **Semilla** | `42` (`SEED`) en partición, CV, KNN y muestreos |
| **Hiperparámetros logística** | `LogisticRegression(C=1e6, max_iter=5000)` sin `class_weight` en el modelo principal |
| **Métrica principal de discriminación** | ROC-AUC (complemento: PR-AUC) |
| **Salida del modelo para operación** | Score de `predict_proba[:, 1]` como **ordenamiento de riesgo relativo** en una muestra con características similares. **No** como probabilidad real de mora en el mercado peruano |

### Variables del escenario B (11)

Numéricas: `edad`, `dias_lab`, `exp_sf`, `nivel_ahorro`, `ingreso`, `linea_sf`, `deuda_sf`, `score`.  
Categóricas: `vivienda`, `zona`, `nivel_educ`.

**Excluidas en producción conceptual:** `atraso`, `clasif_sbs` (comportamiento de pago / clasificación posterior posible).  
**Sensibilidad:** escenario **C** (sin `score`) solo como análisis de robustez en informe, no como modelo final.

### Pipeline (orden fijo)

1. Imputación mediana + indicador de faltante (solo datos de entrenamiento de cada fold).  
2. Estandarización de numéricas.  
3. One-hot (`drop='first'`, agrupación de zonas poco frecuentes según notebook).  
4. Sin paso de balanceo en el modelo final.  
5. Regresión logística.

Implementación de referencia: funciones `construir_pipeline` / `ImbPipeline` en `segunda_entrega_modelamiento_final.ipynb`.

---

## 3. Candidatos congelados (comparación histórica)

En la 2.ª entrega comparamos, en escenario B:

- Línea base (siempre `mora = 1`)
- Regresión logística (**final**)
- LDA (comparador lineal)
- KNN (`k = 151` por CV, solo histórico)
- Naive Bayes (contraste por supuesto de independencia)

**No incorporamos en la entrega final** QDA, Ridge ni Lasso para competir de nuevo por AUC en prueba. La tutora indicó que la comparación ya es suficientemente rica y que la fase 3 debe **cerrar y defender** la solución, no buscar ganadores marginales con el mismo test.

**Justificación resumida de la logística frente a LDA/KNN (criterio de cierre, no nuevo experimento):**

- CV: logística ≈ 0,698 ± 0,017, LDA ≈ 0,699 ± 0,017, KNN ≈ 0,689, Naive Bayes ≈ 0,657.  
- Diferencias entre logística, LDA y KNN menores que la variación de CV e intervalos bootstrap en prueba.  
- Elegimos logística por **interpretabilidad** (coeficientes / logit), alineación con el artículo *Credit/GoodBad*, y comunicación al comité.

---

## 4. Política del conjunto de prueba (20 %)

| Regla | Detalle |
|-------|---------|
| **Partición** | 80 % entrenamiento / 20 % prueba, estratificada por `mora`, **después** de eliminar 2.325 duplicados exactos (6.074 clientes → 4.859 train / 1.215 test) |
| **Semilla** | `42`, no se re-sortea |
| **Uso en 2.ª entrega (ya ocurrido)** | Comparación de cuatro modelos, tablas de prueba, curvas ROC/PR, sensibilidad A/B/C en test, bootstrap de AUC, matrices con umbral 0,5 y Youden **aplicado** en test para **reporte** |
| **Uso en entrega final** | **No** entrenar modelos nuevos ni elegir entre alternativas usando test. **Sí** repetir en informe las métricas de **cierre** del modelo congelado (mismos números de la 2.ª entrega), etiquetadas como evaluación única ya realizada |
| **Selección de variables / modelo** | Escenario B y logística se defendieron con CV, prudencia de fuga y criterios de negocio. La tabla A/B/C en test es **evidencia descriptiva publicada**, no base para nuevas decisiones |
| **Hiperparámetros** | `k` de KNN y comparaciones de balanceo solo con CV en train. Umbral Youden ≈ **0,71** estimado con **OOF en train**, no optimizando en test |

### Métricas de cierre en test (referencia congelada — regresión logística, escenario B)

Fuente: `codigo/salidas/tabla_test_principal.csv` / informe 2.ª entrega.

| Métrica | Valor |
|---------|-------|
| AUC | 0,722 |
| PR-AUC | 0,840 |
| Umbral 0,5: recall mora=1 | 0,942 |
| Umbral 0,5: recall mora=0 | 0,247 |
| Umbral 0,5: FN / FP | 50 / 265 |
| Youden OOF (t ≈ 0,71): recall mora=0 | 0,634 |
| Youden OOF: FN / FP | 268 / 129 |

Cualquier número distinto en la entrega final debe explicarse por error de transcripción, no por re-entrenamiento distinto sin acuerdo.

---

## 5. Umbral operativo (entrega final)

| Regla | Detalle |
|-------|---------|
| **Dónde se elige** | Solo con probabilidades **out-of-fold del entrenamiento** (misma logística, escenario B, CV 5 folds, semilla 42) |
| **Youden (referencia)** | Maximizar \(J = \mathrm{TPR} - \mathrm{FPR}\) → umbral ≈ **0,71** (ya calculado en 2.ª entrega) |
| **Negocio (nuevo en Fase 3)** | Escenarios de costo simbólico: ratio costo(FN)/costo(FP) ∈ **{2, 5}** (y opcional referencia Youden). **No** buscar el umbral que minimiza costo en test |
| **Lectura** | El umbral correcto depende del costo relativo de dejar pasar un moroso frente a rechazar un buen pagador. La base no trae costos monetarios reales |

---

## 6. Lo que haremos en Fase 3 (alcance)

**Juan (técnico):** notebook `tercera_entrega`, costos OOF, `joblib`, Streamlit, informe LaTeX final, carpeta de entregables para presentación.

**Isabella (comunicación):** revisar lo anterior, diapositivas para comité sin perfil de datos, guion oral, empaque de entrega.

**Fuera de alcance explícito:** XGBoost, limpieza de 541 variables del artículo, QDA/Ridge/Lasso como nueva competición en test, calibración poblacional sin muestra representativa.

**Dentro de alcance:** contraste con objetivos del artículo, despliegue Streamlit (guía §8.3), escenarios de umbral, limitaciones visibles (temporalidad, 71 % mora, `score` no verificado), demo en sustentación.

---

## 7. Compromisos de comunicación (evitar errores de la 2.ª entrega)

- No decir que AUC 0,72 = 72 % de aciertos. Es discriminación **moderada**, mejor que azar, lejos de separación perfecta.  
- No vender `predict_proba` como “probabilidad real de mora en Perú”.  
- No generalizar que “el balanceo solo mueve el corte”. En **nuestro** CV el efecto en ROC/PR fue mínimo.  
- No prometer “entrega final = QDA, Ridge, Lasso” en slides ni informe.

---

## 8. Aprobación en reunión corta

Completar al cerrar la reunión Etapa 0 (≈ 45 min).

| Integrante | ¿Aprueba este contrato? | Fecha | Firma / inicial |
|------------|-------------------------|-------|-----------------|
| Juan Angel Triana | | | |
| Isabella Pensamiento | | | |

**Guion 1 minuto (cada uno):**

- **Juan:** Por qué no entran más modelos y qué papel tiene el test a partir de ahora.  
- **Isabella:** Cómo traducimos esto al comité (priorización, umbral, limitaciones).

**Notas de la reunión:**

_(Espacio libre: cambios acordados, dudas de Isabella sobre números, fecha objetivo de handoff para diapositivas.)_

---

## 9. Trazabilidad

| Documento | Rol |
|-----------|-----|
| Este archivo | Fuente de verdad para Fase 3 |
| `documentacion/segunda_entrega/bitacora_decisiones.md` | Historial 2.ª entrega (algunas líneas quedan superseded por §3 y §6 aquí) |
| `documentacion/tercera_entrega/feed_back.txt` | Retroalimentación tutora |
| `documentacion/tercera_entrega/tercera_entrega_limeamientos.txt` | Guía oficial §8 |
