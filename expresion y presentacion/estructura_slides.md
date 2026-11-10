# Expresión y presentación — Segunda entrega (Fase 2)

Este directorio es el **guion de la sustentación oral** (10 % de la rúbrica). No reemplaza el informe PDF ni el notebook: resume lo que deben **decir en voz alta** y qué **mostrar en pantalla** para que coincida con `documentacion/main.tex` y `codigo/segunda_entrega_modelamiento_final.ipynb`.

**Cómo usarlo con otra IA (Gamma, Canva, Beautiful.ai, ChatGPT, etc.)**

1. Adjunta o pega este archivo completo.
2. Adjunta el PDF del informe **o** las tres figuras (ver abajo).
3. Copia la sección **Prompt listo para la IA de diapositivas** al final.
4. Revisa tú mismo: números, nombres de modelos, escenario B, que no aparezcan frases genéricas tipo “en conclusión, la IA transformará…”.

**Figuras que debes subir manualmente a la herramienta de slides** (no las inventes):

| Archivo | Uso en la presentación |
|---------|-------------------------|
| `documentacion/fig_roc.png` o `codigo/salidas/fig_roc.png` | Curvas ROC en prueba |
| `documentacion/fig_pr.png` o `codigo/salidas/fig_pr.png` | Curvas precision–recall |
| `documentacion/fig_coeficientes.png` | Coeficientes logística (escenario B) |
| `codigo/salidas/fig_knn_k.png` | (Opcional) Elección de k en KNN |

**Estilo visual sugerido**

- Paleta sobria: blanco o gris claro de fondo, acento verde oscuro o azul marino (tono universidad), texto negro.
- Una idea fuerte por diapositiva. Pocas viñetas (máximo 4–5).
- Títulos con verbo o resultado (“Auditamos fuga de datos”, no “Fuga de datos”).
- Evitar iconos de robot, cohetes, “100 % accuracy”, gradientes neón, plantillas “startup”.
- Hablar en **primera persona del plural**: buscamos, entrenamos, encontramos.

**Duración orientativa:** 12–15 minutos + preguntas. ~1 min por diapositiva de contenido, ~2 min en resultados y ~2 min en fuga + pipeline.

---

## Reparto sugerido entre integrantes

| Bloque | Diapositivas | Quién puede llevarlo |
|--------|--------------|----------------------|
| Apertura y retroalimentación | 1–2 | Integrante 1 |
| Datos, fuga, pipeline | 3–5 | Integrante 2 |
| Modelos y balanceo | 6–7 | Integrante 1 o 2 |
| Resultados, umbral, escenarios | 8–10 | Integrante 2 |
| Interpretación, límites, cierre | 11–12 | Ambos (alternar) |

En la sustentación el profesor puede preguntar a **cualquiera** por qué excluimos `atraso`, por qué no balanceamos, o cómo elegimos el umbral. Cada uno debe poder responder eso en 30 segundos.

---

## Diapositiva 1 — Portada

**En pantalla**

- Título: *Riesgo de incumplimiento crediticio — Regresión logística en contexto latinoamericano*
- Segunda entrega — Proyecto integrador — Machine Learning I
- Isabella Pensamiento, Juan Angel Triana
- Universidad Externado de Colombia — Ciencia de datos — Octubre 2026

**Qué decir (≈20 s)**

- Presentamos la segunda entrega: modelamiento con cuatro algoritmos del curso, auditoría de fuga y validación estricta.

---

## Diapositiva 2 — Problema y enlace con la primera entrega

**En pantalla**

- **Objetivo:** predecir `mora` (1 = moroso, 0 = al día) en crédito de consumo — Perú, dataset tipo SBS/Kaggle (`BankDefaultAnalysis`).
- **Datos:** 8.399 filas → 2.325 duplicadas eliminadas → **6.074** clientes únicos.
- **Prevalencia:** ~**71 %** mora en esta muestra (no es la tasa real del país).
- **Retroalimentación entrega 1:** revisamos **momento de predicción** y **fuga** (`atraso`, `clasif_sbs`), pipeline reproducible, más modelos del curso (LDA, KNN, Naive Bayes).

**Qué decir (≈1 min)**

- Retomamos el artículo de referencia (modelo Credit/GoodBad) pero con datos peruanos y sin replicar XGBoost ni la limpieza de 541 variables del paper.
- La primera entrega nos señaló que algunos predictores podían ser otra cara del mismo fenómeno. En esta fase lo auditamos con escenarios A, B y C.

---

## Diapositiva 3 — Definición operativa y auditoría de fuga

**En pantalla**

- Pregunta guía: *¿Qué sabe el banco **antes** de conceder o renovar el crédito?*
- **Escenario A (13 vars):** incluye `atraso` y `clasif_sbs` → riesgo de fuga.
- **Escenario B (11 vars) — principal:** sin `atraso` ni `clasif_sbs`.
- **Escenario C (10 vars):** B sin `score` (sensibilidad).
- Mini tabla mental: subir AUC al incluir sospechosas ≈ **+0,007 a +0,012** en prueba, menos que la variación de la CV → nos quedamos en **B**.

**Qué decir (≈1,5 min)**

- `atraso` y clasificación SBS describen comportamiento de pago muy cercano al default. Las usamos solo para demostrar el efecto, no para el modelo que defenderíamos operativamente.
- Este bloque pesa fuerte en la rúbrica (definición y auditoría de fuga).

---

## Diapositiva 4 — Preprocesamiento dentro del pipeline

**En pantalla**

- Diagrama simple (cajas en fila): **Imputación mediana + indicador de faltante → Estandarización → One-Hot (`drop='first'`, `min_frequency=50`) → (opcional balanceo) → Modelo**
- Split **80/20 estratificado**, semilla **42**.
- Regla: **nada se ajusta con datos de prueba** (mediana, escalado, OHE, k de KNN, umbral).

**Qué decir (≈1 min)**

- Todo va en `Pipeline` / `ImbPipeline` para que validación cruzada y prueba sean honestas.
- En el notebook verificamos que mezclar train/test o imputar con toda la base hunde el AUC a ~0,49.

---

## Diapositiva 5 — Balanceo de clases

**En pantalla**

- Probamos **5 opciones** (sin balancear, pesos, sub/sobremuestreo, SMOTE) **solo en train**.
- **Resultado:** ROC-AUC y PR-AUC en CV **casi iguales** → entrenamos **sin balancear**.
- Línea base (clase mayoritaria): AUC **0,500**, recall mora=1 = **1,0**, precision **0,710**.

**Qué decir (≈45 s)**

- El desbalance no mejoró discriminación en AUC. Para tipos de error usamos después umbral Youden o peso de clases, no SMOTE por defecto.

---

## Diapositiva 6 — Modelos entrenados (escenario B)

**En pantalla**

| Modelo | Rol en el proyecto |
|--------|-------------------|
| **Regresión logística** | Principal — comparación con artículo, coeficientes |
| **LDA** | Comparador lineal (supuestos gaussianos) |
| **KNN** | No paramétrico — k elegido por CV (**k = 151**) |
| **Naive Bayes** | **Contraste** — supuesto de independencia (esperamos que sufra) |
| *(No incluidos)* | XGBoost, limpieza 541 vars — fuera del alcance del curso |

**Qué decir (≈1 min)**

- No buscamos “el algoritmo ganador absoluto”, sino entender qué aporta cada familia bajo las mismas reglas de validación.
- Naive Bayes está para mostrar que romper correlaciones entre variables financieras penaliza el AUC.

---

## Diapositiva 7 — Validación cruzada (5 folds)

**En pantalla**

- Tabla resumida:

| Modelo | AUC CV (media ± desv.) | PR-AUC CV |
|--------|------------------------|-----------|
| Regresión logística | 0,698 ± 0,017 | 0,835 |
| LDA | 0,699 ± 0,017 | 0,835 |
| KNN | 0,689 ± 0,015 | 0,827 |
| Naive Bayes | 0,657 ± 0,012 | 0,806 |

- Mensaje: logística y LDA **empatan** dentro del ruido de CV. KNN cerca. NB peor.

**Qué decir (≈45 s)**

- Hiperparámetro KNN se elige dentro del pipeline con CV estratificada.

---

## Diapositiva 8 — Resultados en prueba (discriminación)

**En pantalla**

- Insertar **`fig_roc.png`** (y opcional **`fig_pr.png`** en la siguiente slide o mitad derecha).
- Tabla corta:

| Modelo | AUC test | PR-AUC test |
|--------|----------|-------------|
| Línea base | 0,500 | 0,710 |
| Logística | **0,722** | **0,840** |
| LDA | 0,719 | 0,838 |
| KNN | 0,714 | 0,830 |
| Naive Bayes | 0,685 | 0,828 |

- Bootstrap 95 % IC AUC logística: **[0,689 ; 0,751]** — se solapa con LDA y KNN.

**Qué decir (≈1,5 min)**

- AUC y PR-AUC no dependen del umbral. Pasamos de azar (0,5) a ~0,72.
- PR-AUC importa porque la mora es mayoría: ROC sola puede ser optimista para detectar buenos pagadores.
- No coronamos un único ganador: diferencias menores que la desviación de CV.

---

## Diapositiva 9 — Umbral: Youden con predicciones OOF

**En pantalla**

- Fórmula: **J = sensibilidad + especificidad − 1 = TPR − FPR**
- Umbral elegido en **train vía out-of-fold** (no en test): **t ≈ 0,71**
- Contraste logística en test:
  - Umbral **0,5:** recall mora=1 alto, recall mora=0 **0,247**, FN = **50**
  - **Youden OOF:** recall mora=0 **0,634**, FN = **268**, precision mora=1 **0,822**
- Mensaje: el mejor corte **depende del costo** de FN vs FP — la base no trae esos costos.

**Qué decir (≈1,5 min)**

- Esto responde diseño de validación de la rúbrica: el test se evalúa **una vez** para AUC, el umbral se fijó sin mirar test.
- Si el banco prioriza no prestar a un buen pagador vs dejar pasar un moroso, cambia la historia — por eso mostramos escenarios, no un “threshold mágico”.

---

## Diapositiva 10 — Sensibilidad escenarios A / B / C

**En pantalla**

- Tabla (AUC test logística como ancla):

| Escenario | Logística | LDA | KNN | Naive Bayes |
|-----------|-----------|-----|-----|-------------|
| A completo | 0,734 | 0,730 | 0,721 | 0,692 |
| **B principal** | **0,722** | 0,719 | 0,714 | 0,685 |
| C sin score | 0,709 | 0,707 | 0,702 | 0,677 |

- Conclusión en una línea: **B** equilibra poder predictivo y credibilidad causal/temporal.

**Qué decir (≈45 s)**

---

## Diapositiva 11 — Interpretación preliminar (logística B)

**En pantalla**

- Insertar **`fig_coeficientes.png`**
- Bullets (asociación, no causalidad):
  - Menor mora asociada a: educación universitaria, experiencia, `score`, `nivel_ahorro` (coeficientes negativos en escala estandarizada).
  - Mayor mora asociada a: más deuda, vivienda propia vs alquilada (magnitudes modestas).
  - Edad casi nula.
- Supuesto logística: relación **lineal con el logit**, no con la probabilidad directa.

**Qué decir (≈1 min)**

- Corregimos el feedback de la entrega 1: el supuesto clave es linealidad en el **logit** más independencia y multicolinealidad controlada (VIF bajo).

---

## Diapositiva 12 — Limitaciones y cierre

**En pantalla**

- Sin diccionario oficial → momentos de variables son supuestos.
- Sin fechas ni ID cliente → no hay validación temporal ni certeza sobre duplicados.
- Muestra con 71 % mora → probabilidades sirven para **ordenar** clientes, no para prevalencia poblacional.
- Próximos pasos (entrega final): QDA/regularización si el curso lo incluye, **matriz de costos** para umbral operativo.
- Cierre: “Preguntas”

**Qué decir (≈45 s)**

- Reconocemos límites concretos, no genéricos (“más datos siempre ayuda”).

---

## Diapositiva opcional 13 — Respaldo para preguntas

**En pantalla**

- Matriz de confusión resumida (logística, umbral 0,5).
- Recordatorio: notebook `segunda_entrega_modelamiento_final.ipynb` + informe PDF.
- No presentarla salvo que pregunten por detalle de código.

---

## Checklist antes de exponer

- [ ] Cada integrante puede explicar escenario B y por qué sacamos `atraso`.
- [ ] Sabemos decir por qué no balanceamos (tabla CV) y cómo se eligió **t ≈ 0,71** (OOF, no test).
- [ ] Distinguimos AUC (discriminación) vs recall/precision (dependen del umbral).
- [ ] Sabemos el rol de LDA, KNN y **Naive Bayes como contraste**.
- [ ] Las figuras en slides son las del repo, no recreadas por IA.

---

## Prompt listo para la IA de diapositivas

Copia desde aquí hasta el final del bloque.

```
Contexto: Presentación oral académica (12–15 min) para Machine Learning I, Universidad Externado de Colombia. Proyecto: predicción de mora en crédito de consumo (Perú). Integrantes: Isabella Pensamiento y Juan Angel Triana. Tono: primera persona del plural (buscamos, entrenamos), serio y concreto, sin clichés de IA ni iconos de robots.

Tarea: Genera una presentación de 12 diapositivas en español siguiendo EXACTAMENTE esta estructura y estos números. No inventes métricas ni figuras.

Diapositiva 1: Portada con título "Riesgo de incumplimiento crediticio — Regresión logística en contexto latinoamericano", segunda entrega, nombres, universidad.

Diapositiva 2: Objetivo mora binaria, 6074 clientes tras quitar duplicados, 71% prevalencia mora, retroalimentación entrega 1 (fuga, pipeline).

Diapositiva 3: Escenarios A (13 vars), B principal (11 sin atraso ni clasif_sbs), C (sin score). Mensaje: fuga auditada, nos quedamos en B.

Diapositiva 4: Pipeline imputar→escalar→OHE→modelo, 80/20 estratificado, semilla 42, sin filtrar test.

Diapositiva 5: Balanceo 5 opciones, AUC CV igual → sin balancear. Línea base AUC 0.5.

Diapositiva 6: Cuatro modelos: Regresión logística (principal), LDA, KNN k=151, Naive Bayes (contraste). No XGBoost.

Diapositiva 7: Tabla CV: Logística 0.698, LDA 0.699, KNN 0.689, Naive Bayes 0.657.

Diapositiva 8: Resultados test AUC: Logística 0.722, LDA 0.719, KNN 0.714, NB 0.685. Dejar espacio grande para imagen ROC (fig_roc.png).

Diapositiva 9: Umbral Youden J=sens+esp-1, t≈0.71 OOF en train, trade-off FN 50 vs 268.

Diapositiva 10: Tabla sensibilidad escenarios A/B/C con AUC test.

Diapositiva 11: Coeficientes logística (fig_coeficientes.png), linealidad en logit.

Diapositiva 12: Limitaciones y próximos pasos, slide de preguntas.

Diseño: fondo claro, tipografía sans legible, máximo 5 viñetas por slide, una paleta sobria (verde oscuro o azul marino + gris). Sin gradientes futuristas, sin "100% accuracy", sin stock photos de robots.

Notas del presentador: Al final de cada slide, añade en texto pequeño (notas) 2–3 frases de guion basadas en el contenido anterior.

Adjunto: [sube fig_roc.png, fig_pr.png, fig_coeficientes.png si la herramienta lo permite]
```

---

## ¿Debes modificar este archivo?

Sí, **solo** si cambian números en el notebook o el PDF antes de entregar. Este documento ya está alineado con la versión final de `main.tex` y el notebook `segunda_entrega_modelamiento_final.ipynb`. Si regeneras figuras en `codigo/salidas/`, vuelve a subirlas a la herramienta de slides, no hace falta reescribir el guion salvo que cambien los valores.
