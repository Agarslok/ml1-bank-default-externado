"""Aplica LDA, PR-AUC y umbral OOF al notebook de actualización."""
import json
from pathlib import Path

NB = Path(__file__).resolve().parent / "segunda_entrega_modelamiento_actualizacion.ipynb"

nb = json.loads(NB.read_text(encoding="utf-8"))

for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if cell["cell_type"] != "code" and cell["cell_type"] != "markdown":
        continue

    # --- imports ---
    if "from sklearn.neighbors import KNeighborsClassifier" in src and "LinearDiscriminantAnalysis" not in src:
        src = src.replace(
            "from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate",
            "from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate, cross_val_predict",
        )
        src = src.replace(
            "from sklearn.neighbors import KNeighborsClassifier",
            "from sklearn.neighbors import KNeighborsClassifier\n"
            "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis",
        )
        src = src.replace(
            "from sklearn.metrics import (roc_auc_score, recall_score, precision_score, f1_score,\n"
            "                             accuracy_score, confusion_matrix, roc_curve, make_scorer)",
            "from sklearn.metrics import (roc_auc_score, average_precision_score, recall_score, precision_score, f1_score,\n"
            "                             accuracy_score, confusion_matrix, roc_curve, precision_recall_curve, make_scorer)",
        )

    # --- intro ---
    if "**Modelos usados (solo los vistos en clase hasta el 26 de octubre):**" in src:
        src = src.replace(
            "**Modelos usados (solo los vistos en clase hasta el 26 de octubre):** regresión logística (modelo principal), KNN y Naive Bayes. La línea base es el clasificador de mayoría. Quedan fuera LDA/QDA, Ridge/Lasso y el ajuste fino de hiperparámetros, porque aún no los hemos visto; tampoco se usan XGBoost, árboles ni redes. Para el balanceo de clases se usan las estrategias del curso (peso de clases, submuestreo, sobremuestreo y SMOTE).",
            "**Modelos (vistos en clase):** línea base (mayoría), **regresión logística** (principal, como en la 1.ª entrega y el artículo), **LDA** (prometido en la 1.ª entrega; sesión 34), **KNN** y **Naive Bayes** (contraste probabilístico, visto en agosto). Quedan fuera QDA, Ridge/Lasso y modelos fuera del curso (XGBoost, árboles, redes). Balanceo: peso de clases, sub/sobremuestreo y SMOTE.",
        )
        src = src.replace(
            "| Modelos | Regresión logística, Ridge/Lasso, LDA y KNN | Regresión logística, KNN y Naive Bayes. LDA/QDA y Ridge/Lasso salen porque aún no los vemos en clase |",
            "| Modelos | Regresión logística, Ridge/Lasso, LDA y KNN | Logística, **LDA**, KNN y Naive Bayes (NB como contraste, no principal). QDA/Ridge para la entrega final |",
        )
        src = src.replace(
            "10. Comparación de modelos con validación cruzada\n",
            "10. Comparación de modelos con validación cruzada (ROC-AUC y PR-AUC)\n",
        )
        src = src.replace(
            "11. Evaluación final en el conjunto de prueba (correcciones 5 y 6)\n",
            "11. Evaluación final en prueba + umbral OOF (Youden)\n",
        )

    # --- evaluar ---
    if src.strip().startswith("def evaluar(y_true, proba, pred):"):
        src = src.replace(
            '"AUC": roc_auc_score(y_true, proba),',
            '"AUC": roc_auc_score(y_true, proba),\n'
            '        "PR-AUC": average_precision_score(y_true, proba),',
        )

    # --- sección 7 modelos ---
    if "Lo incluimos solo como **modelo de contraste**" in src and "## 7." not in src:
        pass
    if "## 7. Pipelines, modelos y supuestos" in src and "KNN y Naive Bayes" in src:
        old = (
            "En la primera entrega dejamos Naive Bayes fuera de los modelos principales porque supone que las variables son independientes entre sí, y las nuestras no lo son del todo: hay relaciones moderadas entre las variables financieras (la revisión de abajo muestra, por ejemplo, correlaciones de Spearman cercanas a 0.55 entre `ingreso` y `score`, y entre `exp_sf` y `linea_sf`). **Esa razón sigue siendo válida**, por eso Naive Bayes **no es el modelo principal**. Lo incluimos solo como **modelo de contraste**, porque ya lo vimos en clase y porque nos sirve para medir cuánto cuesta suponer independencia: si la relación entre variables importa, esperamos que sea el peor de los tres. Quedan fuera LDA/QDA y Ridge/Lasso, que aún no hemos visto.\n"
        )
        new = (
            "En la **1.ª entrega** planeamos **regresión logística, LDA y KNN** y dejamos Naive Bayes fuera por el supuesto de independencia. En el desarrollo incorporamos **Naive Bayes** (ya visto en agosto) como **contraste**: mide cuánto cuesta suponer independencia cuando hay correlaciones moderadas (~0.55 entre `ingreso`/`score` o `exp_sf`/`linea_sf`). Para la **2.ª entrega** volvemos a incluir **LDA** (análisis discriminante lineal, sesión del 23 de octubre): frontera lineal entre morosos y buenos pagadores, comparable con la logística. **Naive Bayes se mantiene** como segundo comparador probabilístico. QDA y Ridge/Lasso quedan para la entrega final.\n"
        )
        src = src.replace(old, new)
        src = src.replace(
            "| Regresión logística | Modelo principal; permite leer odds ratio | ninguno (sin penalización) |\n"
            "| KNN | Comparar contra un modelo sin forma fija, que busca clientes parecidos | `k`, elegido con validación cruzada |\n"
            "| Naive Bayes gaussiano | Comparar contra un modelo probabilístico simple | ninguno |",
            "| Regresión logística | Modelo principal; odds ratio | sin penalización (`C` alto) |\n"
            "| LDA | Cumple plan 1.ª entrega; frontera lineal entre clases | ninguno |\n"
            "| KNN | Clientes parecidos | `k` por CV |\n"
            "| Naive Bayes gaussiano | Contraste; supuesto de independencia | ninguno |",
        )

    # --- pipeline cell: NOMBRES_MOD + ajustar_todos ---
    if "def ajustar_todos(cols, verbose=True):" in src:
        src = src.replace(
            "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)   # 5 grupos estratificados\n"
            "K_LISTA = [5, 11, 21, 31, 51, 101, 151, 201]\n",
            "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)   # 5 grupos estratificados\n"
            "K_LISTA = [5, 11, 21, 31, 51, 101, 151, 201]\n"
            'NOMBRES_MOD = ["Regresión logística", "LDA", "KNN", "Naive Bayes"]\n',
        )
        src = src.replace(
            '    """Elige k de KNN con validación cruzada y ajusta los tres modelos con el conjunto de entrenamiento."""',
            '    """Elige k de KNN con CV y ajusta los cuatro modelos en entrenamiento."""',
        )
        src = src.replace(
            "    modelos = {\n"
            '        "Regresión logística": LogisticRegression(C=1e6, max_iter=5000),   # C enorme = sin penalización\n'
            '        "KNN": KNeighborsClassifier(k_mejor),\n'
            '        "Naive Bayes": GaussianNB(),\n'
            "    }\n"
            "    out = {}\n"
            "    for nombre, m in modelos.items():\n"
            "        pipe = construir_pipeline(cols, m)\n"
            '        sc = cross_val_score(pipe, X_tr_full[cols], y_tr, cv=cv, scoring="roc_auc")\n'
            "        pipe.fit(X_tr_full[cols], y_tr)               # ajuste final con TODO el entrenamiento\n"
            '        out[nombre] = {"modelo": pipe, "cv_media": sc.mean(), "cv_std": sc.std(),\n'
            '                       "params": {"k": k_mejor} if nombre == "KNN" else {}}\n'
            "        if verbose:\n"
            '            print(f"{nombre:22s} AUC CV = {sc.mean():.4f} ± {sc.std():.4f} {out[nombre][\'params\']}")\n',
            "    modelos = {\n"
            '        "Regresión logística": LogisticRegression(C=1e6, max_iter=5000),\n'
            '        "LDA": LinearDiscriminantAnalysis(),\n'
            '        "KNN": KNeighborsClassifier(k_mejor),\n'
            '        "Naive Bayes": GaussianNB(),\n'
            "    }\n"
            "    scoring_cv = {\"roc\": \"roc_auc\", \"pr\": \"average_precision\"}\n"
            "    out = {}\n"
            "    for nombre, m in modelos.items():\n"
            "        pipe = construir_pipeline(cols, m)\n"
            "        r = cross_validate(pipe, X_tr_full[cols], y_tr, cv=cv, scoring=scoring_cv)\n"
            "        pipe.fit(X_tr_full[cols], y_tr)\n"
            "        out[nombre] = {\n"
            '            "modelo": pipe, "cv_media": r["test_roc"].mean(), "cv_std": r["test_roc"].std(),\n'
            '            "pr_cv_media": r["test_pr"].mean(), "pr_cv_std": r["test_pr"].std(),\n'
            '            "params": {"k": k_mejor} if nombre == "KNN" else {},\n'
            "        }\n"
            "        if verbose:\n"
            '            print(f"{nombre:22s} ROC-AUC CV = {r[\'test_roc\'].mean():.4f} ± {r[\'test_roc\'].std():.4f} | "\n'
            '                  f"PR-AUC CV = {r[\'test_pr\'].mean():.4f} {out[nombre][\'params\']}")\n',
        )

    # --- permutación ---
    if "for nombre, m in [(\"Regresión logística\"" in src and "y_mezclada" in src:
        src = src.replace(
            '("Naive Bayes", GaussianNB())]:',
            '("Naive Bayes", GaussianNB()), ("LDA", LinearDiscriminantAnalysis())]:',
        )

    # --- balanceo ---
    if 'scoring = {"auc": "roc_auc"' in src:
        src = src.replace(
            'scoring = {"auc": "roc_auc", "rec1": "recall", "rec0": make_scorer(recall_score, pos_label=0)}',
            'scoring = {"auc": "roc_auc", "pr": "average_precision", "rec1": "recall", "rec0": make_scorer(recall_score, pos_label=0)}',
        )
        if '"PR-AUC (CV)"' not in src and "filas.append" in src:
            src = src.replace(
                '"AUC (CV)": r["test_auc"].mean(), "Desv. AUC": r["test_auc"].std(),',
                '"AUC (CV)": r["test_auc"].mean(), "Desv. AUC": r["test_auc"].std(),\n'
                '                  "PR-AUC (CV)": r["test_pr"].mean(), "Desv. PR-AUC": r["test_pr"].std(),',
            )

    # --- tabla cv ---
    if src.strip().startswith('nombres_mod = ["Regresión logística"'):
        src = src.replace(
            'nombres_mod = ["Regresión logística", "KNN", "Naive Bayes"]',
            "nombres_mod = NOMBRES_MOD",
        )
        src = src.replace(
            '    n: {"AUC CV (media)": aj_P[n]["cv_media"], "AUC CV (desv. est.)": aj_P[n]["cv_std"],\n'
            '        "Hiperparámetro elegido": (f"k = {aj_P[n][\'params\'][\'k\']}" if n == "KNN" else "—")}\n',
            '    n: {"ROC-AUC CV (media)": aj_P[n]["cv_media"], "ROC-AUC CV (desv.)": aj_P[n]["cv_std"],\n'
            '        "PR-AUC CV (media)": aj_P[n]["pr_cv_media"], "PR-AUC CV (desv.)": aj_P[n]["pr_cv_std"],\n'
            '        "Hiperparámetro elegido": (f"k = {aj_P[n][\'params\'][\'k\']}" if n == "KNN" else "—")}\n',
        )
        src = src.replace('tabla_cv["AUC CV (media)"]', 'tabla_cv["ROC-AUC CV (media)"]')
        src = src.replace('tabla_cv["AUC CV (desv. est.)"]', 'tabla_cv["ROC-AUC CV (desv.)"]')

    # --- sección 11 markdown ---
    if "## 11. Evaluación final en el conjunto de prueba" in src:
        src = src.replace(
            "El punto de corte es 0.5 (el que trae por defecto cada modelo).",
            "Reportamos métricas con umbral **0.5** y, para la logística, con umbral **Youden** elegido solo con probabilidades **out-of-fold del entrenamiento** (sesión de métricas por umbral). El conjunto de prueba no se usa para elegir el umbral.",
        )

    # --- evaluar en prueba ---
    if src.strip().startswith("def evaluar_en_prueba(ajustados, cols):"):
        src = (
            "def umbral_youden(y_true, proba):\n"
            "    mejor_u, mejor_v = 0.5, -1.0\n"
            "    for u in np.linspace(0.05, 0.95, 181):\n"
            "        pred = (proba >= u).astype(int)\n"
            "        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()\n"
            "        tpr = tp / (tp + fn) if (tp + fn) else 0\n"
            "        fpr = fp / (fp + tn) if (fp + tn) else 0\n"
            "        v = tpr - fpr\n"
            "        if v > mejor_v:\n"
            "            mejor_v, mejor_u = v, u\n"
            "    return mejor_u\n\n"
            "# Umbral solo con OOF en train (regresión logística)\n"
            "_pipe_oof = construir_pipeline(cols_P, LogisticRegression(C=1e6, max_iter=5000))\n"
            "_proba_oof = cross_val_predict(_pipe_oof, X_tr_full[cols_P], y_tr, cv=cv, method=\"predict_proba\")[:, 1]\n"
            "UMBRAL_YOUDEN = umbral_youden(y_tr, _proba_oof)\n"
            "print(\"Umbral Youden (OOF train):\", round(UMBRAL_YOUDEN, 3))\n\n"
            + src
        )
        src = src.replace(
            '    for nombre in nombres_mod:\n',
            "    for nombre in NOMBRES_MOD:\n",
        )
        src = src.replace(
            "    filas[\"Línea base (siempre mora=1)\"] = res_base\n",
            "    filas[\"Línea base (siempre mora=1)\"] = res_base\n"
            "    # Logística con umbral Youden (OOF)\n"
            "    p_lr = probas.get(\"Regresión logística\")\n"
            "    if p_lr is not None:\n"
            "        filas[\"Reg. logística (umbral Youden OOF)\"] = evaluar(y_te, p_lr, (p_lr >= UMBRAL_YOUDEN).astype(int))\n",
        )
        src = src.replace(
            'for c in ["AUC", "Recall (mora=1)"',
            'for c in ["AUC", "PR-AUC", "Recall (mora=1)"',
        )

    if "# Incertidumbre del AUC de prueba" in src:
        src = src.replace("for nombre in nombres_mod:", "for nombre in NOMBRES_MOD:")

    if "for i, nombre in enumerate(nombres_mod):" in src:
        src = src.replace("nombres_mod", "NOMBRES_MOD")

    if "for modelo in nombres_mod:" in src:
        src = src.replace("nombres_mod", "NOMBRES_MOD")
        src = src.replace(".loc[nombres_mod]", ".loc[NOMBRES_MOD]")

    # --- conclusiones ---
    if "## 14. Conclusiones" in src and "lista de chequeo" in src.lower():
        if "LDA" not in src or "Youden" not in src:
            src += (
                "\n\n**Chequeo actualizado:** LDA incorporado; Naive Bayes como contraste; PR-AUC en balanceo y modelos; "
                "umbral Youden con OOF en train; escenario B sin `atraso`/`clasif_sbs`; línea base; feedback de mora, fuga y logit.\n"
            )

    if "Cuando veamos regularización y LDA/QDA" in src:
        src = src.replace(
            "Cuando veamos regularización y LDA/QDA evaluaremos si aportan algo.",
            "QDA y regularización (Ridge/Lasso) se evaluarán en la entrega final.",
        )

    cell["source"] = [line if line.endswith("\n") else line + "\n" for line in src.splitlines()]
    if cell["source"] and not cell["source"][-1].endswith("\n"):
        cell["source"][-1] += "\n"

# clear outputs for re-run
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Patched:", NB)
