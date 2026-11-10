"""
Genera captura_streamlit.png sin abrir el navegador a mano.

Opción A (recomendada si tienes Playwright):
  pip install playwright
  playwright install chromium
  python generar_captura_streamlit.py

Levanta Streamlit en localhost:8501, espera, hace screenshot y guarda en
documentacion/tercera_entrega/entregables_para_informe/captura_streamlit.png

Opción B (solo vista estática del informe, sin UI Streamlit real):
  python generar_captura_streamlit.py --estatica
"""
from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent.parent
OUT = ROOT / "documentacion" / "tercera_entrega" / "entregables_para_informe" / "captura_streamlit.png"
PORT = 8501


def puerto_libre(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def esperar_servidor(timeout: float = 45.0) -> bool:
    import urllib.request

    url = f"http://127.0.0.1:{PORT}/"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def captura_con_playwright() -> None:
    from playwright.sync_api import sync_playwright

    if not esperar_servidor():
        raise RuntimeError("Streamlit no respondió a tiempo en el puerto 8501.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(f"http://127.0.0.1:{PORT}/", wait_until="networkidle", timeout=60000)
        time.sleep(2)
        # Ejemplo: pestaña manual + botón calcular si existe
        try:
            page.get_by_role("tab", name="Ingreso manual").click(timeout=3000)
        except Exception:
            pass
        try:
            page.get_by_role("button", name="Calcular riesgo").click(timeout=5000)
            time.sleep(1)
        except Exception:
            pass
        page.screenshot(path=str(OUT), full_page=True)
        browser.close()
    print("Guardado:", OUT)


def captura_estatica() -> None:
    """Imagen tipo informe con score de ejemplo (mismo pipeline que la app)."""
    import json

    import joblib
    import matplotlib.pyplot as plt
    import pandas as pd

    sys.path.insert(0, str(APP_DIR))
    from paths import CANDIDATOS_PIPELINE, CANDIDATOS_UMBRALES, COLS_B, first_existing

    pipe = joblib.load(first_existing(CANDIDATOS_PIPELINE))
    umb = json.loads(first_existing(CANDIDATOS_UMBRALES).read_text(encoding="utf-8"))
    data = ROOT / "datos" / "database.csv"
    df = pd.read_csv(data).drop_duplicates()
    row = df.iloc[0][COLS_B].to_dict()
    X = pd.DataFrame([row])
    proba = float(pipe.predict_proba(X)[0, 1])
    t = umb["t_youden_oof"]
    pred = proba >= t

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis("off")
    texto = (
        "Demo — Priorización de riesgo de mora (Streamlit)\n"
        "Modelo final: regresión logística · escenario B\n\n"
        f"Índice de riesgo (modelo): {proba:.3f}\n"
        f"Umbral Youden OOF (≈ {t:.2f}): {'Mayor riesgo relativo' if pred else 'Bajo riesgo en este corte'}\n\n"
        "Limitación: score para ordenar clientes, no probabilidad real en Perú.\n"
        "Ejecutar: streamlit run codigo/app_streamlit/app.py"
    )
    ax.text(0.05, 0.95, texto, va="top", ha="left", fontsize=11, family="sans-serif", wrap=True)
    fig.patch.set_facecolor("#fafafa")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Captura estática guardada:", OUT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--estatica", action="store_true", help="Sin Streamlit ni Playwright")
    args = parser.parse_args()

    if args.estatica:
        captura_estatica()
        return

    try:
        import playwright  # noqa: F401
    except ImportError:
        print("Playwright no instalado. Usa:")
        print("  pip install playwright")
        print("  playwright install chromium")
        print("O genera captura estática:")
        print("  python generar_captura_streamlit.py --estatica")
        sys.exit(1)

    proc = None
    try:
        if not puerto_libre(PORT):
            print(f"Puerto {PORT} en uso; asumo Streamlit ya corriendo.")
        else:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(APP_DIR / "app.py"),
                    "--server.headless",
                    "true",
                    "--server.port",
                    str(PORT),
                    "--browser.gatherUsageStats",
                    "false",
                ],
                cwd=str(APP_DIR),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        captura_con_playwright()
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
