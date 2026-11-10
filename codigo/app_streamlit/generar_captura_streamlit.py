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

    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#f8f9fb", edgecolor="#e0e0e0", lw=1))
    titulo = "Priorización de riesgo de mora"
    subt = "U. Externado · ML I · Regresión logística · Escenario B · Pestaña: Ingreso manual"
    cuerpo = (
        f"Umbral operativo: Youden OOF (≈ {t:.2f})\n"
        f"Índice de riesgo (predict_proba): {proba:.3f}\n"
        f"Decisión en este corte: {'Mayor riesgo relativo' if pred else 'Bajo riesgo relativo'}\n"
        f"Ejemplo: cliente índice 0 de database.csv (sin duplicados)\n\n"
        "El score ordena solicitudes; no es probabilidad de mora en el mercado peruano."
    )
    ax.text(0.04, 0.92, titulo, va="top", ha="left", fontsize=14, fontweight="bold", color="#1f1f1f")
    ax.text(0.04, 0.84, subt, va="top", ha="left", fontsize=9, color="#555555")
    ax.text(0.04, 0.72, cuerpo, va="top", ha="left", fontsize=11, color="#222222", family="sans-serif")
    ax.text(0.04, 0.08, "streamlit run codigo/app_streamlit/app.py", va="bottom", ha="left", fontsize=8, color="#888888")
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
