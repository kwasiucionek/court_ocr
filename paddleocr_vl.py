"""PaddleOCR-VL-1.5 — klient HTTP do serwera paddleocr_vl_server.py.

Serwer działa w osobnym venv (transformers>=5.0),
klient komunikuje się przez HTTP — brak konfliktu z Surya/Marker.

Konfiguracja w config.py:
    PADDLEOCR_VL_URL = "http://127.0.0.1:8765"
"""

import io
import logging

import requests
from PIL import Image

logger = logging.getLogger(__name__)

try:
    from config import PADDLEOCR_VL_URL
except ImportError:
    PADDLEOCR_VL_URL = "http://127.0.0.1:8765"


def is_available() -> bool:
    """Sprawdź czy serwer PaddleOCR-VL działa."""
    try:
        r = requests.get(f"{PADDLEOCR_VL_URL}/health", timeout=3)
        return r.ok and r.json().get("status") == "ok"
    except Exception:
        return False


def ocr_image(image: Image.Image, task: str = "ocr") -> str:
    """Wyślij obraz PIL do serwera i zwróć rozpoznany tekst."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    r = requests.post(
        f"{PADDLEOCR_VL_URL}/ocr",
        files={"image": ("image.png", buf, "image/png")},
        data={"task": task},
        timeout=120,
    )
    r.raise_for_status()
    return r.json().get("text", "")


def ocr_image_file(file_path: str, preprocess_fn=None) -> dict:
    """OCR pliku obrazu przez serwer PaddleOCR-VL."""
    img = Image.open(file_path).convert("RGB")
    if preprocess_fn:
        img = preprocess_fn(img)

    text = ocr_image(img)

    return {
        "text": text,
        "pages": [{"page_number": 1, "text": text}],
        "metadata": {
            "engine": "paddleocr-vl",
            "device": "cuda",
        },
    }


def ocr_pdf(file_path: str, preprocess_fn=None) -> dict:
    """OCR PDF-a przez serwer PaddleOCR-VL.

    Jeśli jest preprocess_fn, przetwarza strony lokalnie i wysyła obrazy.
    Jeśli nie — wysyła cały PDF na serwer.
    """
    if preprocess_fn:
        # Preprocessing lokalnie, wysyłaj obrazy po kolei
        import fitz

        doc = fitz.open(file_path)
        pages = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img = preprocess_fn(img)

            logger.info("PaddleOCR-VL: strona %d/%d", page_num + 1, len(doc))
            text = ocr_image(img)
            pages.append({"page_number": page_num + 1, "text": text})

        doc.close()
        return {
            "text": "\n\n".join(p["text"] for p in pages),
            "pages": pages,
            "metadata": {
                "engine": "paddleocr-vl",
                "total_pages": len(pages),
                "device": "cuda",
            },
        }
    else:
        # Wyślij cały PDF na serwer
        with open(file_path, "rb") as f:
            r = requests.post(
                f"{PADDLEOCR_VL_URL}/ocr/pdf",
                files={"file": ("document.pdf", f, "application/pdf")},
                timeout=600,
            )
        r.raise_for_status()
        data = r.json()
        return {
            "text": data.get("text", ""),
            "pages": data.get("pages", []),
            "metadata": {
                "engine": "paddleocr-vl",
                "total_pages": data.get("total_pages", 0),
                "device": "cuda",
            },
        }


def unload_model():
    """Zwolnij model z VRAM na serwerze."""
    try:
        requests.post(f"{PADDLEOCR_VL_URL}/unload", timeout=5)
    except Exception:
        pass
