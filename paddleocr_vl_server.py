"""PaddleOCR-VL-1.5 — standalone HTTP server.

Uruchamiany w osobnym venv z transformers>=4.55.
Komunikacja z court_ocr przez HTTP (jak Ollama).

Uruchomienie:
    source ~/paddleocr-vl-env/bin/activate
    python paddleocr_vl_server.py [--port 8765]

Endpoints:
    POST /ocr     — OCR obrazu (multipart/form-data, pole 'image')
    POST /ocr/pdf — OCR PDF-a (multipart/form-data, pole 'file')
    GET  /health  — Status serwera
    POST /unload  — Zwolnij model z VRAM
"""

import argparse
import gc
import io
import logging
import time

from flask import Flask, jsonify, request
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

MODEL_NAME = "PaddlePaddle/PaddleOCR-VL-1.5"
_model = None
_processor = None

# Oficjalne prompty z dokumentacji PaddleOCR-VL-1.5
# Krótkie tokeny zadań — model jest na nie wytrenowany
PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
    "spotting": "Text Spotting:",
    "seal": "Seal Recognition:",
}

# max_pixels według oficjalnego przykładu
MAX_PIXELS = {
    "spotting": 2048 * 28 * 28,
    "default": 1280 * 28 * 28,
}

# Próg skalowania dla zadania spotting
SPOTTING_UPSCALE_THRESHOLD = 1500


def _load():
    global _model, _processor
    if _model is not None:
        return _model, _processor

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    logger.info("Ładowanie %s...", MODEL_NAME)
    _processor = AutoProcessor.from_pretrained(MODEL_NAME)
    _model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    vram = _model.get_memory_footprint() / 1e9
    logger.info("Model załadowany (%.1f GB VRAM)", vram)
    return _model, _processor


def _unload():
    global _model, _processor
    if _model is None:
        return
    import torch

    del _model, _processor
    _model = None
    _processor = None
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Model zwolniony z VRAM")


def _ocr_single_image(image: Image.Image, task: str = "ocr") -> str:
    import torch

    prompt_text = PROMPTS.get(task, "OCR:")

    # Skalowanie dla zadania spotting (zgodnie z oficjalnym przykładem)
    if task == "spotting":
        orig_w, orig_h = image.size
        if orig_w < SPOTTING_UPSCALE_THRESHOLD and orig_h < SPOTTING_UPSCALE_THRESHOLD:
            try:
                resample = Image.Resampling.LANCZOS
            except AttributeError:
                resample = Image.LANCZOS
            image = image.resize((orig_w * 2, orig_h * 2), resample)

    max_pixels = MAX_PIXELS.get(task, MAX_PIXELS["default"])

    model, processor = _load()

    # Oficjalny sposób przekazywania obrazu — jako część struktury wiadomości
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        max_pixels=max_pixels,
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            repetition_penalty=1.15,
            no_repeat_ngram_size=5,
        )

    # Odcinamy tokeny wejściowe, zostawiamy tylko wygenerowane
    input_len = inputs["input_ids"].shape[1]
    trimmed = generated_ids[:, input_len:]
    result = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return result.strip()


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model": MODEL_NAME,
            "loaded": _model is not None,
        }
    )


@app.route("/ocr", methods=["POST"])
def ocr_image():
    """OCR pojedynczego obrazu."""
    if "image" not in request.files:
        return jsonify({"error": "Brak pola 'image'"}), 400

    task = request.form.get("task", "ocr")
    img = Image.open(request.files["image"]).convert("RGB")

    start = time.time()
    text = _ocr_single_image(img, task=task)
    elapsed = time.time() - start

    return jsonify(
        {
            "text": text,
            "time_seconds": round(elapsed, 2),
            "engine": "paddleocr-vl",
        }
    )


@app.route("/ocr/pdf", methods=["POST"])
def ocr_pdf():
    """OCR PDF-a — strona po stronie."""
    if "file" not in request.files:
        return jsonify({"error": "Brak pola 'file'"}), 400

    import fitz

    task = request.form.get("task", "ocr")
    dpi = int(request.form.get("dpi", 300))

    pdf_bytes = request.files["file"].read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages = []
    start = time.time()

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        logger.info("Strona %d/%d", page_num + 1, len(doc))
        text = _ocr_single_image(img, task=task)
        pages.append({"page_number": page_num + 1, "text": text})

    doc.close()
    elapsed = time.time() - start

    return jsonify(
        {
            "text": "\n\n".join(p["text"] for p in pages),
            "pages": pages,
            "total_pages": len(pages),
            "time_seconds": round(elapsed, 2),
            "engine": "paddleocr-vl",
        }
    )


@app.route("/unload", methods=["POST"])
def unload():
    _unload()
    return jsonify({"status": "unloaded"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PaddleOCR-VL-1.5 Server")
    parser.add_argument("--port", type=int, default=8765, help="Port (domyślnie 8765)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--preload", action="store_true", help="Załaduj model od razu")
    args = parser.parse_args()

    if args.preload:
        _load()

    logger.info("Serwer PaddleOCR-VL na %s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port, threaded=False)
