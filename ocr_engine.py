"""Silnik OCR — Marker dla PDF, Surya dla obrazów, Vision LLM dla odręcznych, z fallbackiem CUDA→CPU."""

import base64
import io
import logging
import os
import re
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

from config import (
    DEVICE,
    MARKER_FORCE_OCR,
    OLLAMA_BASE_URL,
    OLLAMA_TIMEOUT,
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_PDF_EXTENSIONS,
    SURYA_LANGUAGES,
    TEMP_DIR,
    VISION_OCR_MODELS,
    VISION_OCR_PROMPTS,
)
from preprocessor import PreprocessResult, preprocess_image

logger = logging.getLogger(__name__)


def _unload_ollama_model(model_name):
    """Zwolnij model z pamięci GPU Ollama."""
    try:
        requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={"model": model_name, "messages": [], "keep_alive": 0},
            timeout=10,
        )
        logger.info("Zwolniono model %s z pamięci", model_name)
    except Exception as e:
        logger.warning("Nie udało się zwolnić modelu %s: %s", model_name, e)


def _safe_cuda_call(func, *args, **kwargs):
    """Wywołaj funkcję z obsługą błędów CUDA."""
    try:
        return func(*args, **kwargs)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        error_msg = str(e).lower()
        is_cuda_error = any(
            phrase in error_msg
            for phrase in [
                "index out of bounds",
                "cuda error",
                "out of memory",
                "cublas",
                "assertion",
                "device-side assert",
            ]
        )
        if not is_cuda_error:
            raise

        logger.warning("Błąd CUDA: %s — czyszczę cache i retry...", e)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        try:
            return func(*args, **kwargs)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e2:
            logger.warning("Ponowny błąd CUDA: %s — fallback na CPU", e2)
            torch.cuda.empty_cache()
            original_device = os.environ.get("TORCH_DEVICE")
            os.environ["TORCH_DEVICE"] = "cpu"
            try:
                return func(*args, **kwargs)
            finally:
                if original_device:
                    os.environ["TORCH_DEVICE"] = original_device
                elif "TORCH_DEVICE" in os.environ:
                    del os.environ["TORCH_DEVICE"]


def _resize_for_ocr(image: Image.Image, max_dim: int = 2048) -> Image.Image:
    """Resize obrazu do bezpiecznych wymiarów (podzielnych przez 32)."""
    w, h = image.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        w, h = int(w * scale), int(h * scale)
    new_w = max((w // 32) * 32, 32)
    new_h = max((h // 32) * 32, 32)
    if new_w != image.size[0] or new_h != image.size[1]:
        return image.resize((new_w, new_h), Image.LANCZOS)
    return image


def _image_to_base64(image: Image.Image, max_dim: int = 2048) -> str:
    """Konwertuj obraz PIL do base64 dla API Ollama."""
    w, h = image.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _remove_repetitions(text: str) -> str:
    """Usuń powtarzające się frazy — artefakt modeli vision."""
    if not text:
        return text

    # 1. Powtórzone pojedyncze słowa
    text = re.sub(r"\b(\w+)(?:[\s,]+\1){2,}", r"\1", text, flags=re.IGNORECASE)

    # 2. Powtórzone frazy 2-6 słów
    for phrase_len in range(6, 1, -1):
        pattern = r"(\b(?:\w+[\s,]+){" + str(phrase_len - 1) + r"}\w+)(?:[\s,]+\1){2,}"
        text = re.sub(pattern, r"\1", text, flags=re.IGNORECASE)

    # 3. Powtórzone linie
    lines = text.split("\n")
    seen: dict[str, int] = {}
    result = []
    for line in lines:
        stripped = line.strip()
        normalized = re.sub(r"\s+", " ", stripped.lower())
        if not stripped:
            result.append(line)
            continue
        count = seen.get(normalized, 0)
        if count < 2:
            seen[normalized] = count + 1
            result.append(line)

    text = "\n".join(result)

    # 4. Awaryjny detektor pętli
    words = text.split()
    if len(words) > 200:
        unique_ratio = len(set(w.lower() for w in words)) / len(words)
        if unique_ratio < 0.15:
            sentences = re.split(r"(?<=[.!?])\s+", text)
            result_sentences = []
            for s in sentences:
                s_words = s.split()
                if not s_words:
                    continue
                s_unique = len(set(w.lower() for w in s_words)) / len(s_words)
                if s_unique < 0.3 and len(s_words) > 10:
                    break
                result_sentences.append(s)
            text = " ".join(result_sentences)

    return text.strip()


def _get_vision_prompt(model_name):
    """Zwróć prompt odpowiedni dla modelu vision."""
    from config import VISION_OCR_PROMPTS

    base = model_name.split(":")[0].lower()
    for key, prompt in VISION_OCR_PROMPTS.items():
        if key in base:
            return prompt
    return VISION_OCR_PROMPTS.get("default", "Read all text from this image.")


def _apply_preprocess(img: Image.Image, preprocess: bool, preprocess_kwargs: dict) -> tuple[Image.Image, list]:
    """Pomocnik — uruchom preprocessing i zwróć obraz PIL + listę korekt."""
    import cv2
    if not preprocess:
        return img, []
    result = preprocess_image(img, **preprocess_kwargs)
    if len(result.image.shape) == 3:
        img_out = Image.fromarray(cv2.cvtColor(result.image, cv2.COLOR_BGR2RGB))
    else:
        img_out = Image.fromarray(result.image)
    return img_out, result.corrections_applied


class OCREngine:
    """Główny silnik OCR z lazy-loadingiem modeli."""

    def __init__(self):
        self._marker_converter = None
        self._marker_models = None
        self._surya_initialized = False
        self._foundation_predictor = None
        self._det_predictor = None
        self._rec_predictor = None

    def _init_marker(self):
        if self._marker_converter is not None:
            return
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        self._marker_models = create_model_dict()
        self._marker_converter = PdfConverter(
            artifact_dict=self._marker_models,
            config={
                "force_ocr": MARKER_FORCE_OCR,
                "output_format": "json",
                "languages": SURYA_LANGUAGES,
            },
        )

    def _init_surya(self):
        if self._surya_initialized:
            return
        from surya.detection import DetectionPredictor
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor

        self._foundation_predictor = FoundationPredictor()
        self._rec_predictor = RecognitionPredictor(self._foundation_predictor)
        self._det_predictor = DetectionPredictor()
        self._surya_initialized = True

    def process_file(
        self,
        file_path,
        preprocess: bool = True,
        engine: str = "auto",
        vision_model: str | None = None,
        preprocess_kwargs: dict | None = None,
    ):
        preprocess_kwargs = preprocess_kwargs or {}
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        if suffix in SUPPORTED_PDF_EXTENSIONS:
            if engine == "vision":
                return self._process_pdf_as_vision(file_path, preprocess, vision_model, preprocess_kwargs)
            return self._process_pdf(file_path, preprocess, engine, preprocess_kwargs)
        elif suffix in SUPPORTED_IMAGE_EXTENSIONS:
            if engine == "vision":
                return self._process_image_vision(file_path, preprocess, vision_model, preprocess_kwargs)
            return self._process_image(file_path, preprocess, engine, preprocess_kwargs)
        else:
            raise ValueError(f"Nieobsługiwany format: {suffix}")

    # ── PDF ───────────────────────────────────────────────────

    def _process_pdf(self, file_path, preprocess, engine, preprocess_kwargs):
        if engine in ("auto", "marker"):
            return self._process_pdf_marker(file_path, preprocess)
        else:
            return self._process_pdf_as_images(file_path, preprocess, preprocess_kwargs)

    def _process_pdf_marker(self, file_path, preprocess):
        self._init_marker()

        def _do_marker():
            return self._marker_converter(str(file_path))

        rendered = _safe_cuda_call(_do_marker)
        text = rendered.markdown if hasattr(rendered, "markdown") else str(rendered)

        pages = []
        if hasattr(rendered, "children"):
            for i, page in enumerate(rendered.children):
                pages.append({"page_number": i + 1, "text": str(page) if page else ""})
        else:
            pages = [{"page_number": 1, "text": text}]

        return {
            "text": text,
            "pages": pages,
            "metadata": {
                "engine": "marker",
                "source_file": str(file_path),
                "page_count": len(pages),
                "device": DEVICE,
            },
            "preprocessing_info": [],
        }

    def _process_pdf_as_images(self, file_path, preprocess, preprocess_kwargs):
        import fitz

        doc = fitz.open(str(file_path))
        pages, all_text_parts, preprocess_infos = [], [], []

        for page_num in range(len(doc)):
            page = doc[page_num]
            mat = fitz.Matrix(300 / 72, 300 / 72)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            img, prep_info = _apply_preprocess(img, preprocess, preprocess_kwargs)
            img = _resize_for_ocr(img)
            page_text = self._ocr_image_surya(img)

            pages.append({"page_number": page_num + 1, "text": page_text})
            all_text_parts.append(page_text)
            if prep_info:
                preprocess_infos.append({"page": page_num + 1, "corrections": prep_info})

        doc.close()
        return {
            "text": "\n\n".join(all_text_parts),
            "pages": pages,
            "metadata": {
                "engine": "surya",
                "source_file": str(file_path),
                "page_count": len(pages),
                "device": DEVICE,
            },
            "preprocessing_info": preprocess_infos,
        }

    def _process_pdf_as_vision(self, file_path, preprocess, vision_model, preprocess_kwargs):
        import fitz

        doc = fitz.open(str(file_path))
        pages, all_text_parts, preprocess_infos = [], [], []

        for page_num in range(len(doc)):
            page = doc[page_num]
            mat = fitz.Matrix(300 / 72, 300 / 72)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            img, prep_info = _apply_preprocess(img, preprocess, preprocess_kwargs)
            page_text = self._ocr_image_vision(img, vision_model)

            pages.append({"page_number": page_num + 1, "text": page_text})
            all_text_parts.append(page_text)
            if prep_info:
                preprocess_infos.append({"page": page_num + 1, "corrections": prep_info})

        doc.close()
        _unload_ollama_model(vision_model)

        return {
            "text": "\n\n".join(all_text_parts),
            "pages": pages,
            "metadata": {
                "engine": f"vision:{vision_model}",
                "source_file": str(file_path),
                "page_count": len(pages),
                "device": "ollama",
            },
            "preprocessing_info": preprocess_infos,
        }

    # ── Obrazy ────────────────────────────────────────────────

    def _process_image(self, file_path, preprocess, engine, preprocess_kwargs):
        img = Image.open(file_path).convert("RGB")
        img, prep_info = _apply_preprocess(img, preprocess, preprocess_kwargs)
        img = _resize_for_ocr(img)

        if engine in ("auto", "surya"):
            text = self._ocr_image_surya(img)
            engine_used = "surya"
        else:
            text = self._ocr_image_via_marker(img)
            engine_used = "marker"

        return {
            "text": text,
            "pages": [{"page_number": 1, "text": text}],
            "metadata": {
                "engine": engine_used,
                "source_file": str(file_path),
                "page_count": 1,
                "device": DEVICE,
            },
            "preprocessing_info": prep_info,
        }

    def _process_image_vision(self, file_path, preprocess, vision_model, preprocess_kwargs):
        img = Image.open(file_path).convert("RGB")
        img, prep_info = _apply_preprocess(img, preprocess, preprocess_kwargs)
        text = self._ocr_image_vision(img, vision_model)
        _unload_ollama_model(vision_model)

        return {
            "text": text,
            "pages": [{"page_number": 1, "text": text}],
            "metadata": {
                "engine": f"vision:{vision_model}",
                "source_file": str(file_path),
                "page_count": 1,
                "device": "ollama",
            },
            "preprocessing_info": prep_info,
        }

    # ── Silniki OCR ───────────────────────────────────────────

    def _ocr_image_surya(self, image):
        self._init_surya()

        def _do_ocr():
            predictions = self._rec_predictor(
                [image],
                ["ocr_with_boxes"],
                self._det_predictor,
            )
            lines = []
            if predictions:
                for text_line in predictions[0].text_lines:
                    lines.append(text_line.text)
            return "\n".join(lines)

        return _safe_cuda_call(_do_ocr)

    def _ocr_image_via_marker(self, image):
        self._init_marker()
        temp_path = TEMP_DIR / "temp_image.png"
        image.save(str(temp_path))

        def _do_marker():
            rendered = self._marker_converter(str(temp_path))
            return rendered.markdown if hasattr(rendered, "markdown") else str(rendered)

        try:
            return _safe_cuda_call(_do_marker)
        finally:
            temp_path.unlink(missing_ok=True)

    def _ocr_image_vision(self, image, vision_model):
        """OCR obrazu przez Vision LLM (Ollama)."""
        img_b64 = _image_to_base64(image, max_dim=2048)
        prompt = _get_vision_prompt(vision_model)

        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": vision_model,
                    "messages": [
                        {"role": "user", "content": prompt, "images": [img_b64]}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 2048,
                        "repetition_penalty": 1.3,
                        "repeat_last_n": 128,
                    },
                },
                timeout=OLLAMA_TIMEOUT * 2,
            )
            response.raise_for_status()
            result = response.json()
            text = result.get("message", {}).get("content", "").strip()

            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            text = re.sub(r"<\|ref\|>.*?<\|/ref\|>", "", text).strip()
            text = re.sub(r"<\|det\|>.*?<\|/det\|>", "", text).strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:markdown|text)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)

            text = _remove_repetitions(text)
            return text

        except requests.ConnectionError:
            raise RuntimeError(
                f"Ollama niedostępna pod {OLLAMA_BASE_URL}. Uruchom: ollama serve"
            )
        except requests.Timeout:
            raise RuntimeError(
                f"Vision OCR timeout ({OLLAMA_TIMEOUT * 2}s). "
                "Obraz może być zbyt duży lub model zbyt wolny."
            )
        except Exception as e:
            raise RuntimeError(f"Błąd Vision OCR: {e}")


_engine_instance = None


def get_engine():
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = OCREngine()
    return _engine_instance
