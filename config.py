"""Konfiguracja aplikacji Court OCR — w pełni lokalna."""

from pathlib import Path

import torch

# ── Ścieżki ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / "temp"
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# ── GPU / urządzenie ─────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Marker ───────────────────────────────────────────────────
MARKER_WORKERS = 3  # RTX 5090 24GB → ~5GB VRAM per worker
MARKER_FORCE_OCR = True

# ── Surya ────────────────────────────────────────────────────
SURYA_LANGUAGES = ["pl", "en"]

# ── Preprocessing ────────────────────────────────────────────
PREPROCESS_TARGET_DPI = 300
PREPROCESS_MAX_DIMENSION = 4096


PADDLEOCR_VL_URL = "http://127.0.0.1:8765"


# ── Lokalne LLM (Ollama) ────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = (
    "speakleash/bielik-11b-v3.0-instruct:Q4_K_M"  # Domyślny, jeśli dostępny
)
OLLAMA_TIMEOUT = 1200
OLLAMA_TEMPERATURE = 0.1

# Rekomendowane modele do polskich dokumentów sądowych (kolejność = priorytet)
OLLAMA_RECOMMENDED_MODELS = [
    "bielik:7b",
    "qwen3:8b",
    "gemma3:12b",
    "llama3.1:8b",
    "mistral:7b",
]

# ── Vision OCR (Ollama) ──────────────────────────────────────
VISION_OCR_MODELS = [
    "deepseek-ocr:3b",
    "minicpm-v",
    "gemma3:12b",
    "llama3.2-vision:11b",
]

VISION_OCR_PROMPTS = {
    "deepseek-ocr": "\nFree OCR.",
    "default": "Odczytaj i przepisz CAŁY tekst z tego obrazu. Zachowaj oryginalną strukturę, akapity i formatowanie. Zwróć TYLKO tekst, bez komentarzy.",
}


# ── Formaty ──────────────────────────────────────────────────
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_PDF_EXTENSIONS

# ── Bezpieczeństwo ───────────────────────────────────────────
# Wszystko działa lokalnie. Żadne dane nie opuszczają maszyny.
# Marker, Surya, Ollama — 100% offline po pobraniu modeli.
OFFLINE_MODE = True
