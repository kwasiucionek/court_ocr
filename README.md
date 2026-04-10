# court_ocr — Lokalna ekstrakcja dokumentów sądowych

W pełni lokalna aplikacja do OCR polskich dokumentów sądowych.
Żadne dane nie opuszczają maszyny.

Repo zawiera **dwie niezależne aplikacje**:

| Aplikacja | Plik | Przeznaczenie |
|---|---|---|
| Court OCR (pełna) | `app.py` | Multi-engine OCR + Ollama LLM + ekstrakcja struktury |
| dots.ocr | `app_dots.py` | Szybki OCR przez dots.ocr v1.5 + vLLM |

---

## 🏛️ Court OCR — aplikacja pełna (`app.py`)

Rozbudowane narzędzie do masowego OCR z ekstrakcją struktury dokumentów.

### Silniki OCR

| Silnik | Jakość PL | Wymagania |
|---|---|---|
| **dots.ocr v1.5** | ⭐⭐⭐⭐⭐ | vLLM + 24GB VRAM |
| **PaddleOCR-VL 1.5** | ⭐⭐⭐⭐ | osobny venv, serwer HTTP |
| **Marker + Surya** | ⭐⭐⭐⭐ | lokalne modele |
| **Vision OCR (Ollama)** | ⭐⭐⭐ | DeepSeek-OCR, MiniCPM-V |

### Funkcje

- **Multi-engine OCR** — wybór silnika per dokument
- **Ekstrakcja struktury** — Ollama LLM wyodrębnia dane ze skanu
- **Korekta tekstu OCR** — Ollama LLM poprawia błędy rozpoznawania
- **Preprocessing** — deskew (Hough Lines), korekcja perspektywy,
  denoising (fastNlMeans), CLAHE contrast enhancement
- **Wsadowe przetwarzanie** — katalogi plików z automatycznym zapisem
- **Eksport** — MD, TXT, JSON, ZIP

### Uruchomienie

```bash
conda activate court_ocr
streamlit run app.py
```

### Architektura PaddleOCR-VL

PaddleOCR-VL wymaga `transformers>=4.55` — konfliktu z Surya/Marker
rozwiązano przez osobny serwer HTTP (`paddleocr_vl_server.py`):

```bash
# Serwer PaddleOCR-VL (osobny venv)
source ~/paddleocr-vl-env/bin/activate
python paddleocr_vl_server.py --port 8765 --preload
```

---

## 📄 dots.ocr (`app_dots.py`)

Uproszczona aplikacja skoncentrowana na modelu **dots.ocr v1.5** (1.7B).
Lepsza dla pojedynczych dokumentów i szybkiego OCR.

### Funkcje

- Upload pliku lub wsadowy katalog
- Backend vLLM (GPU) lub HuggingFace (CPU)
- Tryby parsowania: pełny / tylko tekst / detekcja layoutu
- Preprocessing: pomijanie nagłówków/stopek, grayscale, DPI
- Eksport: MD, TXT, JSON

### Pobranie modelu

```bash
# dots.ocr v1.5 (zalecany) — ModelScope
pip install modelscope
modelscope download --model rednote-hilab/dots.ocr-1.5 \
    --local_dir ./weights/DotsOCR-1.5
```

### Uruchomienie serwera vLLM

```bash
bash run_vllm.sh
```

### Uruchomienie aplikacji

```bash
streamlit run app_dots.py
```

---

## Wymagania systemowe

- Python 3.10+
- CUDA 12.x + GPU z min. 22GB VRAM (dots.ocr v1.5)
- conda (zalecane)

## Instalacja

```bash
conda create -n court_ocr python=3.11
conda activate court_ocr
pip install -r requirements.txt
```

## Porównanie silników

| Model | Jakość PL | Szybkość | VRAM |
|---|---|---|---|
| dots.ocr v1.5 | ⭐⭐⭐⭐⭐ | ~5s/str | 22GB |
| dots.ocr v1.0 | ⭐⭐⭐⭐ | ~5s/str | 20GB |
| PaddleOCR-VL 1.5 | ⭐⭐⭐⭐ | ~3s/str | 10GB |
| Marker + Surya | ⭐⭐⭐⭐ | ~2s/str | 8GB |

## Uwagi

- Wszystko działa lokalnie — żadne dane nie opuszczają maszyny
- `paddleocr_vl_server.py` musi działać w osobnym venv
  (`transformers>=4.55` jest niekompatybilny z Surya/Marker)
- Katalog `./output` jest nadpisywany przy każdym uruchomieniu
