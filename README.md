# dots.ocr — Dokumenty sądowe

Streamlit app do OCR polskich dokumentów sądowych z wykorzystaniem modelu **dots.ocr** (1.7B).  
Obsługuje skany, zdjęcia telefonem, dokumenty wielostronicowe PDF.

## Wymagania systemowe

- Python 3.10+
- CUDA 12.x (GPU, zalecane) lub CPU (tryb testowy)
- conda (zalecane środowisko: `dots_ocr`)

## Instalacja

```bash
conda activate dots_ocr
pip install -r requirements.txt
```

## Pobranie modelu

### dots.ocr v1.5 (zalecany) — z ModelScope

```bash
pip install modelscope
modelscope download --model rednote-hilab/dots.ocr-1.5 --local_dir ./weights/DotsOCR-1.5
```

### dots.ocr v1.0 — z HuggingFace

```bash
huggingface-cli download rednote-hilab/dots.ocr --local-dir ./weights/DotsOCR
```

## Uruchomienie serwera vLLM (GPU)

```bash
export hf_model_path=./weights/DotsOCR-1.5
export PYTHONPATH=$(dirname "$hf_model_path"):$PYTHONPATH

vllm serve $hf_model_path \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --chat-template-content-format string \
  --served-model-name model
```

### Tryb testowy (CPU)

```bash
vllm serve $hf_model_path \
  --device cpu \
  --dtype bfloat16 \
  --trust-remote-code \
  --chat-template-content-format string \
  --served-model-name model \
  --max-model-len 4096
```

## Uruchomienie aplikacji

```bash
streamlit run app_dots.py
```

Domyślnie dostępna pod: http://localhost:8501

## Funkcje

- **Upload pliku** — pojedynczy lub wiele plików naraz (PDF, JPG, PNG, BMP, TIFF, WEBP)
- **Katalog** — wsadowe przetwarzanie wszystkich plików w katalogu, automatyczny zapis MD i TXT obok oryginałów
- **Backend vLLM** — szybki inference przez lokalny serwer vLLM
- **Backend HuggingFace** — wolniejszy, nie wymaga serwera vLLM
- **Tryby parsowania:**
  - Pełny parsing (tekst + tabele + formuły)
  - Tylko tekst (bez nagłówków/stopek)
  - Tylko detekcja layoutu
- **Opcje preprocessingu:**
  - Pomijanie nagłówków/stopek stron
  - Fitz preprocess (PyMuPDF)
  - Konwersja do skali szarości
  - DPI (100–300, domyślnie 200)
- **Eksport wyników** — MD, TXT, JSON

## Struktura projektu

```
.
├── app_dots.py          # Główna aplikacja Streamlit
├── requirements.txt
├── README.md
├── output/              # Katalog wyjściowy parsera (JSONL, MD, JPG)
└── weights/
    ├── DotsOCR/         # dots.ocr v1.0
    └── DotsOCR-1.5/     # dots.ocr v1.5
```

## Porównanie silników OCR

| Model | Jakość PL | Szybkość | Wymagania |
|---|---|---|---|
| dots.ocr v1.5 | ⭐⭐⭐⭐⭐ | ~5s/str (GPU) | vLLM + 24GB VRAM |
| dots.ocr v1.0 | ⭐⭐⭐⭐ | ~5s/str (GPU) | vLLM + 22GB VRAM |
| PaddleOCR-VL | ⭐⭐⭐ | szybki | PaddlePaddle GPU |

## Uwagi

- Katalog wyjściowy parsera (`./output`) jest nadpisywany przy każdym uruchomieniu — jeśli chcesz zachować wyniki, zmień ścieżkę w ustawieniach lub pobierz pliki przed kolejnym uruchomieniem.
- Parametr `--gpu-memory-utilization 0.9` działa na RTX 5090 24GB. Na kartach z mniejszą pamięcią może wymagać obniżenia.
- Model wymaga `PYTHONPATH` wskazującego na katalog nadrzędny wag (custom modeling code).
