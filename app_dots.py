"""
dots.ocr — Streamlit OCR App

Uruchomienie serwera vLLM:
  conda activate dots_ocr
  export hf_model_path=./weights/DotsOCR-1.5
  export PYTHONPATH=$(dirname "$hf_model_path"):$PYTHONPATH

  CUDA_VISIBLE_DEVICES=0 vllm serve $hf_model_path \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --chat-template-content-format string \
    --served-model-name model \
    --trust-remote-code

Uruchomienie aplikacji:
  streamlit run app_dots.py
"""

import json
import tempfile
from pathlib import Path

import requests
import streamlit as st

st.set_page_config(
    page_title="dots.ocr — Dokumenty sądowe",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

SUPPORTED = {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

PROMPT_MODES = {
    "prompt_layout_all_en": "Pełny parsing (tekst + tabele + formuły)",
    "prompt_ocr": "Tylko tekst (bez nagłówków/stopek)",
    "prompt_layout_only_en": "Tylko detekcja layoutu (bez treści)",
}

SKIP_CATS = {"Page-header", "Page-footer", "Picture"}

# Kategorie → formatowanie Markdown
CATEGORY_FORMAT = {
    "Title": lambda t: "# " + t,
    "Section-header": lambda t: t if t.startswith("#") else "## " + t,
    "List-item": lambda t: "- " + t,
    "Formula": lambda t: "$$\n" + t + "\n$$",
    "Table": lambda t: t,
    "Caption": lambda t: "*" + t + "*",
    "Footnote": lambda t: "> " + t,
    "Text": lambda t: t,
}


# ── Konwersja komórek JSON → tekst ────────────────────────────────────────────
def cells_to_md(cells):
    parts = []
    for cell in cells:
        cat = cell.get("category", "Text")
        text = (cell.get("text") or "").strip()
        if not text:
            continue
        fmt = CATEGORY_FORMAT.get(cat, lambda t: t)
        rendered = fmt(text)
        if rendered:
            parts.append(rendered)
    return "\n\n".join(parts)


def results_to_md(results, fname):
    parts = ["# " + fname + "\n"]
    for r in results:
        if len(results) > 1:
            parts.append("---\n## Strona " + str(r["page_no"] + 1) + "\n")
        parts.append(cells_to_md(r["cells"]))
    return "\n\n".join(parts)


def results_to_txt(results):
    pages = []
    for r in results:
        header = (
            "--- Strona " + str(r["page_no"] + 1) + " ---\n" if len(results) > 1 else ""
        )
        texts = [
            (c.get("text") or "").strip()
            for c in r["cells"]
            if (c.get("text") or "").strip()
        ]
        pages.append(header + "\n".join(texts))
    return "\n\n".join(pages)


# ── Parser ────────────────────────────────────────────────────────────────────
def check_vllm(protocol, ip, port):
    try:
        r = requests.get(protocol + "://" + ip + ":" + str(port) + "/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def run_parser(
    file_path,
    backend,
    vllm_protocol,
    vllm_ip,
    vllm_port,
    prompt_mode,
    skip_hf,
    dots_output_dir="./output",
    dpi=200,
    fitz_preprocess=False,
):
    from dots_ocr.parser import DotsOCRParser

    if backend == "vllm":
        parser = DotsOCRParser(
            protocol=vllm_protocol,
            ip=vllm_ip,
            port=int(vllm_port),
            model_name="model",
            output_dir=dots_output_dir,
            dpi=dpi,
            max_completion_tokens=32000,
        )
    else:
        parser = DotsOCRParser(use_hf=True, output_dir=dots_output_dir, dpi=dpi)

    parser.parse_file(
        file_path,
        output_dir=dots_output_dir,
        prompt_mode=prompt_mode,
        fitz_preprocess=fitz_preprocess,
    )

    # Szukamy JSONL rekurencyjnie, bierzemy najnowszy
    jsonl_files = list(Path(dots_output_dir).rglob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError("Brak pliku JSONL w " + dots_output_dir)
    jsonl_file = max(jsonl_files, key=lambda p: p.stat().st_mtime)

    skip_cats = SKIP_CATS if skip_hf else {"Picture"}
    results = []

    with open(jsonl_file, encoding="utf-8") as jl:
        for line in jl:
            line = line.strip()
            if not line:
                continue
            page_meta = json.loads(line)
            page_no = page_meta.get("page_no", 0)

            # Czytamy komórki z pliku JSON (nie z MD — MD zawiera base64 obrazów)
            layout_path = page_meta.get("layout_info_path", "")
            if layout_path and Path(layout_path).exists():
                with open(layout_path, encoding="utf-8") as cf:
                    cells_raw = json.load(cf)
            else:
                cells_raw = []

            cells = [c for c in cells_raw if c.get("category") not in skip_cats]

            results.append({"page_no": page_no, "cells": cells})

    return results


# ── Renderowanie wyników ──────────────────────────────────────────────────────
def render_results(results, fname):
    total_cells = sum(len(r["cells"]) for r in results)
    cats = {}
    for r in results:
        for c in r["cells"]:
            k = c.get("category", "?")
            cats[k] = cats.get(k, 0) + 1

    col1, col2, col3 = st.columns(3)
    col1.metric("Stron", len(results))
    col2.metric("Bloków", total_cells)
    col3.metric("Kategorii", len(cats))

    if cats:
        with st.expander("Rozkład kategorii"):
            for cat, cnt in sorted(cats.items(), key=lambda x: -x[1]):
                st.text(f"{cat:<20} {cnt}")

    stem = Path(fname).stem

    if len(results) > 1:
        tabs = st.tabs(["Strona " + str(r["page_no"] + 1) for r in results])
        for tab, r in zip(tabs, results):
            with tab:
                st.text_area(
                    "Tekst",
                    cells_to_md(r["cells"]),
                    height=400,
                    key="ta_" + fname + "_" + str(r["page_no"]),
                )
    else:
        st.text_area(
            "Tekst",
            cells_to_md(results[0]["cells"]) if results else "",
            height=400,
            key="ta_" + fname,
        )

    dl1, dl2, dl3 = st.columns(3)
    dl1.download_button(
        "↓ MD",
        results_to_md(results, fname),
        file_name=stem + ".md",
        mime="text/markdown",
        key="dl_md_" + fname,
    )
    dl2.download_button(
        "↓ TXT",
        results_to_txt(results),
        file_name=stem + ".txt",
        mime="text/plain",
        key="dl_txt_" + fname,
    )
    dl3.download_button(
        "↓ JSON",
        json.dumps(results, ensure_ascii=False, indent=2),
        file_name=stem + ".json",
        mime="application/json",
        key="dl_json_" + fname,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚖ dots.ocr")
    st.caption("SOTA Document Parser · 1.7B · vLLM / HF")
    st.divider()

    st.subheader("Backend")
    backend = st.radio(
        "Backend",
        ["vllm", "hf"],
        format_func=lambda x: (
            "vLLM (zalecany)" if x == "vllm" else "HuggingFace (wolniejszy)"
        ),
        label_visibility="collapsed",
    )

    if backend == "vllm":
        vllm_protocol = st.selectbox("Protokół", ["http", "https"], index=0)
        vllm_ip = st.text_input("IP serwera", value="localhost")
        vllm_port = st.number_input(
            "Port", value=8000, min_value=1, max_value=65535, step=1
        )
        if st.button("Sprawdź połączenie"):
            if check_vllm(vllm_protocol, vllm_ip, int(vllm_port)):
                st.success("Serwer działa ✓")
            else:
                st.error("Brak połączenia z serwerem")
    else:
        vllm_protocol, vllm_ip, vllm_port = "http", "localhost", 8000
        st.warning("HF jest znacznie wolniejszy niż vLLM")

    st.divider()
    st.subheader("Tryb parsowania")
    prompt_mode = st.radio(
        "Prompt",
        options=list(PROMPT_MODES.keys()),
        format_func=lambda x: PROMPT_MODES[x],
        label_visibility="collapsed",
    )

    st.divider()
    st.subheader("Opcje")
    skip_hf = st.toggle("Pomijaj nagłówki/stopki stron", value=True)
    fitz_preprocess = st.toggle(
        "Fitz preprocess (lepsza jakość)",
        value=False,
        help="Preprocessing obrazu przez PyMuPDF — może poprawić jakość",
    )
    dpi = st.select_slider(
        "DPI (konwersja PDF)",
        options=[100, 150, 200, 250, 300],
        value=200,
        help="200 to wartość zalecana przez twórców modelu",
    )
    dots_output_dir = st.text_input(
        "Katalog wyjściowy parsera",
        value="./output",
        help="Bezwzględna ścieżka gdzie dots.ocr zapisuje pliki JSONL/MD/JPG",
    )

    st.divider()
    st.caption(
        "**Uruchomienie serwera vLLM:**\n"
        "```\n"
        "export hf_model_path=./weights/DotsOCR\n"
        'export PYTHONPATH=$(dirname "$hf_model_path"):$PYTHONPATH\n'
        "vllm serve $hf_model_path \\\\\n"
        "  --trust-remote-code \\\\\n"
        "  --gpu-memory-utilization 0.95 \\\\\n"
        "  --chat-template-content-format string \\\\\n"
        "  --served-model-name model\n"
        "```"
    )


# ── Nagłówek ──────────────────────────────────────────────────────────────────
st.title("dots.ocr — Dokumenty sądowe")
st.caption(
    "SOTA Multilingual Document Layout Parser · 1.7B · Lepszy niż GPT-4o i Gemini 2.5 Pro"
)

tab_upload, tab_dir = st.tabs(["📎 Upload pliku", "📁 Katalog"])

# ══ Tab 1: Upload ═════════════════════════════════════════════════════════════
with tab_upload:
    uploaded = st.file_uploader(
        "Przeciągnij lub wybierz plik",
        type=["pdf", "jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        if st.button("▶ Uruchom OCR", key="btn_upload"):
            all_results = {}
            progress = st.progress(0, text="Inicjalizacja…")
            error_log = []

            for i, f in enumerate(uploaded):
                progress.progress(i / len(uploaded), text="Przetwarzam: " + f.name)
                suffix = Path(f.name).suffix.lower()
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name
                try:
                    results = run_parser(
                        tmp_path,
                        backend,
                        vllm_protocol,
                        vllm_ip,
                        int(vllm_port),
                        prompt_mode,
                        skip_hf,
                        dots_output_dir,
                        dpi,
                        fitz_preprocess,
                    )
                    all_results[f.name] = results
                except Exception as e:
                    error_log.append(f.name + ": " + str(e))
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

            progress.progress(1.0, text="Gotowe ✓")
            for err in error_log:
                st.error(err)
            st.session_state["upload_results"] = all_results
            st.session_state["upload_meta"] = {
                "backend": backend,
                "prompt_mode": prompt_mode,
            }

    if "upload_results" in st.session_state:
        all_results = st.session_state["upload_results"]
        meta = st.session_state.get("upload_meta", {})
        st.success(
            "Przetworzono " + str(len(all_results)) + " plik(ów) · "
            "backend: " + meta.get("backend", "?").upper() + " · "
            "tryb: " + PROMPT_MODES.get(meta.get("prompt_mode", ""), "?")
        )
        st.divider()
        for fname, results in all_results.items():
            with st.expander("📄 " + fname, expanded=True):
                render_results(results, fname)

# ══ Tab 2: Katalog ════════════════════════════════════════════════════════════
with tab_dir:
    dir_path = st.text_input(
        "Ścieżka do katalogu",
        placeholder="/home/kwasiucionek/dokumenty/",
        label_visibility="collapsed",
    )

    if dir_path:
        p = Path(dir_path)
        if not p.exists():
            st.error("Katalog nie istnieje: " + dir_path)
        elif not p.is_dir():
            st.error("Podana ścieżka nie jest katalogiem")
        else:
            files = sorted(f for f in p.iterdir() if f.suffix.lower() in SUPPORTED)
            if not files:
                st.warning("Brak obsługiwanych plików w katalogu")
            else:
                st.info("Znaleziono " + str(len(files)) + " plików")
                for f in files:
                    st.caption(f.name)

                if st.button("▶ Uruchom OCR na wszystkich", key="btn_dir"):
                    all_results = {}
                    progress = st.progress(0, text="Inicjalizacja…")
                    error_log = []

                    for i, f in enumerate(files):
                        progress.progress(i / len(files), text="Przetwarzam: " + f.name)
                        try:
                            results = run_parser(
                                str(f),
                                backend,
                                vllm_protocol,
                                vllm_ip,
                                int(vllm_port),
                                prompt_mode,
                                skip_hf,
                                dots_output_dir,
                                dpi,
                                fitz_preprocess,
                            )
                            all_results[f.name] = results
                            f.with_suffix(".md").write_text(
                                results_to_md(results, f.name), encoding="utf-8"
                            )
                            f.with_suffix(".txt").write_text(
                                results_to_txt(results), encoding="utf-8"
                            )
                        except Exception as e:
                            error_log.append(f.name + ": " + str(e))

                    progress.progress(1.0, text="Gotowe ✓")
                    for err in error_log:
                        st.error(err)
                    st.session_state["dir_results"] = all_results
                    st.success(
                        "Przetworzono " + str(len(all_results)) + " plików · "
                        "MD i TXT zapisane obok oryginałów"
                    )

                if "dir_results" in st.session_state:
                    st.divider()
                    for fname, results in st.session_state["dir_results"].items():
                        with st.expander("📄 " + fname):
                            render_results(results, fname)
