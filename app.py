"""Court OCR — Masowy OCR dokumentów sądowych.

W pełni lokalna aplikacja:
- OCR: Marker + Surya (lokalne modele)
- Vision OCR: DeepSeek-OCR, MiniCPM-V i inne (przez Ollama)
- Ekstrakcja struktury: Ollama (lokalne LLM)
- Korekta tekstu OCR: Ollama (lokalne LLM)
- Żadne dane nie opuszczają maszyny
"""

import warnings

warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", message=".*no running event loop.*")

import io
import json
import logging
import time
import zipfile
from datetime import datetime
from pathlib import Path

import streamlit as st

from config import (
    DEVICE,
    OLLAMA_DEFAULT_MODEL,
    OLLAMA_RECOMMENDED_MODELS,
    OUTPUT_DIR,
    SUPPORTED_EXTENSIONS,
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_PDF_EXTENSIONS,
    TEMP_DIR,
    VISION_OCR_MODELS,
)
from ocr_engine import get_engine
from structure_parser import check_ollama_status, correct_ocr_text, parse_document

logging.getLogger("torch").setLevel(logging.ERROR)

st.set_page_config(
    page_title="Court OCR",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("⚖️ Court OCR — lokalna ekstrakcja dokumentów")


# ── Status systemu ────────────────────────────────────────────
def _show_status():
    with st.sidebar:
        st.header("📊 Status")
        if DEVICE == "cuda":
            st.success("GPU: CUDA")
        else:
            st.warning("CPU mode")
        ollama_status = check_ollama_status()
        if ollama_status["ollama_running"] and ollama_status["model_available"]:
            st.success("LLM: Ollama ✓")
        elif ollama_status["ollama_running"]:
            st.warning("Ollama ✓, brak modelu")
        else:
            st.info("Ollama wyłączona")
        st.caption("🔒 100% lokalnie")
        return ollama_status


ollama_status = _show_status()


# ── Sidebar — ustawienia ─────────────────────────────────────
with st.sidebar:
    st.divider()
    st.header("⚙️ OCR")

    engine_choice = st.selectbox(
        "Silnik OCR",
        ["auto", "marker", "surya", "vision", "paddleocr-vl"],
        help="auto=Marker/Surya, vision=Vision LLM, paddleocr-vl=PaddleOCR-VL-1.5",
    )

    if engine_choice == "paddleocr-vl":
        from paddleocr_vl import is_available

        if not is_available():
            st.error(
                "❌ Serwer PaddleOCR-VL niedostępny. Uruchom: `python paddleocr_vl_server.py`"
            )

    # Wybór modelu vision
    vision_model = None
    if engine_choice == "vision":
        if not ollama_status.get("ollama_running"):
            st.error("Vision OCR wymaga uruchomionej Ollama!")
        else:
            available = ollama_status.get("available_models", [])
            vision_available = [
                m
                for m in available
                if any(v.split(":")[0] == m.split(":")[0] for v in VISION_OCR_MODELS)
            ]
            other_models = [m for m in available if m not in vision_available]
            all_vision_options = vision_available + other_models

            if not all_vision_options:
                st.error(
                    "Brak modeli w Ollama.\nZainstaluj: `ollama pull deepseek-ocr:3b`"
                )
            else:
                default_idx = 0
                for i, m in enumerate(all_vision_options):
                    if "deepseek-ocr" in m:
                        default_idx = i
                        break

                vision_model = st.selectbox(
                    "Model Vision OCR",
                    all_vision_options,
                    index=default_idx,
                    help="deepseek-ocr — najlepszy do dokumentów, minicpm-v — uniwersalny",
                )
                if vision_available:
                    st.caption(f"✅ Rekomendowane: {', '.join(vision_available)}")

        st.info(
            "💡 Vision OCR wysyła obraz bezpośrednio do modelu AI — najlepszy dla tekstu odręcznego."
        )

    do_preprocess = st.checkbox("Preprocessing obrazu", value=True)

    st.divider()
    st.header("🏛️ Ekstrakcja")

    use_llm = st.checkbox(
        "Użyj lokalnego LLM",
        value=ollama_status.get("model_available", False),
        disabled=not ollama_status.get("ollama_running", False),
    )
    use_ocr_correction = st.checkbox(
        "Korekta tekstu OCR przez LLM",
        value=False,
        disabled=not ollama_status.get("ollama_running", False),
        help="Poprawia błędy OCR — szczególnie dla tekstu odręcznego. Wolniejsze.",
    )

    selected_model = None
    if ollama_status.get("ollama_running"):
        available_models = ollama_status.get("available_models", [])
        if available_models:

            def _model_sort_key(name):
                base = name.split(":")[0]
                for i, rec in enumerate(OLLAMA_RECOMMENDED_MODELS):
                    if rec.split(":")[0] == base:
                        return (0, i)
                return (1, 0)

            sorted_models = sorted(available_models, key=_model_sort_key)
            default_idx = 0
            for i, m in enumerate(sorted_models):
                if OLLAMA_DEFAULT_MODEL.split(":")[0] in m:
                    default_idx = i
                    break

            selected_model = st.selectbox(
                "Model LLM (ekstrakcja)",
                sorted_models,
                index=default_idx,
                disabled=not use_llm,
            )
            model_sizes = {
                "7b": "~5GB",
                "8b": "~5GB",
                "12b": "~8GB",
                "14b": "~9GB",
                "27b": "~17GB",
                "32b": "~20GB",
            }
            for size, vram in model_sizes.items():
                if size in (selected_model or ""):
                    st.caption(f"VRAM: {vram}")
                    break

    st.divider()
    st.header("🖼️ Preprocessing")
    auto_perspective = st.checkbox("Korekcja perspektywy", value=True)
    auto_deskew = st.checkbox("Korekcja nachylenia", value=True)
    auto_denoise = st.checkbox("Usuwanie szumu", value=True)
    auto_contrast = st.checkbox("Poprawa kontrastu", value=True)


# ── Funkcje wspólne ──────────────────────────────────────────


def process_single_file(engine, file_path: Path, filename: str) -> dict:
    start_time = time.time()
    ocr_result = engine.process_file(
        file_path,
        preprocess=do_preprocess,
        engine=engine_choice,
        vision_model=vision_model,
        preprocess_kwargs={
            "auto_perspective": auto_perspective,
            "auto_deskew": auto_deskew,
            "auto_denoise": auto_denoise,
            "auto_contrast": auto_contrast,
        },
    )
    ocr_time = time.time() - start_time

    correction_info = None
    ocr_text = ocr_result["text"]
    if use_ocr_correction:
        correction_start = time.time()
        correction = correct_ocr_text(ocr_text, model=selected_model)
        correction_time = time.time() - correction_start
        ocr_text = correction["corrected_text"]
        correction_info = {
            "was_corrected": correction["was_corrected"],
            "correction_time_seconds": round(correction_time, 2),
        }

    structure_start = time.time()
    structure = parse_document(ocr_text, use_llm=use_llm, model=selected_model)
    structure_time = time.time() - structure_start
    total_elapsed = time.time() - start_time

    return {
        "filename": filename,
        "ocr": {
            "text": ocr_text,
            "text_original": ocr_result["text"] if correction_info else None,
            "pages": ocr_result["pages"],
            "metadata": {
                **ocr_result["metadata"],
                "ocr_time_seconds": round(ocr_time, 2),
                "timestamp": datetime.now().isoformat(),
            },
            "preprocessing_info": ocr_result.get("preprocessing_info", []),
            "correction": correction_info,
        },
        "structure": structure.to_dict(),
        "processing": {
            "total_time_seconds": round(total_elapsed, 2),
            "ocr_time_seconds": round(ocr_time, 2),
            "correction_time_seconds": correction_info["correction_time_seconds"]
            if correction_info
            else 0,
            "structure_extraction_time_seconds": round(structure_time, 2),
            "extraction_method": structure.extraction_method,
        },
    }


def process_uploaded_file(engine, uploaded_file) -> dict:
    temp_path = TEMP_DIR / f"upload_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    try:
        return process_single_file(engine, temp_path, uploaded_file.name)
    finally:
        temp_path.unlink(missing_ok=True)


def display_result(result, idx=0):
    if "error" in result:
        st.error(f"Błąd: {result['error']}")
        return

    structure = result.get("structure", {})
    proc = result.get("processing", {})

    typ = structure.get("typ_dokumentu")
    syg = structure.get("sygnatura_akt")
    header_parts = [
        t
        for t in [typ.upper() if typ else None, syg, structure.get("data_dokumentu")]
        if t
    ]
    if header_parts:
        st.subheader(" | ".join(header_parts))

    streszczenie = structure.get("streszczenie")
    if streszczenie:
        st.info(streszczenie)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if structure.get("sad"):
            st.metric("Sąd", structure["sad"])
    with col2:
        if structure.get("wydzial"):
            st.metric("Wydział", structure["wydzial"])
    with col3:
        if structure.get("nadawca"):
            st.metric("Nadawca", structure["nadawca"])
    with col4:
        if structure.get("adresat"):
            st.metric("Adresat", structure["adresat"])

    strony = structure.get("strony") or {}
    if any(strony.values()):
        typ_lower = (structure.get("typ_dokumentu") or "").lower()
        if typ_lower in ("skarga", "pozew", "apelacja", "zażalenie", "kasacja"):
            role_labels = {
                "powod": "Skarżący/Powód",
                "pozwany": "Organ/Pozwany",
                "pelnomocnicy": "Pełnomocnicy",
            }
        elif typ_lower in ("wyrok", "postanowienie", "nakaz_zaplaty", "zarządzenie"):
            role_labels = {
                "powod": "Powód",
                "pozwany": "Pozwany",
                "pelnomocnicy": "Pełnomocnicy",
            }
        else:
            role_labels = {
                "powod": "Strona/Autor",
                "pozwany": "Strona/Adresat",
                "pelnomocnicy": "Pełnomocnicy/Przedstawiciele",
            }
        st.write("**Strony / uczestnicy:**")
        for role, names in strony.items():
            if names:
                label = role_labels.get(role, role.replace("_", " ").title())
                st.write(f"- **{label}:** {', '.join(names)}")

    sedziowie = structure.get("sedziowie", [])
    if sedziowie:
        st.write(f"**Sędziowie:** {', '.join(sedziowie)}")

    przedmiot = structure.get("przedmiot_sprawy")
    if przedmiot:
        st.write(f"**Przedmiot sprawy:** {przedmiot}")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        for k in structure.get("kwoty", []):
            st.write(f"- 💰 {k}")
    with col_b:
        for p in structure.get("podstawa_prawna", []):
            st.write(f"- 📜 {p}")
    with col_c:
        for t in structure.get("terminy", []):
            st.write(f"- 📅 {t}")

    sentencja = structure.get("sentencja")
    if sentencja:
        with st.expander("📜 Sentencja / rozstrzygnięcie"):
            st.write(sentencja)

    with st.expander("📄 Tekst OCR"):
        ocr_data = result.get("ocr", {})
        ocr_text = ocr_data.get("text", "")
        original_text = ocr_data.get("text_original")
        correction = ocr_data.get("correction")

        if original_text and correction and correction.get("was_corrected"):
            view_mode = st.radio(
                "Widok",
                ["Skorygowany (Markdown)", "Skorygowany (tekst)", "Oryginał OCR"],
                horizontal=True,
                key=f"view_{result.get('filename', '')}_{idx}",
                label_visibility="collapsed",
            )
            st.caption(f"✏️ Korekta: {correction.get('correction_time_seconds', '?')}s")
            if view_mode == "Skorygowany (Markdown)":
                st.markdown(ocr_text)
            elif view_mode == "Skorygowany (tekst)":
                st.code(ocr_text, language="markdown")
            else:
                st.code(original_text, language="markdown")
        else:
            view_mode = st.radio(
                "Widok",
                ["Markdown", "Surowy tekst"],
                horizontal=True,
                key=f"view_{result.get('filename', '')}_{idx}",
                label_visibility="collapsed",
            )
            if view_mode == "Markdown":
                st.markdown(ocr_text)
            else:
                st.code(ocr_text, language="markdown")

    with st.expander("🔧 Pełny JSON"):
        st.json(result)

    engine_info = result.get("ocr", {}).get("metadata", {}).get("engine", "?")
    time_parts = [
        f"Silnik: {engine_info}",
        f"OCR: {proc.get('ocr_time_seconds', '?')}s",
    ]
    if proc.get("correction_time_seconds"):
        time_parts.append(f"Korekta: {proc['correction_time_seconds']}s")
    time_parts.append(
        f"Ekstrakcja: {proc.get('structure_extraction_time_seconds', '?')}s"
    )
    time_parts.append(f"Metoda: {proc.get('extraction_method', '?')}")

    prep_info = result.get("ocr", {}).get("preprocessing_info", [])
    if prep_info:
        if isinstance(prep_info, list) and prep_info and isinstance(prep_info[0], dict):
            # format wielostronicowy: [{page: 1, corrections: [...]}, ...]
            all_corrections = set()
            for p in prep_info:
                all_corrections.update(p.get("corrections", []))
            if all_corrections:
                time_parts.append(f"Preprocessing: {', '.join(sorted(all_corrections))}")
        elif isinstance(prep_info, list) and prep_info and isinstance(prep_info[0], str):
            time_parts.append(f"Preprocessing: {', '.join(prep_info)}")

    st.caption(" | ".join(time_parts))


def display_results(results):
    if not results:
        return
    st.success(f"Przetworzono {len(results)} dokumentów")
    tab_names = [r.get("filename", f"Dok. {i + 1}") for i, r in enumerate(results)]
    tabs = st.tabs(tab_names)
    for i, (tab, result) in enumerate(zip(tabs, results)):
        with tab:
            display_result(result, idx=i)
    st.divider()
    _export_buttons(results)


def _export_buttons(results):
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        st.download_button(
            "⬇️ Wszystko (JSON)",
            data=json.dumps(results, ensure_ascii=False, indent=2),
            file_name=f"court_ocr_{datetime.now():%Y%m%d_%H%M%S}.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_e2:
        structures_only = [
            {
                "filename": r.get("filename"),
                "structure": {
                    k: v
                    for k, v in r.get("structure", {}).items()
                    if k not in ("raw_text", "raw_text_preview")
                },
                "processing": r.get("processing", {}),
            }
            for r in results
        ]
        st.download_button(
            "⬇️ Tylko struktury (JSON)",
            data=json.dumps(structures_only, ensure_ascii=False, indent=2),
            file_name=f"court_structures_{datetime.now():%Y%m%d_%H%M%S}.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_e3:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for r in results:
                fname = Path(r.get("filename", "unknown")).stem
                zf.writestr(
                    f"{fname}.json", json.dumps(r, ensure_ascii=False, indent=2)
                )
        st.download_button(
            "⬇️ ZIP (osobne pliki)",
            data=zip_buffer.getvalue(),
            file_name=f"court_ocr_{datetime.now():%Y%m%d_%H%M%S}.zip",
            mime="application/zip",
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════
tab_upload, tab_batch = st.tabs(["📤 Upload plików", "📁 Batch z katalogu"])

with tab_upload:
    uploaded_files = st.file_uploader(
        "Wgraj dokumenty",
        type=["pdf", "jpg", "jpeg", "png", "tiff", "tif", "bmp", "webp"],
        accept_multiple_files=True,
        help="Wszystko przetwarzane lokalnie",
    )

    if not uploaded_files:
        st.info("Wgraj dokumenty — skargi, orzeczenia, pisma, zdjęcia z telefonu.")
    elif st.button(
        "🚀 Uruchom OCR", type="primary", use_container_width=True, key="btn_upload"
    ):
        engine = get_engine()
        results = []
        total = len(uploaded_files)
        progress = st.progress(0, text="Inicjalizacja...")

        for idx, uploaded_file in enumerate(uploaded_files):
            progress.progress(
                idx / total, text=f"[{idx + 1}/{total}] {uploaded_file.name}"
            )
            with st.status(f"📄 {uploaded_file.name}", expanded=True) as status:
                try:
                    engine_label = (
                        f"vision:{vision_model}"
                        if engine_choice == "vision"
                        else engine_choice
                    )
                    st.write(f"🔍 OCR ({engine_label})...")
                    result = process_uploaded_file(engine, uploaded_file)

                    if use_ocr_correction and result.get("ocr", {}).get(
                        "correction", {}
                    ).get("was_corrected"):
                        st.write("✏️ Tekst skorygowany przez LLM")

                    typ = result.get("structure", {}).get("typ_dokumentu") or "?"
                    syg = result.get("structure", {}).get("sygnatura_akt", "")
                    elapsed = result["processing"]["total_time_seconds"]
                    st.write(f"📋 {typ.upper()} {syg}")
                    status.update(
                        label=f"✅ {uploaded_file.name} ({elapsed:.1f}s)",
                        state="complete",
                        expanded=False,
                    )
                    results.append(result)
                except Exception as e:
                    status.update(label=f"❌ {uploaded_file.name}", state="error")
                    st.error(str(e))
                    results.append({"filename": uploaded_file.name, "error": str(e)})

        progress.progress(1.0, text=f"Gotowe! {total} dokumentów")
        st.session_state["upload_results"] = results

    if "upload_results" in st.session_state:
        display_results(st.session_state["upload_results"])

with tab_batch:
    st.subheader("📁 Przetwarzanie masowe")
    st.caption("Wgraj wiele plików naraz — wyniki do pobrania jako ZIP")

    batch_files = st.file_uploader(
        "Wgraj dokumenty (batch)",
        type=["pdf", "jpg", "jpeg", "png", "tiff", "tif", "bmp", "webp"],
        accept_multiple_files=True,
        key="batch_uploader",
        help="Wybierz wiele plików na raz (Ctrl+A w oknie wyboru)",
    )

    if not batch_files:
        st.info("Wybierz pliki do masowego przetwarzania. Wyniki pobierzesz jako ZIP.")
    else:
        st.metric("Plików do przetworzenia", len(batch_files))

        if st.button(
            f"🚀 Przetwórz {len(batch_files)} plików",
            type="primary",
            use_container_width=True,
            key="btn_batch",
        ):
            engine = get_engine()
            results, errors = [], []
            total = len(batch_files)
            total_time_start = time.time()

            progress = st.progress(0, text="Inicjalizacja...")

            for idx, uploaded_file in enumerate(batch_files):
                progress.progress(
                    idx / total, text=f"[{idx + 1}/{total}] {uploaded_file.name}"
                )
                with st.status(f"📄 {uploaded_file.name}", expanded=False) as status:
                    try:
                        result = process_uploaded_file(engine, uploaded_file)
                        typ = result.get("structure", {}).get("typ_dokumentu") or "?"
                        elapsed = result["processing"]["total_time_seconds"]
                        status.update(
                            label=f"✅ {uploaded_file.name} — {typ.upper()} ({elapsed:.1f}s)",
                            state="complete",
                        )
                        results.append(result)
                    except Exception as e:
                        status.update(label=f"❌ {uploaded_file.name}", state="error")
                        st.error(str(e))
                        errors.append({"filename": uploaded_file.name, "error": str(e)})

            total_time = time.time() - total_time_start
            progress.progress(
                1.0, text=f"Gotowe! {len(results)}/{total} w {total_time:.1f}s"
            )

            st.divider()
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            with col_r1:
                st.metric("Przetworzone", len(results))
            with col_r2:
                st.metric("Błędy", len(errors))
            with col_r3:
                st.metric("Czas łączny", f"{total_time:.0f}s")
            with col_r4:
                st.metric("Średnio/plik", f"{total_time / max(len(results), 1):.1f}s")

            if errors:
                with st.expander(f"❌ Błędy ({len(errors)})"):
                    for err in errors:
                        st.write(f"- **{err['filename']}:** {err['error']}")

            if results:
                st.subheader("📋 Streszczenia")
                summary_data = [
                    {
                        "Plik": r.get("filename", "?"),
                        "Typ": (
                            r.get("structure", {}).get("typ_dokumentu") or "?"
                        ).upper(),
                        "Sygnatura": r.get("structure", {}).get("sygnatura_akt") or "—",
                        "Streszczenie": r.get("structure", {}).get("streszczenie") or "—",
                        "Czas (s)": r.get("processing", {}).get("total_time_seconds", "?"),
                    }
                    for r in results
                ]
                st.dataframe(summary_data, use_container_width=True, hide_index=True)

            st.session_state["batch_results"] = results

        if "batch_results" in st.session_state and st.session_state["batch_results"]:
            results = st.session_state["batch_results"]
            with st.expander("🔍 Szczegóły dokumentów"):
                btabs = st.tabs(
                    [r.get("filename", f"Dok. {i + 1}") for i, r in enumerate(results)]
                )
                for i, (btab, result) in enumerate(zip(btabs, results)):
                    with btab:
                        display_result(result, idx=1000 + i)

            _export_buttons(results)
