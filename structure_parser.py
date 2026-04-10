"""Ekstrakcja struktury polskich dokumentów sądowych.

Podejście hybrydowe:
1. Lokalny LLM (Ollama) — rozumie kontekst, radzi sobie z różnymi typami dokumentów
2. Regex fallback — gdy LLM niedostępny lub dla szybkiej ekstrakcji sygnatury

Obsługiwane typy dokumentów:
- Orzeczenia (wyroki, postanowienia, nakazy)
- Skargi i pozwy
- Pisma procesowe
- Protokoły rozpraw
- Wezwania i zawiadomienia
- Pełnomocnictwa
- Zarządzenia
- Środki odwoławcze (apelacje, zażalenia, kasacje)
"""

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Optional

import requests

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_DEFAULT_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_TIMEOUT,
)

logger = logging.getLogger(__name__)

# ── Prompty ───────────────────────────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """\
Jesteś ekspertem od polskich dokumentów sądowych i prawnych. Twoim zadaniem jest
ekstrakcja strukturalnych danych z tekstu OCR dokumentu.

ZASADY:
1. Zwracaj TYLKO poprawny JSON — bez tekstu przed/po, bez markdown.
2. Jeśli pole nie występuje w dokumencie, ustaw wartość na null.
3. Zachowuj oryginalne brzmienie tekstu — nie poprawiaj treści merytorycznej.
4. Pole "typ_dokumentu" musi być jednym z: wyrok, postanowienie, nakaz_zaplaty,
   zarządzenie, skarga, pozew, apelacja, zażalenie, kasacja, pismo_procesowe,
   protokol, wezwanie, pelnomocnictwo, zawiadomienie, inne.
5. Pole "streszczenie" — krótkie (2-3 zdania) podsumowanie treści dokumentu.

Zwróć JSON o następującej strukturze (pola, których nie ma w dokumencie = null):
{
    "typ_dokumentu": "wyrok | postanowienie | skarga | pozew | apelacja | ...",
    "sygnatura_akt": "np. I C 123/24",
    "sad": "pełna nazwa sądu",
    "wydzial": "nazwa wydziału jeśli podany",
    "data_dokumentu": "data w formacie YYYY-MM-DD jeśli możliwe, inaczej oryginalny tekst",
    "nadawca": "kto wystawił/napisał dokument",
    "adresat": "do kogo skierowany",
    "strony": {
        "powod": ["lista powodów/wnioskodawców/skarżących"],
        "pozwany": ["lista pozwanych/uczestników/organów"],
        "pelnomocnicy": ["pełnomocnicy stron jeśli wymienieni"]
    },
    "sedziowie": ["lista sędziów/asesorów/referendarzy"],
    "przedmiot_sprawy": "czego dotyczy sprawa",
    "sentencja": "treść rozstrzygnięcia (jeśli to orzeczenie)",
    "uzasadnienie_fragment": "pierwsze 500 znaków uzasadnienia (jeśli jest)",
    "streszczenie": "2-3 zdania podsumowania treści dokumentu",
    "kwoty": ["wymienione kwoty pieniężne, np. '10 000 zł', '5 000 EUR'"],
    "terminy": ["daty/terminy wymienione w dokumencie"],
    "podstawa_prawna": ["przywołane przepisy, np. 'art. 233 § 1 k.p.c.'"]
}

WAŻNE: Zwróć TYLKO JSON, bez dodatkowego tekstu."""


OCR_CORRECTION_PROMPT = """Jesteś korektorem po wykonanym OCR.
ZAWSZE:
- Popraw TYLKO literówki i błędy rozpoznawania znaków i słów
- NIE DODAWAJ nowych słów, myślników, list, nawiasów
- Zwróć IDENTYCZNY tekst, tylko z poprawioną pisownią, nie zmieniaj, nie parafrazuj, nie skracaj tekstu
- WYJĄTEK - usuń tzw. pętle powtórzeń
- Nieczytelne słowa oznacz [?]"""


# ── Stałe ─────────────────────────────────────────────────────────────────────

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 150

# ── Regex patterns ────────────────────────────────────────────────────────────

_SYGNATURA_RE = re.compile(
    r"\b([IVXLCDM]+\s+\w{1,5}\s+\d+/\d{2,4})\b",
    re.IGNORECASE,
)
_DATA_RE = re.compile(r"\b(\d{1,2}[\s./-]\w+[\s./-]\d{4}|\d{4}-\d{2}-\d{2})\b")
_SAD_RE = re.compile(
    r"(Sąd\s+(?:Rejonowy|Okręgowy|Apelacyjny|Najwyższy|Administracyjny)"
    r"(?:\s+w\s+[\w\s]+)?)",
    re.IGNORECASE,
)
_TYP_DOKUMENTU_MAP = {
    r"\bwyrok\b": "wyrok",
    r"\bpostanowienie\b": "postanowienie",
    r"\bnakaz\s+zap[łl]aty\b": "nakaz_zaplaty",
    r"\bprotok[oó][łl]\b": "protokol",
    r"\bapelacja\b": "apelacja",
    r"\bza[żz]alenie\b": "zażalenie",
    r"\bkasacja\b": "kasacja",
    r"\bskarga\b": "skarga",
    r"\bpozew\b": "pozew",
    r"\bpismo\s+procesowe\b": "pismo_procesowe",
    r"\bwezwanie\b": "wezwanie",
    r"\bzarz[ąa]dzenie\b": "zarządzenie",
    r"\bpe[łl]nomocnictwo\b": "pelnomocnictwo",
    r"\bzawiadomienie\b": "zawiadomienie",
}


# ── DocumentStructure dataclass ───────────────────────────────────────────────


@dataclass
class DocumentStructure:
    """Struktura wyekstrahowana z dokumentu sądowego."""

    raw_text: str = ""
    extraction_method: str = "none"
    extraction_errors: list = field(default_factory=list)

    # Pola dokumentu
    typ_dokumentu: Optional[str] = None
    sygnatura_akt: Optional[str] = None
    data_dokumentu: Optional[str] = None
    sad: Optional[str] = None
    wydzial: Optional[str] = None
    sedziowie: Optional[list] = None
    strony: Optional[dict] = None
    przedmiot_sprawy: Optional[str] = None
    sentencja: Optional[str] = None
    streszczenie: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ── Wewnętrzne helpery ────────────────────────────────────────────────────────


def _smart_truncate(text: str, max_chars: int = 5000) -> str:
    """Zachowaj nagłówek (2/3) i koniec dokumentu (1/3)."""
    if len(text) <= max_chars:
        return text
    header_size = max_chars * 2 // 3
    tail_size = max_chars - header_size
    return (
        f"{text[:header_size]}\n\n[... fragment pominięty ...]\n\n{text[-tail_size:]}"
    )


def _clean_llm_response(raw: str) -> str:
    """Usuń tagi <think> i markdown z odpowiedzi LLM."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def _parse_llm_json(raw: str) -> Optional[dict]:
    """Bezpiecznie sparsuj JSON z odpowiedzi LLM."""
    raw = _clean_llm_response(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    logger.warning("Nie udało się sparsować JSON z LLM")
    return None


def _split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Dzieli tekst na nakładające się fragmenty, tnąc na końcu akapitu."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            window_start = max(start + chunk_size - 300, start)
            window_end = min(start + chunk_size + 300, len(text))
            search_area = text[window_start:window_end]

            para_match = list(re.finditer(r"\n\n", search_area))
            if para_match:
                end = window_start + para_match[-1].end()
            else:
                line_match = list(re.finditer(r"\n", search_area))
                if line_match:
                    end = window_start + line_match[-1].end()

        chunks.append(text[start:end])
        start = max(start + 1, end - overlap)

    return chunks


def _is_chatbot_response(text: str) -> bool:
    """Sprawdź czy model odpowiedział na tekst zamiast go poprawić."""
    chatbot_markers = [
        r"^(oczywiście|jasne|rozumiem|chętnie|z przyjemnością)",
        r"^(niestety|przepraszam|przykro mi)",
        r"^(oto|poniżej|przedstawiam)",
        r"pytasz o",
        r"w (tym|powyższym) dokumencie",
        r"treść (dokumentu|tekstu) (mówi|dotyczy|opisuje)",
    ]
    text_lower = text.lower().strip()
    return any(re.search(p, text_lower) for p in chatbot_markers)


def _merge_llm_result(doc: DocumentStructure, data: dict) -> DocumentStructure:
    """Scal wynik LLM z obiektem DocumentStructure."""
    for f in (
        "typ_dokumentu",
        "sygnatura_akt",
        "data_dokumentu",
        "sad",
        "wydzial",
        "sedziowie",
        "strony",
        "przedmiot_sprawy",
        "sentencja",
        "streszczenie",
    ):
        value = data.get(f)
        if value is not None:
            setattr(doc, f, value)
    return doc


def _merge_page_results(
    doc: DocumentStructure, results: list[dict]
) -> DocumentStructure:
    """Scal wyniki ekstrakcji z wielu stron do jednego DocumentStructure."""
    for data in results:
        # Pierwsze niepuste wartości dla pól jednokrotnych
        for f in (
            "typ_dokumentu",
            "sygnatura_akt",
            "data_dokumentu",
            "sad",
            "wydzial",
            "przedmiot_sprawy",
            "sentencja",
            "streszczenie",
        ):
            if getattr(doc, f) is None and data.get(f):
                setattr(doc, f, data[f])

        # sedziowie — scalaj listy
        if data.get("sedziowie"):
            existing = doc.sedziowie or []
            for s in data["sedziowie"]:
                if s not in existing:
                    existing.append(s)
            doc.sedziowie = existing or None

        # strony — pierwsza niepusta
        if doc.strony is None and data.get("strony"):
            doc.strony = data["strony"]

    return doc


def _enrich_with_regex(doc: DocumentStructure, text: str) -> None:
    """Uzupełnij brakujące pola regexem (nie nadpisuje istniejących)."""
    if not doc.sygnatura_akt:
        m = _SYGNATURA_RE.search(text)
        if m:
            doc.sygnatura_akt = m.group(1).strip()

    if not doc.data_dokumentu:
        m = _DATA_RE.search(text)
        if m:
            doc.data_dokumentu = m.group(1).strip()

    if not doc.sad:
        m = _SAD_RE.search(text)
        if m:
            doc.sad = m.group(1).strip()

    if not doc.typ_dokumentu:
        text_lower = text.lower()
        for pattern, typ in _TYP_DOKUMENTU_MAP.items():
            if re.search(pattern, text_lower):
                doc.typ_dokumentu = typ
                break
        else:
            doc.typ_dokumentu = "inne"


def _extract_regex_only(doc: DocumentStructure, text: str) -> None:
    """Wypełnij doc wyłącznie przez regex (fallback bez LLM)."""
    m = _SYGNATURA_RE.search(text)
    if m:
        doc.sygnatura_akt = m.group(1).strip()

    m = _DATA_RE.search(text)
    if m:
        doc.data_dokumentu = m.group(1).strip()

    m = _SAD_RE.search(text)
    if m:
        doc.sad = m.group(1).strip()

    text_lower = text.lower()
    for pattern, typ in _TYP_DOKUMENTU_MAP.items():
        if re.search(pattern, text_lower):
            doc.typ_dokumentu = typ
            break
    else:
        doc.typ_dokumentu = "inne"


# ── LLM calls ─────────────────────────────────────────────────────────────────


def _correct_chunk(chunk: str, model: str) -> Optional[str]:
    """Popraw jeden fragment tekstu OCR przez LLM. Zwraca None przy błędzie."""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": OCR_CORRECTION_PROMPT},
                    {"role": "user", "content": chunk},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.05,
                    "num_predict": len(chunk) + 200,
                },
            },
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        corrected = response.json().get("message", {}).get("content", "").strip()
        corrected = _clean_llm_response(corrected)

        if not corrected or _is_chatbot_response(corrected):
            logger.warning("Model odpowiedział zamiast poprawić — zwracam oryginał")
            return None

        return corrected

    except Exception as e:
        logger.warning("Błąd korekty chunka: %s", e)
        return None


def _correct_text(text: str, model: str) -> str:
    """Koryguj tekst (z podziałem na chunki jeśli długi)."""
    if len(text) <= CHUNK_SIZE:
        result = _correct_chunk(text, model)
        return result if result else text

    chunks = _split_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
    corrected_chunks = []

    for i, chunk in enumerate(chunks):
        logger.debug("Chunk %d/%d (%d znaków)", i + 1, len(chunks), len(chunk))
        result = _correct_chunk(chunk, model)
        corrected_chunks.append(result if result else chunk)

    return "\n".join(corrected_chunks)


def _extract_with_llm(text: str, model: str) -> Optional[dict]:
    """Wyciągnij strukturę przez lokalny LLM (Ollama /api/chat)."""
    truncated = _smart_truncate(text, max_chars=5000)

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Przeanalizuj poniższy dokument sądowy i wyodrębnij strukturę:\n\n"
                            + truncated
                        ),
                    },
                ],
                "stream": False,
                "options": {
                    "temperature": OLLAMA_TEMPERATURE,
                    "num_predict": 2048,
                },
                "format": "json",
            },
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        raw = response.json().get("message", {}).get("content", "")
        return _parse_llm_json(raw)

    except requests.ConnectionError:
        logger.warning("Ollama niedostępna pod %s", OLLAMA_BASE_URL)
        return None
    except requests.Timeout:
        logger.warning("Ollama timeout po %ds", OLLAMA_TIMEOUT)
        return None
    except Exception as e:
        logger.error("Błąd LLM: %s", e)
        return None


# ── Główne API ────────────────────────────────────────────────────────────────


def correct_ocr_text(ocr_result, model: str = None) -> dict:
    """Korekcja OCR — strona po stronie.

    Args:
        ocr_result: dict z kluczami 'text' i opcjonalnie 'pages'
                    LUB zwykły string z tekstem OCR
        model: Model Ollama (None = domyślny z configu)

    Returns:
        dict: text, text_original, was_corrected, pages (jeśli wejście miało strony)
    """
    if isinstance(ocr_result, str):
        ocr_result = {"text": ocr_result}

    model = model or OLLAMA_DEFAULT_MODEL
    pages = ocr_result.get("pages", [])

    if not pages:
        original = ocr_result.get("text", "")
        corrected = _correct_text(original, model)
        return {
            "text": corrected,
            "corrected_text": corrected,
            "text_original": original,
            "was_corrected": corrected != original,
        }

    corrected_pages = []
    for page in pages:
        original = page.get("text", "")
        if not original.strip():
            corrected_pages.append(
                {
                    "page_number": page.get("page_number", 0),
                    "text": "",
                }
            )
            continue

        logger.info(
            "Korekcja strony %d/%d (%d znaków)",
            page.get("page_number", 0),
            len(pages),
            len(original),
        )
        corrected = _correct_text(original, model)
        corrected_pages.append(
            {
                "page_number": page.get("page_number", 0),
                "text": corrected,
            }
        )

    full_text = "\n\n".join(p["text"] for p in corrected_pages)
    full_original = "\n\n".join(p.get("text", "") for p in pages)

    return {
        "text": full_text,
        "corrected_text": full_text,
        "text_original": full_original,
        "pages": corrected_pages,
        "was_corrected": full_text != full_original,
    }


def parse_document(
    text: str, use_llm: bool = True, model: str = None
) -> DocumentStructure:
    """Parsuj dokument — strona po stronie jeśli podano pages, inaczej chunki.

    Args:
        text: Tekst po OCR (pełny, bez podziału na strony)
        use_llm: Czy próbować użyć LLM
        model: Model Ollama (None = domyślny z configu)

    Returns:
        DocumentStructure z wypełnionymi polami i metodą to_dict()
    """
    doc = DocumentStructure(raw_text=text)
    model = model or OLLAMA_DEFAULT_MODEL

    if not use_llm:
        _extract_regex_only(doc, text)
        doc.extraction_method = "regex"
        return doc

    # Podziel na strony (podwójny newline jako separator stron)
    # Jeśli tekst jest krótki — jedno wywołanie LLM
    if len(text) <= 5000:
        llm_result = _extract_with_llm(text, model)
        if llm_result:
            doc = _merge_llm_result(doc, llm_result)
            doc.extraction_method = f"llm:{model}"
            _enrich_with_regex(doc, text)
            return doc
        else:
            doc.extraction_errors.append("LLM niedostępny — fallback na regex")
            _extract_regex_only(doc, text)
            doc.extraction_method = "regex"
            return doc

    # Długi dokument — podziel na strony i przetwarzaj po kolei
    logger.info("Długi dokument (%d znaków) — przetwarzanie stronami", len(text))
    pages = _split_into_chunks(text, chunk_size=4000, overlap=0)
    page_results = []

    for i, page_text in enumerate(pages):
        if not page_text.strip():
            continue
        logger.info("Ekstrakcja struktury — strona %d/%d", i + 1, len(pages))
        result = _extract_with_llm(page_text, model)
        if result:
            page_results.append(result)

    if page_results:
        doc = _merge_page_results(doc, page_results)
        doc.extraction_method = f"llm:{model}:pages"
        _enrich_with_regex(doc, text)
    else:
        doc.extraction_errors.append("LLM nie zwrócił wyników — fallback na regex")
        _extract_regex_only(doc, text)
        doc.extraction_method = "regex"

    return doc


def check_ollama_status() -> dict:
    """Sprawdź czy Ollama jest dostępna i jakie modele ma zainstalowane.

    Returns:
        dict: ollama_running, model_available, model_name, url,
              available_models, error
    """
    status = {
        "ollama_running": False,
        "model_available": False,
        "model_name": OLLAMA_DEFAULT_MODEL,
        "url": OLLAMA_BASE_URL,
        "available_models": [],
        "error": None,
    }

    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
        status["ollama_running"] = True

        models = r.json().get("models", [])
        model_names = [m.get("name", "") for m in models]
        status["available_models"] = model_names

        base_name = OLLAMA_DEFAULT_MODEL.split(":")[0]
        status["model_available"] = any(base_name in name for name in model_names)

    except requests.ConnectionError:
        status["error"] = "Brak połączenia z Ollama"
    except Exception as e:
        status["error"] = str(e)

    return status
