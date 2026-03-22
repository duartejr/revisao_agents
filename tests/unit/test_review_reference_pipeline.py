"""Unit tests for deterministic reference pipeline in review chat handlers."""

from pathlib import Path
from unittest.mock import patch

from gradio_app.handlers import (
    _classify_reference_intent,
    _collect_reference_inventory,
    _extract_provided_reference_items,
    _extract_requested_citation_numbers,
    _format_abnt_entry,
    _handle_format_provided_references_request,
    _handle_reference_request,
    _is_metadata_complete,
    _is_reference_request,
    review_chat_turn,
)


def test_extract_requested_citation_numbers_brackets():
    text = "resolva [1] [6] [3] [10] em ABNT"
    assert _extract_requested_citation_numbers(text) == [1, 3, 6, 10]


def test_is_reference_request_with_listing_phrase():
    text = "liste todas as referências sem repetição usadas neste documento"
    assert _is_reference_request(text) is True


def test_classify_reference_intent_list_all():
    text = "liste todas as referências sem repetição usadas neste documento"
    assert _classify_reference_intent(text) == "list_all"


def test_classify_reference_intent_format_provided():
    text = (
        "Formate estas fontes no padrão ABNT:\n"
        "- Bleidorn, 2024\n"
        "- https://doi.org/10.1038/s41586-024-12345-6"
    )
    assert _classify_reference_intent(text) == "format_provided"


def test_classify_reference_intent_format_provided_long_list_message():
    text = (
        "formate a seguinte lista de referencias no padrão abnt: "
        "- [1] Artigo 16315 PT PB. 2025. DOI: 10.3390/ai6090215. "
        "Disponível em: https://www.mdpi.com/2673-2688/6/9/215. "
        "- [2] Probabilistic hierarchical interpolation. [s.d.]. "
        "Disponível em: https://hess.copernicus.org/articles/30/371/2026/."
    )
    assert _classify_reference_intent(text) == "format_provided"


def test_classify_reference_intent_no_false_list_all_for_lista_phrase():
    text = "eu pedi para format apenas a lista que eu forneci"
    assert _classify_reference_intent(text) is None


def test_extract_provided_reference_items_from_colon_block():
    text = "formate em abnt: Bleidorn, 2024; Smith et al., 2023"
    items = _extract_provided_reference_items(text)
    assert len(items) == 2
    assert "Bleidorn" in items[0]


def test_collect_reference_inventory_maps_numbered_and_citations():
    markdown = (
        "## 1. Intro\n\n"
        "Texto com citação [1] e [3].\n\n"
        "Além disso, resultados recentes (Bleidorn et al., 2024) reforçam essa visão.\n\n"
        "### Referências desta seção\n"
        "[1] AUTOR A. Título A.\n"
        "[2] AUTOR B. Título B.\n\n"
        "## 2. Discussão\n\n"
        "Outro texto com [2] e [3].\n"
    )

    inv = _collect_reference_inventory(markdown)

    assert 1 in inv["references_by_number"]
    assert 2 in inv["references_by_number"]
    assert inv["cited_numbers"] == [1, 2, 3]
    assert len(inv["unique_references"]) == 2
    assert any("Bleidorn" in item for item in inv["non_numbered_mentions"])


def test_format_abnt_entry_avoids_duplicated_doi_label():
    metadata = {
        "number": 1,
        "raw": "",
        "title": "Sample Title",
        "doi": "DOI: 10.1234/abc.2025.001",
        "url": "",
        "year": "2025",
        "file_path": "",
    }
    text = _format_abnt_entry(metadata)
    assert "DOI: DOI:" not in text
    assert "DOI: 10.1234/abc.2025.001" in text


def test_format_abnt_entry_cleans_malformed_doi_and_sd_duplication():
    metadata = {
        "number": 1,
        "raw": "",
        "title": "Leonardo Maciel de Sousa TCC. 2025. DOI:",
        "doi": "DOI:. [s.d.]. DOI: 10.1353/jod.2025.a970357",
        "url": "https://hess.copernicus.org/articles/29/6811/2025/hess-29-6811-2025.html.",
        "year": "",
        "file_path": "",
    }
    text = _format_abnt_entry(metadata)
    assert "DOI:." not in text
    assert "DOI: 10.1353/jod.2025.a970357" in text
    assert text.count("[s.d.].") <= 1


def test_format_abnt_entry_does_not_duplicate_raw_fragment():
    metadata = {
        "number": 2,
        "raw": "artigo final do projeto FACEPE FAPESP 24042025 para leitura de prova Val. 2022. DOI: 10.5194/hess-2022-334",
        "title": "artigo final do projeto FACEPE FAPESP 24042025 para leitura de prova Val",
        "doi": "10.5194/hess-2022-334",
        "url": "https://www.frontiersin.org/journals/water/articles/10.3389/frwa.2026.1756052/full",
        "year": "2022",
        "file_path": "",
    }
    text = _format_abnt_entry(metadata)
    assert text.lower().count("artigo final do projeto facepe") == 1


def test_incomplete_metadata_guidance_when_web_disabled():
    markdown = (
        "## 1. Intro\n\n"
        "Texto com citação [10].\n\n"
        "### Referências desta seção\n"
        "[10] /tmp/arquivo_local_sem_metadado.pdf\n"
    )

    reply, _meta = _handle_reference_request(
        markdown, "resolva [10] em ABNT", allow_web=False
    )
    assert "ative **Allow web search**" in reply


def test_provided_format_guidance_when_web_disabled():
    """Formatter agent is called even without web; it uses CrossRef+MongoDB."""
    user_text = """Formate estas fontes no padrão ABNT:
- /tmp/arquivo_local_sem_metadado.pdf
"""
    with (
        patch(
            "gradio_app.handlers.run_reference_extractor_agent",
            return_value="[1] TITLE: Test | AUTHORS: A. Author | YEAR: 2024 | DOI: N/A",
        ),
        patch(
            "gradio_app.handlers.run_reference_formatter_agent",
            return_value="[1] AUTHOR, A. **Test**. 2024.",
        ),
    ):
        reply, meta = _handle_format_provided_references_request(
            user_text, allow_web=False
        )
    assert meta["intent"] == "format_provided"
    assert meta["agent"] == "reference_extractor+reference_formatter"


def test_reference_list_requires_confirmation_before_execution(tmp_path: Path):
    review_file = tmp_path / "review.md"
    review_file.write_text(
        "## 1. Intro\n\nTexto com citação [1].\n\n### Referências desta seção\n[1] AUTOR A. Título A.\n",
        encoding="utf-8",
    )

    history: list = []
    session_state = {
        "working_copy_path": str(review_file),
        "chat_history": [],
    }

    history, session_state, status, _ = review_chat_turn(
        "liste todas as referências sem repetição usadas neste documento",
        history,
        session_state,
        web_enabled=False,
    )
    assert "Aguardando confirmação" in status
    assert session_state.get("awaiting_reference_confirmation") is True
    assert "Responda **sim**" in history[-1]["content"]

    with (
        patch(
            "gradio_app.handlers.run_reference_extractor_agent",
            return_value="[1] TITLE: Título A | AUTHORS: AUTOR, A. | YEAR: N/A | DOI: N/A",
        ),
        patch(
            "gradio_app.handlers.run_reference_formatter_agent",
            return_value="[1] AUTOR, A. **Título A**. [s.d.].",
        ),
    ):
        history, session_state, status, _ = review_chat_turn(
            "sim",
            history,
            session_state,
            web_enabled=False,
        )
    assert "Referências listadas" in status
    assert session_state.get("awaiting_reference_confirmation") is False
    assert "Referências" in history[-1]["content"]


def test_format_confirmation_blocks_without_web(tmp_path: Path):
    review_file = tmp_path / "review.md"
    review_file.write_text("## 1. Intro\n\nTexto simples.\n", encoding="utf-8")

    history: list = []
    session_state = {
        "working_copy_path": str(review_file),
        "chat_history": [],
    }

    history, session_state, status, _ = review_chat_turn(
        "Formate estas fontes no padrão ABNT:\n- /tmp/arquivo_local_sem_metadado.pdf",
        history,
        session_state,
        web_enabled=False,
    )
    assert "Aguardando confirmação" in status
    assert session_state.get("awaiting_reference_confirmation") is True
    assert "habilite **Allow web search**" in history[-1]["content"]

    history, session_state, status, _ = review_chat_turn(
        "sim",
        history,
        session_state,
        web_enabled=False,
    )
    assert "Habilite web" in status
    assert session_state.get("awaiting_reference_confirmation") is True
    assert "Não executei a formatação" in history[-1]["content"]


def test_metadata_complete_with_doi():
    metadata = {
        "title": "Any title",
        "year": "",
        "doi": "10.1111/xyz.2025.10",
        "url": "",
        "derived_from_path": True,
    }
    assert _is_metadata_complete(metadata) is True
