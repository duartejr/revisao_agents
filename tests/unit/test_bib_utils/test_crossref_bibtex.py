"""
Unit tests for bibliography REACT fallback behavior.
"""

from pathlib import Path
from unittest.mock import patch

from revisao_agents.core.schemas.corpus import Chunk
from revisao_agents.utils.bib_utils.crossref_bibtex import (
    get_reference_data_react,
    search_doi_in_mongo_chunks,
)


class _FakeCorpus:
    def __init__(self, chunks):
        self._chunks = chunks

    def query(self, texto_query: str, top_k: int = 10):
        return self._chunks[:top_k]


def test_search_doi_in_mongo_chunks_with_chunk_objects():
    chunks = [
        Chunk(
            texto="This study DOI: 10.1590/s1678-86212019000200316 presents results.",
            url="http://example.com/a",
            titulo="paper",
            fonte_idx=1,
            file_path="/tmp/a.txt",
            chunk_idx="1",
        )
    ]
    corpus = _FakeCorpus(chunks)

    doi = search_doi_in_mongo_chunks("/tmp/local-paper.pdf", corpus)

    assert doi == "10.1590/s1678-86212019000200316"


def test_search_doi_in_mongo_chunks_with_dict_chunks():
    chunks = [
        {"text": "metadata with doi 10.1000/xyz123 in body"},
    ]
    corpus = _FakeCorpus(chunks)

    doi = search_doi_in_mongo_chunks("/tmp/local-paper.pdf", corpus)

    assert doi == "10.1000/xyz123"


@patch(
    "revisao_agents.utils.bib_utils.crossref_bibtex.bibtex_to_abnt", return_value="ABNT"
)
@patch(
    "revisao_agents.utils.bib_utils.crossref_bibtex.get_bibtex_from_doi",
    return_value="@article{a}",
)
def test_get_reference_data_react_url_doi_path(mock_bibtex, mock_abnt):
    result = get_reference_data_react(
        "https://doi.org/10.1590/s1678-86212019000200316",
        mongo_corpus=None,
        tavily_enabled=False,
    )

    assert result["source"] == "url_doi"
    assert result["doi"] == "10.1590/s1678-86212019000200316"
    assert result["bibtex"] == "@article{a}"
    assert result["abnt"] == "ABNT"


@patch(
    "revisao_agents.utils.bib_utils.crossref_bibtex.bibtex_to_abnt", return_value="ABNT"
)
@patch(
    "revisao_agents.utils.bib_utils.crossref_bibtex.get_bibtex_from_doi",
    return_value="@article{mongo}",
)
@patch(
    "revisao_agents.utils.bib_utils.crossref_bibtex.search_doi_in_mongo_chunks",
    return_value="10.2000/mongo",
)
def test_get_reference_data_react_local_pdf_uses_mongo(
    mock_search_mongo,
    mock_get_bibtex,
    mock_abnt,
    tmp_path,
):
    file_path = tmp_path / "paper_local.pdf"
    file_path.write_text("dummy", encoding="utf-8")

    result = get_reference_data_react(
        str(file_path),
        mongo_corpus=object(),
        tavily_enabled=False,
    )

    assert result["source"] == "mongo_chunks"
    assert result["doi"] == "10.2000/mongo"
    assert result["bibtex"] == "@article{mongo}"
    assert result["abnt"] == "ABNT"


@patch(
    "revisao_agents.utils.bib_utils.crossref_bibtex._generate_fallback_abnt",
    return_value="FALLBACK",
)
@patch(
    "revisao_agents.utils.bib_utils.crossref_bibtex.search_crossref_by_title",
    return_value=None,
)
@patch(
    "revisao_agents.utils.bib_utils.crossref_bibtex.search_doi_in_mongo_chunks",
    return_value=None,
)
def test_get_reference_data_react_fallback_when_no_doi(
    mock_mongo,
    mock_crossref,
    mock_fallback,
    tmp_path,
):
    file_path = tmp_path / "very_long_local_document_name.pdf"
    file_path.write_text("dummy", encoding="utf-8")

    result = get_reference_data_react(
        str(file_path),
        mongo_corpus=object(),
        tavily_enabled=False,
    )

    assert result["source"] == "fallback"
    assert result["abnt"] == "FALLBACK"
