# src/revisao_agents/utils/pdf_ingestor.py
"""
PDF Ingestor — processa uma pasta de PDFs, extrai texto com pdfplumber
e indexa os chunks no MongoDB via CorpusMongoDB.build().

O campo `url` de cada chunk é preenchido com o caminho absoluto do PDF,
mantendo compatibilidade total com todo o pipeline de busca e verificação.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pdfplumber

from ..config import EXTRACT_MIN_CHARS
from .mongodb_corpus import CorpusMongoDB


# ── Extração de texto ─────────────────────────────────────────────────────────

def _extrair_texto_pdf(pdf_path: Path) -> str:
    """
    Extrai todo o texto de um PDF usando pdfplumber.

    Páginas sem texto extraível são silenciosamente ignoradas.
    Retorna string vazia se o arquivo não puder ser lido.
    """
    paginas: List[str] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                texto = page.extract_text()
                if texto:
                    paginas.append(texto.strip())
    except Exception as e:
        print(f"   ⚠️  Erro ao ler {pdf_path.name}: {e}")
        return ""
    return "\n\n".join(paginas)


# ── Função principal ──────────────────────────────────────────────────────────

def ingest_pdf_folder(folder_path: str) -> dict:
    """
    Processa todos os PDFs em uma pasta (recursivamente) e os indexa no MongoDB.

    O campo `url` de cada chunk é o caminho absoluto do PDF, mantendo
    compatibilidade com o pipeline existente (busca vetorial, citações,
    referências). Arquivos já indexados são detectados por url_exists()
    e pulados automaticamente.

    Args:
        folder_path: Caminho para a pasta contendo os PDFs.

    Returns:
        {
            "indexed"     : int,  # novos PDFs indexados nesta execução
            "skipped"     : int,  # PDFs com texto insuficiente (< EXTRACT_MIN_CHARS)
            "already"     : int,  # PDFs já presentes no MongoDB
            "total_chunks": int,  # chunks inseridos no total desta sessão
            "errors"      : int,  # PDFs que falharam na leitura
        }
    """
    pasta = Path(folder_path).resolve()

    pdfs = sorted(pasta.rglob("*.pdf"))
    if not pdfs:
        print(f"   ℹ️  Nenhum PDF encontrado em: {pasta}")
        return {"indexed": 0, "skipped": 0, "already": 0, "total_chunks": 0, "errors": 0}

    print(f"\n📂 Pasta: {pasta}")
    print(f"   {len(pdfs)} PDF(s) encontrado(s)\n")

    corpus = CorpusMongoDB()
    extraidos: List[dict] = []

    counts = {"indexed": 0, "skipped": 0, "already": 0, "errors": 0}

    for pdf_path in pdfs:
        abs_path = str(pdf_path)
        nome = pdf_path.stem  # filename sem extensão

        # Verifica se já indexado (url = caminho absoluto do PDF)
        if corpus.url_exists(abs_path):
            print(f"   ⏭️  Já indexado: {pdf_path.name}")
            counts["already"] += 1
            continue

        print(f"   📄 Extraindo: {pdf_path.name}")
        texto = _extrair_texto_pdf(pdf_path)

        if not texto:
            print(f"      ❌ Texto vazio — arquivo inválido ou protegido")
            counts["errors"] += 1
            continue

        if len(texto) < EXTRACT_MIN_CHARS:
            print(f"      ⚠️  Texto muito curto ({len(texto)} chars < {EXTRACT_MIN_CHARS}) — ignorado")
            counts["skipped"] += 1
            continue

        print(f"      ✅ {len(texto):,} chars extraídos")
        extraidos.append({
            "url":      abs_path,    # filepath como identificador único
            "conteudo": texto,
            "titulo":   nome,
        })
        counts["indexed"] += 1

    if not extraidos:
        total_chunks = 0
        print("\n   ℹ️  Nenhum novo PDF para indexar.")
    else:
        print(f"\n🔷 Indexando {len(extraidos)} PDF(s) no MongoDB…")
        before = corpus._total_chunks
        corpus.build(extraidos, snippets=[], prefixo="pdf")
        total_chunks = corpus._total_chunks - before
        print(f"   ✅ {total_chunks} chunk(s) inserido(s)")

    counts["total_chunks"] = total_chunks
    return counts
