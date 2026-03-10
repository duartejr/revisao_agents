"""
consolidar_node — consolidates all written sections into a final document
Part of the nodes/writing subpackage.
"""
import re
import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

from ...state import EscritaTecnicaState
from ...config import (
    llm_call, parse_json_safe,
    TECNICO_MAX_RESULTS, MAX_CORPUS_PROMPT, EXTRACT_MIN_CHARS,
    MAX_URLS_EXTRACT, CTX_RESUMO_CHARS, SECAO_MIN_PARAGRAFOS,
    DELAY_ENTRE_SECOES, MAX_REACT_ITERATIONS, TOP_K_OBSERVACAO,
)
from ...core.schemas.techinical_writing import RespostaSecao, Fonte
from ...utils.vector_utils.mongodb_corpus import CorpusMongoDB
from ...utils.file_utils.helpers import resumir_secao, parse_plano_tecnico, parse_plano_academico
from ...core.schemas.writer_config import WriterConfig
from ...utils.llm_utils.prompt_loader import load_prompt
from .text_filters import _strip_justification_blocks, _strip_meta_sentences, _strip_figure_table_refs
from .anchor_helpers import _ANCORA_PATTERN, _extrair_ancora_principal, _extrair_citacao_ancora, _extrair_todas_ancoras_com_citacoes
from .phase_runners import _fase_pensamento, _fase_observacao, _fase_rascunho, _extrair_com_fallback
from .verification import (
    _contar_claims_verificaveis, _juiz_paragrafo_melhorado,
    _monitorar_taxa_verificacao, _buscar_conteudo_complementar,
    _verificar_e_corrigir_secao_adaptativa,
    _verificar_paragrafo_com_ancora, _verificar_e_corrigir_secao_com_ancora,
)

def consolidar_node(state: EscritaTecnicaState) -> dict:
    """Consolidates written sections into a final document."""
    config = WriterConfig.from_dict(state.get("writer_config", {}))
    tema = state["tema"]
    secoes = sorted(state["secoes_escritas"], key=lambda s: s["indice"])
    all_urls = list(dict.fromkeys(state.get("refs_urls", [])))
    all_imagens = state.get("refs_imagens", [])
    react_log = state.get("react_log", [])
    stats_global = state.get("stats_verificacao", [])
    resumo_final = state.get("resumo_acumulado", "")[:1000]

    print(f"\n📚 Consolidando {len(secoes)} seções...")

    total_par = sum(s.get("total", 0) for s in stats_global)
    total_aprov = sum(s.get("aprovados", 0) for s in stats_global)
    total_ajust = sum(s.get("ajustados", 0) for s in stats_global)
    total_corr = sum(s.get("corrigidos", 0) for s in stats_global)
    total_verif = total_aprov + total_ajust
    taxa_global = (total_verif / total_par * 100) if total_par > 0 else 100

    print(f"   📊 {total_verif}/{total_par} verificados ({taxa_global:.0f}%) "
          f"— ✅{total_aprov} aprovados  🔵{total_ajust} ajustados  "
          f"🔧{total_corr} corrigidos | {len(all_urls)} fontes")

    titulos = [s["titulo"] for s in secoes]
    p_intro = load_prompt(
        f"{config.prompt_dir}/consolidar_intro",
        tema=tema,
        titulos=", ".join(titulos),
        language=config.language,
    )
    resp_intro = llm_call(p_intro.text, temperature=p_intro.temperature)
    p_concl = load_prompt(
        f"{config.prompt_dir}/consolidar_conclusao",
        tema=tema,
        resumo_final=resumo_final,
        language=config.language,
    )
    resp_concl = llm_call(p_concl.text, temperature=p_concl.temperature)

    partes = [
        f"# {tema}\n",
        f"> **Tipo:** {config.review_type_label}\n",
        f"> **Verificação por parágrafo:** {total_verif}/{total_par} verificados "
        f"({taxa_global:.0f}%) — {total_aprov} aprovados, {total_ajust} ajustados, "
        f"{total_corr} corrigidos | "
        f"**Fontes:** {len(all_urls)} | **Seções:** {len(secoes)}\n",
        "\n---\n", "## Sumário\n", "- Introdução",
    ]
    for s in secoes:
        partes.append(f"- {s['titulo']}")
    partes += ["- Conclusão", "\n\n---\n",
               "## Introdução\n", resp_intro.strip(), "\n\n---\n"]

    for s in secoes:
        stats_s = next(
            (x for x in stats_global if x.get("secao") == s["titulo"]), {}
        )
        t_s = stats_s.get("total", 0)
        a_s = stats_s.get("aprovados", 0) + stats_s.get("ajustados", 0)
        r_s = stats_s.get("corrigidos", 0)
        aj_s = stats_s.get("ajustados", 0)
        tx_s = (a_s / t_s * 100) if t_s > 0 else 100
        partes.append(
            f"<!-- Parágrafos: {a_s}/{t_s} verificados ({tx_s:.0f}%) "
            f"| {stats_s.get('aprovados', 0)} aprovados, {aj_s} ajustados, "
            f"{r_s} corrigidos -->\n"
        )
        partes.append(s["texto"].strip())
        partes.append("\n\n---\n")

    partes += ["## Conclusão\n", resp_concl.strip(), "\n\n"]

    # ══════════════════════════════════════════════════════════════════
    # GLOBAL CITATION SYNCHRONIZATION + PER-SECTION REFERENCE REBUILD
    # ══════════════════════════════════════════════════════════════════
    print(f"\n  🔗 Sincronizando citações globais...")

    # 1. Build consolidated fonte_map: {original_citation_number: url}
    #    Merge all per-section fonte_maps; keep the first URL seen per index.
    #    Keys may be int or str depending on serialization — normalize to int.
    fonte_map_consolidado: dict = {}
    for s in secoes:
        s_map = s.get("fonte_map", {})
        for idx, url in s_map.items():
            idx_int = int(idx)
            if idx_int not in fonte_map_consolidado:
                fonte_map_consolidado[idx_int] = url

    # Also add URLs from corpus that might be cited but not in fonte_maps
    for i, url in enumerate(all_urls, 1):
        if i not in fonte_map_consolidado:
            fonte_map_consolidado[i] = url

    documento_raw = "\n".join(partes)

    # 2. Strip old "### Referências desta seção" blocks before renumbering
    documento_clean = re.sub(
        r'\n*### Referências desta seção\s*\n(?:\[?\d+\]?[^\n]*\n?)*',
        '',
        documento_raw,
    )

    # 3. Strip invalid figure/table/equation references
    documento_clean = _strip_figure_table_refs(documento_clean)

    # 4. Extract all [N] from entire document and create global renumbering
    citacoes_originais = re.findall(r'\[(\d+)\]', documento_clean)
    citacoes_unicas = []
    seen = set()
    for c in citacoes_originais:
        n = int(c)
        if n not in seen:
            seen.add(n)
            citacoes_unicas.append(n)

    # old_idx → new_idx (first-appearance order)
    mapa_global: dict = {}
    for new_idx, old_idx in enumerate(citacoes_unicas, 1):
        mapa_global[old_idx] = new_idx

    # Build synchronized global fonte map: {new_idx: url}
    global_fonte_map_sync: dict = {}
    for old_idx, new_idx in mapa_global.items():
        url = fonte_map_consolidado.get(old_idx, "")
        if url:
            global_fonte_map_sync[new_idx] = url

    # 5. Renumber all [N] in the document
    def _renumber(match):
        old = int(match.group(1))
        new = mapa_global.get(old, old)
        return f"[{new}]"

    documento_sync = re.sub(r'\[(\d+)\]', _renumber, documento_clean)
    # Also handle [N, M] compound citations
    def _renumber_compound(match):
        nums = re.findall(r'\d+', match.group(0))
        new_nums = [str(mapa_global.get(int(n), int(n))) for n in nums]
        return "[" + ", ".join(new_nums) + "]"
    documento_sync = re.sub(r'\[\d+(?:\s*,\s*\d+)+\]', _renumber_compound, documento_sync)

    n_global_sources = len(global_fonte_map_sync)
    print(f"     ✅ {n_global_sources} fontes globais | {len(mapa_global)} citações remapeadas")

    # 6. Rebuild per-section "### Referências desta seção" blocks
    #    First, split out the conclusion so it doesn't contaminate the
    #    last section block (the old code skipped any block containing
    #    '## Conclusão', silently dropping the last section's refs).
    _CONCLUSAO_MARKER = "\n## Conclusão"
    if _CONCLUSAO_MARKER in documento_sync:
        _c_idx = documento_sync.index(_CONCLUSAO_MARKER)
        doc_sections_part = documento_sync[:_c_idx]
        doc_conclusao_part = documento_sync[_c_idx:]
    else:
        doc_sections_part = documento_sync
        doc_conclusao_part = ""

    section_pattern = re.compile(r'(?=\n<!-- Parágrafos:)')
    section_blocks = section_pattern.split(doc_sections_part)

    rebuilt_parts = []
    for block in section_blocks:
        # Only process blocks that contain a numbered section heading
        if not re.search(r'## \d', block):
            rebuilt_parts.append(block)
            continue

        # Extract all [N] referenced in block body
        cits_in_block = set()
        for m in re.finditer(r'\[(\d+)\]', block):
            cits_in_block.add(int(m.group(1)))
        # Also handle [N, M]
        for m in re.finditer(r'\[(\d+(?:\s*,\s*\d+)+)\]', block):
            for n in re.findall(r'\d+', m.group(1)):
                cits_in_block.add(int(n))

        if cits_in_block:
            refs_lines = []
            for idx in sorted(cits_in_block):
                url = global_fonte_map_sync.get(idx, "")
                if url:
                    refs_lines.append(f"[{idx}] {url}")
            if refs_lines:
                # Remove trailing --- if present, we'll re-add it
                block_trimmed = block.rstrip()
                if block_trimmed.endswith("---"):
                    block_trimmed = block_trimmed[:-3].rstrip()
                block = (
                    block_trimmed
                    + "\n\n### Referências desta seção\n\n"
                    + "\n".join(refs_lines)
                    + "\n\n\n---\n"
                )
        rebuilt_parts.append(block)

    documento = "".join(rebuilt_parts) + doc_conclusao_part

    # Update all_urls count for header
    all_urls_final = list(global_fonte_map_sync.values())
    # Update the header line with correct source count
    documento = re.sub(
        r'\*\*Fontes:\*\* \d+',
        f'**Fontes:** {len(all_urls_final)}',
        documento,
        count=1,
    )

    print(f"\n  ℹ️  Referências reconstruídas por seção ({n_global_sources} fontes globais)")
    print(
        "\n  ℹ️  A seção final '## Referências' não é mais gerada automaticamente.\n"
        "      Use a opção [5] do menu principal para formatar suas referências\n"
        "      no padrão desejado (ABNT, APA, IEEE, etc.) a partir de um arquivo\n"
        "      YAML/JSON. Consulte references/README.md para detalhes."
    )

    slug = re.sub(r"[^\w\s-]", "", tema[:40]).strip().replace(" ", "_").lower()
    output_path = f"reviews/{config.output_prefix}_{slug}.md"
    log_path = f"reviews/{config.output_prefix}_{slug}.log"

    try:
        os.makedirs("reviews", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(documento)
        print(f"\n💾 {output_path} ({len(documento):,} chars)")
    except Exception as e:
        print(f"⚠️  Erro ao salvar: {e}")

    try:
        cabecalho = [
            "=" * 70, f"REACT AUDIT LOG — {tema}",
            f"Gerado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Seções: {len(secoes)} | Fontes: {len(all_urls)}",
            f"Verificados: {total_verif}/{total_par} ({taxa_global:.0f}%) "
            f"— {total_aprov} aprovados, {total_ajust} ajustados, {total_corr} corrigidos",
            "=" * 70, "\n── STATS POR SEÇÃO ──",
        ]
        for s in stats_global:
            t = s.get("total", 0)
            a = s.get("aprovados", 0) + s.get("ajustados", 0)
            r = s.get("corrigidos", 0)
            aj = s.get("ajustados", 0)
            tx = (a / t * 100) if t > 0 else 100
            cabecalho.append(
                f"  [{a}/{t} = {tx:.0f}% | {s.get('aprovados', 0)} aprov "
                f"{aj} ajust {r} corrig] {s.get('secao', '?')[:55]}"
            )
        os.makedirs("reviews", exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(cabecalho + [""] + react_log))
        print(f"📋 {log_path}")
    except Exception as e:
        print(f"⚠️  Erro ao salvar log: {e}")

    return {"status": "concluido"}
