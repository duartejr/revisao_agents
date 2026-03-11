"""
handlers.py — Business logic bridges between Gradio UI and revisao_agents.

Each function wraps one of the five main workflow options and adapts
CLI-style interactions to Gradio's generator / state model.
"""

from __future__ import annotations

import glob
import os
import sys
from typing import Any, Generator

# ---------------------------------------------------------------------------
# Ensure the src/ directory is on sys.path when this module is imported
# directly (e.g. from run_ui.py).
# ---------------------------------------------------------------------------
_SRC = os.path.join(os.path.dirname(__file__), "..")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from revisao_agents.state import RevisaoState, EscritaTecnicaState
from revisao_agents.workflows import build_academico_workflow, build_tecnico_workflow
from revisao_agents.workflows.technical_writing_workflow import (
    build_workflow as build_escrita_workflow,
)
from revisao_agents.utils.vector_utils.pdf_ingestor import ingest_pdf_folder
from revisao_agents.core.schemas.writer_config import WriterConfig
from revisao_agents.tools.reference_formatter import format_references_from_file


# ═══════════════════════════════════════════════════════════════════════════
# Option 1 & 2 — Planning (Acadêmica / Técnica)          Human-in-the-Loop
# ═══════════════════════════════════════════════════════════════════════════

def start_planning(
    tema: str,
    tipo: str,         # "academico" | "tecnico" | "ambos"
    rodadas: int,
) -> tuple[list, dict, str]:
    """
    Launch the planning workflow and run until the first HITL pause.

    Returns
    -------
    history      : Gradio chatbot history (list of [user, bot] pairs)
    session_state: Serialisable dict keeping the LangGraph app + config alive
    status_msg   : Short status string for display
    """
    if not tema.strip():
        return [], {}, "❌ Por favor, informe o tema antes de iniciar."

    tipos_list = ["academico", "tecnico"] if tipo == "ambos" else [tipo]

    # For simplicity handle one tipo at a time — we process the first only here.
    # If "ambos", the UI will cycle through automatically after the first finishes.
    tipo_atual = tipos_list[0]
    label = "ACADÊMICA" if tipo_atual == "academico" else "TÉCNICA"

    state_init: RevisaoState = {
        "tema": tema,
        "tipo_revisao": tipo_atual,
        "chunks_relevantes": [],
        "snippets_tecnicos": [],
        "urls_tecnicos": [],
        "plano_atual": "",
        "historico_entrevista": [],
        "perguntas_feitas": 0,
        "max_perguntas": int(rodadas),
        "plano_final": "",
        "plano_final_path": "",
        "status": "iniciando",
    }

    thread_id = f"revisao_{tipo_atual}_{tema[:20]}"
    config = {"configurable": {"thread_id": thread_id}}

    app = build_academico_workflow() if tipo_atual == "academico" else build_tecnico_workflow()

    # Stream until the first HITL pause
    try:
        for _ in app.stream(state_init, config):
            pass
    except Exception as exc:
        return [], {}, f"❌ Erro ao iniciar: {exc}"

    graph_state = app.get_state(config)

    if not graph_state.next:
        return (
            [{"role": "assistant", "content": f"✅ Planejamento {label} concluído! Plano salvo em plans/"}],
            {},
            "✅ Concluído",
        )

    # Extract first agent question
    agent_question = ""
    for role, content in reversed(graph_state.values.get("historico_entrevista", [])):
        if role == "assistant":
            agent_question = content
            break

    p  = graph_state.values.get("perguntas_feitas", 0)
    mp = graph_state.values.get("max_perguntas", rodadas)
    header = f"[Rodada {p}/{mp} — {tipo_atual}]"
    bot_msg = f"{header}\n\n{agent_question}"

    history = [{"role": "assistant", "content": bot_msg}]
    session_state = {
        "app": app,
        "config": config,
        "tipo": tipo_atual,
        "tipos_pendentes": tipos_list[1:],
        "tema": tema,
        "rodadas": rodadas,
    }

    return history, session_state, f"🔄 {label} em andamento — aguardando resposta…"


def continue_planning(
    user_msg: str,
    history: list,
    session_state: dict,
) -> tuple[list, dict, str]:
    """
    Feed a user response back into the HITL loop and advance the workflow.

    Returns updated (history, session_state, status_msg).
    """
    if not session_state or "app" not in session_state:
        return history, session_state, "❌ Nenhuma sessão ativa. Inicie o planejamento primeiro."

    app    = session_state["app"]
    config = session_state["config"]
    tipo   = session_state["tipo"]
    label  = "ACADÊMICA" if tipo == "academico" else "TÉCNICA"

    history = history + [{"role": "user", "content": user_msg}, {"role": "assistant", "content": None}]

    # Update state with user response
    hist = app.get_state(config).values.get("historico_entrevista", [])
    app.update_state(
        config,
        {"historico_entrevista": hist + [("user", user_msg)]},
        as_node="pausa_humana",
    )

    # Resume streaming
    try:
        for _ in app.stream(None, config):
            pass
    except Exception as exc:
        history[-1]["content"] = f"❌ Erro: {exc}"
        return history, session_state, f"❌ Erro: {exc}"

    graph_state = app.get_state(config)

    if not graph_state.next:
        # This tipo is finished — check if there are more
        finished_msg = f"✅ Planejamento {label} concluído! Plano salvo em plans/"
        history[-1]["content"] = finished_msg

        tipos_pendentes = session_state.get("tipos_pendentes", [])
        if tipos_pendentes:
            # Kick off the next tipo automatically
            next_history, next_state, next_status = start_planning(
                tema=session_state["tema"],
                tipo=tipos_pendentes[0],
                rodadas=session_state["rodadas"],
            )
            next_state["tipos_pendentes"] = tipos_pendentes[1:]
            return history + next_history, next_state, next_status

        return history, {}, "✅ Todos os planejamentos concluídos!"

    # Extract next agent question
    agent_question = ""
    for role, content in reversed(graph_state.values.get("historico_entrevista", [])):
        if role == "assistant":
            agent_question = content
            break

    p  = graph_state.values.get("perguntas_feitas", 0)
    mp = graph_state.values.get("max_perguntas", session_state.get("rodadas", 3))
    header = f"[Rodada {p}/{mp} — {tipo}]"
    history[-1]["content"] = f"{header}\n\n{agent_question}"

    return history, session_state, f"🔄 {label} em andamento — rodada {p}/{mp}"


# ═══════════════════════════════════════════════════════════════════════════
# Option 3 — Execute Writing from existing plan
# ═══════════════════════════════════════════════════════════════════════════

def list_plan_files(mode: str) -> list[str]:
    """Return a list of plan .md files matching the given mode."""
    os.makedirs("plans", exist_ok=True)
    pattern = "plans/plano_revisao_tecnica_*.md" if mode == "Técnica" else "plans/plano_revisao_*.md"
    files = sorted(glob.glob(pattern))
    if not files:
        files = sorted(glob.glob("plans/plano_revisao_*.md"))
    if not files:
        files = sorted(glob.glob("plano_revisao_*.md"))
    return files if files else ["(nenhum plano encontrado)"]


def _find_newest_md(folder: str) -> str | None:
    """Return the path of the most recently modified .md in folder, or None."""
    import glob as _glob
    files = _glob.glob(os.path.join(folder, "*.md"))
    return max(files, key=os.path.getmtime) if files else None


def _list_md(folder: str) -> list[str]:
    """Return all .md files in folder."""
    import glob as _glob
    return _glob.glob(os.path.join(folder, "*.md"))


def start_writing(
    plan_path: str,
    mode: str,         # "Técnica" | "Acadêmica"
    language: str,     # "pt" | "en"
    min_src: int,
    tavily_enabled: bool,
    history: list,
) -> Generator[tuple[list, str, str], None, None]:
    """
    Stream writing progress to the Gradio chatbot.

    Yields (updated_history, status_msg, rendered_content) at each workflow step.
    rendered_content is empty during streaming and contains the final file when done.
    """
    os.makedirs("reviews", exist_ok=True)

    if not plan_path or not os.path.exists(plan_path):
        yield history + [{"role": "assistant", "content": f"❌ Arquivo de plano não encontrado: {plan_path}"}], "❌ Erro", ""
        return

    if mode == "Acadêmica":
        writer_config = WriterConfig.academic(language=language)
    else:
        writer_config = WriterConfig.technical(language=language)
    writer_config.min_sources_per_section = max(0, int(min_src))

    state_init: EscritaTecnicaState = {
        "tema": "",
        "resumo_plano": "",
        "secoes": [],
        "caminho_plano": plan_path,
        "secoes_escritas": [],
        "refs_urls": [],
        "refs_imagens": [],
        "resumo_acumulado": "",
        "react_log": [],
        "stats_verificacao": [],
        "status": "iniciando",
        "writer_config": writer_config.to_dict(),
        "tavily_enabled": tavily_enabled,
    }

    app = build_escrita_workflow()
    snapshot_before = set(_list_md("reviews"))

    history = history + [{"role": "assistant", "content": f"▶ Iniciando escrita {mode} — `{os.path.basename(plan_path)}`"}]
    yield history, "🔄 Iniciando…", ""

    try:
        for event in app.stream(state_init):
            node = list(event.keys())[0] if event else "?"
            if node != "__end__":
                st = event.get(node, {}).get("status", "")
                if st:
                    history = history + [{"role": "assistant", "content": f"**[{node}]** → {st}"}]
                    yield history, f"🔄 {node}: {st}", ""
    except KeyboardInterrupt:
        history = history + [{"role": "assistant", "content": "⚠️ Cancelado pelo usuário."}]
        yield history, "⚠️ Cancelado", ""
        return
    except Exception as exc:
        history = history + [{"role": "assistant", "content": f"❌ Erro: {exc}"}]
        yield history, f"❌ Erro: {exc}", ""
        return

    # Find the newly created file
    new_files = set(_list_md("reviews")) - snapshot_before
    output_file = max(new_files, key=os.path.getmtime) if new_files else _find_newest_md("reviews")
    rendered = open(output_file, encoding="utf-8").read() if output_file and os.path.exists(output_file) else ""
    link_msg = f"✅ Escrita concluída! Arquivo salvo: `{output_file}`" if output_file else "✅ Escrita concluída! Arquivo salvo em reviews/"
    history = history + [{"role": "assistant", "content": link_msg}]
    yield history, "✅ Concluído", rendered


# ═══════════════════════════════════════════════════════════════════════════
# Option 4 — Index local PDFs
# ═══════════════════════════════════════════════════════════════════════════

def index_pdfs(folder_path: str) -> str:
    """Index all PDFs in the given folder to MongoDB."""
    if not folder_path.strip():
        return "❌ Informe o caminho da pasta."

    folder_path = os.path.expanduser(folder_path.strip())
    if not os.path.isdir(folder_path):
        return f"❌ Pasta não encontrada: {folder_path}"

    try:
        result = ingest_pdf_folder(folder_path)
    except Exception as exc:
        return f"❌ Erro durante indexação: {exc}"

    return (
        f"✅ Indexação concluída!\n\n"
        f"- Novos PDFs indexados : **{result['indexed']}**\n"
        f"- Já no banco          : **{result['already']}**\n"
        f"- Texto insuficiente   : **{result['skipped']}**\n"
        f"- Erros de leitura     : **{result['errors']}**\n"
        f"- Chunks inseridos     : **{result['total_chunks']}**"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Option 5 — Format References
# ═══════════════════════════════════════════════════════════════════════════

def format_references(
    yaml_file_obj: Any,   # Gradio file upload object (has .name attribute)
    tavily_enabled: bool,
    output_dir: str,
) -> tuple[str, str]:
    """
    Format references from a YAML/JSON file uploaded via Gradio.

    Returns (formatted_markdown, status_message).
    """
    if yaml_file_obj is None:
        return "", "❌ Nenhum arquivo selecionado."

    # Gradio provides a temp path via the .name attribute
    input_path = yaml_file_obj if isinstance(yaml_file_obj, str) else yaml_file_obj.name

    output_path = None
    if output_dir.strip():
        os.makedirs(output_dir.strip(), exist_ok=True)
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir.strip(), f"{base}_formatted.md")

    try:
        result_md = format_references_from_file(
            input_path=input_path,
            tavily_enabled=tavily_enabled,
            output_path=output_path,
        )
    except Exception as exc:
        return "", f"❌ Erro ao formatar referências: {exc}"

    status = "✅ Referências formatadas com sucesso!"
    if output_path:
        status += f"\n\nArquivo salvo em: `{output_path}`"
    return result_md, status
