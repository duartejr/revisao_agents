"""
handlers.py — Business logic bridges between Gradio UI and revisao_agents.

Each function wraps one of the five main workflow options and adapts
CLI-style interactions to Gradio's generator / state model.

Live-log streaming
------------------
`start_writing` runs the LangGraph workflow in a background thread and
captures every print() call via `_StdoutCapture`, funnelling the lines
through a thread-safe queue that the Gradio generator drains between
yields.  This produces true, line-by-line live output in the UI.
"""

from __future__ import annotations

import glob
import os
import queue
import sys
import threading
from typing import Any, Generator

_SRC = os.path.join(os.path.dirname(__file__), "..")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from revisao_agents.state import ReviewState, TechnicalWriterState
from revisao_agents.workflows import build_academic_workflow, build_technical_workflow
from revisao_agents.workflows.technical_writing_workflow import (
    build_technical_writing_workflow,
)
from revisao_agents.config import validate_runtime_config
from revisao_agents.utils.vector_utils.pdf_ingestor import ingest_pdf_folder
from revisao_agents.core.schemas.writer_config import WriterConfig
from revisao_agents.tools.reference_formatter import format_references_from_file


# ═══════════════════════════════════════════════════════════════════════════
# Live stdout capture
# ═══════════════════════════════════════════════════════════════════════════

class _StdoutCapture:
    """
    Context manager that redirects sys.stdout to a queue so the caller
    can read lines as they are produced by any print() inside the block.
    """

    def __init__(self, q: "queue.Queue[str]"):
        self._q = q
        self._buf = ""
        self._original: Any = None

    def __enter__(self) -> "_StdoutCapture":
        self._original = sys.stdout
        sys.stdout = self  # type: ignore[assignment]
        return self

    def __exit__(self, *_: Any) -> None:
        if self._buf.strip():
            self._q.put(self._buf.rstrip())
            self._buf = ""
        sys.stdout = self._original

    def write(self, text: str) -> int:
        self._original.write(text)
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            stripped = line.rstrip()
            if stripped:
                self._q.put(stripped)
        return len(text)

    def flush(self) -> None:
        self._original.flush()

    @property
    def encoding(self) -> str:
        return getattr(self._original, "encoding", "utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════

def _list_md(folder: str) -> list[str]:
    return glob.glob(os.path.join(folder, "*.md"))


def _find_newest_md(folder: str) -> str | None:
    files = _list_md(folder)
    return max(files, key=os.path.getmtime) if files else None


def _read_md(path: str | None) -> str:
    if not path or not os.path.exists(path):
        return ""
    try:
        return open(path, encoding="utf-8").read()
    except Exception:
        return ""


# ═══════════════════════════════════════════════════════════════════════════
# Option 1 & 2 — Planning                                Human-in-the-Loop
# ═══════════════════════════════════════════════════════════════════════════

def start_planning(
    tema: str,
    tipo: str,
    rodadas: int,
) -> tuple[list, dict, str, str]:
    """Launch planning workflow until the first HITL pause.

    Returns (history, session_state, status_msg, rendered_plan).
    rendered_plan is empty until the workflow fully completes.
    """
    if not tema.strip():
        return [], {}, "❌ Por favor, informe o tema antes de iniciar.", ""

    req_mongodb = tipo in ("academico", "ambos")
    req_openai_embeddings = tipo in ("academico", "ambos")
    req_tavily = tipo in ("tecnico", "ambos")
    cfg_issues = validate_runtime_config(
        require_mongodb=req_mongodb,
        require_tavily=req_tavily,
        require_openai_embeddings=req_openai_embeddings,
        strict=False,
    )
    if cfg_issues:
        msg = "❌ Configuração incompleta para este modo:\n- " + "\n- ".join(cfg_issues)
        return [], {}, msg, ""

    tipos_list = ["academico", "tecnico"] if tipo == "ambos" else [tipo]
    tipo_atual = tipos_list[0]
    label = "ACADÊMICA" if tipo_atual == "academico" else "TÉCNICA"

    state_init: ReviewState = {
        "theme": tema,
        "review_type": tipo_atual,
        "relevant_chunks": [],
        "technical_snippets": [],
        "technical_urls": [],
        "current_plan": "",
        "interview_history": [],
        "questions_asked": 0,
        "max_questions": int(rodadas),
        "final_plan": "",
        "final_plan_path": "",
        "status": "starting",
    }

    thread_id = f"revisao_{tipo_atual}_{tema[:20]}"
    config = {"configurable": {"thread_id": thread_id}}
    app = build_academic_workflow() if tipo_atual == "academico" else build_technical_workflow()

    log_q: queue.Queue[str] = queue.Queue()
    with _StdoutCapture(log_q):
        try:
            for _ in app.stream(state_init, config):
                pass
        except Exception as exc:
            return [], {}, f"❌ Erro ao iniciar: {exc}", ""

    history: list[dict] = []
    lines = []
    while not log_q.empty():
        lines.append(log_q.get_nowait())
    if lines:
        history.append({"role": "assistant", "content": "```\n" + "\n".join(lines) + "\n```"})

    graph_state = app.get_state(config)

    if not graph_state.next:
        plan_path = graph_state.values.get("final_plan_path", "")
        rendered = _read_md(plan_path)
        history.append({"role": "assistant", "content": f"✅ Planejamento {label} concluído! Plano salvo em `{plan_path}`"})
        return history, {}, "✅ Concluído", rendered

    agent_question = ""
    for role, content in reversed(graph_state.values.get("interview_history", [])):
        if role == "assistant":
            agent_question = content
            break

    p  = graph_state.values.get("questions_asked", 0)
    mp = graph_state.values.get("max_questions", rodadas)
    history.append({"role": "assistant", "content": f"[Rodada {p}/{mp} — {tipo_atual}]\n\n{agent_question}"})

    session_state = {
        "app": app,
        "config": config,
        "tipo": tipo_atual,
        "tipos_pendentes": tipos_list[1:],
        "theme": tema,
        "rodadas": rodadas,
    }

    return history, session_state, f"🔄 {label} em andamento — aguardando resposta…", ""


def continue_planning(
    user_msg: str,
    history: list,
    session_state: dict,
) -> tuple[list, dict, str, str]:
    """Feed user response back into the HITL loop.

    Returns (history, session_state, status_msg, rendered_plan).
    """
    if not session_state or "app" not in session_state:
        return history, session_state, "❌ Nenhuma sessão ativa.", ""

    app    = session_state["app"]
    config = session_state["config"]
    tipo   = session_state["tipo"]
    label  = "ACADÊMICA" if tipo == "academico" else "TÉCNICA"

    history = history + [{"role": "user", "content": user_msg}]

    hist = app.get_state(config).values.get("interview_history", [])
    app.update_state(
        config,
        {"interview_history": hist + [("user", user_msg)]},
        as_node="human_pause",
    )

    log_q: queue.Queue[str] = queue.Queue()
    with _StdoutCapture(log_q):
        try:
            for _ in app.stream(None, config):
                pass
        except Exception as exc:
            history = history + [{"role": "assistant", "content": f"❌ Erro: {exc}"}]
            return history, session_state, f"❌ Erro: {exc}", ""

    lines = []
    while not log_q.empty():
        lines.append(log_q.get_nowait())
    if lines:
        history = history + [{"role": "assistant", "content": "```\n" + "\n".join(lines) + "\n```"}]

    graph_state = app.get_state(config)

    if not graph_state.next:
        plan_path = graph_state.values.get("final_plan_path", "")
        rendered = _read_md(plan_path)
        finished_msg = (
            f"✅ Planejamento {label} concluído! Plano salvo em `{plan_path}`"
            if plan_path else f"✅ Planejamento {label} concluído!"
        )
        history = history + [{"role": "assistant", "content": finished_msg}]

        tipos_pendentes = session_state.get("tipos_pendentes", [])
        if tipos_pendentes:
            next_history, next_state, next_status, _ = start_planning(
                tema=session_state["theme"],
                tipo=tipos_pendentes[0],
                rodadas=session_state["rodadas"],
            )
            next_state["tipos_pendentes"] = tipos_pendentes[1:]
            return history + next_history, next_state, next_status, rendered

        return history, {}, "✅ Todos os planejamentos concluídos!", rendered

    agent_question = ""
    for role, content in reversed(graph_state.values.get("interview_history", [])):
        if role == "assistant":
            agent_question = content
            break

    p  = graph_state.values.get("questions_asked", 0)
    mp = graph_state.values.get("max_questions", session_state.get("rodadas", 3))
    history = history + [{"role": "assistant", "content": f"[Rodada {p}/{mp} — {tipo}]\n\n{agent_question}"}]

    return history, session_state, f"🔄 {label} em andamento — rodada {p}/{mp}", ""


# ═══════════════════════════════════════════════════════════════════════════
# Option 3 — Execute Writing from existing plan
# ═══════════════════════════════════════════════════════════════════════════

def list_plan_files(mode: str) -> list[str]:
    os.makedirs("plans", exist_ok=True)
    pattern = "plans/plano_revisao_tecnica_*.md" if mode == "Técnica" else "plans/plano_revisao_*.md"
    files = sorted(glob.glob(pattern))
    if not files:
        files = sorted(glob.glob("plans/plano_revisao_*.md"))
    if not files:
        files = sorted(glob.glob("plano_revisao_*.md"))
    return files if files else ["(nenhum plano encontrado)"]


def start_writing(
    plan_path: str,
    mode: str,
    language: str,
    min_src: int,
    tavily_enabled: bool,
    history: list,
) -> Generator[tuple[list, str, str], None, None]:
    """
    Stream writing progress with live logs to the Gradio chatbot.

    Runs the LangGraph workflow in a background thread and captures every
    print() call via _StdoutCapture, funnelling lines through a queue so
    the UI updates line-by-line as the agent works.

    Yields
    ------
    (updated_history, status_msg, rendered_content)
    rendered_content is empty during streaming; it contains the final .md
    document when the workflow finishes.
    """
    os.makedirs("reviews", exist_ok=True)

    cfg_issues = validate_runtime_config(
        require_mongodb=True,
        require_tavily=bool(tavily_enabled),
        require_openai_embeddings=True,
        strict=False,
    )
    if cfg_issues:
        yield (
            history + [{"role": "assistant", "content": "❌ Configuração incompleta:\n- " + "\n- ".join(cfg_issues)}],
            "❌ Erro", "",
        )
        return

    if not plan_path or not os.path.exists(plan_path):
        yield (
            history + [{"role": "assistant", "content": f"❌ Plano não encontrado: `{plan_path}`"}],
            "❌ Erro", "",
        )
        return

    if mode == "Acadêmica":
        writer_config = WriterConfig.academic(language=language)
    else:
        writer_config = WriterConfig.technical(language=language)
    writer_config.min_sources_per_section = max(0, int(min_src))

    state_init: TechnicalWriterState = {
        "theme": "",
        "plan_summary": "",
        "sections": [],
        "plan_path": plan_path,
        "written_sections": [],
        "refs_urls": [],
        "refs_images": [],
        "cumulative_summary": "",
        "react_log": [],
        "verification_stats": [],
        "status": "starting",
        "writer_config": writer_config.to_dict(),
        "tavily_enabled": tavily_enabled,
    }

            app = build_technical_writing_workflow()
    snapshot_before = set(_list_md("reviews"))

    result_q: queue.Queue[tuple[str, Any]] = queue.Queue()
    _DONE = object()

    def _worker() -> None:
        log_q: queue.Queue[str] = queue.Queue()

        def _drain() -> None:
            while not log_q.empty():
                result_q.put(("log", log_q.get_nowait()))

        with _StdoutCapture(log_q):
            try:
                for event in app.stream(state_init):
                    _drain()
                    result_q.put(("event", event))
            except Exception as exc:
                _drain()
                result_q.put(("error", exc))
            finally:
                _drain()

        result_q.put(("done", _DONE))

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    history = history + [
        {"role": "assistant", "content": f"▶ Iniciando escrita **{mode}** — `{os.path.basename(plan_path)}`"}
    ]
    yield history, "🔄 Iniciando…", ""

    while True:
        try:
            kind, data = result_q.get(timeout=120)
        except queue.Empty:
            history = history + [{"role": "assistant", "content": "⏳ Aguardando agente…"}]
            yield history, "⏳ Aguardando…", ""
            continue

        if data is _DONE:
            break

        if kind == "log":
            history = history + [{"role": "assistant", "content": f"`{data}`"}]
            yield history, "🔄 …", ""

        elif kind == "event":
            node = list(data.keys())[0] if data else "?"
            if node != "__end__":
                st = data.get(node, {}).get("status", "")
                if st:
                    history = history + [{"role": "assistant", "content": f"**[{node}]** → {st}"}]
                    yield history, f"🔄 {node}", ""

        elif kind == "error":
            history = history + [{"role": "assistant", "content": f"❌ Erro: {data}"}]
            yield history, "❌ Erro", ""
            return

    thread.join(timeout=5)

    new_files = set(_list_md("reviews")) - snapshot_before
    output_file = max(new_files, key=os.path.getmtime) if new_files else _find_newest_md("reviews")
    rendered = _read_md(output_file)

    link_msg = (
        f"✅ Escrita concluída!  📄 `{output_file}`"
        if output_file else "✅ Escrita concluída!"
    )
    history = history + [{"role": "assistant", "content": link_msg}]
    yield history, "✅ Concluído", rendered


# ═══════════════════════════════════════════════════════════════════════════
# Option 4 — Index local PDFs
# ═══════════════════════════════════════════════════════════════════════════

def index_pdfs(folder_path: str) -> str:
    cfg_issues = validate_runtime_config(
        require_mongodb=True,
        require_openai_embeddings=True,
        strict=False,
    )
    if cfg_issues:
        return "❌ Configuração incompleta:\n- " + "\n- ".join(cfg_issues)

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
        "✅ Indexação concluída!\n\n"
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
    yaml_file_obj: Any,
    tavily_enabled: bool,
    output_dir: str,
) -> tuple[str, str]:
    if yaml_file_obj is None:
        return "", "❌ Nenhum arquivo selecionado."

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
