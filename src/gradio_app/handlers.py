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
import logging
import os
import queue
import re
import shutil
import sys
import tempfile
import threading
from datetime import datetime
from typing import Any, Generator, Optional

_SRC = os.path.join(os.path.dirname(__file__), "..")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from revisao_agents.state import ReviewState, TechnicalWriterState
from revisao_agents.workflows import build_academic_workflow, build_technical_workflow
from revisao_agents.workflows.technical_writing_workflow import (
    build_technical_writing_workflow,
)
from revisao_agents.config import (
    get_runtime_config_summary,
    llm_call,
    validate_runtime_config,
)
from revisao_agents.utils.vector_utils.pdf_ingestor import ingest_pdf_folder
from revisao_agents.utils.vector_utils.vector_store import search_chunks
from revisao_agents.core.schemas.writer_config import WriterConfig
from revisao_agents.tools.tavily_web_search import extract_tavily, search_tavily_incremental
from revisao_agents.tools.reference_formatter import format_references_from_file
from revisao_agents.agents.review_agent import run_review_agent


_SUPPORTED_LLM_PROVIDERS = ("gemini", "groq", "openai", "openrouter")


def list_llm_providers() -> list[str]:
    """Return supported LLM providers for UI selector."""
    return list(_SUPPORTED_LLM_PROVIDERS)


def get_current_llm_provider() -> str:
    """Return normalized current LLM provider from env."""
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    return provider if provider in _SUPPORTED_LLM_PROVIDERS else "groq"


def get_llm_provider_status() -> str:
    """Build concise status line for the global LLM selector."""
    summary = get_runtime_config_summary()
    provider = summary["llm_provider"]
    model = summary["llm_model"]
    key_ok = summary["llm_provider_key_present"]
    key_name = summary["llm_provider_key"]
    marker = "✅" if key_ok else "⚠️"
    key_msg = "key ok" if key_ok else f"missing {key_name}"
    return f"{marker} Provider: {provider} | Model: {model} | {key_msg}"


def set_llm_provider(provider: str) -> tuple[str, str]:
    """Switch active provider globally for the current UI process.

    Returns:
        (normalized_provider_value_for_dropdown, status_message)
    """
    normalized = (provider or "").strip().lower()
    if normalized not in _SUPPORTED_LLM_PROVIDERS:
        normalized = "groq"

    current = get_current_llm_provider()
    switched = normalized != current

    os.environ["LLM_PROVIDER"] = normalized

    if switched and os.getenv("LLM_MODEL"):
        os.environ.pop("LLM_MODEL", None)

    status = get_llm_provider_status()
    if switched and "Model: <default>" in status:
        status = status + " (modelo redefinido para padrão do provedor)"
    return normalized, status


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


class _StderrCapture:
    """
    Context manager that redirects sys.stderr to a queue so exceptions,
    warnings and direct stderr writes also appear in the live UI stream.
    """

    def __init__(self, q: "queue.Queue[str]"):
        self._q = q
        self._buf = ""
        self._original: Any = None

    def __enter__(self) -> "_StderrCapture":
        self._original = sys.stderr
        sys.stderr = self  # type: ignore[assignment]
        return self

    def __exit__(self, *_: Any) -> None:
        if self._buf.strip():
            self._q.put(self._buf.rstrip())
            self._buf = ""
        sys.stderr = self._original

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


class _QueueLogHandler(logging.Handler):
    def __init__(self, q: "queue.Queue[str]"):
        super().__init__(level=logging.NOTSET)
        self._q = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        if msg.strip():
            self._q.put(msg.rstrip())


class _LoggingCapture:
    """
    Context manager that adds a queue-backed logging handler to the root
    logger so logging calls are streamed live to the UI.
    """

    def __init__(self, q: "queue.Queue[str]"):
        self._q = q
        self._handler = _QueueLogHandler(q)
        self._logger = logging.getLogger()

    def __enter__(self) -> "_LoggingCapture":
        self._handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        self._logger.addHandler(self._handler)
        return self

    def __exit__(self, *_: Any) -> None:
        self._logger.removeHandler(self._handler)


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
        stop_logs = threading.Event()

        def _forward_logs_live() -> None:
            while not stop_logs.is_set() or not log_q.empty():
                try:
                    line = log_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                result_q.put(("log", line))

        log_thread = threading.Thread(target=_forward_logs_live, daemon=True)
        log_thread.start()

        with _StdoutCapture(log_q), _StderrCapture(log_q), _LoggingCapture(log_q):
            try:
                for event in app.stream(state_init):
                    result_q.put(("event", event))
            except Exception as exc:
                result_q.put(("error", exc))
            finally:
                stop_logs.set()
                log_thread.join(timeout=1.0)

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


# ═══════════════════════════════════════════════════════════════════════════
# Interactive Review Chatbot
# ═══════════════════════════════════════════════════════════════════════════

def list_review_files() -> list[str]:
    os.makedirs("reviews", exist_ok=True)
    return sorted(glob.glob("reviews/*.md"))


def _working_copy_path(review_file: str) -> str:
    base_dir = os.path.dirname(review_file) or "reviews"
    base_name = os.path.basename(review_file)
    name, ext = os.path.splitext(base_name)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    return os.path.join(base_dir, f"{name}__review_edit_{ts}{ext}")


def _atomic_write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=os.path.dirname(path) or ".") as temp_file:
        temp_file.write(content)
        tmp_path = temp_file.name
    os.replace(tmp_path, path)


def _split_sections(markdown: str) -> list[dict]:
    lines = markdown.splitlines(keepends=True)
    if not lines:
        return []

    line_offsets: list[int] = []
    acc = 0
    for line in lines:
        line_offsets.append(acc)
        acc += len(line)

    headers: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        match = re.match(r"^##\s+(.+?)\s*$", line.strip("\n"))
        if match:
            headers.append((idx, match.group(1).strip()))

    sections: list[dict] = []
    for header_idx, (start_line, title) in enumerate(headers):
        next_start_line = headers[header_idx + 1][0] if header_idx + 1 < len(headers) else len(lines)
        section_start = line_offsets[start_line]
        section_end = line_offsets[next_start_line] if next_start_line < len(line_offsets) else len(markdown)
        section_text = markdown[section_start:section_end]

        references_start_line: Optional[int] = None
        for i in range(start_line + 1, next_start_line):
            if lines[i].strip().lower().startswith("### referências desta seção"):
                references_start_line = i
                break

        body_end_line = references_start_line if references_start_line is not None else next_start_line
        body_start = line_offsets[start_line + 1] if start_line + 1 < len(line_offsets) else section_start
        body_end = line_offsets[body_end_line] if body_end_line < len(line_offsets) else section_end
        body_text = markdown[body_start:body_end].strip()

        references: list[str] = []
        if references_start_line is not None:
            for i in range(references_start_line + 1, next_start_line):
                ref_line = lines[i].strip()
                if re.match(r"^\[\d+\]", ref_line):
                    references.append(ref_line)

        paragraphs: list[dict] = []
        current_lines: list[str] = []
        current_start: Optional[int] = None
        for i in range(start_line + 1, body_end_line):
            stripped = lines[i].strip()
            if not stripped or stripped.startswith("<!--") or stripped.startswith("### "):
                if current_lines:
                    para_text = "".join(current_lines).strip()
                    if para_text:
                        paragraphs.append(
                            {
                                "text": para_text,
                                "start": current_start,
                                "end": line_offsets[i],
                            }
                        )
                    current_lines = []
                    current_start = None
                continue
            if current_start is None:
                current_start = line_offsets[i]
            current_lines.append(lines[i])

        if current_lines and current_start is not None:
            para_text = "".join(current_lines).strip()
            if para_text:
                paragraphs.append(
                    {
                        "text": para_text,
                        "start": current_start,
                        "end": body_end,
                    }
                )

        sections.append(
            {
                "title": title,
                "start": section_start,
                "end": section_end,
                "text": section_text,
                "body": body_text,
                "paragraphs": paragraphs,
                "references": references,
            }
        )

    return sections


def _resolve_section_index(user_text: str, sections: list[dict]) -> Optional[int]:
    text = user_text.lower()
    sec_match = re.search(r"(?:section|sec|seção)\s*(\d+)", text)
    if sec_match:
        number = sec_match.group(1)
        for idx, section in enumerate(sections):
            if re.match(rf"^{number}[\.)\s]", section["title"], flags=re.IGNORECASE):
                return idx
    if "conclusion" in text or "conclusão" in text:
        for idx, section in enumerate(sections):
            t = section["title"].lower()
            if "conclusion" in t or "conclusão" in t:
                return idx
    return None


def _resolve_paragraph_index(user_text: str, paragraph_count: int) -> Optional[int]:
    if paragraph_count <= 0:
        return None
    text = user_text.lower()
    if "last paragraph" in text or "último parágrafo" in text:
        return paragraph_count - 1
    para_match = re.search(r"(?:paragraph|parágrafo)\s*(\d+)", text)
    if para_match:
        idx = int(para_match.group(1)) - 1
        return idx if 0 <= idx < paragraph_count else None

    ordinals = {
        "first": 0,
        "second": 1,
        "third": 2,
        "fourth": 3,
        "fifth": 4,
        "primeiro": 0,
        "segundo": 1,
        "terceiro": 2,
        "quarto": 3,
        "quinto": 4,
    }
    for token, idx in ordinals.items():
        if token in text and ("paragraph" in text or "parágrafo" in text):
            return idx if idx < paragraph_count else None
    return None


def _extract_quoted_snippet(user_text: str) -> str:
    match = re.search(r'"([^"]{12,})"', user_text)
    if match:
        return match.group(1).strip()
    match = re.search(r"'([^']{12,})'", user_text)
    return match.group(1).strip() if match else ""


def _resolve_target_hint(
    user_text: str,
    sections: list[dict],
    last_target: dict | None = None,
) -> dict | None:
    """Resolve target paragraph for edit proposals.

    Priority: quoted snippet match > explicit section/paragraph > last target.
    """
    if not sections:
        return None

    snippet = _extract_quoted_snippet(user_text)
    if snippet:
        for section in sections:
            for p_idx, paragraph in enumerate(section.get("paragraphs", [])):
                if snippet.lower() in paragraph.get("text", "").lower():
                    return {
                        "section_title": section.get("title", ""),
                        "paragraph_index": p_idx,
                        "start": paragraph.get("start", 0),
                        "end": paragraph.get("end", 0),
                        "before": paragraph.get("text", ""),
                    }

    sec_idx = _resolve_section_index(user_text, sections)
    if sec_idx is None and last_target:
        target_section = str(last_target.get("section", ""))
        for idx, section in enumerate(sections):
            if section.get("title", "") == target_section:
                sec_idx = idx
                break

    if sec_idx is None or sec_idx < 0 or sec_idx >= len(sections):
        return None

    section = sections[sec_idx]
    paragraphs = section.get("paragraphs", [])
    if not paragraphs:
        return None

    para_idx = _resolve_paragraph_index(user_text, len(paragraphs))
    if para_idx is None and last_target:
        maybe_idx = int(last_target.get("paragraph_index", -1))
        if 0 <= maybe_idx < len(paragraphs):
            para_idx = maybe_idx
    if para_idx is None:
        para_idx = 0

    paragraph = paragraphs[para_idx]
    return {
        "section_title": section.get("title", ""),
        "paragraph_index": para_idx,
        "start": paragraph.get("start", 0),
        "end": paragraph.get("end", 0),
        "before": paragraph.get("text", ""),
    }


def _intent(user_text: str) -> str:
    text = user_text.lower().strip()
    if text in {"confirm", "confirm edit", "apply edit", "yes apply", "yes"}:
        return "apply_pending_edit"
    if text in {"cancel", "cancel edit", "discard edit", "no"}:
        return "cancel_pending_edit"
    if "main finding" in text or "main findings" in text or "principais achados" in text:
        return "summarize_main_findings"
    if "cited in section" in text or "papers are cited" in text or "artigos citados" in text:
        return "list_section_citations"
    if "confirmed" in text and "paragraph" in text:
        return "confirm_paragraph_by_authors"
    if ("more documents" in text or "more sources" in text or "mais documentos" in text) and ("phrase" in text or "frase" in text):
        return "suggest_more_documents_for_phrase"
    if any(word in text for word in ["edit", "fix", "add", "rewrite", "melhore", "corrija", "adicionar"]):
        return "propose_targeted_edit"
    return "summarize_main_findings"


def _explicit_web_request(user_text: str) -> bool:
    text = user_text.lower()
    return any(k in text for k in ["internet", "web", "online", "tavily", "search on internet", "busque na internet"])


def _summarize_findings(markdown: str) -> str:
    sections = _split_sections(markdown)
    bullets = []
    for sec in sections[:6]:
        paras = sec.get("paragraphs", [])
        if not paras:
            continue
        first_sentence = paras[0]["text"].split(". ")[0].strip()
        if first_sentence:
            bullets.append(f"- **{sec['title']}**: {first_sentence}.")
    if not bullets:
        return "Não encontrei conteúdo suficiente para sintetizar os principais achados."
    return "\n".join(bullets)


def _list_section_citations(markdown: str, user_text: str) -> str:
    sections = _split_sections(markdown)
    sec_idx = _resolve_section_index(user_text, sections)
    if sec_idx is None:
        return "Não consegui identificar a seção pedida. Use, por exemplo, 'section 2'."
    refs = sections[sec_idx].get("references", [])
    if not refs:
        return f"A seção **{sections[sec_idx]['title']}** não tem bloco de referências detectado."
    return f"### Referências da seção {sections[sec_idx]['title']}\n\n" + "\n".join(refs)


def _extract_citation_number(user_text: str) -> Optional[int]:
    match = re.search(r"\[(\d+)\]", user_text)
    if match:
        return int(match.group(1))

    text = user_text.lower()
    match = re.search(r"(?:source|citation|refer(?:e|ê)ncia|fonte)\s*#?\s*(\d+)", text)
    if match:
        return int(match.group(1))
    return None


def _is_citation_usage_query(user_text: str) -> bool:
    """Returns True only for queries asking *which paragraphs* currently use [N].

    Queries about finding/replacing sources are intentionally excluded so they
    reach the ReAct review agent with its web-search tools.
    """
    text = user_text.lower()
    if _extract_citation_number(user_text) is None:
        return False

    # Queries with these words are about finding or replacing sources — let the
    # ReAct agent handle them.
    exclusions = [
        "replace", "substitut", "alternative", "instead",
        "find source", "find new", "new source", "search for",
        "not yet used", "not used yet", "haven't been used",
        "can be used to", "could replace", "suggest", "recommend",
        "look for", "related with", "related to",
    ]
    if any(kw in text for kw in exclusions):
        return False

    # Require both a listing-intent word AND a usage-verb — avoids false
    # positives such as "what would be a good source for [2]?".
    listing_words = [
        "paragraph", "paragraphs", "parágrafo", "parágrafos",
        "where", "which", "what", "list", "show",
    ]
    usage_words = [
        "using", "uses", "used", "cite", "cites", "cited",
        "referência", "referencia", "mention", "mentions",
    ]
    return any(w in text for w in listing_words) and any(w in text for w in usage_words)


def _list_paragraphs_using_citation(markdown: str, user_text: str) -> str:
    citation_number = _extract_citation_number(user_text)
    if citation_number is None:
        return "Não consegui identificar a citação pedida. Use algo como [2]."

    sections = _split_sections(markdown)
    token = f"[{citation_number}]"
    matches: list[str] = []
    reference_hits: list[str] = []

    for section in sections:
        refs = section.get("references", [])
        for ref in refs:
            if ref.startswith(token):
                reference_hits.append(f"- **{section['title']}**: {ref}")

        for paragraph_index, paragraph in enumerate(section.get("paragraphs", []), start=1):
            text = paragraph.get("text", "")
            if token not in text:
                continue
            snippet = re.sub(r"\s+", " ", text).strip()
            if len(snippet) > 280:
                snippet = snippet[:277].rstrip() + "..."
            matches.append(
                f"- **{section['title']}**, parágrafo **{paragraph_index}**: {snippet}"
            )

    if not matches:
        return f"Nenhum parágrafo na cópia de trabalho usa a citação **{token}**."

    lines = [
        f"### Parágrafos que usam {token}",
        "",
        *matches,
    ]
    if reference_hits:
        lines += ["", "### Referência detectada", "", *reference_hits[:8]]
    return "\n".join(lines)


def _confirm_paragraph(markdown: str, user_text: str) -> tuple[str, dict]:
    sections = _split_sections(markdown)
    snippet = _extract_quoted_snippet(user_text)

    target_para = None
    target_sec = None
    if snippet:
        for section in sections:
            for paragraph in section.get("paragraphs", []):
                if snippet.lower() in paragraph["text"].lower():
                    target_para = paragraph
                    target_sec = section
                    break
            if target_para:
                break

    if target_para is None:
        sec_idx = _resolve_section_index(user_text, sections)
        if sec_idx is not None:
            p_idx = _resolve_paragraph_index(user_text, len(sections[sec_idx].get("paragraphs", [])))
            if p_idx is not None:
                target_sec = sections[sec_idx]
                target_para = target_sec["paragraphs"][p_idx]

    if target_para is None:
        return (
            "Não consegui resolver o parágrafo alvo. Informe seção + parágrafo ou envie o trecho entre aspas.",
            {},
        )

    chunks = search_chunks(target_para["text"][:600], k=6)
    refs = target_sec.get("references", []) if target_sec else []
    ref_labels = [re.sub(r"^\[(\d+)\]\s*", "", r) for r in refs[:5]]
    authors_hint = [os.path.basename(r).replace(".pdf", "") for r in ref_labels]
    evidence = "\n\n".join(chunks[:3]) if chunks else "Sem chunks relevantes retornados no momento."

    msg = (
        "### Verificação do parágrafo\n"
        f"- Seção: **{target_sec['title'] if target_sec else 'N/A'}**\n"
        f"- Evidências MongoDB: **{len(chunks)} chunks**\n"
        f"- Fontes/autores (aproximação pelos arquivos/links citados): {', '.join(authors_hint[:6]) if authors_hint else 'não identificado'}\n\n"
        f"**Trecho alvo:**\n{target_para['text'][:700]}\n\n"
        f"**Evidência principal:**\n{evidence[:1800]}"
    )
    return msg, {
        "section": target_sec["title"] if target_sec else "",
        "chunks": len(chunks),
        "references": len(refs),
    }


def _suggest_more_documents(user_text: str, allow_web: bool) -> tuple[str, dict]:
    snippet = _extract_quoted_snippet(user_text)
    query = snippet or user_text

    local_chunks = search_chunks(query[:600], k=5)
    local_msg = "\n".join(f"- Local evidence chunk {i+1}: {chunk[:180]}..." for i, chunk in enumerate(local_chunks[:3]))

    if not allow_web:
        msg = (
            "### Documentos relacionados (modo local)\n"
            "Use 'search on internet' na pergunta para incluir documentos web.\n\n"
            f"{local_msg or '- Sem evidência local retornada.'}"
        )
        return msg, {"source": "mongo", "chunks": len(local_chunks)}

    web = search_tavily_incremental(query=query[:400], urls_anteriores=[], max_results=5)
    urls = web.get("urls_novos", [])[:3]
    extracted = extract_tavily(urls, incluir_imagens=False) if urls else {"extraidos": []}

    lines = ["### Documentos relacionados (local + web)"]
    if local_msg:
        lines += ["**MongoDB**", local_msg]
    if urls:
        lines += ["\n**Web (Tavily)**"]
        for idx, item in enumerate(extracted.get("extraidos", [])[:3], start=1):
            lines.append(f"- [{idx}] {item.get('title','(sem título)')} — {item.get('url','')}")
    else:
        lines.append("- Nenhum novo URL web encontrado.")

    return "\n".join(lines), {
        "source": "mongo+web",
        "chunks": len(local_chunks),
        "web_urls": len(urls),
    }


def _build_edit_proposal(markdown: str, user_text: str, allow_web: bool) -> tuple[str, dict]:
    sections = _split_sections(markdown)
    sec_idx = _resolve_section_index(user_text, sections)
    if sec_idx is None:
        return "Não consegui identificar a seção alvo para edição.", {}

    section = sections[sec_idx]
    p_idx = _resolve_paragraph_index(user_text, len(section.get("paragraphs", [])))
    if p_idx is None:
        p_idx = 0 if section.get("paragraphs") else None
    if p_idx is None:
        return "A seção alvo não possui parágrafos editáveis.", {}

    paragraph = section["paragraphs"][p_idx]
    evidence_chunks = search_chunks(paragraph["text"][:600], k=5)

    web_context = ""
    if allow_web:
        web = search_tavily_incremental(query=paragraph["text"][:350], urls_anteriores=[], max_results=3)
        urls = web.get("urls_novos", [])[:2]
        if urls:
            ext = extract_tavily(urls, incluir_imagens=False)
            web_context = "\n\nWEB SOURCES:\n" + "\n\n".join(
                f"URL: {item.get('url','')}\nTITLE: {item.get('title','')}\nCONTENT: {str(item.get('content',''))[:1200]}"
                for item in ext.get("extraidos", [])
            )

    prompt = (
        "You are editing a scientific review paragraph. "
        "Return only the revised paragraph text, no extra commentary.\n\n"
        f"USER INSTRUCTION:\n{user_text}\n\n"
        f"CURRENT YEAR: {datetime.now().year}\n\n"
        f"ORIGINAL PARAGRAPH:\n{paragraph['text']}\n\n"
        f"MONGODB EVIDENCE:\n{chr(10).join(evidence_chunks[:3])}\n"
        f"{web_context}"
    )

    try:
        proposed = str(llm_call(prompt=prompt, temperature=0.2)).strip()
    except Exception:
        proposed = paragraph["text"]

    proposal = {
        "section_title": section["title"],
        "paragraph_index": p_idx,
        "start": paragraph["start"],
        "end": paragraph["end"],
        "before": paragraph["text"],
        "after": proposed,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    preview = (
        "### Proposta de edição (pendente)\n"
        f"- Alvo: **{section['title']}**, parágrafo **{p_idx+1}**\n"
        "- Ação necessária: clique em **Confirm Edit** para aplicar.\n\n"
        f"**Antes**\n{proposal['before'][:1200]}\n\n"
        f"**Depois (proposto)**\n{proposal['after'][:1200]}"
    )
    return preview, proposal


def start_review_session(
    review_file: str,
    history: list,
    session_state: dict,
) -> tuple[list, dict, str, str]:
    if not review_file or not os.path.exists(review_file):
        return history, session_state, "❌ Arquivo não encontrado.", ""

    normalized = os.path.normpath(review_file)
    if not normalized.startswith("reviews/"):
        return history, session_state, "❌ Apenas arquivos em reviews/ são permitidos.", ""

    working_copy = _working_copy_path(normalized)
    shutil.copyfile(normalized, working_copy)
    content = _read_md(working_copy)

    state = {
        "original_file_path": normalized,
        "working_copy_path": working_copy,
        "current_markdown": content,
        "chat_history": [],
        "pending_edit": {},
        "last_target_resolution": {},
        "retrieval_trace": [],
        "status": "ready",
    }

    history = history + [
        {
            "role": "assistant",
            "content": (
                "✅ Sessão de revisão iniciada.\n"
                f"- Original: `{normalized}`\n"
                f"- Cópia editável: `{working_copy}`\n"
                "Pergunte sobre achados, referências, confirmação de parágrafos ou peça propostas de edição."
            ),
        }
    ]
    return history, state, "✅ Sessão pronta", content


def review_chat_turn(
    user_msg: str,
    history: list,
    session_state: dict,
    web_enabled: bool = False,
) -> tuple[list, dict, str, str]:
    if not session_state or not session_state.get("working_copy_path"):
        return history, session_state, "❌ Inicie uma sessão selecionando um arquivo.", ""
    if not user_msg.strip():
        return history, session_state, "⚠️ Mensagem vazia.", _read_md(session_state.get("working_copy_path"))

    working_copy = session_state["working_copy_path"]
    markdown = _read_md(working_copy)
    session_state["current_markdown"] = markdown
    sections = _split_sections(markdown)
    allow_web = bool(web_enabled) or _explicit_web_request(user_msg)
    pending_edit = session_state.get("pending_edit") or {}
    target_hint = _resolve_target_hint(
        user_msg,
        sections,
        session_state.get("last_target_resolution") or {},
    )

    if _is_citation_usage_query(user_msg):
        reply = _list_paragraphs_using_citation(markdown, user_msg)
        history = history + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": reply},
        ]
        session_state["chat_history"] = history
        session_state.setdefault("retrieval_trace", []).append({
            "action": "local_citation_lookup",
            "web": False,
            "at": datetime.now().isoformat(timespec="seconds"),
            "tool_calls": [],
        })
        return history, session_state, "✅ Sessão ativa", _read_md(working_copy)

    # ── Run the ReAct review agent ────────────────────────────────────
    try:
        result = run_review_agent(
            document_content=markdown,
            document_sections=sections,
            user_message=user_msg,
            chat_history=session_state.get("chat_history", []),
            allow_web=allow_web,
            pending_edit=pending_edit or None,
            target_hint=target_hint,
        )
    except Exception as exc:
        reply = f"⚠️ Erro do agente: {exc}"
        history = history + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": reply},
        ]
        session_state["chat_history"] = history
        return history, session_state, "❌ Erro no agente", _read_md(working_copy)

    action = result.get("action", "answer")
    reply = result.get("reply", "")

    # ── Handle actions ────────────────────────────────────────────────
    if action == "apply_edit":
        proposal = session_state.get("pending_edit") or {}
        if not proposal:
            reply = "Não há edição pendente para confirmar."
        else:
            start = int(proposal["start"])
            end = int(proposal["end"])
            updated = markdown[:start] + proposal["after"] + "\n\n" + markdown[end:]
            _atomic_write(working_copy, updated)
            markdown = _read_md(working_copy)
            session_state["current_markdown"] = markdown
            session_state["pending_edit"] = {}
            session_state["last_target_resolution"] = {
                "section": proposal.get("section_title", ""),
                "paragraph_index": proposal.get("paragraph_index", -1),
            }
            reply = (
                "✅ Edição aplicada na cópia de trabalho.\n"
                f"- Seção: **{proposal.get('section_title', '')}**\n"
                f"- Parágrafo: **{int(proposal.get('paragraph_index', 0)) + 1}**\n"
                f"- Arquivo: `{working_copy}`"
            )

    elif action == "cancel_edit":
        has_pending = bool(session_state.get("pending_edit"))
        session_state["pending_edit"] = {}
        reply = "🗑️ Edição pendente cancelada." if has_pending else "Não havia edição pendente para cancelar."

    elif action == "edit_proposal":
        proposal = result.get("edit_proposal")
        if proposal:
            session_state["pending_edit"] = proposal
            session_state["last_target_resolution"] = {
                "section": proposal.get("section_title", ""),
                "paragraph_index": proposal.get("paragraph_index", -1),
            }
            reply = (
                "### Proposta de edição (pendente)\n"
                f"- Alvo: **{proposal.get('section_title', '')}**, "
                f"parágrafo **{int(proposal.get('paragraph_index', 0)) + 1}**\n"
                "- Ação necessária: clique em **Confirm Edit** ou diga "
                "'confirmar' para aplicar.\n\n"
                f"**Antes**\n{proposal['before'][:1200]}\n\n"
                f"**Depois (proposto)**\n{proposal['after'][:1200]}"
            )
    # else: action == "answer" → reply already set by agent

    # ── Update trace & history ────────────────────────────────────────
    trace = {
        "action": action,
        "web": allow_web,
        "at": datetime.now().isoformat(timespec="seconds"),
        "tool_calls": result.get("trace", []),
    }
    session_state.setdefault("retrieval_trace", []).append(trace)

    history = history + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": reply},
    ]
    session_state["chat_history"] = history

    pending = session_state.get("pending_edit")
    status = "🟡 Edição pendente — confirme ou cancele" if pending else "✅ Sessão ativa"
    return history, session_state, status, _read_md(working_copy)


def confirm_review_edit(
    history: list,
    session_state: dict,
) -> tuple[list, dict, str, str]:
    return review_chat_turn("confirm edit", history, session_state)


def cancel_review_edit(
    history: list,
    session_state: dict,
) -> tuple[list, dict, str, str]:
    return review_chat_turn("cancel edit", history, session_state)


def save_review_manual_edit(
    edited_text: str,
    history: list,
    session_state: dict,
) -> tuple[list, dict, str, str]:
    """Save manual edits made directly in the text editor."""
    if not session_state or not session_state.get("working_copy_path"):
        return history, session_state, "❌ Nenhuma sessão ativa.", ""
    if not edited_text.strip():
        return history, session_state, "⚠️ Texto vazio — nada salvo.", _read_md(session_state.get("working_copy_path"))

    working_copy = session_state["working_copy_path"]
    _atomic_write(working_copy, edited_text)
    refreshed = _read_md(working_copy)
    session_state["current_markdown"] = refreshed
    session_state["pending_edit"] = {}

    history = history + [
        {"role": "assistant", "content": f"💾 Edição manual salva em `{working_copy}`."},
    ]
    session_state["chat_history"] = history
    return history, session_state, "✅ Edição manual salva", refreshed
