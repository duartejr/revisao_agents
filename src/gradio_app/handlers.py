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

# 1. Standard Library Imports
import glob
import hashlib
import logging
import os
import queue
import re
import shutil
import sys
import tempfile
import threading
from collections.abc import Generator
from datetime import datetime
from typing import Any

# 2. Local Codebase Imports
from revisao_agents.agents.image_suggestion_agent import run_image_suggestion_agent
from revisao_agents.agents.reference_extractor_agent import (
    run_reference_extractor_agent,
)
from revisao_agents.agents.reference_formatter_agent import (
    run_reference_formatter_agent,
)
from revisao_agents.agents.review_agent import run_review_agent
from revisao_agents.config import (
    get_runtime_config_summary,
    llm_call,
    validate_runtime_config,
)
from revisao_agents.core.schemas.writer_config import WriterConfig
from revisao_agents.state import ReviewState, TechnicalWriterState
from revisao_agents.tools.reference_formatter import format_references_from_file
from revisao_agents.tools.tavily_web_search import (
    extract_tavily,
    search_tavily_incremental,
)
from revisao_agents.utils.bib_utils.doi_utils import (
    extract_doi_from_url,
    get_bibtex_from_doi,
    search_crossref_by_title,
    search_doi_in_text,
)
from revisao_agents.utils.llm_utils.prompt_loader import load_prompt
from revisao_agents.utils.vector_utils.pdf_ingestor import ingest_pdf_folder
from revisao_agents.utils.vector_utils.vector_store import (
    search_chunk_records,
    search_chunks,
)
from revisao_agents.workflows import build_academic_workflow, build_technical_workflow
from revisao_agents.workflows.technical_writing_workflow import (
    build_technical_writing_workflow,
)

_SRC = os.path.join(os.path.dirname(__file__), "..")

if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_SUPPORTED_LLM_PROVIDERS = ("google", "groq", "openai", "openrouter")


def list_llm_providers() -> list[str]:
    """Return supported LLM providers for UI selector."""
    return list(_SUPPORTED_LLM_PROVIDERS)


def get_current_llm_provider() -> str:
    """Return normalized current LLM provider from env."""
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    return provider if provider in _SUPPORTED_LLM_PROVIDERS else "openai"


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

    Args:
        provider: The name of the provider to switch to (e.g., "google", "groq", "openai", "openrouter").

    Returns:
        (normalized_provider_value_for_dropdown, status_message)
    """
    normalized = (provider or "").strip().lower()
    if normalized not in _SUPPORTED_LLM_PROVIDERS:
        normalized = "openai"

    current = get_current_llm_provider()
    switched = normalized != current

    os.environ["LLM_PROVIDER"] = normalized

    # Set LLM_MODEL to "" (not pop) so subsequent load_dotenv() calls cannot
    # restore the old model from the .env file (load_dotenv skips vars that
    # already exist, even if empty).
    if switched:
        os.environ["LLM_MODEL"] = ""

    status = get_llm_provider_status()
    if switched and "Model: <default>" in status:
        status = status + " (model reset to provider default)"
    return normalized, status


# ═══════════════════════════════════════════════════════════════════════════
# Live stdout capture
# ═══════════════════════════════════════════════════════════════════════════


class _StdoutCapture:
    """
    Context manager that redirects sys.stdout to a queue so the caller
    can read lines as they are produced by any print() inside the block.
    """

    def __init__(self, q: queue.Queue[str]):
        """Initialize the stdout capture with a queue to receive lines."""
        self._q = q
        self._buf = ""
        self._original: Any = None

    def __enter__(self) -> _StdoutCapture:
        """Redirect sys.stdout to this object, which funnels lines into the queue."""
        self._original = sys.stdout
        sys.stdout = self  # type: ignore[assignment]
        return self

    def __exit__(self, *_: Any) -> None:
        """Restore original sys.stdout and flush any remaining buffer to the queue."""
        if self._buf.strip():
            self._q.put(self._buf.rstrip())
            self._buf = ""
        sys.stdout = self._original

    def write(self, text: str) -> int:
        """Write text to the original stdout and also capture it in the buffer. '

        Args:
            text: The string to write (may contain multiple lines).

        Returns:
            The number of characters written.
        """
        self._original.write(text)
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            stripped = line.rstrip()
            if stripped:
                self._q.put(stripped)
        return len(text)

    def flush(self) -> None:
        """Flush the original stdout."""
        self._original.flush()

    @property
    def encoding(self) -> str:
        """Return the encoding of the original stdout, defaulting to 'utf-8' if not available."""
        return getattr(self._original, "encoding", "utf-8")


class _StderrCapture:
    """
    Context manager that redirects sys.stderr to a queue so exceptions,
    warnings and direct stderr writes also appear in the live UI stream.
    """

    def __init__(self, q: queue.Queue[str]):
        """Initialize the stderr capture with a queue to receive lines."""
        self._q = q
        self._buf = ""
        self._original: Any = None

    def __enter__(self) -> _StderrCapture:
        """Redirect sys.stderr to this object, which funnels lines into the queue."""
        self._original = sys.stderr
        sys.stderr = self  # type: ignore[assignment]
        return self

    def __exit__(self, *_: Any) -> None:
        """Restore original sys.stderr and flush any remaining buffer to the queue."""
        if self._buf.strip():
            self._q.put(self._buf.rstrip())
            self._buf = ""
        sys.stderr = self._original

    def write(self, text: str) -> int:
        """Write text to the original stderr and also capture it in the buffer.

        Args:
            text: The string to write (may contain multiple lines).

        Returns:
            The number of characters written.
        """
        self._original.write(text)
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            stripped = line.rstrip()
            if stripped:
                self._q.put(stripped)
        return len(text)

    def flush(self) -> None:
        """Flush the original stderr."""
        self._original.flush()

    @property
    def encoding(self) -> str:
        """Return the encoding of the original stderr, defaulting to 'utf-8' if not available."""
        return getattr(self._original, "encoding", "utf-8")


class _QueueLogHandler(logging.Handler):
    """A logging handler that sends log records to a queue, allowing logs to be captured and
    streamed in real-time to the UI."""

    def __init__(self, q: queue.Queue[str]):
        """Initialize the queue log handler with a queue to receive log messages."""
        super().__init__(level=logging.NOTSET)
        self._q = q

    def emit(self, record: logging.LogRecord) -> None:
        """Format the log record and put it into the queue.

        Args:
            record: The log record to emit.
        """
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

    def __init__(self, q: queue.Queue[str]):
        """Initialize the logging capture with a queue to receive log messages."""
        self._q = q
        self._handler = _QueueLogHandler(q)
        self._logger = logging.getLogger()

    def __enter__(self) -> _LoggingCapture:
        """Add the queue log handler to the root logger."""
        self._handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        self._logger.addHandler(self._handler)
        return self

    def __exit__(self, *_: Any) -> None:
        """Remove the queue log handler from the root logger."""
        self._logger.removeHandler(self._handler)


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════


def _list_md(folder: str) -> list[str]:
    """List all .md files in the given folder.

    Args:
        folder: The path to the folder to search for .md files.

    Returns:
         A list of file paths to .md files in the folder."""
    return glob.glob(os.path.join(folder, "*.md"))


def _find_newest_md(folder: str) -> str | None:
    """Find the newest .md file in the given folder.

    Args:
        folder: The path to the folder to search for .md files.

    Returns:
        The file path to the newest .md file, or None if no .md files are found.
    """
    files = _list_md(folder)
    return max(files, key=os.path.getmtime) if files else None


def _read_md(path: str | None) -> str:
    """Read the content of a markdown file, returning an empty string if the file does not
    exist or cannot be read.

    Args:
        path: The file path to the markdown file to read.

    Returns:
        The content of the markdown file as a string, or an empty string if the file does not
        exist or cannot be read.
    """
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
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

    Args:
        tema: The review topic/theme provided by the user.
        tipo: The review type ("academico", "tecnico", or "ambos").
        rodadas: The number of refinement rounds for HITL steps.

    Returns:
        tuple: A tuple containing:
            - history: A list of message dicts representing the conversation history.
            - session_state: A dict containing the session state for continuing the workflow.
            - status_msg: A string message indicating the current status or next steps.
            - rendered_plan: A string with the rendered plan in markdown, empty until completion.
    """
    if not tema.strip():
        return [], {}, "❌ Please provide a topic before starting.", ""

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
        msg = "❌ Incomplete configuration for this mode:\n- " + "\n- ".join(cfg_issues)
        return [], {}, msg, ""

    tipos_list = ["academico", "tecnico"] if tipo == "ambos" else [tipo]
    tipo_atual = tipos_list[0]
    label = "ACADEMIC" if tipo_atual == "academico" else "TECHNICAL"

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
            return [], {}, f"❌ Error starting: {exc}", ""

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
        history.append(
            {
                "role": "assistant",
                "content": f"✅ {label} planning complete! Plan saved at `{plan_path}`",
            }
        )
        return history, {}, "✅ Done", rendered

    agent_question = ""
    for role, content in reversed(graph_state.values.get("interview_history", [])):
        if role == "assistant":
            agent_question = content
            break

    p = graph_state.values.get("questions_asked", 0)
    mp = graph_state.values.get("max_questions", rodadas)
    history.append(
        {
            "role": "assistant",
            "content": f"[Round {p}/{mp} — {tipo_atual}]\n\n{agent_question}",
        }
    )

    session_state = {
        "app": app,
        "config": config,
        "tipo": tipo_atual,
        "tipos_pendentes": tipos_list[1:],
        "theme": tema,
        "rodadas": rodadas,
    }

    return history, session_state, f"🔄 {label} in progress — waiting for reply…", ""


def continue_planning(
    user_msg: str,
    history: list,
    session_state: dict,
) -> tuple[list, dict, str, str]:
    """Feed user response back into the HITL loop.

    Args:
        user_msg: The message from the user responding to the agent's question.
        history: The current conversation history, to which the user message will be appended.
        session_state: The current session state containing the app and config needed to continue the workflow.

    Returns:
        history, session_state, status_msg, rendered_plan.
    """
    if not session_state or "app" not in session_state:
        return history, session_state, "❌ No active session.", ""

    app = session_state["app"]
    config = session_state["config"]
    tipo = session_state["tipo"]
    label = "ACADEMIC" if tipo == "academico" else "TECHNICAL"

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
            history = history + [{"role": "assistant", "content": f"❌ Error: {exc}"}]
            return history, session_state, f"❌ Error: {exc}", ""

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
            f"✅ {label} planning complete! Plan saved at `{plan_path}`"
            if plan_path
            else f"✅ {label} planning complete!"
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

        return history, {}, "✅ All planning complete!", rendered

    agent_question = ""
    for role, content in reversed(graph_state.values.get("interview_history", [])):
        if role == "assistant":
            agent_question = content
            break

    p = graph_state.values.get("questions_asked", 0)
    mp = graph_state.values.get("max_questions", session_state.get("rodadas", 3))
    history = history + [
        {
            "role": "assistant",
            "content": f"[Round {p}/{mp} — {tipo}]\n\n{agent_question}",
        }
    ]

    return history, session_state, f"🔄 {label} in progress — round {p}/{mp}", ""


# ═══════════════════════════════════════════════════════════════════════════
# Option 3 — Execute Writing from existing plan
# ═══════════════════════════════════════════════════════════════════════════


def list_plan_files(mode: str) -> list[str]:
    """List available plan files for the given mode (Academic or Technical)."""
    os.makedirs("plans", exist_ok=True)
    pattern = (
        "plans/plano_revisao_tecnica_*.md" if mode == "Technical" else "plans/plano_revisao_*.md"
    )
    files = sorted(glob.glob(pattern))
    if not files:
        files = sorted(glob.glob("plans/plano_revisao_*.md"))
    if not files:
        files = sorted(glob.glob("plano_revisao_*.md"))
    return files if files else ["(no plan files found)"]


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
            history
            + [
                {
                    "role": "assistant",
                    "content": "❌ Configuração incompleta:\n- " + "\n- ".join(cfg_issues),
                }
            ],
            "❌ Erro",
            "",
        )
        return

    if not plan_path or not os.path.exists(plan_path):
        yield (
            history
            + [
                {
                    "role": "assistant",
                    "content": f"❌ Plano não encontrado: `{plan_path}`",
                }
            ],
            "❌ Erro",
            "",
        )
        return

    if mode == "Academic":
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
        {
            "role": "assistant",
            "content": f"▶ Iniciando escrita **{mode}** — `{os.path.basename(plan_path)}`",
        }
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
        f"✅ Escrita concluída!  📄 `{output_file}`" if output_file else "✅ Escrita concluída!"
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
    with tempfile.NamedTemporaryFile(
        "w", delete=False, encoding="utf-8", dir=os.path.dirname(path) or "."
    ) as temp_file:
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
        next_start_line = (
            headers[header_idx + 1][0] if header_idx + 1 < len(headers) else len(lines)
        )
        section_start = line_offsets[start_line]
        section_end = (
            line_offsets[next_start_line] if next_start_line < len(line_offsets) else len(markdown)
        )
        section_text = markdown[section_start:section_end]

        references_start_line: int | None = None
        for i in range(start_line + 1, next_start_line):
            if re.match(
                r"^###\s+(?:References\s+for\s+this\s+section|Refer[êe]ncias\s+desta\s+se[çc][ãa]o)\s*$",
                lines[i].strip(),
                re.IGNORECASE,
            ):
                references_start_line = i
                break

        body_end_line = (
            references_start_line if references_start_line is not None else next_start_line
        )
        body_start = (
            line_offsets[start_line + 1] if start_line + 1 < len(line_offsets) else section_start
        )
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
        current_start: int | None = None
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


def _resolve_section_index(user_text: str, sections: list[dict]) -> int | None:
    text = user_text.lower()

    # Priority 1 — explicit keyword + number, allowing non-whitespace between
    # them (e.g. "seção ## 4", "section #4", "seção4", "section 4")
    sec_match = re.search(
        r"(?:section|sec|se[çc][ãa]o|chapter|cap[ií]tulo)\s*[^\w\d]*(\d+)",
        text,
    )
    if sec_match:
        number = sec_match.group(1)
        for idx, section in enumerate(sections):
            if re.match(rf"^{number}[\.)\s]", section["title"], flags=re.IGNORECASE):
                return idx

    # Priority 2 — bare markdown heading number in user text: "## 4", "##4."
    md_match = re.search(r"##\s*(\d+)[.\s]", text)
    if md_match:
        number = md_match.group(1)
        for idx, section in enumerate(sections):
            if re.match(rf"^{number}[\.)\s]", section["title"], flags=re.IGNORECASE):
                return idx

    # Priority 3 — conclusion keyword
    if re.search(r"\b(?:conclusion|conclus[ãa]o)\b", text):
        for idx, section in enumerate(sections):
            t = section["title"].lower()
            if re.search(r"\b(?:conclusion|conclus[ãa]o)\b", t):
                return idx

    # Priority 4 — match by section title word (e.g. "discussion", "methodology")
    # Strip stopwords so short common words don't cause false positives.
    _STOPWORDS = {
        "a",
        "o",
        "e",
        "de",
        "da",
        "do",
        "em",
        "para",
        "no",
        "na",
        "com",
        "the",
        "of",
        "in",
        "on",
        "at",
        "to",
        "and",
        "for",
        "or",
        "an",
        "quero",
        "contexto",
        "seção",
        "section",
        "apenas",
        "only",
        "sobre",
        "about",
        "this",
        "esse",
        "este",
        "essa",
    }
    user_words = {
        w
        for w in re.findall(r"[a-záàâãéèêíïóôõöúüçñ]+", text)
        if w not in _STOPWORDS and len(w) > 3
    }
    best_idx: int | None = None
    best_hits = 0
    for idx, section in enumerate(sections):
        title_words = set(re.findall(r"[a-záàâãéèêíïóôõöúüçñ]+", section["title"].lower()))
        hits = len(user_words & title_words)
        if hits > best_hits:
            best_hits = hits
            best_idx = idx
    if best_hits >= 1:
        return best_idx

    return None


def _resolve_paragraph_index(user_text: str, paragraph_count: int) -> int | None:
    if paragraph_count <= 0:
        return None
    text = user_text.lower()
    if re.search(r"\b(?:last\s+paragraph|[úu]ltimo\s+par[áa]grafo)\b", text):
        return paragraph_count - 1
    para_match = re.search(r"(?:paragraph|par[áa]grafo)\s*(\d+)", text)
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
        if token in text and re.search(r"\b(?:paragraph|par[áa]grafo)\b", text):
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


def _detect_user_language(user_text: str, fallback: str = "pt") -> str:
    padded = f" {user_text.lower()} "
    pt_markers = [
        " seção ",
        " parágrafo ",
        " referências ",
        " referência ",
        " citação ",
        " fonte ",
        " internet ",
        " confirmar ",
        " confirme ",
        " cancelar ",
        " edição ",
        " achados ",
        " frase ",
        " trecho ",
        " artigos ",
        " mais ",
    ]
    en_markers = [
        " section ",
        " paragraph ",
        " references ",
        " reference ",
        " citation ",
        " source ",
        " internet ",
        " confirm ",
        " cancel ",
        " edit ",
        " findings ",
        " phrase ",
        " snippet ",
        " papers ",
        " more ",
    ]
    pt_score = sum(marker in padded for marker in pt_markers)
    en_score = sum(marker in padded for marker in en_markers)
    if en_score > pt_score:
        return "en"
    if pt_score > en_score:
        return "pt"
    return fallback


def _localized_text(language: str, pt_text: str, en_text: str) -> str:
    return en_text if language == "en" else pt_text


def _intent(user_text: str) -> str:
    text = user_text.lower().strip()
    if text in {
        "confirm",
        "confirm edit",
        "apply edit",
        "yes apply",
        "yes",
        "confirmar",
        "confirmar edição",
        "aplicar edição",
        "aplicar edicao",
        "sim",
    }:
        return "apply_pending_edit"
    if text in {
        "cancel",
        "cancel edit",
        "discard edit",
        "no",
        "cancelar",
        "cancelar edição",
        "cancelar edicao",
        "descartar edição",
        "descartar edicao",
        "não",
        "nao",
    }:
        return "cancel_pending_edit"
    if any(
        phrase in text
        for phrase in [
            "main finding",
            "main findings",
            "key finding",
            "key findings",
            "principais achados",
            "achado principal",
            "achados principais",
        ]
    ):
        return "summarize_main_findings"
    if any(
        phrase in text
        for phrase in [
            "cited in section",
            "papers are cited",
            "references in section",
            "sources in section",
            "artigos citados",
            "referências na seção",
            "referencias na secao",
            "fontes na seção",
            "fontes na secao",
        ]
    ):
        return "list_section_citations"
    if ("confirmed" in text and "paragraph" in text) or (
        "confirmado" in text and ("parágrafo" in text or "paragrafo" in text)
    ):
        return "confirm_paragraph_by_authors"
    if any(
        phrase in text
        for phrase in [
            "more documents",
            "more sources",
            "additional documents",
            "additional sources",
            "mais documentos",
            "mais fontes",
        ]
    ) and any(phrase in text for phrase in ["phrase", "excerpt", "snippet", "frase", "trecho"]):
        return "suggest_more_documents_for_phrase"
    if any(
        word in text
        for word in [
            "edit",
            "fix",
            "add",
            "rewrite",
            "improve",
            "update",
            "modify",
            "replace",
            "remove",
            "melhore",
            "corrija",
            "adicionar",
            "reescreva",
            "atualize",
            "modifique",
            "substitua",
            "remova",
        ]
    ):
        return "propose_targeted_edit"
    return "summarize_main_findings"


def _explicit_web_request(user_text: str) -> bool:
    text = user_text.lower()
    return any(
        k in text
        for k in [
            "internet",
            "web",
            "online",
            "tavily",
            "search on internet",
            "search the web",
            "busque na internet",
            "pesquise na internet",
            "busque online",
        ]
    )


# ═══════════════════════════════════════════════════════════════════════════
# Image suggestion intent helpers
# ═══════════════════════════════════════════════════════════════════════════

_IMAGE_REQUEST_KEYWORDS = [
    # Portuguese
    "imagem",
    "imagens",
    "figur",
    "ilustr",
    "foto",
    "diagrama",
    "inserir imagem",
    "adicionar imagem",
    "sugerir imagem",
    "buscar imagem",
    "encontrar imagem",
    "colocar imagem",
    "incluir imagem",
    "imagem para",
    "imagens para",
    # English
    "image",
    "images",
    "figure",
    "illustration",
    "diagram",
    "picture",
    "insert image",
    "add image",
    "suggest image",
    "find image",
    "search image",
    "place image",
]


def _is_image_request(user_text: str) -> bool:
    """Return True when the user is asking for image suggestions."""
    text = user_text.lower()
    return any(kw in text for kw in _IMAGE_REQUEST_KEYWORDS)


def _build_image_scope_description(
    user_text: str, sections: list[dict], language: str = "en"
) -> tuple[str, str]:
    """Derive a human-readable scope and a document excerpt for the image agent.

    Returns (scope_description, document_excerpt).
    The excerpt clearly delimits each paragraph with [PARAGRAPH N] markers so
    the image agent can reproduce them verbatim in the PARAGRAPH template block.
    """
    text = user_text.lower()

    def _paragraphs_excerpt(section: dict, max_chars: int = 3500) -> str:
        """Build a numbered-paragraph excerpt for a section."""
        accumulated = f"## {section['title']}\n\n"
        for i, para in enumerate(section.get("paragraphs", []), 1):
            para_text = para.get("text", "").strip()
            if not para_text:
                continue
            block = f"[PARAGRAPH {i}]\n{para_text}\n\n"
            if len(accumulated) + len(block) > max_chars:
                break
            accumulated += block
        return accumulated

    # Check for specific section request
    sec_idx = _resolve_section_index(user_text, sections)
    if sec_idx is not None and 0 <= sec_idx < len(sections):
        section = sections[sec_idx]
        scope = _localized_text(
            language,
            f"seção {sec_idx + 1} — {section['title']}",
            f"section {sec_idx + 1} — {section['title']}",
        )
        excerpt = _paragraphs_excerpt(section)
        return scope, excerpt

    # Check for specific paragraph request
    para_match = re.search(r"par[áa]grafo\s*(\d+)|paragraph\s*(\d+)", text)
    if para_match and sections:
        num_str = para_match.group(1) or para_match.group(2)
        para_num = int(num_str) - 1
        # Try to infer the intended section from the user text by title,
        # falling back to the first section if nothing matches.
        section = None
        for candidate in sections:
            title = str(candidate.get("title", "")).strip()
            if title and title.lower() in text:
                section = candidate
                break
        if section is None:
            section = sections[0]
        paragraphs = section.get("paragraphs", [])
        if 0 <= para_num < len(paragraphs):
            para = paragraphs[para_num]
            scope = _localized_text(
                language,
                f"parágrafo {para_num + 1} da seção '{section['title']}'",
                f"paragraph {para_num + 1} of section '{section['title']}'",
            )
            excerpt = (
                f"## {section['title']}\n\n[PARAGRAPH {para_num + 1}]\n{para.get('text', '')}\n"
            )
            return scope, excerpt

    # Check for quoted snippet
    snippet = _extract_quoted_snippet(user_text)
    if snippet and sections:
        for section in sections:
            for p_idx, paragraph in enumerate(section.get("paragraphs", [])):
                if snippet.lower() in paragraph.get("text", "").lower():
                    scope = _localized_text(
                        language,
                        f"parágrafo contendo \"{snippet[:60]}\"... na seção '{section['title']}'",
                        f"paragraph containing \"{snippet[:60]}\"... in section '{section['title']}'",
                    )
                    excerpt = (
                        f"## {section['title']}\n\n"
                        f"[PARAGRAPH {p_idx + 1}]\n{paragraph.get('text', '')}\n"
                    )
                    return scope, excerpt

    # Default: all sections (condensed, showing first paragraph of each)
    scope = _localized_text(
        language, "todas as seções do documento", "all sections of the document"
    )
    parts: list[str] = []
    total = 0
    for sec in sections[:6]:
        paragraphs = sec.get("paragraphs", [])
        if not paragraphs:
            continue
        block = f"## {sec['title']}\n\n"
        for i, para in enumerate(paragraphs[:3], 1):
            para_text = para.get("text", "").strip()
            if para_text:
                block += f"[PARAGRAPH {i}]\n{para_text}\n\n"
        if total + len(block) > 4000:
            break
        parts.append(block)
        total += len(block)
    excerpt = "\n".join(parts)
    return scope, excerpt


def _build_image_confirmation_prompt(scope: str, language: str) -> str:
    """Return a confirmation prompt asking user to confirm image search scope."""
    return _localized_text(
        language,
        f"Vou buscar imagens para ilustrar: **{scope}**.\n\n"
        "Confirme o escopo ou especifique uma seção/parágrafo diferente.\n"
        "Responda **sim** para confirmar ou descreva o escopo desejado.",
        f"I will search for images to illustrate: **{scope}**.\n\n"
        "Confirm the scope or specify a different section/paragraph.\n"
        "Reply **yes** to confirm or describe the desired scope.",
    )


def _summarize_findings(markdown: str) -> str:
    language = _detect_user_language(markdown)
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
        return _localized_text(
            language,
            "Não encontrei conteúdo suficiente para sintetizar os principais achados.",
            "I couldn't find enough content to summarize the main findings.",
        )
    return "\n".join(bullets)


def _list_section_citations(markdown: str, user_text: str) -> str:
    language = _detect_user_language(user_text)
    sections = _split_sections(markdown)
    sec_idx = _resolve_section_index(user_text, sections)
    if sec_idx is None:
        return _localized_text(
            language,
            "Não consegui identificar a seção pedida. Use, por exemplo, 'section 2'.",
            "I couldn't identify the requested section. Use, for example, 'section 2'.",
        )
    refs = sections[sec_idx].get("references", [])
    if not refs:
        return _localized_text(
            language,
            f"A seção **{sections[sec_idx]['title']}** não tem bloco de referências detectado.",
            f"The section **{sections[sec_idx]['title']}** has no detected references block.",
        )
    heading = _localized_text(language, "Referências da seção", "Section references")
    return f"### {heading} {sections[sec_idx]['title']}\n\n" + "\n".join(refs)


def _extract_citation_number(user_text: str) -> int | None:
    match = re.search(r"\[(\d+)\]", user_text)
    if match:
        return int(match.group(1))

    text = user_text.lower()
    match = re.search(r"(?:source|citation|reference|refer(?:e|ê)ncia|fonte)\s*#?\s*(\d+)", text)
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
        "replace",
        "substitut",
        "alternative",
        "instead",
        "find source",
        "find new",
        "new source",
        "search for",
        "not yet used",
        "not used yet",
        "haven't been used",
        "can be used to",
        "could replace",
        "suggest",
        "recommend",
        "look for",
        "related with",
        "related to",
        # Portuguese source-search / rewrite intents
        "procurar",
        "buscar",
        "busque",
        "pesquise",
        "pesquisar",
        "ainda não usada",
        "ainda nao usada",
        "não usada ainda",
        "nao usada ainda",
        "fontes não usadas",
        "fontes nao usadas",
        "novas fontes",
        "nova fonte",
        "sugerir",
        "recomendar",
        "substituir",
        "alternativa",
        "relacionado com",
        "relacionado a",
        # Rewrite / improve intents (Portuguese)
        "reescreva",
        "reescrever",
        "melhore",
        "melhorar",
        "melhorar o",
        "adicionando",
        "adicione",
        "adicionar",
        # Rewrite / improve intents (English)
        "rewrite",
        "improve",
        "add new",
        "new references",
        "abnt",
    ]
    if any(kw in text for kw in exclusions):
        return False

    # Require both a listing-intent word AND a usage-verb — avoids false
    # positives such as "what would be a good source for [2]?".
    listing_words = [
        "paragraph",
        "paragraphs",
        "parágrafo",
        "parágrafos",
        "paragrafo",
        "paragrafos",
        "where",
        "which",
        "what",
        "list",
        "show",
        "onde",
        "qual",
        "quais",
        "listar",
        "mostre",
        "mostrar",
    ]
    usage_words = [
        "using",
        "uses",
        "used",
        "cite",
        "cites",
        "cited",
        "referência",
        "referencia",
        "referências",
        "referencias",
        "mention",
        "mentions",
        "mentioned",
        "menciona",
        "mencionado",
        "usando",
        r"\busa\b",
        "usado",
        "citado",
        "citam",
    ]
    return any(w in text for w in listing_words) and any(re.search(w, text) for w in usage_words)


def _matches_intent_keyword(text: str, keyword: str) -> bool:
    """Match a keyword as a whole token or exact phrase inside text."""
    text = (text or "").lower()
    keyword = (keyword or "").strip().lower()
    if not keyword:
        return False
    if re.search(r"\s", keyword):
        return keyword in text
    return bool(re.search(rf"\b{re.escape(keyword)}\b", text))


def _classify_phrase_reference_intent(user_text: str) -> tuple[bool, dict[str, bool]]:
    """Return whether the user asks for the source of a specific phrase/snippet.

    This stays deterministic, using semantic signals with boundary-aware matching
    plus explicit rewrite exclusions to avoid false positives like "rephrase".
    """
    text = (user_text or "").lower()
    has_citation_number = _extract_citation_number(user_text) is not None
    has_quoted_snippet = bool(_extract_quoted_snippet(user_text))

    source_markers = [
        "reference",
        "source",
        "citation",
        "referência",
        "referencia",
        "fonte",
        "citação",
        "citacao",
    ]
    phrase_markers = [
        "phrase",
        "snippet",
        "excerpt",
        "trecho",
        "frase",
    ]
    rewrite_exclusions = [
        "rephrase",
        "paraphrase",
        "paráfrase",
        "parafrasear",
        "parafraseie",
        "reescreva",
        "reescrever",
        "rewrite",
    ]

    has_source_marker = any(_matches_intent_keyword(text, marker) for marker in source_markers)
    has_phrase_marker = has_quoted_snippet or any(
        _matches_intent_keyword(text, marker) for marker in phrase_markers
    )
    has_rewrite_exclusion = any(
        _matches_intent_keyword(text, marker) for marker in rewrite_exclusions
    )

    debug = {
        "has_citation_number": has_citation_number,
        "has_source_marker": has_source_marker,
        "has_phrase_marker": has_phrase_marker,
        "has_quoted_snippet": has_quoted_snippet,
        "has_rewrite_exclusion": has_rewrite_exclusion,
    }
    return (
        has_citation_number
        and has_source_marker
        and has_phrase_marker
        and not has_rewrite_exclusion,
        debug,
    )


def _is_phrase_reference_query(user_text: str) -> bool:
    """Detect requests asking for the source/reference of a specific phrase/snippet."""
    return _classify_phrase_reference_intent(user_text)[0]


def _build_phrase_reference_query_seed(user_text: str) -> str:
    """Build the best-effort query seed from quoted text or from the full user message."""
    snippet = _extract_quoted_snippet(user_text)
    if snippet:
        return snippet

    marker_match = re.search(
        r"(?:frase|phrase|trecho)\s*[:?]\s*(.+)$",
        user_text,
        flags=re.IGNORECASE,
    )
    if marker_match:
        candidate = marker_match.group(1).strip()
        if candidate:
            return candidate

    return (user_text or "").strip()


def _search_reference_in_mongo_by_phrase(
    user_text: str, missing_numbers: list[int]
) -> tuple[str, dict]:
    """Search MongoDB vectors for likely source metadata for an unresolved phrase citation."""
    language = _detect_user_language(user_text)
    query_seed = _build_phrase_reference_query_seed(user_text)
    if not query_seed:
        return (
            _localized_text(
                language,
                "Não consegui extrair um trecho para busca no MongoDB.",
                "I couldn't extract a phrase to search in MongoDB.",
            ),
            {"found": False, "mongo_queries": 0, "mongo_hits": 0},
        )

    records = search_chunk_records(query_seed[:500], k=5)
    if not records:
        return (
            _localized_text(
                language,
                "Não encontrei candidato no MongoDB para essa frase.",
                "I couldn't find a MongoDB candidate for that phrase.",
            ),
            {"found": False, "mongo_queries": 1, "mongo_hits": 0},
        )

    best = records[0]
    title = str(best.get("source_title") or "").strip()
    doi = str(best.get("doi") or "").strip()
    url = str(best.get("source_url") or "").strip()
    file_path = str(best.get("file_path") or "").strip()

    lines = [
        _localized_text(
            language,
            f"### Referência candidata para {', '.join(f'[{n}]' for n in missing_numbers)} (MongoDB)",
            f"### Candidate reference for {', '.join(f'[{n}]' for n in missing_numbers)} (MongoDB)",
        ),
        "",
        f"- {_localized_text(language, 'Título', 'Title')}: {title or _localized_text(language, '(não identificado)', '(not identified)')}",
    ]
    if doi:
        lines.append(f"- DOI: {doi}")
    if url:
        lines.append(f"- URL: {url}")
    if file_path:
        lines.append(f"- {_localized_text(language, 'Arquivo', 'File')}: {file_path}")

    return "\n".join(lines), {"found": True, "mongo_queries": 1, "mongo_hits": 1}


def _search_reference_on_web_by_phrase(
    user_text: str, missing_numbers: list[int]
) -> tuple[str, dict]:
    """Search the internet (Tavily) for likely source metadata for an unresolved phrase citation."""
    language = _detect_user_language(user_text)
    query_seed = _build_phrase_reference_query_seed(user_text)
    if not query_seed:
        return (
            _localized_text(
                language,
                "Não consegui extrair um trecho para busca na internet.",
                "I couldn't extract a phrase to search on the internet.",
            ),
            {"found": False, "web_queries": 0, "web_hits": 0},
        )

    web = search_tavily_incremental(query=query_seed[:400], previous_urls=[], max_results=3)
    urls = web.get("new_urls", [])[:3]
    if not urls:
        return (
            _localized_text(
                language,
                "Não encontrei resultados web para essa frase.",
                "I couldn't find web results for that phrase.",
            ),
            {"found": False, "web_queries": 1, "web_hits": 0},
        )

    extracted = extract_tavily.invoke({"urls": urls, "include_images": False})
    items = extracted.get("extracted", []) if isinstance(extracted, dict) else []
    if not items:
        return (
            _localized_text(
                language,
                "Encontrei URLs, mas não consegui extrair metadados suficientes.",
                "I found URLs, but I couldn't extract enough metadata.",
            ),
            {"found": False, "web_queries": 1, "web_hits": 0},
        )

    first = items[0]
    title = str(first.get("title") or "").strip()
    url = str(first.get("url") or "").strip()

    lines = [
        _localized_text(
            language,
            f"### Referência candidata para {', '.join(f'[{n}]' for n in missing_numbers)} (Internet)",
            f"### Candidate reference for {', '.join(f'[{n}]' for n in missing_numbers)} (Internet)",
        ),
        "",
        f"- {_localized_text(language, 'Título', 'Title')}: {title or _localized_text(language, '(não identificado)', '(not identified)')}",
    ]
    if url:
        lines.append(f"- URL: {url}")

    return "\n".join(lines), {"found": True, "web_queries": 1, "web_hits": 1}


def _extract_requested_citation_numbers(user_text: str) -> list[int]:
    numbers = [int(match) for match in re.findall(r"\[(\d+)\]", user_text)]
    if numbers:
        return sorted(dict.fromkeys(numbers))

    # fallback patterns: "fonte 10", "reference #3", etc.
    fallback = [
        int(match)
        for match in re.findall(
            r"(?:source|citation|reference|refer(?:e|ê)ncia|fonte)\s*#?\s*(\d+)",
            user_text.lower(),
        )
    ]
    return sorted(dict.fromkeys(fallback))


def _is_reference_request(user_text: str) -> bool:
    intent = _classify_reference_intent(user_text)
    return intent in {"list_all", "format_provided", "resolve_numbers"}


def _contains_keyword(text: str, keyword: str) -> bool:
    keyword = (keyword or "").strip().lower()
    if not keyword:
        return False
    if re.search(r"\s", keyword):
        return keyword in text
    return bool(re.search(rf"\b{re.escape(keyword)}\b", text))


def _classify_reference_intent(user_text: str) -> str | None:
    text = user_text.lower()
    numbers = _extract_requested_citation_numbers(user_text)

    # 1) First, prioritize explicit "format provided list" requests.
    format_keywords = [
        "abnt",
        "format",
        "formate",
        "formatar",
        "norma",
        "padrão",
        "padrao",
    ]
    has_format_keyword = any(_contains_keyword(text, keyword) for keyword in format_keywords)
    if has_format_keyword:
        provided_items = _extract_provided_reference_items(user_text)
        if provided_items:
            return "format_provided"

    # 2) Then detect explicit "list references in document" requests.
    explicit_list_all_phrases = [
        "todas as referências",
        "todas as referencias",
        "all references",
        "all sources",
        "sem repetição",
        "sem repeticao",
        "without duplicates",
        "deduplicate",
        "used in this document",
        "used in document",
        "usadas neste documento",
        "referências usadas no documento",
        "referencias usadas no documento",
    ]
    list_all_action_words = [
        "liste",
        "listar",
        "list",
        "show",
        "mostre",
        "retorne",
        "return",
    ]
    has_explicit_phrase = any(
        _contains_keyword(text, phrase) for phrase in explicit_list_all_phrases
    )
    has_list_action = any(_contains_keyword(text, keyword) for keyword in list_all_action_words)
    has_reference_word = any(
        _contains_keyword(text, keyword)
        for keyword in [
            "referência",
            "referencias",
            "referências",
            "references",
            "fontes",
            "sources",
        ]
    )
    if has_explicit_phrase or (has_list_action and has_reference_word and "document" in text):
        return "list_all"

    # 3) Numbered citation requests.
    if numbers:
        return "resolve_numbers"

    return None


def _extract_provided_reference_items(user_text: str) -> list[str]:
    lines = [line.strip() for line in (user_text or "").splitlines() if line.strip()]
    items: list[str] = []

    for line in lines:
        stripped = re.sub(r"^(?:[-*]|\d+[\).]|\[\d+\])\s*", "", line).strip()
        if not stripped:
            continue
        if re.search(
            r"\b(formate|formatar|abnt|liste|listar|all references|todas as refer)\b",
            stripped,
            flags=re.IGNORECASE,
        ):
            continue
        if ";" in stripped and len(stripped) > 30 and "http" not in stripped.lower():
            parts = [p.strip() for p in stripped.split(";") if p.strip()]
            items.extend(parts)
            continue
        items.append(stripped)

    if len(items) >= 2:
        return items

    body_after_colon = user_text.split(":", 1)[1].strip() if ":" in user_text else ""
    if body_after_colon:
        chunks = [p.strip() for p in re.split(r"\n+|;", body_after_colon) if p.strip()]
        filtered = [p for p in chunks if len(p) > 6]
        if len(filtered) >= 1:
            return filtered
    return []


def _reference_request_fingerprint(user_text: str) -> str:
    normalized = re.sub(r"\s+", " ", (user_text or "").strip().lower())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


def _is_affirmative_confirmation(user_text: str) -> bool:
    text = (user_text or "").strip().lower()
    patterns = [
        r"^(sim|s|yes|y|ok|okay|confirmo|confirmar|pode|prosseguir|continue|go ahead)\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def _is_negative_confirmation(user_text: str) -> bool:
    text = (user_text or "").strip().lower()
    patterns = [
        r"^(nao|não|n|no|cancelar|cancela|pare|stop|cancel)\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def _precheck_provided_requires_web(user_text: str) -> dict:
    provided = _extract_provided_reference_items(user_text)
    incomplete_items: list[int] = []
    for idx, item in enumerate(provided, start=1):
        metadata = _metadata_from_raw_reference(idx, item)
        if not _is_metadata_complete(metadata):
            incomplete_items.append(idx)

    return {
        "provided_count": len(provided),
        "incomplete_items": incomplete_items,
        "requires_web": bool(incomplete_items),
    }


def _build_reference_confirmation_prompt(
    intent: str, user_text: str, allow_web: bool
) -> tuple[str, dict]:
    """Builds a confirmation prompt for reference formatting requests, analyzing the user's input to determine
    the intent and whether additional web search is needed for incomplete metadata. The function first classifies
    the intent of the user's request (e.g., listing all references, formatting provided items, resolving citation
    numbers) and then checks for any explicitly provided reference items. It assesses the completeness of the metadata
    for these items and determines if a web search would be necessary to fill in missing information. Based on this
    analysis, it constructs a localized confirmation prompt that informs the user of the next steps and what will be
    formatted, while also providing details about any incomplete items that may require web search if allowed.

    Args:
    - intent: A string representing the classified intent of the user's reference request (e.g., "list_all", "format_provided", "resolve_numbers").

    """
    language = _detect_user_language(user_text)
    fingerprint = _reference_request_fingerprint(user_text)

    if intent == "list_all":
        prompt = _localized_text(
            language,
            "Vou listar as referências usadas no documento em ABNT.\n\n"
            "Responda **sim** para confirmar ou **não** para cancelar.",
            "I will list the references used in the document in ABNT.\n\n"
            "Reply **yes** to confirm or **no** to cancel.",
        )
        return prompt, {
            "intent": intent,
            "original_message": user_text,
            "fingerprint": fingerprint,
            "requires_web": False,
            "incomplete_items": [],
            "provided_count": 0,
        }

    precheck = _precheck_provided_requires_web(user_text)
    incomplete_items = precheck["incomplete_items"]
    provided_count = int(precheck["provided_count"])
    requires_web = bool(precheck["requires_web"])

    if requires_web and not allow_web:
        prompt = _localized_text(
            language,
            "Detectei itens com metadados incompletos para ABNT "
            f"({', '.join(f'[{idx}]' for idx in incomplete_items)}).\n"
            "Para evitar resultado parcial incorreto, habilite **Allow web search** antes de confirmar.\n\n"
            "Depois responda **sim** para executar ou **não** para cancelar.",
            "I detected items with incomplete ABNT metadata "
            f"({', '.join(f'[{idx}]' for idx in incomplete_items)}).\n"
            "To avoid incorrect partial output, enable **Allow web search** before confirming.\n\n"
            "Then reply **yes** to execute or **no** to cancel.",
        )
    else:
        prompt = _localized_text(
            language,
            "Vou formatar somente os itens enviados por você em ABNT "
            f"({provided_count} item(ns)).\n\n"
            "Responda **sim** para confirmar ou **não** para cancelar.",
            "I will format only the items you provided in ABNT "
            f"({provided_count} item(s)).\n\n"
            "Reply **yes** to confirm or **no** to cancel.",
        )

    return prompt, {
        "intent": intent,
        "original_message": user_text,
        "fingerprint": fingerprint,
        "requires_web": requires_web,
        "incomplete_items": incomplete_items,
        "provided_count": provided_count,
    }


_BIBTEX_FIELD_RE = re.compile(r'(\w+)\s*=\s*["{]([^"}]+)["}]', re.IGNORECASE)


def _parse_bibtex_fields(bibtex: str) -> dict[str, str]:
    """Parses a BibTeX entry string and extracts its fields into a dictionary. The function
    uses a regular expression to identify key-value pairs in the BibTeX format, where keys are
    typically alphanumeric identifiers (e.g., title, author, year) and values are enclosed in
    either double quotes or curly braces. The extracted keys are normalized to lowercase, and
    the values are stripped of leading/trailing whitespace. This allows for flexible parsing of
    BibTeX entries, even when they contain varying amounts of whitespace or different field orderings.

    Args:
        - bibtex: A string containing the raw BibTeX entry, which may include various fields such
            as title, author, year, doi, and url.

    Returns:
        A dictionary where the keys are the normalized field names (in lowercase) and the values are
            the corresponding field values extracted from the BibTeX entry. If the input string is
            empty or does not contain valid BibTeX fields, an empty dictionary is returned.
    """
    if not bibtex:
        return {}
    return {m.group(1).lower(): m.group(2).strip() for m in _BIBTEX_FIELD_RE.finditer(bibtex)}


def _metadata_from_bibtex(number: int | None, bibtex: str) -> dict:
    """Extracts metadata from a BibTeX entry string, attempting to identify the title, year, DOI,
    and URL. This function uses regular expressions to parse common BibTeX field patterns and
    normalizes the extracted values by stripping extraneous whitespace and punctuation. The
    resulting metadata dictionary is structured to facilitate comparison with reference entries
    in a document, allowing for more accurate matching even when the input BibTeX is incomplete or
    formatted inconsistently.

    Args:
        number: An optional integer representing the reference number (e.g., from [1], [2], etc.).
        bibtex: A string containing the raw BibTeX entry, which may include various fields such as title, year, doi, and url.

    Returns:
        A dictionary with the following keys
        - "number": The provided reference number.
        - "raw": The original raw BibTeX string, stripped of leading/trailing whitespace.
        - "title": The extracted title from the BibTeX entry, or an empty string if not found.
        - "year": The extracted publication year from the BibTeX entry, or an empty string if not found.
        - "url": The extracted URL from the BibTeX entry, stripped of extraneous punctuation, or an empty string if not found.
        - "doi": The extracted DOI from the BibTeX entry, normalized to a standard format and stripped of extraneous punctuation, or an empty string if not found.
        - "file_path": An empty string (reserved for potential future use if a file path can be derived from the BibTeX entry).
        - "derived_from_path": A boolean set to False (reserved for potential future use to indicate whether the title was derived from a file path).
    """
    fields = _parse_bibtex_fields(bibtex)
    title = fields.get("title", "")
    year = fields.get("year", "")
    url = (fields.get("url", "") or "").strip().rstrip(".,;")
    doi = fields.get("doi", "")
    doi_match = re.search(r"(10\.\d{4,9}/[^\s,;]+)", doi, flags=re.IGNORECASE)
    doi_clean = doi_match.group(1).rstrip(".)],;") if doi_match else ""
    return {
        "number": number,
        "raw": bibtex.strip(),
        "title": title,
        "year": year,
        "url": url,
        "doi": doi_clean,
        "file_path": "",
        "derived_from_path": False,
    }


def _normalize_reference_key(raw: str) -> str:
    """Normalizes a raw reference string by removing leading numbering, condensing whitespace, stripping common metadata patterns (like DOIs and URLs), and removing punctuation. This function is designed to produce a clean, lowercase string that can be used for comparison or matching purposes, while ignoring common formatting variations and extraneous information that often accompanies reference entries.

    Args:
        - raw: A string containing the raw reference text, which may include numbering (e.g., "[1]"), DOIs, URLs, and various punctuation.

    Returns:
        A normalized string with numbering, DOIs, URLs, and punctuation removed, and whitespace condensed.
    """
    text = re.sub(r"^\[\d+\]\s*", "", raw or "")
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"doi:\s*10\.[^\s,;]+", "", text)
    text = re.sub(r"https?://\S+", "", text)
    return re.sub(r"[^\w\s]", "", text).strip()


def _title_from_file_path(path: str) -> str:
    """Given a file path, this function attempts to derive a clean title by extracting the base name, removing common delimiters and file extensions, and normalizing whitespace. This is particularly useful for cases where the reference metadata is incomplete but a file path (e.g., to a PDF) is available, allowing for a best-effort guess at the reference title.

    Args:
        - path: A string representing the file path.

    Returns:
        A string containing the derived title.
    """
    base = os.path.basename(path or "")
    base = re.sub(r"\.pdf$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"[_+\-]", " ", base)
    base = re.sub(r"\s+", " ", base).strip()
    return base


def _metadata_from_raw_reference(number: int | None, raw_reference: str) -> dict:
    """Extracts metadata from a raw reference string, attempting to identify the title, year, DOI,
      URL, and file path.

    Args:
        - number: An optional integer representing the reference number (e.g., from [1], [2], etc.).
        - raw_reference: A string containing the raw reference text, which may include various formats and metadata.

    Returns:
        A dictionary with the following keys:
        - "number": The provided reference number.
        - "raw": The original raw reference string, stripped of leading/trailing whitespace.
        - "title": A best-effort guess at the reference title, derived from the raw text and cleaned of common metadata artifacts.
        - "doi": The extracted DOI if present, normalized to a standard format and stripped of extraneous punctuation.
        - "url": The extracted URL if present, stripped of extraneous punctuation.
        - "year": The extracted publication year if present.
        - "file_path": The extracted file path if a PDF link is detected.
        - "derived_from_path": A boolean indicating whether the title was derived from a file path (which may indicate lower confidence in the title extraction).
    """
    raw = (raw_reference or "").strip()
    body = re.sub(r"^\[\d+\]\s*", "", raw).strip()

    doi_match = re.search(r"(10\.\d{4,9}/[^\s,;]+)", body, flags=re.IGNORECASE)
    url_match = re.search(r"(https?://\S+)", body, flags=re.IGNORECASE)
    path_match = re.search(r"(/[^\n]*?\.pdf)", body, flags=re.IGNORECASE)
    year_match = re.search(r"\b(19|20)\d{2}\b", body)

    file_path = path_match.group(1).strip() if path_match else ""
    title_guess = ""

    if file_path:
        title_guess = _title_from_file_path(file_path)
    else:
        text_no_url = re.sub(r"https?://\S+", "", body)
        text_no_doi = re.sub(r"10\.\d{4,9}/[^\s,;]+", "", text_no_url, flags=re.IGNORECASE)
        text_no_path = re.sub(r"/[^\n]*?\.pdf", "", text_no_doi, flags=re.IGNORECASE)
        text_no_labels = re.sub(
            r"\b(?:dispon[ií]vel em|arquivo local|citado em)\b:?.*",
            "",
            text_no_path,
            flags=re.IGNORECASE,
        )
        title_guess = re.sub(r"\s+", " ", text_no_labels).strip(" .;,")

    # Clean title candidate from trailing metadata artifacts
    title_guess = re.sub(
        r"\bDOI\b\s*:?\s*10\.\d{4,9}/[^\s,;]+", "", title_guess, flags=re.IGNORECASE
    )
    title_guess = re.sub(r"https?://\S+", "", title_guess)
    title_guess = re.sub(r"\s+", " ", title_guess).strip(" .;,")

    return {
        "number": number,
        "raw": raw,
        "title": title_guess,
        "doi": doi_match.group(1).rstrip(".)],;") if doi_match else "",
        "url": (url_match.group(1).rstrip(".)],;")) if url_match else "",
        "year": year_match.group(0) if year_match else "",
        "file_path": file_path,
        "derived_from_path": bool(file_path),
    }


def _is_metadata_complete(metadata: dict) -> bool:
    """Determines if the provided metadata dictionary contains sufficient information to be considered complete for ABNT formatting purposes.

    Args:
        metadata: A dictionary containing reference metadata, which may include keys such as 'title', 'year',
             'doi', 'url', and 'derived_from_path'. The function evaluates the presence and validity of these
              fields to determine if the metadata is complete enough for formatting.

    Returns:
        A boolean value indicating whether the metadata is considered complete. The metadata is deemed
        complete if it contains a valid DOI, or if it has both a title and year (or title and URL) without
        relying solely on a derived title from a file path. Incomplete metadata may lack critical information
        needed for proper ABNT formatting.
    """
    title = (metadata.get("title") or "").strip()
    year = (metadata.get("year") or "").strip()
    doi = (metadata.get("doi") or "").strip()
    url = (metadata.get("url") or "").strip()
    derived_from_path = bool(metadata.get("derived_from_path"))

    if doi:
        return True
    is_valid = bool(title and not derived_from_path and (year or url))
    return is_valid


def _format_abnt_entry(metadata: dict) -> str:
    """Formats a reference entry in ABNT style based on the provided metadata dictionary.

    Args:
        metadata: A dictionary containing reference metadata, which may include keys such as 'number',
            'title', 'year', 'doi', 'url', 'file_path', and 'raw'. The function will attempt to construct
            a properly formatted ABNT reference entry using this information, applying normalization and
            cleaning steps to handle common issues such as malformed DOIs, extraneous punctuation, and
            missing fields.

    Returns:
        A string representing the formatted reference entry in ABNT style. The function will prioritize
        using the title, year, DOI, and URL from the metadata, while also handling cases where certain
        fields may be missing or incomplete. The output will be structured according to ABNT guidelines,
        with appropriate punctuation and formatting based on the available metadata.
    """
    number = metadata.get("number")
    title = (metadata.get("title") or "").strip()
    year = (metadata.get("year") or "").strip()
    doi = (metadata.get("doi") or "").strip()
    url = (metadata.get("url") or "").strip()
    file_path = (metadata.get("file_path") or "").strip()
    raw = (metadata.get("raw") or "").strip()

    # Normalize malformed DOI fragments and duplicated punctuation
    doi_match = re.search(r"(10\.\d{4,9}/[^\s,;]+)", doi, flags=re.IGNORECASE)
    doi = doi_match.group(1).rstrip(".)],;") if doi_match else ""
    url = url.rstrip(".)],;")
    year = (
        re.search(r"\b(19|20)\d{2}\b", year).group(0)
        if re.search(r"\b(19|20)\d{2}\b", year)
        else ""
    )
    title = re.sub(r"\bDOI\b\s*:?.*$", "", title, flags=re.IGNORECASE).strip(" .;,")
    title = re.sub(r"\s+", " ", title).strip()

    if not year:
        year_in_title = re.search(r"\b(19|20)\d{2}\b", title)
        if year_in_title:
            year = year_in_title.group(0)
            title = re.sub(rf"\b{re.escape(year)}\b", "", title).strip(" .;,")
            title = re.sub(r"\s+", " ", title).strip()
    if raw and not title:
        m_author_year = re.match(r"^([^,]{2,80}),\s*((?:19|20)\d{2})$", raw)
        if m_author_year:
            author_stub = m_author_year.group(1).strip().upper()
            year = year or m_author_year.group(2)
            title = "TÍTULO NÃO IDENTIFICADO"
            raw = f"{author_stub}."

    prefix = f"[{number}] " if isinstance(number, int) else ""
    core = title or "TÍTULO NÃO IDENTIFICADO"

    fragments: list[str] = []
    fragments.append(core.rstrip(".;,") + ".")

    if year and year not in core:
        fragments.append(f"{year}.")
    else:
        fragments.append("[s.d.].")

    if doi:
        fragments.append(f"DOI: {doi}.")

    if url:
        fragments.append(f"Disponível em: {url.rstrip('.,;')}.")
    elif file_path:
        fragments.append(f"Documento local: {file_path}.")

    output = " ".join(fragment.strip() for fragment in fragments if fragment.strip())
    output = re.sub(r"\bDOI:\s*DOI:\s*", "DOI: ", output, flags=re.IGNORECASE)
    output = re.sub(r"\bDOI:\s*\.(?=\s|$)", "", output, flags=re.IGNORECASE)
    output = re.sub(r"(\[s\.d\.\]\.\s*){2,}", "[s.d.]. ", output, flags=re.IGNORECASE)
    output = re.sub(r"\.{2,}", ".", output)
    output = re.sub(r"\s+", " ", output).strip()
    return f"{prefix}{output}" if not output.startswith(prefix) else output


def _merge_metadata(base: dict, extra: dict) -> dict:
    """Merges two metadata dictionaries, giving priority to non-empty values in the base dictionary.

    Args:
        base: The primary metadata dictionary, which may contain keys like 'title', 'doi', 'url', 'year', and 'file_path'. Values in this dictionary take precedence if they are non-empty.
        extra: The secondary metadata dictionary, which may provide additional information to fill in missing values in the base dictionary. This dictionary is consulted for keys that are missing or empty in the base.

    Returns:
        A new dictionary that combines the information from both base and extra, where values from the base are retained if present, and values from extra are used to fill in any gaps. The resulting dictionary will have the same keys as the base, with values taken from either the base or extra as appropriate.
    """
    merged = dict(base)
    for key in ("title", "doi", "url", "year", "file_path"):
        if not merged.get(key) and extra.get(key):
            merged[key] = extra[key]
    return merged


def _extract_non_numbered_mentions(markdown: str) -> list[str]:
    """Extracts reference mentions from the markdown content that do not follow the numbered citation format.
    This includes author-year citations in the body text and non-numbered lines within reference blocks.
    The function uses regex patterns to identify potential reference mentions and applies cleaning and deduplication to

    Args:
        markdown: The markdown content of the document, which may contain sections, paragraphs, and reference blocks.

    Returns:
         A list of unique reference mentions extracted from the markdown, which may include author-year citations and other informal references that do not follow the numbered format.
    """
    mentions: list[str] = []

    # Author-year patterns in the body, e.g., (Bleidorn et al., 2024)
    patterns = [
        r"\(([A-Z][A-Za-zÀ-ÿ'’\-]+(?:\s+et\s+al\.)?(?:\s*&\s*[A-Z][A-Za-zÀ-ÿ'’\-]+)?\s*,\s*(?:19|20)\d{2})\)",
        r"\(([A-Z][^()\n]{6,120},\s*(?:19|20)\d{2})\)",
    ]

    for pattern in patterns:
        for match in re.findall(pattern, markdown):
            text = re.sub(r"\s+", " ", match).strip(" .;,")
            if text:
                mentions.append(text)

    # Non-numbered lines inside reference blocks
    lines = markdown.splitlines()
    in_refs = False
    for line in lines:
        stripped = line.strip()
        if re.match(
            r"^###\s+(?:References\s+for\s+this\s+section|Refer[êe]ncias\s+desta\s+se[çc][ãa]o)\s*$",
            stripped,
            flags=re.IGNORECASE,
        ):
            in_refs = True
            continue
        if in_refs and re.match(r"^##\s+", stripped):
            in_refs = False
        if not in_refs:
            continue
        if not stripped or stripped.startswith("<!--"):
            continue
        if re.match(r"^\[\d+\]", stripped):
            continue
        cleaned = re.sub(r"^[-*]\s+", "", stripped)
        if (
            cleaned
            and len(cleaned) <= 180
            and "http" not in cleaned.lower()
            and "doi" not in cleaned.lower()
        ):
            mentions.append(cleaned)

    # de-duplicate while preserving order
    dedup: list[str] = []
    seen = set()
    for mention in mentions:
        key = _normalize_reference_key(mention)
        if not key or key in seen:
            continue
        seen.add(key)
        dedup.append(mention)
    return dedup


def _collect_reference_inventory(markdown: str) -> dict:
    """Parses the markdown content to extract a structured inventory of references,
    including their numbering, associated paragraphs, and unique entries.
    This function identifies reference blocks, extracts citation numbers and their corresponding text,
    and organizes them for further processing such as enrichment or formatting.

    Args:
        markdown: The markdown content of the document, which may contain sections, paragraphs, and reference blocks.

    Returns:
        A dictionary containing:
        - 'references_by_number': A mapping of citation numbers to their raw reference text extracted from the markdown.
        - 'citation_paragraphs': A mapping of citation numbers to lists of paragraphs that cite them, used for contextual enrichment.
        - 'unique_references': A list of unique reference entries formatted as "[N] Reference text", deduplicated based on normalized keys.
        - 'cited_numbers': A sorted list of citation numbers that are actually cited in the paragraphs, used for identifying which references are in use.
        - 'non_numbered_mentions': A list of reference mentions that do not follow the numbered format, extracted from both the body text and reference blocks, which may include author-year citations or other informal references.
    """
    sections = _split_sections(markdown)
    references_by_number: dict[int, str] = {}
    citation_paragraphs: dict[int, list[str]] = {}

    for section in sections:
        for ref in section.get("references", []):
            match = re.match(r"^\[(\d+)\]\s*(.+)$", ref.strip())
            if not match:
                continue
            number = int(match.group(1))
            text = f"[{number}] {match.group(2).strip()}"
            references_by_number[number] = text

        for paragraph in section.get("paragraphs", []):
            p_text = paragraph.get("text", "")
            for number_token in re.findall(r"\[(\d+)\]", p_text):
                number = int(number_token)
                citation_paragraphs.setdefault(number, []).append(p_text)

    unique_refs: list[str] = []
    seen_keys: set[str] = set()
    for number in sorted(references_by_number.keys()):
        ref = references_by_number[number]
        key = _normalize_reference_key(ref)
        if key and key in seen_keys:
            continue
        if key:
            seen_keys.add(key)
        unique_refs.append(ref)

    cited_numbers = sorted(citation_paragraphs.keys())
    non_numbered_mentions = _extract_non_numbered_mentions(markdown)
    return {
        "references_by_number": references_by_number,
        "citation_paragraphs": citation_paragraphs,
        "unique_references": unique_refs,
        "cited_numbers": cited_numbers,
        "non_numbered_mentions": non_numbered_mentions,
    }


def _enrich_reference_from_mongo(number: int, paragraphs: list[str]) -> tuple[dict, dict]:
    """Attempts to enrich the reference metadata by performing a search in the MongoDB vector store using the paragraphs that cite the reference number.

    Args:
        number: The citation number associated with the reference, used for metadata structuring.
        paragraphs: A list of text paragraphs that cite the reference number, used for performing the search in the MongoDB vector store.

    Returns:
        Tuple containing the enriched metadata dictionary with fields such as 'title', 'doi', 'url', and 'file_path' if found, and a dictionary with counts of MongoDB queries and hits for tracking the enrichment process.
    """
    if not paragraphs:
        return {}, {"mongo_queries": 0, "mongo_hits": 0}

    mongo_queries = 0
    best: dict | None = None
    for paragraph in paragraphs[:4]:
        query = paragraph[:600]
        mongo_queries += 1
        records = search_chunk_records(query, k=6)
        if not records:
            continue
        candidate = records[0]
        if best is None or float(candidate.get("score", 0.0) or 0.0) > float(
            best.get("score", 0.0) or 0.0
        ):
            best = candidate

    if not best:
        return {}, {"mongo_queries": mongo_queries, "mongo_hits": 0}

    title = best.get("source_title", "") or "(untitled source)"
    doi = best.get("doi", "")
    url = best.get("source_url", "")
    file_path = best.get("file_path", "")

    metadata = {
        "number": number,
        "title": title,
        "doi": doi,
        "url": url,
        "file_path": file_path,
    }
    return metadata, {"mongo_queries": mongo_queries, "mongo_hits": 1}


def _enrich_reference_from_web(number: int, query: str) -> tuple[dict, dict]:
    """Attempts to enrich the reference metadata by performing a web search using Tavily.

    Args:
        number: The citation number associated with the reference, used for metadata structuring.
        query: The text query derived from the reference item, which may include title, raw text, or other metadata, used for performing the web search.

    Returns:
        Tuple containing the enriched metadata dictionary with fields such as 'title', 'doi', 'url', and 'year' if found, and a dictionary with counts of web queries and hits for tracking the enrichment process.
    """
    if not query.strip():
        return {}, {"web_queries": 0, "web_hits": 0}

    web = search_tavily_incremental(query=query[:400], previous_urls=[], max_results=5)
    urls = web.get("new_urls", [])[:2]
    if not urls:
        return {}, {"web_queries": 1, "web_hits": 0}

    extracted = extract_tavily.invoke({"urls": urls, "include_images": False})
    items = extracted.get("extracted", []) if isinstance(extracted, dict) else []
    if not items:
        return {}, {"web_queries": 1, "web_hits": 0}

    first = items[0]
    title = first.get("title", "") or "(untitled source)"
    url = first.get("url", "")
    content = str(first.get("content", ""))
    doi_match = re.search(r"(10\.\d{4,9}/[^\s)]+)", content)
    doi = doi_match.group(1) if doi_match else ""

    year_match = re.search(r"\b(19|20)\d{2}\b", content)
    metadata = {
        "number": number,
        "title": title,
        "doi": doi,
        "url": url,
        "year": year_match.group(0) if year_match else "",
    }
    return metadata, {"web_queries": 1, "web_hits": 1}


def _collect_all_raw_references_text(markdown: str) -> list[str]:
    """Extract every reference line from ALL reference/bibliography sections in the markdown.

    Unlike ``_split_sections`` (which only recognises ``### Referências desta seção``),
    this function scans for ANY heading that looks like a references or bibliography
    heading at any ``#`` depth — including numbered sections like ``## 4. Referências``
    or ``## Referências Bibliográficas`` — and collects every non-blank line below it
    until the next heading of the same or higher level.

    Args:
        - markdown: The full markdown content of the document, which may contain multiple reference or bibliography sections with varying heading levels and formats.

    Returns:
        List of collected reference lines (may be empty).
    """
    # Optional numeric prefix (e.g. "4. " or "4 ") before keyword
    ref_heading_re = re.compile(
        r"^(#+)\s+(?:[\d]+[\s\.]+)?(refer[eê]ncias|references|bibliography|bibliograf\w+|bibliog\w+)\b",
        re.IGNORECASE,
    )
    any_heading_re = re.compile(r"^(#+)\s+")

    lines = markdown.splitlines()
    collected: list[str] = []
    collecting = False
    current_depth = 0

    for line in lines:
        stripped = line.strip()
        ref_match = ref_heading_re.match(stripped)
        if ref_match:
            collecting = True
            current_depth = len(ref_match.group(1))
            continue

        if collecting:
            any_match = any_heading_re.match(stripped)
            if any_match and len(any_match.group(1)) <= current_depth:
                # A heading at the same or higher level ends this section
                collecting = False
            elif stripped:
                collected.append(stripped)

    return collected


def _collect_all_citation_paragraphs(markdown: str) -> dict[int, list[str]]:
    """Scan the full markdown body for paragraphs that cite numbered references.

    Returns a mapping ``{ref_number: [paragraph1, paragraph2]}`` (max 2 paragraphs
    per reference number) suitable for passing as ``citation_context`` to the
    extractor agent.

    Args:
        - markdown: The full markdown content of the document, which may contain paragraphs with in-text citations in the format [N].

    Returns:
        A dictionary mapping each cited reference number to a list of up to two paragraphs that contain citations
        of that reference number. The paragraphs are extracted from the markdown content and are intended to provide
        context for the extractor agent when enriching reference metadata.
    """
    result: dict[int, list[str]] = {}
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("<!--"):
            continue
        nums = {int(n) for n in re.findall(r"\[(\d+)\]", stripped)}
        for num in nums:
            paragraphs = result.setdefault(num, [])
            if len(paragraphs) < 2:
                paragraphs.append(stripped)
    return result


def _handle_resolve_numbers_request(
    markdown: str, user_text: str, allow_web: bool = True
) -> tuple[str, dict]:
    """Resolve specific numbered references via the extractor\u2192formatter agent pipeline.

    Extracts the citation numbers from the user message, fetches their entries
    from the reference inventory, and runs them through the same agent pipeline
    used by ``_handle_list_all_references_request``.

    Args:
        - markdown: The full markdown content of the document.
        - user_text: The original user message, used for language detection and localization.
        - allow_web: Whether to allow web search for metadata enrichment in the extractor agent.

    Returns:
        A tuple containing the reply string with the formatted reference list for the requested
        numbers and a metadata dictionary
    """
    language = _detect_user_language(user_text)
    requested = _extract_requested_citation_numbers(user_text)
    inventory = _collect_reference_inventory(markdown)
    references_by_number: dict[int, str] = inventory.get("references_by_number", {})

    entries = (
        {n: references_by_number[n] for n in requested if n in references_by_number}
        if requested
        else references_by_number
    )

    if not entries:
        msg = _localized_text(
            language,
            "Nenhuma refer\u00eancia encontrada para os n\u00fameros solicitados.",
            "No references found for the requested numbers.",
        )
        return msg, {"intent": "resolve_numbers", "count": 0, "agent": "none"}

    raw_block = "\n".join(entries.values())
    citation_context = _collect_all_citation_paragraphs(markdown)

    enriched = run_reference_extractor_agent(
        raw_block, citation_context=citation_context, allow_web=allow_web
    )
    abnt_list = run_reference_formatter_agent(enriched, allow_web=allow_web)

    heading = _localized_text(language, "### Refer\u00eancias (ABNT)", "### References (ABNT)")
    reply = f"{heading}\n\n{abnt_list}"
    return reply, {
        "intent": "resolve_numbers",
        "count": len(entries),
        "agent": "reference_extractor+reference_formatter",
    }


def _handle_list_all_references_request(
    markdown: str, user_text: str, allow_web: bool = True
) -> tuple[str, dict]:
    """Collect every reference from the document, then run extractor→formatter
    agent pipeline for ABNT output.

    Uses two complementary collectors:
    - ``_collect_reference_inventory``: handles ``### Referências desta seção`` blocks
    - ``_collect_all_raw_references_text``: catches any other reference/bibliography
      headings (e.g. ``## Referências``, ``## 4. Referências Bibliográficas``)

    No deduplication is applied — every entry is passed to the agents as-is.
    HTML comments (``<!-- ... -->``) are stripped before forwarding.
    Citation paragraphs from the full document body are collected and passed
    to the extractor agent as context for Type-B in-text citations.

    Args:
        - markdown: The full markdown content of the document.
        - user_text: The original user message, used for language detection and localization.
        - allow_web: Whether to allow web search for metadata enrichment in the extractor agent.

    Returns:
        A tuple containing the reply string with the formatted reference list and a metadata dictionary
        with details about the operation, such as intent, count of references processed, and agents used.
    """
    language = _detect_user_language(user_text)

    # ── Primary: inventory-based collector (handles standard section format) ──
    inventory = _collect_reference_inventory(markdown)
    primary_refs: list[str] = list(inventory.get("references_by_number", {}).values())

    # ── Supplementary: any reference heading _split_sections may not recognise ──
    extra_lines = _collect_all_raw_references_text(markdown)
    # Add all extra lines — no deduplication
    primary_refs.extend(extra_lines)

    # ── Filter HTML comments ──
    primary_refs = [r for r in primary_refs if not r.strip().startswith("<!--")]

    if not primary_refs:
        msg = _localized_text(
            language,
            "Nenhuma referência encontrada no documento. Verifique se o arquivo contém seções de referências.",
            "No references found in the document. Check that the file contains reference sections.",
        )
        return msg, {"intent": "list_all", "count": 0, "agent": "none"}

    # ── Build raw block: preserve existing [N] or assign sequential numbers ──
    numbered_lines: list[str] = []
    counter = 1
    for ref in primary_refs:
        if re.match(r"^\[\d+\]", ref):
            numbered_lines.append(ref)
        else:
            numbered_lines.append(f"[{counter}] {ref}")
        counter += 1
    raw_block = "\n".join(numbered_lines)

    # ── Collect citation context (full-doc scan + inventory) ──
    citation_context: dict[int, list[str]] = _collect_all_citation_paragraphs(markdown)
    for num, paras in inventory.get("citation_paragraphs", {}).items():
        existing = citation_context.setdefault(num, [])
        for para in paras:
            if para not in existing and len(existing) < 2:
                existing.append(para)

    # ── Extractor agent: enrich raw entries with full metadata ──
    enriched = run_reference_extractor_agent(
        raw_block, citation_context=citation_context, allow_web=allow_web
    )

    # ── Formatter agent: apply ABNT NBR 6023 formatting ──
    abnt_list = run_reference_formatter_agent(enriched, allow_web=allow_web)

    heading = _localized_text(
        language,
        "### Referências do documento (ABNT)",
        "### Document references (ABNT)",
    )
    reply = f"{heading}\n\n{abnt_list}"
    meta = {
        "intent": "list_all",
        "count": len(primary_refs),
        "agent": "reference_extractor+reference_formatter",
    }
    return reply, meta


def _enrich_metadata_doi_first(metadata: dict, allow_web: bool) -> tuple[dict, dict]:
    """Attempts to enrich the metadata of a reference item, prioritizing DOI extraction and lookup.
    The function first tries to find a DOI from the raw text, URL, or title.

    Args:
        metadata: A dictionary containing the initial metadata of the reference item, which may include 'raw', 'title', 'url', and 'doi'.
        allow_web: Whether to allow web search for metadata enrichment.

    Returns:
        A tuple containing the enriched metadata dictionary and a dictionary with query and hit counts.
    """
    mongo_queries = 0
    mongo_hits = 0
    web_queries = 0
    web_hits = 0

    query = (metadata.get("raw") or metadata.get("title") or "").strip()
    if query:
        mongo_queries += 1
        records = search_chunk_records(query[:500], k=4)
        if records:
            best = records[0]
            mongo_hits += 1
            metadata = _merge_metadata(
                metadata,
                {
                    "title": best.get("source_title", ""),
                    "doi": best.get("doi", ""),
                    "url": best.get("source_url", ""),
                    "file_path": best.get("file_path", ""),
                },
            )

    doi = (metadata.get("doi") or "").strip()
    if not doi:
        doi = extract_doi_from_url(metadata.get("url", "") or "") or ""
    if not doi:
        doi = search_doi_in_text(metadata.get("raw", "") or "") or ""

    if allow_web and not doi and (metadata.get("title") or ""):
        web_queries += 1
        doi = search_crossref_by_title((metadata.get("title") or "")[:200]) or ""
        if doi:
            web_hits += 1
            metadata["doi"] = doi

    bibtex_success = False
    if allow_web and doi:
        web_queries += 1
        bibtex = get_bibtex_from_doi(doi, timeout=10)
        if bibtex:
            web_hits += 1
            metadata = _merge_metadata(
                metadata, _metadata_from_bibtex(metadata.get("number"), bibtex)
            )
            metadata["doi"] = metadata.get("doi") or doi
            bibtex_success = True

    weak_metadata = not (
        metadata.get("title")
        and (metadata.get("year") or metadata.get("doi") or metadata.get("url"))
    )
    should_try_tavily = allow_web and (
        not bibtex_success or not _is_metadata_complete(metadata) or weak_metadata
    )

    if should_try_tavily:
        query_seed = (
            query or metadata.get("title") or metadata.get("url") or metadata.get("doi") or ""
        )
        web_ref, web_meta = _enrich_reference_from_web(metadata.get("number") or 0, str(query_seed))
        web_queries += int(web_meta.get("web_queries", 0))
        web_hits += int(web_meta.get("web_hits", 0))
        if web_ref:
            metadata = _merge_metadata(metadata, web_ref)

    return metadata, {
        "mongo_queries": mongo_queries,
        "mongo_hits": mongo_hits,
        "web_queries": web_queries,
        "web_hits": web_hits,
    }


def _handle_format_provided_references_request(user_text: str, allow_web: bool) -> tuple[str, dict]:
    """Runs extractor then formatter on the user-provided reference list.

    The extractor resolves paths, in-text citations, and partial entries into
    structured metadata. The formatter then applies ABNT NBR 6023 rules.
    """
    language = _detect_user_language(user_text)

    # ── Extractor: resolve any paths, in-text citations, or partial entries ──
    enriched = run_reference_extractor_agent(user_text, allow_web=allow_web)

    # ── Formatter: apply ABNT NBR 6023 ──
    abnt_list = run_reference_formatter_agent(enriched, allow_web=allow_web)

    heading = _localized_text(
        language,
        "### Fontes formatadas (ABNT)",
        "### Formatted sources (ABNT)",
    )
    reply = f"{heading}\n\n{abnt_list}"
    meta = {
        "intent": "format_provided",
        "agent": "reference_extractor+reference_formatter",
    }
    return reply, meta


def _handle_reference_request(markdown: str, user_text: str, allow_web: bool) -> tuple[str, dict]:
    """This is the main handler for resolving numbered references in the document based on user requests.
    It collects the reference inventory, identifies which numbers to target, and attempts to enrich their metadata
    using MongoDB and web search as needed, then formats the results in ABNT style.

    The response includes sections for unique references, non-numbered mentions, resolved numbered references, and pending items.

    Args:
        markdown: The full markdown content of the document.
        user_text: The user's request text that may contain instructions and specific reference numbers.
        allow_web: Whether to allow web search for metadata enrichment.

    Returns:
        A tuple of (response_markdown, metadata_dict) where response_markdown is the formatted
        markdown string to reply with, and metadata_dict contains details about the processing.
    """
    language = _detect_user_language(user_text)
    inventory = _collect_reference_inventory(markdown)
    references_by_number = inventory["references_by_number"]
    citation_paragraphs = inventory["citation_paragraphs"]
    unique_references = inventory["unique_references"]
    cited_numbers = inventory["cited_numbers"]
    non_numbered_mentions = inventory.get("non_numbered_mentions", [])

    requested_numbers = _extract_requested_citation_numbers(user_text)
    target_numbers = requested_numbers or cited_numbers

    mongo_queries = 0
    mongo_hits = 0
    web_queries = 0
    web_hits = 0

    resolved_numbered: list[str] = []
    unresolved: list[int] = []

    complete_count = 0
    for number in target_numbers:
        raw_ref = references_by_number.get(number, f"[{number}]")
        metadata = _metadata_from_raw_reference(number, raw_ref)

        mongo_ref, mongo_meta = _enrich_reference_from_mongo(
            number, citation_paragraphs.get(number, [])
        )
        mongo_queries += int(mongo_meta.get("mongo_queries", 0))
        mongo_hits += int(mongo_meta.get("mongo_hits", 0))
        if mongo_ref:
            metadata = _merge_metadata(metadata, mongo_ref)

        need_web = allow_web and (not _is_metadata_complete(metadata))
        if need_web:
            query_seed = " ".join(citation_paragraphs.get(number, [])[:1])
            if not query_seed:
                query_seed = metadata.get("title", "")
            web_ref, web_meta = _enrich_reference_from_web(number, query_seed)
            web_queries += int(web_meta.get("web_queries", 0))
            web_hits += int(web_meta.get("web_hits", 0))
            if web_ref:
                metadata = _merge_metadata(metadata, web_ref)

        formatted = _format_abnt_entry(metadata)
        if formatted.strip():
            resolved_numbered.append(formatted)
            if _is_metadata_complete(metadata):
                complete_count += 1
            else:
                unresolved.append(number)
        else:
            unresolved.append(number)

    # Deduplicate resolved numbered list by normalized text
    dedup_resolved: list[str] = []
    seen = set()
    for ref in resolved_numbered:
        key = _normalize_reference_key(ref)
        if key in seen:
            continue
        seen.add(key)
        dedup_resolved.append(ref)

    lines: list[str] = []
    lines.append(
        _localized_text(
            language,
            "### Referências únicas (deduplicadas) — padrão ABNT",
            "### Unique references (deduplicated) — ABNT style",
        )
    )
    lines.append("")
    if unique_references:
        unique_abnt: list[str] = []
        unique_seen: set[str] = set()
        for ref in unique_references:
            metadata = _metadata_from_raw_reference(None, ref)
            formatted = _format_abnt_entry(metadata)
            key = _normalize_reference_key(formatted)
            if key and key in unique_seen:
                continue
            if key:
                unique_seen.add(key)
            unique_abnt.append(formatted)
        lines.extend(f"- {ref}" for ref in unique_abnt)
    else:
        lines.append(
            _localized_text(
                language,
                "- Nenhuma referência explícita detectada no bloco de referências.",
                "- No explicit references were detected in references blocks.",
            )
        )

    lines += [
        "",
        _localized_text(
            language,
            "### Referências não numeradas detectadas",
            "### Detected non-numbered references",
        ),
        "",
    ]
    if non_numbered_mentions:
        non_numbered_abnt = [
            _format_abnt_entry(_metadata_from_raw_reference(None, mention))
            for mention in non_numbered_mentions
        ]
        lines.extend(f"- {ref}" for ref in non_numbered_abnt)
    else:
        lines.append(
            _localized_text(
                language,
                "- Nenhuma referência não numerada detectada no texto.",
                "- No non-numbered references detected in the text.",
            )
        )

    lines += [
        "",
        _localized_text(language, "### Referências numeradas [n]", "### Numbered references [n]"),
        "",
    ]
    if dedup_resolved:
        lines.extend(f"- {ref}" for ref in dedup_resolved)
    else:
        lines.append(
            _localized_text(
                language,
                "- Nenhuma referência numerada foi resolvida.",
                "- No numbered references were resolved.",
            )
        )

    if unresolved:
        lines += ["", _localized_text(language, "### Pendências", "### Pending")]
        lines.append(
            _localized_text(
                language,
                f"- Não foi possível resolver completamente: {', '.join(f'[{n}]' for n in unresolved)}",
                f"- Could not fully resolve: {', '.join(f'[{n}]' for n in unresolved)}",
            )
        )
        if not allow_web:
            lines.append(
                _localized_text(
                    language,
                    "- Para completar essas referências no padrão ABNT, ative **Allow web search** e repita o comando.",
                    "- To complete these references in ABNT format, enable **Allow web search** and run the command again.",
                )
            )

    lines += [
        "",
        _localized_text(language, "### Rastreabilidade da busca", "### Search traceability"),
        _localized_text(
            language,
            f"- MongoDB: {mongo_queries} consulta(s), {mongo_hits} item(ns) resolvido(s)",
            f"- MongoDB: {mongo_queries} query(ies), {mongo_hits} item(s) resolved",
        ),
        _localized_text(
            language,
            f"- Tavily: {web_queries} consulta(s), {web_hits} item(ns) resolvido(s)",
            f"- Tavily: {web_queries} query(ies), {web_hits} item(s) resolved",
        ),
        _localized_text(
            language,
            f"- Cobertura de [n] solicitados: {len(target_numbers) - len(unresolved)}/{len(target_numbers)}",
            f"- Requested [n] coverage: {len(target_numbers) - len(unresolved)}/{len(target_numbers)}",
        ),
        _localized_text(
            language,
            f"- Completude ABNT de [n]: {complete_count}/{len(target_numbers)}",
            f"- ABNT completeness for [n]: {complete_count}/{len(target_numbers)}",
        ),
    ]

    meta = {
        "requested_numbers": target_numbers,
        "unresolved_numbers": unresolved,
        "mongo_queries": mongo_queries,
        "mongo_hits": mongo_hits,
        "web_queries": web_queries,
        "web_hits": web_hits,
    }
    return "\n".join(lines), meta


def _list_paragraphs_using_citation(markdown: str, user_text: str) -> str:
    """Lists paragraphs in the document that use a specific citation number, based on user input that references the citation by its number (e.g., [2]). It also checks if the citation is mentioned in the reference sections and provides localized feedback.

    Args:
        markdown (str): The full markdown text of the document, which may contain sections, paragraphs and references.
        user_text (str): The user's input text, which should contain a reference to a citation
            number in the format [n], where n is the citation number to look for.

    Returns:
        str: A formatted string listing the paragraphs that use the specified citation, along with any detected
            references that match the citation number. If no paragraphs or references are found, it returns a localized message indicating that the citation could not be identified or used.
    """
    language = _detect_user_language(user_text)
    citation_number = _extract_citation_number(user_text)
    if citation_number is None:
        return _localized_text(
            language,
            "Não consegui identificar a citação pedida. Use algo como [2].",
            "I couldn't identify the requested citation. Use something like [2].",
        )

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
                _localized_text(
                    language,
                    f"- **{section['title']}**, parágrafo **{paragraph_index}**: {snippet}",
                    f"- **{section['title']}**, paragraph **{paragraph_index}**: {snippet}",
                )
            )

    if not matches:
        return _localized_text(
            language,
            f"Nenhum parágrafo na cópia de trabalho usa a citação **{token}**.",
            f"No paragraph in the working copy uses citation **{token}**.",
        )

    lines = [
        _localized_text(
            language,
            f"### Parágrafos que usam {token}",
            f"### Paragraphs using {token}",
        ),
        "",
        *matches,
    ]
    if reference_hits:
        lines += [
            "",
            _localized_text(language, "### Referência detectada", "### Detected reference"),
            "",
            *reference_hits[:8],
        ]
    return "\n".join(lines)


def _confirm_paragraph(markdown: str, user_text: str) -> tuple[str, dict]:
    """Provides evidence and context for a specific paragraph in the document, based on user input that may reference the paragraph by section/paragraph number or by quoting a snippet of its text. It retrieves relevant chunks from MongoDB and identifies potential sources/authors based on cited files/links.

    Args:
        markdown (str): The full markdown text of the document, which may contain sections, paragraphs
            and references.
        user_text (str): The user's input text, which may contain instructions to identify a specific
            paragraph either by section/paragraph number or by quoting a snippet of its text.

    Returns:
        Tuple [str, dict]: A tuple containing the message with evidence and context for the identified
        paragraph, and a metadata dictionary with details about the identified section, number of chunks
        retrieved, and number of references found.
    """
    language = _detect_user_language(user_text)
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
            p_idx = _resolve_paragraph_index(
                user_text, len(sections[sec_idx].get("paragraphs", []))
            )
            if p_idx is not None:
                target_sec = sections[sec_idx]
                target_para = target_sec["paragraphs"][p_idx]

    if target_para is None:
        return (
            _localized_text(
                language,
                "Não consegui resolver o parágrafo alvo. Informe seção + parágrafo ou envie o trecho entre aspas.",
                "I couldn't resolve the target paragraph. Provide section + paragraph or send the excerpt in quotes.",
            ),
            {},
        )

    chunks = search_chunks(target_para["text"][:600], k=6)
    refs = target_sec.get("references", []) if target_sec else []
    ref_labels = [re.sub(r"^\[(\d+)\]\s*", "", r) for r in refs[:5]]
    authors_hint = [os.path.basename(r).replace(".pdf", "") for r in ref_labels]
    evidence = (
        "\n\n".join(chunks[:3])
        if chunks
        else _localized_text(
            language,
            "Sem chunks relevantes retornados no momento.",
            "No relevant chunks were returned at the moment.",
        )
    )

    msg = (
        _localized_text(language, "### Verificação do parágrafo\n", "### Paragraph verification\n")
        + _localized_text(
            language,
            f"- Seção: **{target_sec['title'] if target_sec else 'N/A'}**\n",
            f"- Section: **{target_sec['title'] if target_sec else 'N/A'}**\n",
        )
        + _localized_text(
            language,
            f"- Evidências MongoDB: **{len(chunks)} chunks**\n",
            f"- MongoDB evidence: **{len(chunks)} chunks**\n",
        )
        + _localized_text(
            language,
            f"- Fontes/autores (aproximação pelos arquivos/links citados): {', '.join(authors_hint[:6]) if authors_hint else 'não identificado'}\n\n",
            f"- Sources/authors (approximated from cited files/links): {', '.join(authors_hint[:6]) if authors_hint else 'not identified'}\n\n",
        )
        + _localized_text(
            language,
            f"**Trecho alvo:**\n{target_para['text'][:700]}\n\n",
            f"**Target excerpt:**\n{target_para['text'][:700]}\n\n",
        )
        + _localized_text(
            language,
            f"**Evidência principal:**\n{evidence[:1800]}",
            f"**Primary evidence:**\n{evidence[:1800]}",
        )
    )
    return msg, {
        "section": target_sec["title"] if target_sec else "",
        "chunks": len(chunks),
        "references": len(refs),
    }


def _suggest_more_documents(user_text: str, allow_web: bool) -> tuple[str, dict]:
    """Suggests more documents related to the user's query, using both local MongoDB evidence and optional web search.

    Args:
        user_text (str): The user's input text, which may contain a query and/or a quoted snippet.
        allow_web (bool): A flag indicating whether web search is enabled for finding related documents.

    Returns:
        Tuple [str, dict]: A tuple containing the message with suggested documents and a metadata dictionary with search details.
    """
    language = _detect_user_language(user_text)
    snippet = _extract_quoted_snippet(user_text)
    query = snippet or user_text

    local_chunks = search_chunks(query[:600], k=5)
    local_msg = "\n".join(
        f"- Local evidence chunk {i + 1}: {chunk[:180]}..."
        for i, chunk in enumerate(local_chunks[:3])
    )

    if not allow_web:
        msg = (
            _localized_text(
                language,
                "### Documentos relacionados (modo local)\n",
                "### Related documents (local mode)\n",
            )
            + _localized_text(
                language,
                "Use 'search on internet' na pergunta para incluir documentos web.\n\n",
                "Use 'search on internet' in your request to include web documents.\n\n",
            )
            + f"{local_msg or _localized_text(language, '- Sem evidência local retornada.', '- No local evidence returned.')}"
        )
        return msg, {"source": "mongo", "chunks": len(local_chunks)}

    web = search_tavily_incremental(query=query[:400], previous_urls=[], max_results=5)
    urls = web.get("new_urls", [])[:3]
    extracted = (
        extract_tavily.invoke({"urls": urls, "include_images": False})
        if urls
        else {"extracted": []}
    )

    lines = [
        _localized_text(
            language,
            "### Documentos relacionados (local + web)",
            "### Related documents (local + web)",
        )
    ]
    if local_msg:
        lines += ["**MongoDB**", local_msg]
    if urls:
        lines += ["\n**Web (Tavily)**"]
        for idx, item in enumerate(extracted.get("extracted", [])[:3], start=1):
            lines.append(f"- [{idx}] {item.get('title', '(sem título)')} — {item.get('url', '')}")
    else:
        lines.append(
            _localized_text(
                language,
                "- Nenhum novo URL web encontrado.",
                "- No new web URL was found.",
            )
        )

    return "\n".join(lines), {
        "source": "mongo+web",
        "chunks": len(local_chunks),
        "web_urls": len(urls),
    }


def _build_edit_proposal(markdown: str, user_text: str, allow_web: bool) -> tuple[str, dict]:
    """
    Builds an edit proposal for a given paragraph in the markdown content based on the user's input.

    Args:
        markdown (str): The markdown content of the document.
        user_text (str): The message input by the user in the chat.
        allow_web (bool): A flag indicating whether web search is enabled for reference retrieval.

    Returns:
        tuple: A tuple containing the preview of the edit proposal (str) and the proposal details (dict).
    """
    language = _detect_user_language(user_text)
    sections = _split_sections(markdown)
    sec_idx = _resolve_section_index(user_text, sections)
    if sec_idx is None:
        return (
            _localized_text(
                language,
                "Não consegui identificar a seção alvo para edição.",
                "I couldn't identify the target section for editing.",
            ),
            {},
        )

    section = sections[sec_idx]
    p_idx = _resolve_paragraph_index(user_text, len(section.get("paragraphs", [])))
    if p_idx is None:
        p_idx = 0 if section.get("paragraphs") else None
    if p_idx is None:
        return (
            _localized_text(
                language,
                "A seção alvo não possui parágrafos editáveis.",
                "The target section has no editable paragraphs.",
            ),
            {},
        )

    paragraph = section["paragraphs"][p_idx]
    evidence_chunks = search_chunks(paragraph["text"][:600], k=5)

    web_context = ""
    if allow_web:
        web = search_tavily_incremental(
            query=paragraph["text"][:350], previous_urls=[], max_results=3
        )
        urls = web.get("new_urls", [])[:2]
        if urls:
            ext = extract_tavily.invoke({"urls": urls, "include_images": False})
            web_context = "\n\nWEB SOURCES:\n" + "\n\n".join(
                f"URL: {item.get('url', '')}\nTITLE: {item.get('title', '')}\nCONTENT: {str(item.get('content', ''))[:1200]}"
                for item in ext.get("extracted", [])
            )

    prompt_obj = load_prompt(
        "academic/edit_paragraph_suggestions",
        user_instruction=user_text,
        current_year=datetime.now().year,
        original_paragraph=paragraph["text"],
        evidence_chunks="\n".join(evidence_chunks[:3]),
        web_context=web_context,
    )

    try:
        proposed = str(llm_call(prompt=prompt_obj.text, temperature=0.2)).strip()
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
        _localized_text(
            language,
            "### Proposta de edição (pendente)\n",
            "### Edit proposal (pending)\n",
        )
        + _localized_text(
            language,
            f"- Alvo: **{section['title']}**, parágrafo **{p_idx + 1}**\n",
            f"- Target: **{section['title']}**, paragraph **{p_idx + 1}**\n",
        )
        + _localized_text(
            language,
            "- Ação necessária: clique em **Confirm Edit** para aplicar.\n\n",
            "- Required action: click **Confirm Edit** to apply it.\n\n",
        )
        + _localized_text(
            language,
            f"**Antes**\n{proposal['before'][:1200]}\n\n",
            f"**Before**\n{proposal['before'][:1200]}\n\n",
        )
        + _localized_text(
            language,
            f"**Depois (proposto)**\n{proposal['after'][:1200]}",
            f"**After (proposed)**\n{proposal['after'][:1200]}",
        )
    )
    return preview, proposal


def start_review_session(
    review_file: str,
    history: list,
    session_state: dict,
) -> tuple[list, dict, str, str]:
    """
    Starts a review session by initializing the session state and preparing the working copy of the review file.

    Args:
        review_file (str): The path to the review file.
        history (list): The chat history.
        session_state (dict): The current session state.

    Returns:
        tuple: A tuple containing the updated chat history, session state, status message, and the content of the working copy.
    """
    language = _detect_user_language(
        " ".join(
            str(msg.get("content", "")) for msg in (history or [])[-3:] if isinstance(msg, dict)
        )
    )
    if not review_file or not os.path.exists(review_file):
        return (
            history,
            session_state,
            _localized_text(language, "❌ Arquivo não encontrado.", "❌ File not found."),
            "",
        )

    normalized = os.path.normpath(review_file)
    if not normalized.startswith("reviews/"):
        return (
            history,
            session_state,
            _localized_text(
                language,
                "❌ Apenas arquivos em reviews/ são permitidos.",
                "❌ Only files inside reviews/ are allowed.",
            ),
            "",
        )

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
            "content": _localized_text(
                language,
                "✅ Sessão de revisão iniciada.\n"
                f"- Original: `{normalized}`\n"
                f"- Cópia editável: `{working_copy}`\n"
                "Pergunte sobre achados, referências, confirmação de parágrafos ou peça propostas de edição.",
                "✅ Review session started.\n"
                f"- Original: `{normalized}`\n"
                f"- Editable copy: `{working_copy}`\n"
                "Ask about findings, references, paragraph confirmation, or request edit proposals.",
            ),
        }
    ]
    return (
        history,
        state,
        _localized_text(language, "✅ Sessão pronta", "✅ Session ready"),
        content,
    )


def review_chat_turn(
    user_msg: str,
    history: list,
    session_state: dict,
    web_enabled: bool = False,
) -> tuple[list, dict, str, str]:
    """Handles a chat turn during the review session, processing the user's message and updating the session state accordingly.

    Args:
        user_msg (str): The message input by the user in the chat.
        history (list): The list of previous messages in the chat history, where each message is a dictionary with 'role' and 'content' keys.
        session_state (dict): The current state of the review session, containing information such as the working copy path, current markdown content, pending edits, and retrieval trace.
        web_enabled (bool, optional): A flag indicating whether web search is enabled for reference retrieval. Defaults to False.

    Returns:
        tuple: A tuple containing the updated chat history (list), the updated session state (dict
    """
    language = _detect_user_language(user_msg)
    session_state["last_language"] = language
    if not session_state or not session_state.get("working_copy_path"):
        return (
            history,
            session_state,
            _localized_text(
                language,
                "❌ Inicie uma sessão selecionando um arquivo.",
                "❌ Start a session by selecting a file.",
            ),
            "",
        )
    if not user_msg.strip():
        return (
            history,
            session_state,
            _localized_text(language, "⚠️ Mensagem vazia.", "⚠️ Empty message."),
            _read_md(session_state.get("working_copy_path")),
        )

    working_copy = session_state["working_copy_path"]
    markdown = _read_md(working_copy)
    session_state["current_markdown"] = markdown
    sections = _split_sections(markdown)
    allow_web = bool(web_enabled) or _explicit_web_request(user_msg)
    pending_edit = session_state.get("pending_edit") or {}
    pending_reference_action = session_state.get("pending_reference_action") or {}
    awaiting_reference_confirmation = bool(session_state.get("awaiting_reference_confirmation"))
    pending_phrase_reference_action = session_state.get("pending_phrase_reference_action") or {}
    awaiting_phrase_reference_confirmation = bool(
        session_state.get("awaiting_phrase_reference_confirmation")
    )
    target_hint = _resolve_target_hint(
        user_msg,
        sections,
        session_state.get("last_target_resolution") or {},
    )

    reference_intent = _classify_reference_intent(user_msg)
    if awaiting_reference_confirmation and pending_reference_action:
        pending_intent = str(pending_reference_action.get("intent") or "")

        if _is_negative_confirmation(user_msg):
            session_state["pending_reference_action"] = {}
            session_state["awaiting_reference_confirmation"] = False
            reply = _localized_text(
                language,
                "🛑 Ação de referências cancelada.",
                "🛑 Reference action canceled.",
            )
            history = history + [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": reply},
            ]
            session_state["chat_history"] = history
            return (
                history,
                session_state,
                _localized_text(language, "✅ Cancelado", "✅ Canceled"),
                _read_md(working_copy),
            )

        if _is_affirmative_confirmation(user_msg):
            if pending_intent == "list_all":
                reply, ref_meta = _handle_list_all_references_request(
                    markdown,
                    str(pending_reference_action.get("original_message") or ""),
                    allow_web=allow_web,
                )
                status_msg = _localized_text(
                    language, "✅ Referências listadas", "✅ References listed"
                )
                trace_action = "reference_pipeline_list_all"
            elif pending_intent == "format_provided":
                requires_web = bool(pending_reference_action.get("requires_web"))
                if requires_web and not allow_web:
                    incomplete_items = pending_reference_action.get("incomplete_items") or []
                    reply = _localized_text(
                        language,
                        "Não executei a formatação para evitar saída parcial incorreta.\n"
                        f"Itens incompletos: {', '.join(f'[{idx}]' for idx in incomplete_items)}\n"
                        "Ative **Allow web search** e confirme novamente com **sim**.",
                        "I did not execute formatting to avoid incorrect partial output.\n"
                        f"Incomplete items: {', '.join(f'[{idx}]' for idx in incomplete_items)}\n"
                        "Enable **Allow web search** and confirm again with **yes**.",
                    )
                    history = history + [
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": reply},
                    ]
                    session_state["chat_history"] = history
                    session_state["awaiting_reference_confirmation"] = True
                    return (
                        history,
                        session_state,
                        _localized_text(
                            language,
                            "⚠️ Habilite web para continuar",
                            "⚠️ Enable web to continue",
                        ),
                        _read_md(working_copy),
                    )

                reply, ref_meta = _handle_format_provided_references_request(
                    str(pending_reference_action.get("original_message") or ""),
                    allow_web=allow_web,
                )
                status_msg = _localized_text(
                    language, "✅ Fontes formatadas", "✅ Sources formatted"
                )
                trace_action = "reference_pipeline_format_provided"
            else:
                reply = _localized_text(
                    language,
                    "Ação pendente inválida. Reinicie o comando.",
                    "Invalid pending action. Please send the command again.",
                )
                history = history + [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": reply},
                ]
                session_state["chat_history"] = history
                session_state["pending_reference_action"] = {}
                session_state["awaiting_reference_confirmation"] = False
                return (
                    history,
                    session_state,
                    _localized_text(language, "❌ Erro de estado", "❌ State error"),
                    _read_md(working_copy),
                )

            history = history + [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": reply},
            ]
            session_state["chat_history"] = history
            session_state["pending_reference_action"] = {}
            session_state["awaiting_reference_confirmation"] = False
            session_state.setdefault("retrieval_trace", []).append(
                {
                    "action": trace_action,
                    "web": allow_web,
                    "at": datetime.now().isoformat(timespec="seconds"),
                    "tool_calls": [],
                    "meta": ref_meta,
                }
            )
            return history, session_state, status_msg, _read_md(working_copy)

        if reference_intent in {"list_all", "format_provided"}:
            prompt, pending_data = _build_reference_confirmation_prompt(
                reference_intent, user_msg, allow_web=allow_web
            )
            session_state["pending_reference_action"] = pending_data
            session_state["awaiting_reference_confirmation"] = True
            history = history + [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": prompt},
            ]
            session_state["chat_history"] = history
            return (
                history,
                session_state,
                _localized_text(language, "⏳ Aguardando confirmação", "⏳ Awaiting confirmation"),
                _read_md(working_copy),
            )

        reply = _localized_text(
            language,
            "Estou aguardando sua confirmação da ação de referências. Responda **sim** para continuar ou **não** para cancelar.",
            "I'm waiting for your confirmation of the reference action. Reply **yes** to continue or **no** to cancel.",
        )
        history = history + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": reply},
        ]
        session_state["chat_history"] = history
        return (
            history,
            session_state,
            _localized_text(language, "⏳ Aguardando confirmação", "⏳ Awaiting confirmation"),
            _read_md(working_copy),
        )

    if awaiting_phrase_reference_confirmation and pending_phrase_reference_action:
        stage = str(pending_phrase_reference_action.get("stage") or "ask_mongo")
        missing_numbers = list(pending_phrase_reference_action.get("missing_numbers") or [])
        original_message = str(pending_phrase_reference_action.get("original_message") or user_msg)
        pending_action_language = pending_phrase_reference_action.get("action_language")
        if pending_action_language:
            action_language = str(pending_action_language)
            language_source = "pending_phrase_reference_action.action_language"
        else:
            action_language = _detect_user_language(original_message, fallback=language)
            language_source = "detected_from_original_message"

        if _is_affirmative_confirmation(user_msg):
            if stage == "ask_mongo":
                reply, meta = _search_reference_in_mongo_by_phrase(
                    original_message, missing_numbers
                )
                if meta.get("found"):
                    session_state["pending_phrase_reference_action"] = {}
                    session_state["awaiting_phrase_reference_confirmation"] = False
                    session_state.setdefault("retrieval_trace", []).append(
                        {
                            "action": "phrase_reference_mongo",
                            "web": False,
                            "at": datetime.now().isoformat(timespec="seconds"),
                            "tool_calls": [],
                            "meta": {
                                **meta,
                                "intent_source": pending_phrase_reference_action.get(
                                    "intent_source", "deterministic"
                                ),
                                "language_source": language_source,
                            },
                        }
                    )
                    history = history + [
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": reply},
                    ]
                    session_state["chat_history"] = history
                    return (
                        history,
                        session_state,
                        _localized_text(
                            action_language,
                            "✅ Referência candidata encontrada no MongoDB",
                            "✅ Candidate reference found in MongoDB",
                        ),
                        _read_md(working_copy),
                    )

                if allow_web:
                    followup = _localized_text(
                        action_language,
                        "Não encontrei no MongoDB. Deseja buscar na internet? Responda **sim** ou **não**.",
                        "I couldn't find it in MongoDB. Do you want to search on the internet? Reply **yes** or **no**.",
                    )
                    pending_phrase_reference_action["stage"] = "ask_internet"
                    session_state["pending_phrase_reference_action"] = (
                        pending_phrase_reference_action
                    )
                    history = history + [
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": followup},
                    ]
                    session_state["chat_history"] = history
                    return (
                        history,
                        session_state,
                        _localized_text(
                            action_language,
                            "⏳ Aguardando confirmação",
                            "⏳ Awaiting confirmation",
                        ),
                        _read_md(working_copy),
                    )

                reply = _localized_text(
                    action_language,
                    "Não encontrei no MongoDB e a busca web está desativada. Ative **Allow web search** se quiser tentar internet.",
                    "I couldn't find it in MongoDB and web search is disabled. Enable **Allow web search** if you want to try internet search.",
                )
                pending_phrase_reference_action["stage"] = "ask_internet"
                session_state["pending_phrase_reference_action"] = pending_phrase_reference_action
                session_state["awaiting_phrase_reference_confirmation"] = True
                history = history + [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": reply},
                ]
                session_state["chat_history"] = history
                return (
                    history,
                    session_state,
                    _localized_text(action_language, "⚠️ Web desativado", "⚠️ Web disabled"),
                    _read_md(working_copy),
                )

            if stage == "ask_internet":
                if not allow_web:
                    reply = _localized_text(
                        action_language,
                        "A busca na internet está desativada. Ative **Allow web search** para continuar.",
                        "Internet search is disabled. Enable **Allow web search** to continue.",
                    )
                    history = history + [
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": reply},
                    ]
                    session_state["chat_history"] = history
                    return (
                        history,
                        session_state,
                        _localized_text(action_language, "⚠️ Web desativado", "⚠️ Web disabled"),
                        _read_md(working_copy),
                    )

                reply, meta = _search_reference_on_web_by_phrase(original_message, missing_numbers)
                session_state["pending_phrase_reference_action"] = {}
                session_state["awaiting_phrase_reference_confirmation"] = False
                session_state.setdefault("retrieval_trace", []).append(
                    {
                        "action": "phrase_reference_web",
                        "web": True,
                        "at": datetime.now().isoformat(timespec="seconds"),
                        "tool_calls": [],
                        "meta": {
                            **meta,
                            "intent_source": pending_phrase_reference_action.get(
                                "intent_source", "deterministic"
                            ),
                            "language_source": language_source,
                        },
                    }
                )
                history = history + [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": reply},
                ]
                session_state["chat_history"] = history
                return (
                    history,
                    session_state,
                    _localized_text(
                        action_language,
                        "✅ Busca na internet concluída",
                        "✅ Internet search completed",
                    ),
                    _read_md(working_copy),
                )

        if _is_negative_confirmation(user_msg):
            if stage == "ask_mongo" and allow_web:
                followup = _localized_text(
                    action_language,
                    "Ok, sem MongoDB. Deseja buscar na internet? Responda **sim** ou **não**.",
                    "Okay, skipping MongoDB. Do you want to search on the internet? Reply **yes** or **no**.",
                )
                pending_phrase_reference_action["stage"] = "ask_internet"
                session_state["pending_phrase_reference_action"] = pending_phrase_reference_action
                history = history + [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": followup},
                ]
                session_state["chat_history"] = history
                return (
                    history,
                    session_state,
                    _localized_text(
                        action_language,
                        "⏳ Aguardando confirmação",
                        "⏳ Awaiting confirmation",
                    ),
                    _read_md(working_copy),
                )

            session_state["pending_phrase_reference_action"] = {}
            session_state["awaiting_phrase_reference_confirmation"] = False
            reply = _localized_text(
                action_language,
                "Busca de referência por frase cancelada.",
                "Phrase reference search canceled.",
            )
            history = history + [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": reply},
            ]
            session_state["chat_history"] = history
            return (
                history,
                session_state,
                _localized_text(action_language, "✅ Cancelado", "✅ Canceled"),
                _read_md(working_copy),
            )

        wait_msg = _localized_text(
            action_language,
            "Responda **sim** ou **não** para continuar a busca da referência por frase.",
            "Reply **yes** or **no** to continue the phrase reference lookup.",
        )
        history = history + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": wait_msg},
        ]
        session_state["chat_history"] = history
        return (
            history,
            session_state,
            _localized_text(
                action_language, "⏳ Aguardando confirmação", "⏳ Awaiting confirmation"
            ),
            _read_md(working_copy),
        )

    if reference_intent == "list_all":
        reply, pending_data = _build_reference_confirmation_prompt(
            reference_intent, user_msg, allow_web=allow_web
        )
        history = history + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": reply},
        ]
        session_state["chat_history"] = history
        session_state["pending_reference_action"] = pending_data
        session_state["awaiting_reference_confirmation"] = True
        return (
            history,
            session_state,
            _localized_text(language, "⏳ Aguardando confirmação", "⏳ Awaiting confirmation"),
            _read_md(working_copy),
        )

    if reference_intent == "format_provided":
        reply, pending_data = _build_reference_confirmation_prompt(
            reference_intent, user_msg, allow_web=allow_web
        )
        history = history + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": reply},
        ]
        session_state["chat_history"] = history
        session_state["pending_reference_action"] = pending_data
        session_state["awaiting_reference_confirmation"] = True
        return (
            history,
            session_state,
            _localized_text(language, "⏳ Aguardando confirmação", "⏳ Awaiting confirmation"),
            _read_md(working_copy),
        )

    if reference_intent == "resolve_numbers":
        requested_numbers = _extract_requested_citation_numbers(user_msg)
        phrase_reference_match, phrase_reference_debug = _classify_phrase_reference_intent(user_msg)
        if requested_numbers and phrase_reference_match:
            inventory = _collect_reference_inventory(markdown)
            refs_by_number = inventory.get("references_by_number", {})
            missing_numbers = [n for n in requested_numbers if n not in refs_by_number]
            if missing_numbers:
                prompt = _localized_text(
                    language,
                    "Não encontrei essas referências na lista atual: "
                    f"{', '.join(f'[{n}]' for n in missing_numbers)}.\n"
                    "Deseja que eu busque no MongoDB? Responda **sim** ou **não**.",
                    "I couldn't find these references in the current list: "
                    f"{', '.join(f'[{n}]' for n in missing_numbers)}.\n"
                    "Do you want me to search in MongoDB? Reply **yes** or **no**.",
                )
                session_state["pending_phrase_reference_action"] = {
                    "stage": "ask_mongo",
                    "missing_numbers": missing_numbers,
                    "original_message": user_msg,
                    "action_language": language,
                    "intent_source": "deterministic",
                    "intent_debug": phrase_reference_debug,
                }
                session_state["awaiting_phrase_reference_confirmation"] = True
                history = history + [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": prompt},
                ]
                session_state["chat_history"] = history
                return (
                    history,
                    session_state,
                    _localized_text(
                        language, "⏳ Aguardando confirmação", "⏳ Awaiting confirmation"
                    ),
                    _read_md(working_copy),
                )

        reply, ref_meta = _handle_resolve_numbers_request(markdown, user_msg, allow_web=allow_web)
        history = history + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": reply},
        ]
        session_state["chat_history"] = history
        session_state.setdefault("retrieval_trace", []).append(
            {
                "action": "reference_pipeline",
                "web": allow_web,
                "at": datetime.now().isoformat(timespec="seconds"),
                "tool_calls": [],
                "meta": ref_meta,
            }
        )
        return (
            history,
            session_state,
            _localized_text(language, "✅ Referências processadas", "✅ References processed"),
            _read_md(working_copy),
        )

    if _is_citation_usage_query(user_msg):
        reply = _list_paragraphs_using_citation(markdown, user_msg)
        history = history + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": reply},
        ]
        session_state["chat_history"] = history
        session_state.setdefault("retrieval_trace", []).append(
            {
                "action": "local_citation_lookup",
                "web": False,
                "at": datetime.now().isoformat(timespec="seconds"),
                "tool_calls": [],
            }
        )
        return (
            history,
            session_state,
            _localized_text(language, "✅ Sessão ativa", "✅ Session active"),
            _read_md(working_copy),
        )

    # ── Image suggestion flow ─────────────────────────────────────────
    awaiting_image_confirmation = bool(session_state.get("awaiting_image_confirmation"))
    pending_image_action = session_state.get("pending_image_action") or {}

    if awaiting_image_confirmation and pending_image_action:
        if _is_negative_confirmation(user_msg):
            session_state["pending_image_action"] = {}
            session_state["awaiting_image_confirmation"] = False
            reply = _localized_text(
                language,
                "🛑 Busca de imagens cancelada.",
                "🛑 Image search canceled.",
            )
            history = history + [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": reply},
            ]
            session_state["chat_history"] = history
            return (
                history,
                session_state,
                _localized_text(language, "✅ Cancelado", "✅ Canceled"),
                _read_md(working_copy),
            )

        # Affirmative or new scope — run the image agent
        original_request = str(pending_image_action.get("original_request", user_msg))
        pending_excerpt = pending_image_action.get("excerpt")
        if pending_excerpt:
            excerpt = str(pending_excerpt)
        else:
            # Rebuild excerpt only (scope stays from pending_image_action — it
            # was already confirmed by the user).
            _confirmed_scope, excerpt = _build_image_scope_description(
                original_request, sections, language
            )
        scope = str(pending_image_action.get("scope", "all sections"))

        # Allow user to override scope in the same message
        if not _is_affirmative_confirmation(user_msg):
            scope, excerpt = _build_image_scope_description(user_msg, sections, language)
            session_state["pending_image_action"]["scope"] = scope
            session_state["pending_image_action"]["excerpt"] = excerpt

        if not allow_web:
            session_state["pending_image_action"] = {}
            session_state["awaiting_image_confirmation"] = False
            reply = _localized_text(
                language,
                "A sugestão de imagens requer busca na web. "
                "Ative **Allow web search** e tente novamente.",
                "Image suggestion requires web search. Enable **Allow web search** and try again.",
            )
            history = history + [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": reply},
            ]
            session_state["chat_history"] = history
            return (
                history,
                session_state,
                _localized_text(language, "⚠️ Web desativado", "⚠️ Web disabled"),
                _read_md(working_copy),
            )

        reply = run_image_suggestion_agent(
            document_excerpt=excerpt,
            user_request=original_request,
            scope_description=scope,
        )
        session_state["pending_image_action"] = {}
        session_state["awaiting_image_confirmation"] = False
        history = history + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": reply},
        ]
        session_state["chat_history"] = history
        return (
            history,
            session_state,
            _localized_text(language, "✅ Imagens sugeridas", "✅ Images suggested"),
            _read_md(working_copy),
        )

    if _is_image_request(user_msg):
        if not allow_web:
            reply = _localized_text(
                language,
                "A sugestão de imagens requer busca na web. "
                "Ative **Allow web search** e tente novamente.",
                "Image suggestion requires web search. Enable **Allow web search** and try again.",
            )
            history = history + [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": reply},
            ]
            session_state["chat_history"] = history
            return (
                history,
                session_state,
                _localized_text(language, "⚠️ Web desativado", "⚠️ Web disabled"),
                _read_md(working_copy),
            )
        # First image request — ask for scope confirmation
        scope, excerpt = _build_image_scope_description(user_msg, sections, language)
        confirm_prompt = _build_image_confirmation_prompt(scope, language)
        session_state["pending_image_action"] = {
            "scope": scope,
            "excerpt": excerpt,
            "original_request": user_msg,
        }
        session_state["awaiting_image_confirmation"] = True
        history = history + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": confirm_prompt},
        ]
        session_state["chat_history"] = history
        return (
            history,
            session_state,
            _localized_text(language, "⏳ Aguardando confirmação", "⏳ Awaiting confirmation"),
            _read_md(working_copy),
        )

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
        return (
            history,
            session_state,
            _localized_text(language, "❌ Erro no agente", "❌ Agent error"),
            _read_md(working_copy),
        )

    action = result.get("action", "answer")
    reply = result.get("reply", "")

    # ── Handle actions ────────────────────────────────────────────────
    if action == "apply_edit":
        proposal = session_state.get("pending_edit") or {}
        if not proposal:
            reply = _localized_text(
                language,
                "Não há edição pendente para confirmar.",
                "There is no pending edit to confirm.",
            )
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
            reply = _localized_text(
                language,
                "✅ Edição aplicada na cópia de trabalho.\n"
                f"- Seção: **{proposal.get('section_title', '')}**\n"
                f"- Parágrafo: **{int(proposal.get('paragraph_index', 0)) + 1}**\n"
                f"- Arquivo: `{working_copy}`",
                "✅ Edit applied to the working copy.\n"
                f"- Section: **{proposal.get('section_title', '')}**\n"
                f"- Paragraph: **{int(proposal.get('paragraph_index', 0)) + 1}**\n"
                f"- File: `{working_copy}`",
            )

    elif action == "cancel_edit":
        has_pending = bool(session_state.get("pending_edit"))
        session_state["pending_edit"] = {}
        reply = _localized_text(
            language,
            (
                "🗑️ Edição pendente cancelada."
                if has_pending
                else "Não havia edição pendente para cancelar."
            ),
            ("🗑️ Pending edit canceled." if has_pending else "There was no pending edit to cancel."),
        )

    elif action == "edit_proposal":
        proposal = result.get("edit_proposal")
        if proposal:
            session_state["pending_edit"] = proposal
            session_state["last_target_resolution"] = {
                "section": proposal.get("section_title", ""),
                "paragraph_index": proposal.get("paragraph_index", -1),
            }
            reply = _localized_text(
                language,
                "### Proposta de edição (pendente)\n"
                f"- Alvo: **{proposal.get('section_title', '')}**, "
                f"parágrafo **{int(proposal.get('paragraph_index', 0)) + 1}**\n"
                "- Ação necessária: clique em **Confirm Edit** ou diga "
                "'confirmar' para aplicar.\n\n"
                f"**Antes**\n{proposal['before'][:1200]}\n\n"
                f"**Depois (proposto)**\n{proposal['after'][:1200]}",
                "### Edit proposal (pending)\n"
                f"- Target: **{proposal.get('section_title', '')}**, "
                f"paragraph **{int(proposal.get('paragraph_index', 0)) + 1}**\n"
                "- Required action: click **Confirm Edit** or say "
                "'confirm' to apply it.\n\n"
                f"**Before**\n{proposal['before'][:1200]}\n\n"
                f"**After (proposed)**\n{proposal['after'][:1200]}",
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
    status = _localized_text(
        language,
        "🟡 Edição pendente — confirme ou cancele" if pending else "✅ Sessão ativa",
        "🟡 Pending edit — confirm or cancel" if pending else "✅ Session active",
    )
    return history, session_state, status, _read_md(working_copy)


def confirm_review_edit(
    history: list,
    session_state: dict,
) -> tuple[list, dict, str, str]:
    """Confirm and apply the pending edit in the review session.

    Args:
        history: The current chat history.
        session_state: The current session state, expected to contain 'pending_edit'.

    Returns:
        Updated history, session_state, status message, and the refreshed markdown content.
    """
    language = (session_state or {}).get("last_language", "pt")
    msg = "confirm edit" if language == "en" else "confirmar edição"
    return review_chat_turn(msg, history, session_state)


def cancel_review_edit(
    history: list,
    session_state: dict,
) -> tuple[list, dict, str, str]:
    """Cancel the pending edit in the review session.

    Args:
        history: The current chat history.
        session_state: The current session state, expected to contain 'pending_edit'.

    Returns:
        Updated history, session_state, status message, and the current markdown content.
    """
    language = (session_state or {}).get("last_language", "pt")
    msg = "cancel edit" if language == "en" else "cancelar edição"
    return review_chat_turn(msg, history, session_state)


def save_review_manual_edit(
    edited_text: str,
    history: list,
    session_state: dict,
) -> tuple[list, dict, str, str]:
    """Save manual edits made directly in the text editor.

    Args:
        edited_text: The full text from the editor after manual changes.
        history: The current chat history.
        session_state: The current session state, expected to contain 'working_copy_path'.

    Returns:
        Updated history, session_state, status message, and the refreshed markdown content.
    """
    language = (session_state or {}).get("last_language", "pt")
    if not session_state or not session_state.get("working_copy_path"):
        return (
            history,
            session_state,
            _localized_text(language, "❌ Nenhuma sessão ativa.", "❌ No active session."),
            "",
        )
    if not edited_text.strip():
        return (
            history,
            session_state,
            _localized_text(
                language,
                "⚠️ Texto vazio — nada salvo.",
                "⚠️ Empty text — nothing was saved.",
            ),
            _read_md(session_state.get("working_copy_path")),
        )

    working_copy = session_state["working_copy_path"]
    _atomic_write(working_copy, edited_text)
    refreshed = _read_md(working_copy)
    session_state["current_markdown"] = refreshed
    session_state["pending_edit"] = {}

    history = history + [
        {
            "role": "assistant",
            "content": _localized_text(
                language,
                f"💾 Edição manual salva em `{working_copy}`.",
                f"💾 Manual edit saved to `{working_copy}`.",
            ),
        },
    ]
    session_state["chat_history"] = history
    return (
        history,
        session_state,
        _localized_text(language, "✅ Edição manual salva", "✅ Manual edit saved"),
        refreshed,
    )
