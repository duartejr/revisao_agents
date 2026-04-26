from __future__ import annotations

import glob
import os
import queue
import threading
from collections.abc import Generator
from typing import Any

from observability import workflow_run
from observability.mlflow_config import EXP_WRITING_ACADEMIC, EXP_WRITING_TECHNICAL

from revisao_agents.config import PLANS_DIR, validate_runtime_config
from revisao_agents.core.schemas.writer_config import WriterConfig
from revisao_agents.state import TechnicalWriterState
from revisao_agents.workflows.technical_writing_workflow import (
    build_technical_writing_workflow,
)

from .base import (
    _find_newest_md,
    _list_md,
    _LoggingCapture,
    _read_md,
    _StderrCapture,
    _StdoutCapture,
)


def list_plan_files(mode: str) -> list[str]:
    """List available plan files for the given mode.

    Args:
        mode: The writing mode, either "Academic" or "Technical", to filter the plan files accordingly.

    Returns:
        A list of file paths to the available plan files matching the specified mode.
    """
    pattern = (
        os.path.join(PLANS_DIR, "plano_revisao_tecnica_*.md")
        if mode == "Technical"
        else os.path.join(PLANS_DIR, "plano_revisao_*.md")
    )
    files = sorted(glob.glob(pattern))
    if not files:
        files = sorted(glob.glob(os.path.join(PLANS_DIR, "plano_revisao_*.md")))
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
    """Stream writing progress with live logs to the Gradio chatbot.

    Args:
        plan_path: The file path to the plan markdown file to be used for writing.
        mode: The writing mode, either "Academic" or "Technical", which determines the writer
                configuration.
        language: The language to be used for writing.
        min_src: The minimum number of sources required per section.
        tavily_enabled: A boolean indicating whether Tavily integration is enabled for enhanced writing capabilities.
        history: The current chat history to which new messages will be appended during streaming.

    Yields:
        A tuple containing the updated chat history, a status message for the UI, and the final rendered markdown output once writing is complete.
    """
    cfg_issues = validate_runtime_config(strict=False)
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
            history + [{"role": "assistant", "content": f"❌ Plano não encontrado: `{plan_path}`"}],
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
        exp = EXP_WRITING_ACADEMIC if mode == "Academic" else EXP_WRITING_TECHNICAL
        run_name = f"{mode.lower()}/{os.path.basename(plan_path)[:40]}"

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
                with workflow_run(
                    exp,
                    run_name,
                    params={
                        "plan": os.path.basename(plan_path),
                        "mode": mode,
                        "language": language,
                    },
                ):
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
