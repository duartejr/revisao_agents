"""Gradio event handlers for the planning workflow tab.

Exposes ``start_planning``, ``continue_planning``, ``load_thread_state``,
and ``list_available_threads`` for driving the academic/technical planning
HITL loop from the Gradio UI.
"""

from __future__ import annotations

import queue
import traceback
from datetime import datetime

from revisao_agents.config import validate_runtime_config
from revisao_agents.graphs.checkpoints import get_checkpointer
from revisao_agents.state import ReviewState
from revisao_agents.workflows import build_academic_workflow, build_technical_workflow

from .base import _read_md, _StdoutCapture


def list_available_threads() -> list[str]:
    """List available thread IDs for debugging purposes.

    Returns:
        A list of thread IDs currently stored in the checkpointer.
    """
    from revisao_agents.graphs.checkpoints import list_thread_ids

    return list_thread_ids()


def _format_chatbot_history(interview_history: list[tuple[str, str]]) -> list[dict]:
    """Convert LangGraph history [("role", "content")] to Gradio [{"role": "role", "content": "content"}].

    Args:
        interview_history: A list of tuples representing the history of the conversation, where each tuple contains a role ("human" or "assistant") and the corresponding content.

    Returns:
        A list of dictionaries formatted for Gradio chatbot, where each dictionary has a "role" key (with values "user" or "assistant") and a "content" key with the message content.
    """
    formatted = []
    for role, content in interview_history:
        gradio_role = "user" if role in ("human", "user") else "assistant"
        formatted.append({"role": gradio_role, "content": content})
    return formatted


def load_thread_state(thread_id: str) -> tuple[str, str, int, dict, list[dict]]:
    """Load the state of a planning thread and return formatted components for UI restoration.

    Args:
        thread_id: The ID of the thread to load.

    Returns:
        A tuple containing the theme, review type, number of rounds, session state, and formatted interview history.
    """
    if not thread_id:
        return "", "academico", 3, {}, []

    saver = get_checkpointer()
    config = {"configurable": {"thread_id": thread_id}}
    checkpoint_tuple = saver.get_tuple(config)

    if not checkpoint_tuple:
        return "", "academico", 3, {}, []

    state_values = checkpoint_tuple.checkpoint.get("channel_values", {})
    theme = state_values.get("theme", "")
    review_type = state_values.get("review_type", "academico")
    rounds = state_values.get("max_questions", 3)
    interview_history = state_values.get("interview_history", [])

    if review_type == "academico":
        app = build_academic_workflow(checkpointer=saver)
    else:
        app = build_technical_workflow(checkpointer=saver)

    session_state = {
        "app": app,
        "config": config,
        "type": review_type,
        "theme": theme,
        "rounds": rounds,
        "thread_id": thread_id,
    }

    formatted_history = _format_chatbot_history(interview_history)

    # Add round info if available
    if interview_history:
        p = state_values.get("questions_asked", 0)
        mp = state_values.get("max_questions", rounds)
        status_prefix = f"[Round {p}/{mp} — {review_type}]\n\n"
        if (
            formatted_history
            and formatted_history[-1]["role"] == "assistant"
            and status_prefix not in formatted_history[-1]["content"]
        ):
            formatted_history[-1]["content"] = status_prefix + formatted_history[-1]["content"]

    return theme, review_type, rounds, session_state, formatted_history


def start_planning(
    theme: str,
    type: str,
    rounds: int,
) -> tuple[list, dict, str, str, str | None]:
    """Launch planning workflow until the first HITL pause.

    Args:
        theme: The theme or topic for the planning session.
        type: The type of planning (``"academico"``, ``"tecnico"``, or ``"ambos"``).
        rounds: The maximum number of HITL rounds before stopping.

    Returns:
        A 5-tuple of: updated chat history, session state dict, status
        message string, rendered plan markdown, and the new thread ID.
    """

    from openai import AuthenticationError

    if not theme.strip():
        return [], {}, "❌ Por favor, forneça um tema antes de iniciar.", "", ""

    cfg_issues = validate_runtime_config(strict=False)
    if cfg_issues:
        msg = "❌ Configuração incompleta:\n- " + "\n- ".join(cfg_issues)
        return [], {}, msg, "", ""

    types_list = ["academico", "tecnico"] if type == "ambos" else [type]
    current_type = types_list[0]
    label = "ACADEMIC" if current_type == "academico" else "TECHNICAL"

    state_init: ReviewState = {
        "theme": theme,
        "review_type": current_type,
        "relevant_chunks": [],
        "technical_snippets": [],
        "technical_urls": [],
        "current_plan": "",
        "interview_history": [],
        "questions_asked": 0,
        "max_questions": int(rounds),
        "final_plan": "",
        "final_plan_path": "",
        "status": "starting",
        "detected_language": "",
        "user_language_choice": "",
        "is_theme_vague": True,
        "is_theme_refined": False,
        "confidence_score": 0.0,
        "refinement_feedback": [],
    }

    thread_id = f"revisao_{current_type}_{theme[:20]}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    config = {"configurable": {"thread_id": thread_id}}

    if current_type == "academico":
        app = build_academic_workflow(checkpointer=get_checkpointer())
    else:
        app = build_technical_workflow(checkpointer=get_checkpointer())

    log_q: queue.Queue[str] = queue.Queue()
    with _StdoutCapture(log_q):
        try:
            for _ in app.stream(state_init, config):
                pass
        except AuthenticationError:
            return [], {}, "❌ Error in Authentication: Your API Key is invalid.", "", ""
        except Exception as exc:
            tb = traceback.format_exc()
            short_tb = "\n".join(tb.splitlines()[-5:])
            return [], {}, f"❌ Unexpected error: {exc}\n\n{short_tb}", "", ""

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
        return history, {}, "✅ Done", rendered, thread_id

    agent_question = ""
    for role, content in reversed(graph_state.values.get("interview_history", [])):
        if role == "assistant":
            agent_question = content
            break

    p = graph_state.values.get("questions_asked", 0)
    mp = graph_state.values.get("max_questions", rounds)
    is_refinement_phase = p == 0 and not graph_state.values.get("is_theme_refined", False)
    prefix = "" if is_refinement_phase else f"[Round {p}/{mp} — {current_type}]\n\n"
    history.append(
        {
            "role": "assistant",
            "content": f"{prefix}{agent_question}",
        }
    )

    session_state = {
        "app": app,
        "config": config,
        "type": current_type,
        "types_pending": types_list[1:],
        "theme": theme,
        "rounds": rounds,
    }

    return history, session_state, f"🔄 {label} planning in progress...", "", thread_id


def continue_planning(
    user_msg: str,
    history: list,
    session_state: dict,
) -> tuple[list, dict, str, str]:
    """Feed user response back into the HITL loop.

    Args:
        user_msg: The message input by the user in response to the agent's question.
        history: The current chat history to which the new message will be appended.
        session_state: Active session dict containing the compiled workflow
            ``app`` and its LangGraph ``config``.

    Returns:
        A 4-tuple of: updated chat history, updated session state, status
        message string, and the rendered plan markdown (empty while the
        workflow is still running).
    """
    if not session_state or "app" not in session_state:
        return history, session_state, "❌ No active session.", ""

    app = session_state["app"]
    config = session_state["config"]
    type = session_state["type"]
    label = "ACADEMIC" if type == "academico" else "TECHNICAL"

    history = history + [{"role": "user", "content": user_msg}]
    hist = app.get_state(config).values.get("interview_history", [])
    app.update_state(
        config, {"interview_history": hist + [("user", user_msg)]}, as_node="human_pause"
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
        history = history + [{"role": "assistant", "content": f"✅ {label} planning complete!"}]

        tipos_pendentes = session_state.get("types_pending", [])
        if tipos_pendentes:
            next_hist, next_state, next_status, _, _ = start_planning(
                session_state["theme"], tipos_pendentes[0], session_state["rounds"]
            )
            next_state["tipos_pending"] = tipos_pendentes[1:]
            return history + next_hist, next_state, next_status, rendered

        return history, {}, "✅ All planning complete!", rendered

    agent_question = ""
    for role, content in reversed(graph_state.values.get("interview_history", [])):
        if role == "assistant":
            agent_question = content
            break

    p = graph_state.values.get("questions_asked", 0)
    mp = graph_state.values.get("max_questions", session_state.get("rounds", 3))
    is_refinement_phase = p == 0 and not graph_state.values.get("is_theme_refined", False)
    prefix = "" if is_refinement_phase else f"[Round {p}/{mp} — {type}]\n\n"
    history = history + [
        {
            "role": "assistant",
            "content": f"{prefix}{agent_question}",
        }
    ]

    return history, session_state, f"🔄 {label} in progress...", ""
