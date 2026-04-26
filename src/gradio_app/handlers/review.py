"""Gradio event handlers for the interactive review tab.

Exposes the public session lifecycle functions (``start_review_session``,
``review_chat_turn``, ``confirm_review_edit``, ``cancel_review_edit``,
``save_review_manual_edit``) and the ``ReviewManager`` orchestration class.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime

import mlflow
from observability import workflow_run
from observability.mlflow_config import EXP_REVIEW_CHAT, get_tracking_uri

from revisao_agents.agents.image_suggestion_agent import run_image_suggestion_agent
from revisao_agents.agents.review_agent import run_review_agent

from .base import _atomic_write, _detect_user_language, _localized_text, _read_md
from .review_parts import references as review_refs
from .review_parts.document import (
    _resolve_target_hint,
    _split_sections,
    _working_copy_path,
)
from .review_parts.images import (
    _build_image_confirmation_prompt,
    _build_image_scope_description,
    _is_image_request,
)
from .review_parts.intent import (
    _classify_phrase_reference_intent,
    _classify_reference_intent,
    _explicit_web_request,
    _extract_requested_citation_numbers,
    _is_affirmative_confirmation,
    _is_citation_usage_query,
    _is_negative_confirmation,
)


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
        "mlflow_run_id": None,
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
        tuple: A tuple containing the updated chat history, session state, status message, and working copy content.
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
                reply, ref_meta = review_refs._handle_list_all_references_request(
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

                reply, ref_meta = review_refs._handle_format_provided_references_request(
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
            prompt, pending_data = review_refs._build_reference_confirmation_prompt(
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
                reply, meta = review_refs._search_reference_in_mongo_by_phrase(
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

                reply, meta = review_refs._search_reference_on_web_by_phrase(
                    original_message, missing_numbers
                )
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
        reply, pending_data = review_refs._build_reference_confirmation_prompt(
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
        reply, pending_data = review_refs._build_reference_confirmation_prompt(
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
            inventory = review_refs._collect_reference_inventory(markdown)
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

        reply, ref_meta = review_refs._handle_resolve_numbers_request(
            markdown, user_msg, allow_web=allow_web
        )
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
        reply = review_refs._list_paragraphs_using_citation(markdown, user_msg)
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
        doc_name = os.path.basename(session_state.get("original_file_path", "unknown"))
        run_id = session_state.get("mlflow_run_id")
        mlflow.set_tracking_uri(get_tracking_uri())
        if run_id:
            run_ctx = mlflow.start_run(run_id=run_id)
        else:
            run_name = f"review_chat/{doc_name[:40]}"
            run_ctx = workflow_run(
                EXP_REVIEW_CHAT, run_name, params={"document": doc_name, "allow_web": allow_web}
            )
        with run_ctx as active_run:
            session_state["mlflow_run_id"] = active_run.info.run_id
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


class ReviewManager:
    """Orchestrate review session actions with a consistent API."""

    def start_session(
        self,
        review_file: str,
        history: list,
        session_state: dict,
    ) -> tuple[list, dict, str, str]:
        return start_review_session(review_file, history, session_state)

    def chat_turn(
        self,
        user_msg: str,
        history: list,
        session_state: dict,
        web_enabled: bool = False,
    ) -> tuple[list, dict, str, str]:
        return review_chat_turn(user_msg, history, session_state, web_enabled=web_enabled)

    def confirm_edit(
        self,
        history: list,
        session_state: dict,
    ) -> tuple[list, dict, str, str]:
        return confirm_review_edit(history, session_state)

    def cancel_edit(
        self,
        history: list,
        session_state: dict,
    ) -> tuple[list, dict, str, str]:
        return cancel_review_edit(history, session_state)

    def save_manual_edit(
        self,
        edited_text: str,
        history: list,
        session_state: dict,
    ) -> tuple[list, dict, str, str]:
        return save_review_manual_edit(edited_text, history, session_state)


review_manager = ReviewManager()
