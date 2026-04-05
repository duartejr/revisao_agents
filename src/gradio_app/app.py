"""
app.py — Gradio-based chatbot UI for the revisao_agents project.

Provides a clean, ChatGPT-style web interface for all workflow options:

  Tab 1  📋 Plan        — Plan a literature review (Academic / Technical)
  Tab 2  ✍️  Write       — Execute writing from an existing plan file
  Tab 3  📁 Index PDFs  — Index local PDFs into the MongoDB vector store
  Tab 4  📚 References  — Format a reference list from a YAML/JSON file
  Tab 5  📄 View        — Browse and render any generated plan or review file

Run via:  python run_ui.py
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# sys.path bootstrap — needed when launched from the project root
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import gradio as gr  # noqa: E402

from gradio_app.handlers import (  # noqa: E402
    cancel_review_edit,
    confirm_review_edit,
    continue_planning,
    format_references,
    get_current_llm_provider,
    get_llm_provider_status,
    index_pdfs,
    list_llm_providers,
    list_plan_files,
    list_review_files,
    review_chat_turn,
    save_review_manual_edit,
    set_llm_provider,
    start_planning,
    start_review_session,
    start_writing,
)

# ═══════════════════════════════════════════════════════════════════════════
# Custom CSS — minimal ChatGPT-inspired theme
# ═══════════════════════════════════════════════════════════════════════════
_CSS = """
body, .gradio-container {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}
.tab-nav button {
    font-size: 0.95rem;
    font-weight: 600;
}
#app-header {
    text-align: center;
    padding: 1.2rem 0 0.4rem;
}
#app-header h1 {
    font-size: 1.8rem;
    font-weight: 800;
    margin: 0;
}
#app-header p {
    color: #6b7280;
    margin: 0.2rem 0 0;
}
#llm-switch-bar {
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 0.75rem;
    margin: 0.4rem 0 1rem;
    background: #f9fafb;
}
.status-bar {
    border-radius: 6px;
    padding: 0.4rem 0.8rem;
    font-size: 0.85rem;
    color: #374151;
    background: #f3f4f6;
}
"""

# ═══════════════════════════════════════════════════════════════════════════
# Helper: refresh plan file list based on mode selection
# ═══════════════════════════════════════════════════════════════════════════


def refresh_plan_list(mode: str) -> gr.update:
    """Refresh the list of available plan files based on the selected mode (Technical/Academic).

    Args:
        mode: The selected writing mode, expected to be "Technical" or "Academic".

    Returns:
        A gr.update object with the updated choices and default value for the plan dropdown.
    """
    files = list_plan_files(mode)
    return gr.update(choices=files, value=files[0] if files else None)


def _auto_refresh_plan_list(mode: str, current_value: str | None) -> gr.update:
    """Auto-refresh plan dropdown preserving current selection when possible.

    Args:
        mode: The selected writing mode, expected to be "Technical" or "Academic".
        current_value: The currently selected plan file path (if any).

    Returns:
        A gr.update object with the updated choices and value for the plan dropdown.
    """
    files = list_plan_files(mode)

    value = current_value if current_value in files else (files[0] if files else None)

    return gr.update(choices=files, value=value)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers: list and load output files for the View tab
# ═══════════════════════════════════════════════════════════════════════════


def _list_output_files(folder: str) -> list[str]:
    """Return sorted .md files from plans/ or reviews/.

    Args:
        folder: The folder to list files from, expected to be "plans" or "reviews".

    Returns:
        A list of file paths for .md files in the specified folder, sorted alphabetically.
    """
    os.makedirs(folder, exist_ok=True)
    files = sorted(f for f in os.listdir(folder) if f.endswith(".md"))
    return [os.path.join(folder, f) for f in files] if files else []


def _load_file(path: str) -> str:
    """Read and return the markdown content of a file.

    Args:
        path: The file path to read.

    Returns:
        The content of the file as a string, or an error message if the file cannot be read.
    """
    if not path or not os.path.exists(path):
        return f"*(Arquivo {path} não encontrado)*"
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except Exception as exc:
        return f"❌ Erro ao ler o arquivo {path}: {exc}"


def _refresh_file_list(folder: str) -> gr.update:
    """Refresh the file list for the View tab based on the selected folder.

    Args:
        folder: The selected folder, expected to be "plans" or "reviews".

    Returns:
        A gr.update object with the updated choices and default value for the file dropdown.
    """
    files = _list_output_files(folder)
    return gr.update(choices=files, value=files[-1] if files else None)


def _auto_refresh_review_list(current_value: str | None) -> gr.update:
    """Auto-refresh review dropdown preserving current selection when possible.

    Args:
        current_value: The currently selected review file path (if any).

    Returns:
        A gr.update object with the updated choices and value for the review dropdown.
    """
    files = list_review_files()

    value = current_value if current_value in files else (files[-1] if files else None)

    return gr.update(choices=files, value=value)


def _auto_refresh_view_file_list(folder: str, current_value: str | None) -> gr.update:
    """Auto-refresh view file dropdown preserving current selection when possible.

    Args:
        folder: The selected folder, expected to be "plans" or "reviews".
        current_value: The currently selected file path (if any).

    Returns:
        A gr.update object with the updated choices and value for the view file dropdown.
    """
    files = _list_output_files(folder)

    value = current_value if current_value in files else (files[-1] if files else None)

    return gr.update(choices=files, value=value)


# ═══════════════════════════════════════════════════════════════════════════
# Build Gradio App
# ═══════════════════════════════════════════════════════════════════════════


def build_app() -> gr.Blocks:
    """Build the Gradio app with multiple tabs for planning, writing, and reviewing."""
    with gr.Blocks(title="Literature Review Agent") as demo:
        # Global periodic auto-refresh (avoids restart when files are added manually)
        auto_refresh_timer = gr.Timer(value=3.0)

        # ── Header ─────────────────────────────────────────────────────────
        gr.HTML("""
            <div id="app-header">
              <h1>🔬 Literature Review Agent</h1>
              <p>Plan, write and manage academic and technical reviews with AI</p>
            </div>
            """)

        with gr.Row(elem_id="llm-switch-bar"):
            llm_provider_selector = gr.Dropdown(
                label="LLM Provider (global)",
                choices=list_llm_providers(),
                value=get_current_llm_provider(),
                allow_custom_value=False,
                scale=1,
            )
            llm_provider_status = gr.Textbox(
                label="Provider Status",
                value=get_llm_provider_status(),
                interactive=False,
                elem_classes="status-bar",
                scale=2,
            )

        llm_provider_selector.change(
            fn=set_llm_provider,
            inputs=[llm_provider_selector],
            outputs=[llm_provider_selector, llm_provider_status],
        )

        # ══════════════════════════════════════════════════════════════════
        # TAB 1 — Plan Review
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("📋 Plan"):
            gr.Markdown(
                "### Plan Literature Review\n"
                "The agent will ask refinement questions to improve the plan. "
                "Answer in the field below and click **Reply**."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    plan_tema = gr.Textbox(
                        label="Topic",
                        placeholder="e.g.: Streamflow forecasting with deep learning models",
                        lines=2,
                    )
                    plan_tipo = gr.Radio(
                        label="Review Type",
                        choices=[
                            ("Academic (literature narrative)", "academico"),
                            ("Technical (didactic chapter)", "tecnico"),
                            ("Both", "ambos"),
                        ],
                        value="academico",
                    )
                    plan_rodadas = gr.Slider(
                        label="Refinement rounds",
                        minimum=1,
                        maximum=6,
                        step=1,
                        value=3,
                    )
                    plan_start_btn = gr.Button("🚀 Start Planning", variant="primary")

                with gr.Column(scale=2):
                    plan_chatbot = gr.Chatbot(
                        label="Agent Conversation",
                        height=350,
                        layout="bubble",
                        buttons=["copy"],
                    )
                    plan_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        elem_classes="status-bar",
                    )
                    plan_user_input = gr.Textbox(
                        label="Your answer",
                        placeholder="Answer the agent's question…",
                        lines=3,
                        visible=False,
                    )
                    plan_reply_btn = gr.Button("💬 Reply", variant="secondary", visible=False)
                    plan_rendered = gr.Markdown(
                        label="Generated Plan",
                        value="*(the plan will appear here when planning is complete)*",
                        height=300,
                        visible=False,
                    )

            # Persistent state for the LangGraph session
            plan_session = gr.State({})

            # ── Wire up ──────────────────────────────────────────────────

            def _on_start(tema, tipo, rodadas):
                history, state, status, rendered = start_planning(tema, tipo, int(rodadas))
                has_session = bool(state)
                has_rendered = bool(rendered)
                return (
                    history,
                    state,
                    status,
                    gr.update(visible=has_session),
                    gr.update(visible=has_session),
                    gr.update(value=""),
                    gr.update(value=rendered, visible=has_rendered),
                )

            plan_start_btn.click(
                fn=_on_start,
                inputs=[plan_tema, plan_tipo, plan_rodadas],
                outputs=[
                    plan_chatbot,
                    plan_session,
                    plan_status,
                    plan_user_input,
                    plan_reply_btn,
                    plan_user_input,
                    plan_rendered,
                ],
            )

            def _on_reply(user_msg, history, state):
                history, state, status, rendered = continue_planning(user_msg, history, state)
                has_session = bool(state)
                has_rendered = bool(rendered)
                return (
                    history,
                    state,
                    status,
                    gr.update(visible=has_session),
                    gr.update(visible=has_session),
                    gr.update(value=""),
                    gr.update(value=rendered, visible=has_rendered),
                )

            plan_reply_btn.click(
                fn=_on_reply,
                inputs=[plan_user_input, plan_chatbot, plan_session],
                outputs=[
                    plan_chatbot,
                    plan_session,
                    plan_status,
                    plan_user_input,
                    plan_reply_btn,
                    plan_user_input,
                    plan_rendered,
                ],
            )
            plan_user_input.submit(
                fn=_on_reply,
                inputs=[plan_user_input, plan_chatbot, plan_session],
                outputs=[
                    plan_chatbot,
                    plan_session,
                    plan_status,
                    plan_user_input,
                    plan_reply_btn,
                    plan_user_input,
                    plan_rendered,
                ],
            )

        # ══════════════════════════════════════════════════════════════════
        # TAB 2 — Write
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("✍️ Write"):
            gr.Markdown(
                "### Run Writing from a Plan\n"
                "Select an existing plan (generated in the **Plan** tab) and configure the writing options."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    write_mode = gr.Radio(
                        label="Writing Mode",
                        choices=["Technical", "Academic"],
                        value="Technical",
                    )
                    write_plan = gr.Dropdown(
                        label="Plan (.md)",
                        choices=list_plan_files("Technical"),
                        allow_custom_value=True,
                    )
                    write_lang = gr.Radio(
                        label="Language",
                        choices=[("Português (pt-BR)", "pt"), ("English", "en")],
                        value="pt",
                    )
                    write_min_src = gr.Slider(
                        label="Min. sources per section",
                        minimum=0,
                        maximum=10,
                        step=1,
                        value=0,
                    )
                    write_tavily = gr.Checkbox(
                        label="Allow web search (Tavily)",
                        value=False,
                    )
                    write_start_btn = gr.Button("✍️ Start Writing", variant="primary")

                with gr.Column(scale=2):
                    write_chatbot = gr.Chatbot(
                        label="Writing Progress",
                        height=360,
                        layout="bubble",
                        buttons=["copy", "copy_all"],
                    )
                    write_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        elem_classes="status-bar",
                    )
                    write_rendered = gr.Markdown(
                        label="Generated Document",
                        value="*(the document will appear here when writing is complete)*",
                        height=360,
                    )

            # Refresh plan list when mode changes
            write_mode.change(
                fn=refresh_plan_list,
                inputs=[write_mode],
                outputs=[write_plan],
            )

            auto_refresh_timer.tick(
                fn=_auto_refresh_plan_list,
                inputs=[write_mode, write_plan],
                outputs=[write_plan],
            )

            def _on_write(
                plan: str,
                mode: str,
                lang: str,
                min_src: int,
                tavily: bool,
                history: list,
            ) -> tuple:
                """
                Delegate the writing process to the start_writing generator.

                This function acts as a bridge, streaming the writing progress,
                intermediate states, and the final rendered output back to the UI.

                Args:
                    plan (str): The structured plan or theme to be processed.
                    mode (str): The writing mode (e.g., 'academic' or 'technical').
                    lang (str): The target language for the output (e.g., 'pt', 'en').
                    min_src (int): Minimum number of sources required for validation.
                    tavily (bool): Whether to enable web search via Tavily.
                    history (list): The conversation or state history for context.

                Yields:
                    tuple: A tuple containing (history, state, rendered_markdown)
                        at each step of the generation process.
                """
                yield from start_writing(plan, mode, lang, min_src, tavily, history)

            write_start_btn.click(
                fn=_on_write,
                inputs=[
                    write_plan,
                    write_mode,
                    write_lang,
                    write_min_src,
                    write_tavily,
                    write_chatbot,
                ],
                outputs=[write_chatbot, write_status, write_rendered],
            )

        # ══════════════════════════════════════════════════════════════════
        # TAB 3 — Revisão Interativa
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("🤖 Revisão Interativa"):
            gr.Markdown(
                "### Review Document with Chat\n"
                "Select a review in **reviews/**. The system creates a working copy and "
                "only applies changes after explicit confirmation.\n"
                "For reference commands (list/format), the assistant now always asks for confirmation before execution."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    review_file = gr.Dropdown(
                        label="Review file",
                        choices=list_review_files(),
                        allow_custom_value=False,
                    )
                    review_refresh = gr.Button("🔄 Refresh files", size="sm")
                    review_start = gr.Button("▶ Start session", variant="primary")
                    review_web_toggle = gr.Checkbox(
                        label="🌐 Allow web search",
                        value=False,
                        info="Enable this so the agent can search for papers online.",
                    )
                    review_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        elem_classes="status-bar",
                    )
                    review_confirm = gr.Button("✅ Confirm Edit", variant="secondary")
                    review_cancel = gr.Button("🗑️ Cancel Edit", variant="secondary")

                with gr.Column(scale=2):
                    review_chatbot = gr.Chatbot(
                        label="Review Assistant",
                        height=360,
                        layout="bubble",
                        buttons=["copy", "copy_all"],
                    )
                    review_input = gr.Textbox(
                        label="Your question/command",
                        placeholder="Ex.: what are the papers cited in section 2?",
                        lines=3,
                    )
                    review_send = gr.Button("💬 Send", variant="primary")

                with gr.Column(scale=2):
                    review_preview = gr.Textbox(
                        label="Working document (editable)",
                        value="",
                        placeholder="Start the session to load the document...",
                        lines=22,
                        max_lines=50,
                        interactive=True,
                    )
                    review_save = gr.Button("💾 Save manual edit", variant="secondary")

            review_session = gr.State({})

            review_refresh.click(
                fn=lambda: gr.update(choices=list_review_files(), value=None),
                inputs=[],
                outputs=[review_file],
            )

            auto_refresh_timer.tick(
                fn=_auto_refresh_review_list,
                inputs=[review_file],
                outputs=[review_file],
            )

            review_start.click(
                fn=start_review_session,
                inputs=[review_file, review_chatbot, review_session],
                outputs=[review_chatbot, review_session, review_status, review_preview],
            )

            def _on_review_send(user_msg, chatbot, session, web_toggle):
                history, session_state, status, preview = review_chat_turn(
                    user_msg, chatbot, session, web_toggle
                )
                return history, session_state, status, preview, gr.update(value="")

            review_send.click(
                fn=_on_review_send,
                inputs=[
                    review_input,
                    review_chatbot,
                    review_session,
                    review_web_toggle,
                ],
                outputs=[
                    review_chatbot,
                    review_session,
                    review_status,
                    review_preview,
                    review_input,
                ],
            )
            review_input.submit(
                fn=_on_review_send,
                inputs=[
                    review_input,
                    review_chatbot,
                    review_session,
                    review_web_toggle,
                ],
                outputs=[
                    review_chatbot,
                    review_session,
                    review_status,
                    review_preview,
                    review_input,
                ],
            )

            review_confirm.click(
                fn=confirm_review_edit,
                inputs=[review_chatbot, review_session],
                outputs=[review_chatbot, review_session, review_status, review_preview],
            )

            review_cancel.click(
                fn=cancel_review_edit,
                inputs=[review_chatbot, review_session],
                outputs=[review_chatbot, review_session, review_status, review_preview],
            )

            review_save.click(
                fn=save_review_manual_edit,
                inputs=[review_preview, review_chatbot, review_session],
                outputs=[review_chatbot, review_session, review_status, review_preview],
            )

        # ══════════════════════════════════════════════════════════════════
        # TAB 5 — View (rendered output viewer)
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("📄 View"):
            gr.Markdown(
                "### View Generated Files\n"
                "Select the folder and file to render its Markdown content."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    view_folder = gr.Radio(
                        label="Folder",
                        choices=[("📋 Plans", "plans"), ("📝 Reviews", "reviews")],
                        value="reviews",
                    )
                    initial_review_files = _list_output_files("reviews")
                    view_file = gr.Dropdown(
                        label="File",
                        choices=initial_review_files,
                        value=(initial_review_files[-1] if initial_review_files else None),
                        allow_custom_value=True,
                    )
                    view_refresh_btn = gr.Button("🔄 Refresh list", size="sm")
                    view_load_btn = gr.Button("👁️ View", variant="primary")

                with gr.Column(scale=3):
                    view_output = gr.Markdown(
                        value="*(select a file and click View)*",
                        label="Content",
                        height=620,
                    )

            # Wire up folder change → refresh file list
            view_folder.change(
                fn=_refresh_file_list,
                inputs=[view_folder],
                outputs=[view_file],
            )

            auto_refresh_timer.tick(
                fn=_auto_refresh_view_file_list,
                inputs=[view_folder, view_file],
                outputs=[view_file],
            )
            # Refresh button → re-scan folder
            view_refresh_btn.click(
                fn=_refresh_file_list,
                inputs=[view_folder],
                outputs=[view_file],
            )
            # Load button → render file
            view_load_btn.click(
                fn=_load_file,
                inputs=[view_file],
                outputs=[view_output],
            )
            # Also auto-load on dropdown selection change
            view_file.change(
                fn=_load_file,
                inputs=[view_file],
                outputs=[view_output],
            )

        # ══════════════════════════════════════════════════════════════════
        # TAB 3 — Index PDFs
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("📁 Index PDFs"):
            gr.Markdown(
                "### Index Local PDFs\n"
                "Vectorizes the PDFs in the indicated folder and saves the chunks in MongoDB for use in the corpus."
            )

            with gr.Row(), gr.Column():
                idx_folder = gr.Textbox(
                    label="Path to PDF folder",
                    placeholder="e.g.: /home/user/articles  or  ~/papers",
                    lines=1,
                )
                idx_btn = gr.Button("📂 Index", variant="primary")
                idx_result = gr.Markdown(label="Resultado")

            idx_btn.click(
                fn=index_pdfs,
                inputs=[idx_folder],
                outputs=[idx_result],
            )

        # ══════════════════════════════════════════════════════════════════
        # TAB 4 — Format References
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("📚 References"):
            gr.Markdown(
                "### Format Reference List\n"
                "Upload a **YAML** or **JSON** file with the references and the desired format "
                "(abnt, apa, ieee, vancouver, mla, chicago). "
                "See the example at `references/example_references.yaml`."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    ref_file = gr.File(
                        label="Arquivo YAML / JSON",
                        file_types=[".yaml", ".yml", ".json"],
                    )
                    ref_tavily = gr.Checkbox(
                        label="Allow web search (Tavily) to resolve metadata",
                        value=False,
                    )
                    ref_output_dir = gr.Textbox(
                        label="Output folder (optional)",
                        placeholder="e.g.: references/output",
                        lines=1,
                    )
                    ref_btn = gr.Button("📚 Format References", variant="primary")
                    ref_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        elem_classes="status-bar",
                    )

                with gr.Column(scale=2):
                    ref_output = gr.Markdown(
                        label="Formatted Result",
                        value="*(the result will appear here)*",
                    )

            ref_btn.click(
                fn=format_references,
                inputs=[ref_file, ref_tavily, ref_output_dir],
                outputs=[ref_output, ref_status],
            )

        # ── Footer ─────────────────────────────────────────────────────────
        gr.HTML(
            "<div style='text-align:center; color:#9ca3af; font-size:0.8rem; padding:1rem 0'>"
            "Literature Review Agent · Powered by LangGraph + Gradio"
            "</div>"
        )

    return demo


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════


def main(share: bool = False, port: int = 7860):
    """Launch the Gradio app.

    Args:
        share: Whether to create a public share link (useful when running in a remote environment)
        port: The local port to serve the app on (default: 7860)
    """
    os.makedirs("plans", exist_ok=True)
    os.makedirs("reviews", exist_ok=True)
    demo = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        show_error=True,
        css=_CSS,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    )


if __name__ == "__main__":
    main()
