"""
app.py — Gradio-based chatbot UI for the revisao_agents project.

Provides a clean, ChatGPT-style web interface for all workflow options:

  Tab 1  📋 Planejar    — Plan a literature review (Academic / Technical)
  Tab 2  ✍️  Escrever    — Execute writing from an existing plan file
  Tab 3  📁 Indexar     — Index local PDFs into the MongoDB vector store
  Tab 4  📚 Referências — Format a reference list from a YAML/JSON file
  Tab 5  📄 Visualizar  — Browse and render any generated plan or review file

Run via:  python run_ui.py
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# sys.path bootstrap — needed when launched from the project root
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.join(_HERE, "..")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import gradio as gr

from gradio_app.handlers import (
    cancel_review_edit,
    confirm_review_edit,
    get_current_llm_provider,
    get_llm_provider_status,
    save_review_manual_edit,
    list_llm_providers,
    start_planning,
    start_review_session,
    continue_planning,
    list_review_files,
    list_plan_files,
    review_chat_turn,
    set_llm_provider,
    start_writing,
    index_pdfs,
    format_references,
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
    files = list_plan_files(mode)
    return gr.update(choices=files, value=files[0] if files else None)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers: list and load output files for the Visualizar tab
# ═══════════════════════════════════════════════════════════════════════════

def _list_output_files(folder: str) -> list[str]:
    """Return sorted .md files from plans/ or reviews/."""
    os.makedirs(folder, exist_ok=True)
    files = sorted(f for f in os.listdir(folder) if f.endswith(".md"))
    return [os.path.join(folder, f) for f in files] if files else []


def _load_file(path: str) -> str:
    """Read and return the markdown content of a file."""
    if not path or not os.path.exists(path):
        return "*(arquivo não encontrado)*"
    try:
        return open(path, encoding="utf-8").read()
    except Exception as exc:
        return f"❌ Erro ao ler arquivo: {exc}"


def _refresh_file_list(folder: str) -> gr.update:
    files = _list_output_files(folder)
    return gr.update(choices=files, value=files[-1] if files else None)


# ═══════════════════════════════════════════════════════════════════════════
# Build Gradio App
# ═══════════════════════════════════════════════════════════════════════════

def build_app() -> gr.Blocks:
    with gr.Blocks(title="Agente de Revisão da Literatura") as demo:

        # ── Header ─────────────────────────────────────────────────────────
        gr.HTML(
            """
            <div id="app-header">
              <h1>🔬 Agente de Revisão da Literatura</h1>
              <p>Planeje, escreva e gerencie revisões académicas e técnicas com IA</p>
            </div>
            """
        )

        with gr.Row(elem_id="llm-switch-bar"):
            llm_provider_selector = gr.Dropdown(
                label="LLM Provider (global)",
                choices=list_llm_providers(),
                value=get_current_llm_provider(),
                allow_custom_value=False,
                scale=1,
            )
            llm_provider_status = gr.Textbox(
                label="Status do Provider",
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
        # TAB 1 — Planejar Revisão
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("📋 Planejar"):
            gr.Markdown(
                "### Planejar Revisão da Literatura\n"
                "O agente fará perguntas de refinamento para melhorar o plano. "
                "Responda no campo abaixo e clique em **Responder**."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    plan_tema = gr.Textbox(
                        label="Tema",
                        placeholder="Ex.: Previsão de vazão com modelos de aprendizado profundo",
                        lines=2,
                    )
                    plan_tipo = gr.Radio(
                        label="Tipo de Revisão",
                        choices=[
                            ("Acadêmica (narrativa da literatura)", "academico"),
                            ("Técnica (capítulo didático)", "tecnico"),
                            ("Ambas", "ambos"),
                        ],
                        value="academico",
                    )
                    plan_rodadas = gr.Slider(
                        label="Rodadas de refinamento",
                        minimum=1, maximum=6, step=1, value=3,
                    )
                    plan_start_btn = gr.Button("🚀 Iniciar Planejamento", variant="primary")

                with gr.Column(scale=2):
                    plan_chatbot = gr.Chatbot(
                        label="Conversa com o Agente",
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
                        label="Sua resposta",
                        placeholder="Responda a pergunta do agente…",
                        lines=3,
                        visible=False,
                    )
                    plan_reply_btn = gr.Button("💬 Responder", variant="secondary", visible=False)
                    plan_rendered = gr.Markdown(
                        label="Plano Gerado",
                        value="*(o plano aparecerá aqui quando o planejamento for concluído)*",
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
                outputs=[plan_chatbot, plan_session, plan_status, plan_user_input, plan_reply_btn, plan_user_input, plan_rendered],
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
                outputs=[plan_chatbot, plan_session, plan_status, plan_user_input, plan_reply_btn, plan_user_input, plan_rendered],
            )
            plan_user_input.submit(
                fn=_on_reply,
                inputs=[plan_user_input, plan_chatbot, plan_session],
                outputs=[plan_chatbot, plan_session, plan_status, plan_user_input, plan_reply_btn, plan_user_input, plan_rendered],
            )

        # ══════════════════════════════════════════════════════════════════
        # TAB 2 — Escrever
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("✍️ Escrever"):
            gr.Markdown(
                "### Executar Escrita a partir de um Plano\n"
                "Selecione um plano existente (gerado na aba **Planejar**) e configure as opções de escrita."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    write_mode = gr.Radio(
                        label="Modo de Escrita",
                        choices=["Técnica", "Acadêmica"],
                        value="Técnica",
                    )
                    write_plan = gr.Dropdown(
                        label="Plano (.md)",
                        choices=list_plan_files("Técnica"),
                        allow_custom_value=True,
                    )
                    write_lang = gr.Radio(
                        label="Idioma",
                        choices=[("Português (pt-BR)", "pt"), ("English", "en")],
                        value="pt",
                    )
                    write_min_src = gr.Slider(
                        label="Mínimo de fontes por seção",
                        minimum=0, maximum=10, step=1, value=0,
                    )
                    write_tavily = gr.Checkbox(
                        label="Permitir busca web (Tavily)",
                        value=False,
                    )
                    write_start_btn = gr.Button("✍️ Iniciar Escrita", variant="primary")

                with gr.Column(scale=2):
                    write_chatbot = gr.Chatbot(
                        label="Progresso da Escrita",
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
                        label="Documento Gerado",
                        value="*(o documento aparecerá aqui quando a escrita for concluída)*",
                        height=360,
                    )

            # Refresh plan list when mode changes
            write_mode.change(
                fn=refresh_plan_list,
                inputs=[write_mode],
                outputs=[write_plan],
            )

            def _on_write(plan, mode, lang, min_src, tavily, history):
                for h, s, rendered in start_writing(plan, mode, lang, min_src, tavily, history):
                    yield h, s, rendered

            write_start_btn.click(
                fn=_on_write,
                inputs=[write_plan, write_mode, write_lang, write_min_src, write_tavily, write_chatbot],
                outputs=[write_chatbot, write_status, write_rendered],
            )

        # ══════════════════════════════════════════════════════════════════
        # TAB 3 — Revisão Interativa
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("🤖 Revisão Interativa"):
            gr.Markdown(
                "### Revisar Documento com Chat\n"
                "Selecione uma revisão em **reviews/**. O sistema cria uma cópia de trabalho e "
                "só aplica mudanças após confirmação explícita."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    review_file = gr.Dropdown(
                        label="Arquivo de revisão",
                        choices=list_review_files(),
                        allow_custom_value=False,
                    )
                    review_refresh = gr.Button("🔄 Atualizar arquivos", size="sm")
                    review_start = gr.Button("▶ Iniciar sessão", variant="primary")
                    review_web_toggle = gr.Checkbox(
                        label="🌐 Permitir busca na internet",
                        value=False,
                        info="Ative para que o agente possa pesquisar artigos online.",
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
                        label="Sua pergunta/comando",
                        placeholder="Ex.: what are the papers cited in section 2?",
                        lines=3,
                    )
                    review_send = gr.Button("💬 Enviar", variant="primary")

                with gr.Column(scale=2):
                    review_preview = gr.Textbox(
                        label="Documento de trabalho (editável)",
                        value="",
                        placeholder="Inicie a sessão para carregar o documento...",
                        lines=22,
                        max_lines=50,
                        interactive=True,
                    )
                    review_save = gr.Button("💾 Salvar edição manual", variant="secondary")

            review_session = gr.State({})

            review_refresh.click(
                fn=lambda: gr.update(choices=list_review_files(), value=None),
                inputs=[],
                outputs=[review_file],
            )

            review_start.click(
                fn=start_review_session,
                inputs=[review_file, review_chatbot, review_session],
                outputs=[review_chatbot, review_session, review_status, review_preview],
            )

            review_send.click(
                fn=review_chat_turn,
                inputs=[review_input, review_chatbot, review_session, review_web_toggle],
                outputs=[review_chatbot, review_session, review_status, review_preview],
            )
            review_input.submit(
                fn=review_chat_turn,
                inputs=[review_input, review_chatbot, review_session, review_web_toggle],
                outputs=[review_chatbot, review_session, review_status, review_preview],
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
        # TAB 5 — Visualizar (rendered output viewer)
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("📄 Visualizar"):
            gr.Markdown(
                "### Visualizar Arquivos Gerados\n"
                "Selecione a pasta e o arquivo para renderizar seu conteúdo em Markdown."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    view_folder = gr.Radio(
                        label="Pasta",
                        choices=[("📋 Planos", "plans"), ("📝 Revisões", "reviews")],
                        value="reviews",
                    )
                    initial_review_files = _list_output_files("reviews")
                    view_file = gr.Dropdown(
                        label="Arquivo",
                        choices=initial_review_files,
                        value=initial_review_files[-1] if initial_review_files else None,
                        allow_custom_value=True,
                    )
                    view_refresh_btn = gr.Button("🔄 Atualizar lista", size="sm")
                    view_load_btn = gr.Button("👁️ Visualizar", variant="primary")

                with gr.Column(scale=3):
                    view_output = gr.Markdown(
                        value="*(selecione um arquivo e clique em Visualizar)*",
                        label="Conteúdo",
                        height=620,
                    )

            # Wire up folder change → refresh file list
            view_folder.change(
                fn=_refresh_file_list,
                inputs=[view_folder],
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
        # TAB 3 — Indexar PDFs
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("📁 Indexar PDFs"):
            gr.Markdown(
                "### Indexar PDFs Locais\n"
                "Vectoriza os PDFs da pasta indicada e salva os chunks no MongoDB para uso no corpus."
            )

            with gr.Row():
                with gr.Column():
                    idx_folder = gr.Textbox(
                        label="Caminho da pasta com PDFs",
                        placeholder="Ex.: /home/user/artigos  ou  ~/papers",
                        lines=1,
                    )
                    idx_btn = gr.Button("📂 Indexar", variant="primary")
                    idx_result = gr.Markdown(label="Resultado")

            idx_btn.click(
                fn=index_pdfs,
                inputs=[idx_folder],
                outputs=[idx_result],
            )

        # ══════════════════════════════════════════════════════════════════
        # TAB 4 — Formatar Referências
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("📚 Referências"):
            gr.Markdown(
                "### Formatar Lista de Referências\n"
                "Faça upload de um arquivo **YAML** ou **JSON** com as referências e o padrão desejado "
                "(abnt, apa, ieee, vancouver, mla, chicago). "
                "Veja o exemplo em `references/example_references.yaml`."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    ref_file = gr.File(
                        label="Arquivo YAML / JSON",
                        file_types=[".yaml", ".yml", ".json"],
                    )
                    ref_tavily = gr.Checkbox(
                        label="Permitir busca web (Tavily) para resolver metadados",
                        value=False,
                    )
                    ref_output_dir = gr.Textbox(
                        label="Pasta de saída (opcional)",
                        placeholder="Ex.: references/output",
                        lines=1,
                    )
                    ref_btn = gr.Button("📚 Formatar Referências", variant="primary")
                    ref_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        elem_classes="status-bar",
                    )

                with gr.Column(scale=2):
                    ref_output = gr.Markdown(
                        label="Resultado Formatado",
                        value="*(o resultado aparecerá aqui)*",
                    )

            ref_btn.click(
                fn=format_references,
                inputs=[ref_file, ref_tavily, ref_output_dir],
                outputs=[ref_output, ref_status],
            )

        # ── Footer ─────────────────────────────────────────────────────────
        gr.HTML(
            "<div style='text-align:center; color:#9ca3af; font-size:0.8rem; padding:1rem 0'>"
            "Agente de Revisão da Literatura · Powered by LangGraph + Gradio"
            "</div>"
        )

    return demo


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main(share: bool = False, port: int = 7860):
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
