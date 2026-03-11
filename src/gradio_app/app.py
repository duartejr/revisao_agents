"""
app.py — Gradio-based chatbot UI for the revisao_agents project.

Provides a clean, ChatGPT-style web interface for all five workflow options:

  Tab 1  📋 Planejar   — Plan a literature review (Academic / Technical)
  Tab 2  ✍️  Escrever   — Execute writing from an existing plan file
  Tab 3  📁 Indexar    — Index local PDFs into the MongoDB vector store
  Tab 4  📚 Referências — Format a reference list from a YAML/JSON file

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
    start_planning,
    continue_planning,
    list_plan_files,
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
# Build Gradio App
# ═══════════════════════════════════════════════════════════════════════════

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="Agente de Revisão da Literatura",
        css=_CSS,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    ) as demo:

        # ── Header ─────────────────────────────────────────────────────────
        gr.HTML(
            """
            <div id="app-header">
              <h1>🔬 Agente de Revisão da Literatura</h1>
              <p>Planeje, escreva e gerencie revisões académicas e técnicas com IA</p>
            </div>
            """
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
                        height=420,
                        show_copy_button=True,
                        bubble_full_width=False,
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

            # Persistent state for the LangGraph session
            plan_session = gr.State({})

            # ── Wire up ──────────────────────────────────────────────────

            def _on_start(tema, tipo, rodadas):
                history, state, status = start_planning(tema, tipo, int(rodadas))
                has_session = bool(state)
                return (
                    history,
                    state,
                    status,
                    gr.update(visible=has_session),  # user input textbox
                    gr.update(visible=has_session),  # reply button
                    gr.update(value=""),             # clear user input
                )

            plan_start_btn.click(
                fn=_on_start,
                inputs=[plan_tema, plan_tipo, plan_rodadas],
                outputs=[plan_chatbot, plan_session, plan_status, plan_user_input, plan_reply_btn, plan_user_input],
            )

            def _on_reply(user_msg, history, state):
                history, state, status = continue_planning(user_msg, history, state)
                has_session = bool(state)
                return (
                    history,
                    state,
                    status,
                    gr.update(visible=has_session),
                    gr.update(visible=has_session),
                    gr.update(value=""),
                )

            plan_reply_btn.click(
                fn=_on_reply,
                inputs=[plan_user_input, plan_chatbot, plan_session],
                outputs=[plan_chatbot, plan_session, plan_status, plan_user_input, plan_reply_btn, plan_user_input],
            )
            plan_user_input.submit(
                fn=_on_reply,
                inputs=[plan_user_input, plan_chatbot, plan_session],
                outputs=[plan_chatbot, plan_session, plan_status, plan_user_input, plan_reply_btn, plan_user_input],
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
                        height=480,
                        show_copy_button=True,
                        bubble_full_width=False,
                    )
                    write_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        elem_classes="status-bar",
                    )

            # Refresh plan list when mode changes
            write_mode.change(
                fn=refresh_plan_list,
                inputs=[write_mode],
                outputs=[write_plan],
            )

            def _on_write(plan, mode, lang, min_src, tavily, history):
                for h, s in start_writing(plan, mode, lang, min_src, tavily, history):
                    yield h, s

            write_start_btn.click(
                fn=_on_write,
                inputs=[write_plan, write_mode, write_lang, write_min_src, write_tavily, write_chatbot],
                outputs=[write_chatbot, write_status],
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
    )


if __name__ == "__main__":
    main()
