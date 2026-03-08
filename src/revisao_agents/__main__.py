import sys
import glob
import os

from .state import RevisaoState, EscritaTecnicaState
from .workflows import build_academico_workflow, build_tecnico_workflow
from .workflows.technical_writing_workflow import build_workflow as build_escrita_workflow
from .hitl import run_hitl_loop
from .utils.pdf_ingestor import ingest_pdf_folder
from .core.schemas.writer_config import WriterConfig


def main():
    # Garante que os diretórios de saída existem
    os.makedirs("plans", exist_ok=True)
    os.makedirs("reviews", exist_ok=True)

    print("\n" + "=" * 70)
    print("AGENTE DE PLANEJAMENTO DA REVISAO DA LITERATURA")
    print("=" * 70)
    print("\nOpções:")
    print("  [1] Planejar Revisão Acadêmica (narrativa)")
    print("  [2] Planejar Revisão Técnica (capítulo)")
    print("  [3] Executar Escrita a partir de plano existente (Técnica ou Acadêmica)")
    print("  [4] Indexar PDFs locais → vetorizar e salvar no MongoDB")
    escolha = input("\nEscolha [1/2/3/4]: ").strip()

    if escolha == "4":
        print("\n" + "=" * 70)
        print("INDEXAR PDFs LOCAIS")
        print("=" * 70)
        pasta = input("\nCaminho da pasta com PDFs: ").strip()
        if not pasta:
            print("❌ Caminho vazio.")
            return
        pasta = os.path.expanduser(pasta)
        if not os.path.isdir(pasta):
            print(f"❌ Pasta não encontrada: {pasta}")
            return
        resultado = ingest_pdf_folder(pasta)
        print("\n" + "=" * 70)
        print("RESULTADO DA INDEXAÇÃO")
        print("=" * 70)
        print(f"  ✅ Novos PDFs indexados : {resultado['indexed']}")
        print(f"  ⏭️  Já no banco          : {resultado['already']}")
        print(f"  ⚠️  Texto insuficiente  : {resultado['skipped']}")
        print(f"  ❌ Erros de leitura     : {resultado['errors']}")
        print(f"  📦 Chunks inseridos     : {resultado['total_chunks']}")
        print("=" * 70)
        return

    if escolha == "3":
        # --- Writing mode sub-menu ---
        print("\n" + "-" * 70)
        print("ESTILO DE ESCRITA:")
        print("  [a] Técnica   — capítulo didático (busca web + MongoDB)")
        print("  [b] Acadêmica — revisão narrativa da literatura (corpus-first)")
        escolha_modo = input("\nEscolha [a/b, padrão=a]: ").strip().lower() or "a"
        if escolha_modo == "b":
            writer_config = WriterConfig.academic()
            glob_pattern_primary = "plans/plano_revisao_*.md"
            mode_label = "ACADÊMICA"
        else:
            writer_config = WriterConfig.technical()
            glob_pattern_primary = "plans/plano_revisao_tecnica_*.md"
            mode_label = "TÉCNICA"

        # --- Tavily search option ---
        print("\n" + "-" * 70)
        print("Deseja permitir busca web/imagens via Tavily?")
        print("  [y] Sim (busca web e imagens)")
        print("  [n] Não (apenas corpus local)")
        tavily_opt = input("\nPermitir Tavily? [y/N]: ").strip().lower() or "n"
        tavily_enabled = tavily_opt == "y"

        print(f"\n" + "=" * 70)
        print(f"EXECUÇÃO DE ESCRITA {mode_label}")
        print("=" * 70)

        # --- Find plan files ---
        planos = sorted(glob.glob(glob_pattern_primary))
        if not planos:
            planos = sorted(glob.glob("plans/plano_revisao_*.md"))  # broader fallback
        if not planos:
            planos = sorted(glob.glob("plano_revisao_*.md"))  # root fallback
        if planos:
            print("\nPlanos encontrados:")
            for i, p in enumerate(planos, 1):
                print(f"  [{i}] {p}")
            idx = input(f"\n👤 Escolha [1-{len(planos)} ou caminho]: ").strip()
            if idx.isdigit() and 1 <= int(idx) <= len(planos):
                caminho = planos[int(idx) - 1]
            else:
                caminho = idx
        else:
            caminho = input("\n👤 Caminho do plano (.md): ").strip()

        if not os.path.exists(caminho):
            print(f"❌ Arquivo não encontrado: {caminho}")
            return

        state_init: EscritaTecnicaState = {
            "tema": "",
            "resumo_plano": "",
            "secoes": [],
            "caminho_plano": caminho,
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
        try:
            for event in app.stream(state_init):
                node = list(event.keys())[0] if event else "?"
                if node != "__end__":
                    st = event.get(node, {}).get("status", "")
                    if st:
                        print(f"\n   ▶ [{node}] → {st}")
            print("\n" + "=" * 70)
            print("✅ REVISÃO TÉCNICA CONCLUÍDA")
            print("=" * 70)
        except KeyboardInterrupt:
            print("\nCancelado.")
        return

    # --- Planejamento (opções 1 e 2) ---
    tema = input("\nTema da revisao: ").strip()
    if not tema:
        print("Tema vazio.")
        return

    print("\n[1] Revisao Academica (narrativa da literatura) — busca no corpus MongoDB")
    print("[2] Revisao Tecnica   (capitulo didatico/detalhado) — busca na internet")
    print("[3] Ambas")
    while True:
        e = escolha if escolha in ("1", "2", "3") else input("\nEscolha [1/2/3]: ").strip()
        if e in ("1", "2", "3"):
            break
        escolha = ""  # limpa para pedir novamente
        print("Digite 1, 2 ou 3.")

    tipos = {"1": ["academico"], "2": ["tecnico"], "3": ["academico", "tecnico"]}[e]

    max_p = 3
    try:
        n = input("Rodadas de refinamento por plano [padrao 3]: ").strip()
        if n.isdigit() and int(n) > 0:
            max_p = min(int(n), 6)
    except Exception:
        pass

    for tipo in tipos:
        label = "ACADEMICA" if tipo == "academico" else "TECNICA"
        print("\n" + "=" * 70)
        print(f"Iniciando: REVISAO {label} | {repr(tema)} | {max_p} rodadas")
        print("-" * 70)

        state_init: RevisaoState = {
            "tema":                   tema,
            "tipo_revisao":           tipo,
            "chunks_relevantes":      [],
            "snippets_tecnicos":      [],
            "urls_tecnicos":          [],
            "plano_atual":            "",
            "historico_entrevista":   [],
            "perguntas_feitas":       0,
            "max_perguntas":          max_p,
            "plano_final":            "",
            "plano_final_path":       "",
            "status":                 "iniciando",
        }
        config = {"configurable": {"thread_id": f"revisao_{tipo}_{tema[:20]}"}}

        if tipo == "academico":
            app = build_academico_workflow()
        else:
            app = build_tecnico_workflow()

        try:
            run_hitl_loop(app, config, state_init)
        except KeyboardInterrupt:
            print("\nCancelado.")
            break
        except Exception as ex:
            import traceback
            print("\nErro:", str(ex))
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Sessao de planejamento concluida.")
    print("=" * 70)


if __name__ == "__main__":
    main()