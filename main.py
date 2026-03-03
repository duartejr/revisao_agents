import sys
import glob
import os
from state import RevisaoState, EscritaTecnicaState
from workflows import build_academico_workflow, build_tecnico_workflow
from workflows.technical_writing_workflow import build_workflow as build_escrita_workflow
from hitl import run_hitl_loop

def main():
    print("\n" + "=" * 70)
    print("AGENTE DE PLANEJAMENTO DA REVISAO DA LITERATURA")
    print("=" * 70)
    print("\nOpções:")
    print("  [1] Planejar Revisão Acadêmica (narrativa)")
    print("  [2] Planejar Revisão Técnica (capítulo)")
    print("  [3] Executar Escrita Técnica a partir de plano existente")
    escolha = input("\nEscolha [1/2/3]: ").strip()

    if escolha == "3":
        # Modo escrita técnica
        planos = sorted(glob.glob("plano_revisao_tecnica_*.md"))
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
            print("✅ REVISÃO TÉCNICA CONCLUÍDA")
            print("=" * 70)
        except KeyboardInterrupt:
            print("\nCancelado.")
        return

    # Restante do código original (planejamento)
    tema = input("\nTema da revisao: ").strip()
    if not tema:
        print("Tema vazio.")
        return

    print("\n[1] Revisao Academica (narrativa da literatura)")
    print("[2] Revisao Tecnica   (capitulo didatico/detalhado)")
    print("[3] Ambas")
    while True:
        e = input("\nEscolha [1/2/3]: ").strip()
        if e in ("1", "2", "3"):
            break
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