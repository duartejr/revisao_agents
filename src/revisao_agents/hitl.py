def run_hitl_loop(app, config, state_init):
    """Loop de interação humana (Human-in-the-loop)."""
    for _ in app.stream(state_init, config):
        pass
    while True:
        state = app.get_state(config)
        if not state.next:
            print("\nPlanejamento concluido!")
            break
        if "pausa_humana" not in state.next:
            print("\nNo inesperado:", str(state.next))
            break
        # Mostra a última pergunta do agente
        for role, c in reversed(state.values.get("historico_entrevista", [])):
            if role == "assistant":
                print("\n🤖", c)
                break
        p  = state.values.get("perguntas_feitas", 0)
        mp = state.values.get("max_perguntas", 3)
        tp = state.values.get("tipo_revisao", "academico")
        print(f"\n   [Rodada {p}/{mp} — {tp} — ok para finalizar]")
        resp = input("👤 ").strip() or "Manter o plano atual."
        hist = state.values.get("historico_entrevista", [])
        app.update_state(config,
            {"historico_entrevista": hist + [("user", resp)]},
            as_node="pausa_humana")
        print("\n[Refinando plano...]")
        for _ in app.stream(None, config):
            pass
        if not app.get_state(config).next:
            print("\nPlanejamento concluido!")
            break