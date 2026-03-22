## Plan: Guardião de Referências no Chat

Objetivo: criar um fluxo especializado dentro do chat de revisão para pedidos de listar e formatar fontes, com confirmação obrigatória antes de executar e bloqueio explícito quando faltar metadado sem busca web habilitada. A abordagem escolhida é um tool/handler dedicado no chat atual, sem criar modo separado agora.

**Steps**
1. Fase 1 — Gate de intenção + confirmação obrigatória (bloqueia execução direta)
   - Expandir o roteamento de intenção em [revisao_agent/src/gradio_app/handlers.py](revisao_agent/src/gradio_app/handlers.py) para um estado de pré-execução em pedidos de referências.
   - Criar estado de conversa para confirmação obrigatória em qualquer pedido de listar/formatar fontes.
   - Persistir no session_state: ação pendente, parâmetros extraídos, e hash da mensagem-base para evitar confirmações cruzadas.
   - Dependência: nenhuma.

2. Fase 2 — Normalização de comando e escopo (depende da Fase 1)
   - Separar de forma determinística três intenções operacionais:
     - listar referências do documento sem repetição;
     - formatar somente a lista enviada pelo usuário;
     - resolver referências numeradas [n].
   - Adicionar um parser robusto para lista fornecida pelo usuário (linhas, bullets, itens com DOI/URL/local path, ruído de markdown malformado).
   - Quando houver ambiguidade de escopo, responder com pergunta fechada e não executar nada até confirmação.

3. Fase 3 — Tool especializado de referência (depende da Fase 2)
   - Encapsular o pipeline bibliográfico em uma função-orquestradora única no chat, chamada somente após confirmação.
   - Executar DOI-first para itens fornecidos: DOI explícito no texto/url, depois título→Crossref, depois enriquecimento local.
   - Centralizar deduplicação e limpeza ABNT para evitar mistura de referências do documento com lista enviada.
   - Paralelismo: pode evoluir em paralelo com Fase 4 (teste unitário) após assinatura estável.

4. Fase 4 — Política de web obrigatória para incompletos (depende da Fase 3)
   - Inserir checagem prévia de completude mínima por item antes da formatação final.
   - Se web estiver desabilitada e houver itens incompletos, parar execução e retornar pedido objetivo para habilitar web; não emitir lista parcial final.
   - Com web habilitada, rodar enriquecimento e só então formatar ABNT.

5. Fase 5 — Contrato de resposta e rastreabilidade (depende das Fases 3 e 4)
   - Definir payload de saída por intenção:
     - listar: somente referências do documento deduplicadas;
     - formatar lista enviada: somente itens enviados, na ordem enviada;
     - resolver [n]: somente números solicitados.
   - Padronizar seção curta de rastreabilidade (consultas local/web, itens completos/incompletos).
   - Proibir texto ambíguo que indique ação não executada.

6. Fase 6 — Integração UI mínima (paralela com Fase 5)
   - Ajustar mensagens na aba de revisão em [revisao_agent/src/gradio_app/app.py](revisao_agent/src/gradio_app/app.py) para explicitar o fluxo: comando → confirmação → execução.
   - Exibir estado “aguardando confirmação” e “aguardando habilitar web” sem alterar layout estrutural.

7. Fase 7 — Validação (última fase)
   - Ampliar testes unitários em [revisao_agent/tests/unit/test_review_reference_pipeline.py](revisao_agent/tests/unit/test_review_reference_pipeline.py) para:
     - confirmação obrigatória antes de executar;
     - bloqueio sem web para itens incompletos;
     - isolamento estrito da lista fornecida.
   - Adicionar teste de fluxo em 2 turnos (pedido + confirmação) no chat de revisão.
   - Validar regressão do roteamento existente para intents não bibliográficas.

**Relevant files**
- `/home/duartejr/paper_reviwer/revisao_agent/src/gradio_app/handlers.py` — ponto central para roteamento de intenção, confirmação e execução do pipeline especializado.
- `/home/duartejr/paper_reviwer/revisao_agent/src/gradio_app/app.py` — mensagens de UX e sinalização de estado no chat de revisão.
- `/home/duartejr/paper_reviwer/revisao_agent/src/revisao_agents/utils/bib_utils/doi_utils.py` — resolução DOI/Crossref para estratégia DOI-first.
- `/home/duartejr/paper_reviwer/revisao_agent/tests/unit/test_review_reference_pipeline.py` — regressão e novos cenários de confirmação/web.

**Verification**
1. Cenário listar: pedido inicial deve receber confirmação obrigatória; após confirmação, retorna apenas referências do documento deduplicadas.
2. Cenário formatar lista enviada: pedido inicial deve receber confirmação; após confirmação, saída contém apenas itens enviados em ABNT.
3. Cenário sem web: se incompleto, resposta deve interromper execução e pedir habilitação web explicitamente.
4. Cenário com web: após habilitar, itens incompletos devem ser enriquecidos e formatados com melhoria mensurável de completude.
5. Executar testes unitários do pipeline e verificar ausência de regressão no chat normal.

**Decisions**
- Arquitetura escolhida: tool/handler especializado dentro do chat atual (não subagente separado neste ciclo).
- Confirmação obrigatória antes de qualquer ação de listar/formatar fontes.
- Política sem web: parar e pedir habilitação, sem resultado final parcial.
- Escopo incluído: roteamento, confirmação, política web, contrato de saída, testes.
- Escopo excluído: nova aba, novo modo de agente, redesign de UI, mudanças em workflows não relacionados.

**Further Considerations**
1. Futuro opcional: promover o handler para subagente dedicado somente se o fluxo atual atingir limite de complexidade.
2. Definir timeout e limite de tentativas web por item para manter latência previsível.
3. Adicionar telemetria de taxa de retrabalho (pedidos refeitos) para medir redução de frustração do usuário.
