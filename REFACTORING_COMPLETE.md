# Refactoring Status — Updated

## Current Status

The project evolved to a stable runtime centered on `src/revisao_agents/workflows` + `src/revisao_agents/nodes`.
`graphs/review_graph.py` is currently a compatibility wrapper that delegates to those workflows.

This file replaces earlier claims that all runtime nodes had fully migrated to `agents/`.

## Canonical Runtime Structure (Current)

```
src/revisao_agents/
├── nodes/                     # ← Canonical node implementations (current)
│   ├── academic.py
│   ├── technical.py
│   ├── common.py
│   ├── technical_writing.py
│   └── writing/
│
├── workflows/                 # ← Canonical orchestration for planning/writing
│   ├── academic_workflow.py
│   ├── technical_workflow.py
│   └── technical_writing_workflow.py

├── graphs/
│   └── review_graph.py        # compatibility layer delegating to workflows
│
├── core/
│   └── schemas/             # Data models
│
├── prompts/                 # Prompt templates
├── tools/                   # Tool integrations
├── utils/                   # Core utilities
├── workflows/               # Workflow definitions
├── state.py                # State definitions
└── config.py               # Configuration
```

## Key Decisions (Phases 1–7)

### 1. Runtime Canonicalization ✅
- `workflows/*` + `nodes/*` are now the supported runtime paths.
- `graphs/review_graph.py` no longer maintains an independent graph implementation.
- Compatibility builders delegate to canonical workflows.

### 2. CLI Contract Repair ✅
- CLI now targets planning outputs (`plano_final`, `plano_final_path`) and auto-HITL loop execution.

### 3. Test Realignment ✅
- Tests updated to current module paths and state schema.
- Integration smoke tests now validate graph/workflow buildability without requiring OpenAI key.

### 4. Robustness & Validation ✅
- Bibliography/corpus API mismatch fixed (`query_similar` → `query`).
- Tavily empty/degraded output contracts stabilized.
- `llm_call` unified with provider factory and explicit error semantics.
- Runtime preflight validation and startup diagnostics added.

## Supported Entry Points

### Menu (canonical)

```bash
python -m revisao_agents
```

### UI

```bash
python run_ui.py
```

### CLI

```bash
PYTHONPATH=src python -m revisao_agents.cli "Tema" --tipo academico
```

## Testing Baseline

```bash
PYTHONPATH=src python -m pytest -q
```

## Benefits Achieved

✅ **Improved Scalability** - Organized structure makes it easy to add new agents
✅ **Better Maintainability** - Clear separation of agents, helpers, utils, and tools
✅ **Easier Debugging** - Hierarchical module structure makes tracing easier
✅ **Standardized Structure** - Follows Python package best practices
✅ **Package-Ready** - Can be distributed/installed as a proper Python package

## Notes

- Earlier documentation snapshots that claim full migration to `agents/` should be treated as historical context.
- For current architecture and commands, prioritize this file and `README.md` in workspace root.

**Status:** ✅ Runtime stabilized through phase 7
