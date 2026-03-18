# Architecture & Migration Guide (Current)

**Date:** March 7, 2026  
**Status:** Updated after hardening phases 1–7

## Overview

This document reflects the current runtime architecture of `revisao_agent`.
The canonical path is based on `src/revisao_agents/workflows` + `src/revisao_agents/nodes`.
`src/revisao_agents/graphs/review_graph.py` is a compatibility wrapper that delegates to workflow builders.

## Canonical Runtime Structure

```
revisao_agent/
├── src/revisao_agents/
│   ├── nodes/
│   │   ├── academic.py
│   │   ├── technical.py
│   │   ├── common.py
│   │   ├── technical_writing.py
│   │   └── writing/
│   │
│   ├── workflows/
│   │   ├── academic_workflow.py
│   │   ├── technical_workflow.py
│   │   └── technical_writing_workflow.py
│   │
│   ├── graphs/
│   │   └── review_graph.py   # compatibility delegation layer
│   │
│   ├── tools/
│   ├── utils/
│   ├── prompts/
│   ├── config.py
│   └── state.py
└── run_ui.py
```

## Key Stabilization Changes

### 1. Runtime path unification ✅
- Removed divergence between graph and workflow implementations.
- Kept graph API surface for backward compatibility.

### 2. CLI contract alignment ✅
- Planning output keys aligned to real state (`plano_final`, `plano_final_path`).
- HITL progression is handled in the execution loop.

### 3. Retrieval and bibliography hardening ✅
- Bibliography corpus lookup corrected to use supported API path.
- Tavily handling made resilient for empty/degraded scenarios.

### 4. LLM and config reliability ✅
- `llm_call` invocation path unified.
- Explicit typed failure (`LLMInvocationError`) for invocation issues.
- Startup and UI preflight checks provide clear configuration diagnostics.

## Migration Notes

### Import guidance

Prefer imports that reference canonical modules:

```python
from src.revisao_agents.nodes.academic import consulta_vetorial_node
from src.revisao_agents.workflows.academic_workflow import build_academico_workflow
```

When inside package modules, use relative imports:

```python
from ..state import ReviewState
```

### Compatibility guidance

- Treat `graphs/review_graph.py` as compatibility API.
- Implement new runtime behavior in `nodes`/`workflows` first.

## Verification Commands

Run from `revisao_agent/`:

```bash
# basic workflow import sanity
PYTHONPATH=src python -c "from revisao_agents.workflows.academic_workflow import build_academico_workflow; print('ok')"

# unit/integration tests
PYTHONPATH=src python -m pytest -q
```

## Next Steps

1. Keep all docs consistent with root `README.md` and `TESTING_GUIDE.md`.
2. Avoid reintroducing parallel orchestration paths.
3. Extend tests around changed contracts before broad feature work.

---

**Status:** Current architecture documented and aligned with hardening phases 1–7.
