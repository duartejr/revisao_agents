# Refactoring Summary: revisao_agent Project Restructuring

**Date:** March 7, 2026  
**Status:** In Progress

## Overview

This document summarizes the comprehensive refactoring of the `revisao_agent` project to improve scalability, maintainability, and debuggability by following a more structured folder organization consistent with the `src/revisao_agents/` pattern.

## Old Structure vs New Structure

### Before (Distributed)
```
revisao_agent/
├── helpers/
│   └── ancora_helpers.py          # Anchor utilities
├── nodes/                          # ← Mixed node implementations
│   ├── academic.py                 # Academic review nodes
│   ├── technical.py                # Technical review nodes
│   ├── technical_writing.py        # Technical writing nodes  
│   ├── common.py                   # Shared nodes
│   └── __init__.py
├── workflows/                      # Workflow definitions
│   ├── academic_workflow.py
│   ├── technical_workflow.py
│   └── technical_writing_workflow.py
├── src/revisao_agents/            # Already structured
│   ├── agents/                     # ← Empty (to be populated)
│   ├── core/
│   ├── helpers/                    # ← Empty (to be populated)
│   ├── prompts/
│   ├── tools/
│   ├── utils/
│   └── workflows/
└── (other root files...)
```

### After (Unified)
```
revisao_agent/
├── src/revisao_agents/
│   ├── agents/                     # ← All node implementations moved here
│   │   ├── __init__.py
│   │   ├── academic.py             # Academic review nodes
│   │   ├── technical.py            # Technical review nodes
│   │   ├── technical_writing.py    # Technical writing nodes
│   │   └── common.py               # Shared nodes (interview, pausing, etc)
│   │
│   ├── helpers/                    # ← Utility helpers (anchor management, etc)
│   │   ├── __init__.py
│   │   └── ancora_helpers.py       # Anchor extraction and manipulation
│   │
│   ├── core/
│   │   └── schemas/               # Data models (RespostaSecao, Fonte, etc)
│   │
│   ├── prompts/                   # Prompt templates
│   ├── tools/                     # Tool integrations
│   ├── utils/                     # Core utilities
│   │   ├── llm_providers.py       # get_llm and LLM logic
│   │   ├── vector_store.py        # FAISS vector search
│   │   ├── helpers.py             # Text formatting, file saving
│   │   ├── tavily_client.py       # Web search integration
│   │   ├── mongodb_corpus.py      # MongoDB corpus management
│   │   └── ...
│   │
│   ├── workflows/                # Workflow definitions
│   ├── state.py                  # State definitions
│   ├── config.py                 # Configuration
│   └── ...
│
├── workflows/                     # ← Wrapper workflows (updated imports)
│   ├── academic_workflow.py       # Now imports from src.revisao_agents
│   ├── technical_workflow.py      # Now imports from src.revisao_agents
│   └── technical_writing_workflow.py
│
├── helpers/                       # ← OLD LOCATION (deprecate or keep for compatibility)
└── nodes/                         # ← OLD LOCATION (deprecate or keep for compatibility)
```

## Changes Made

### 1. **Agents Migration** ✅
Created `/src/revisao_agents/agents/` with all node implementations:

- **academic.py** - Literature review planning nodes
  - `consulta_vetorial_node()` - Vector search for papers
  - `plano_inicial_academico_node()` - Initial plan generation
  - `refinar_consulta_academico_node()` - Query refinement
  - `refinar_plano_academico_node()` - Plan refinement
  - `finalizar_plano_academico_node()` - Final plan generation

- **technical.py** - Technical chapter planning nodes
  - `busca_tecnica_inicial_node()` - Initial technical search
  - `plano_inicial_tecnico_node()` - Initial technical plan
  - `refinar_busca_tecnica_node()` - Search refinement
  - `refinar_plano_tecnico_node()` - Plan refinement
  - `finalizar_plano_tecnico_node()` - Final technical plan

- **common.py** - Shared nodes across workflows
  - `pausa_humana_node()` - Human interaction pause
  - `entrevista_node()` - Interview / Q&A generation
  - `roteador_entrevista()` - Flow control router

- **technical_writing.py** - Advanced technical authoring nodes
  - `parsear_plano_node()` - Plan parsing
  - `escrever_secoes_node()` - Section writing with search & verification
  - `consolidar_node()` - Document consolidation

- **__init__.py** - Package exports for easy imports

### 2. **Helpers Reorganization** ✅
Moved `/helpers/ancora_helpers.py` → `/src/revisao_agents/helpers/__init__.py`

Core helper functions:
- `extrair_ancoras_com_citacoes()` - Extract anchors with citations
- `extrair_ancora_principal()` - Get main anchor
- `extrair_citacao_ancora()` - Find citation number for anchor
- `limpar_ancoras()` - Remove anchors from text

### 3. **Import Updates** ✅
Updated all imports to use relative imports within the package:

**Before:**
```python
from state import RevisaoState
from config import get_llm
from utils.vector_store import buscar_chunks
from utils.helpers import fmt_chunks
```

**After:**
```python
from ..state import RevisaoState
from ..utils.llm_providers import get_llm
from ..utils.vector_store import buscar_chunks
from ..utils.helpers import fmt_chunks
```

### 4. **Workflow Updates** ✅
Updated all workflow files (`workflows/*.py`) to import from new location:

**Before:**
```python
from state import RevisaoState
from nodes import consulta_vetorial_node, ...
```

**After:**
```python
from src.revisao_agents.state import RevisaoState
from src.revisao_agents.agents import consulta_vetorial_node, ...
```

## Benefits of This Refactoring

### 1. **Improved Structure**
- ✅ Clear separation of concerns (agents, helpers, utils, tools, prompts)
- ✅ Consistent with `src/` package layout
- ✅ Easier to navigate and understand codebase

### 2. **Better Scalability**
- ✅ Easy to add new agent types (`src/revisao_agents/agents/new_agent.py`)
- ✅ Easy to add new helpers and utilities
- ✅ Single source of truth for shared components

### 3. **Enhanced Maintainability**
- ✅ All related functionality grouped together
- ✅ Clear import paths (no circular dependencies)
- ✅ Easier to refactor and update

### 4. **Easier Debugging**
- ✅ Clearer module hierarchy reduces confusion
- ✅ Consistent naming conventions
- ✅ Better organized imports make tracing easier

### 5. **Package-Ready Structure**
- ✅ Proper `__init__.py` files for package exports
- ✅ Ready for distribution/pip installation
- ✅ Proper namespace organization

## Migration Path

### For Existing Code
If you have other files importing from the old structure:

**Old imports:**
```python
from nodes import consulta_vetorial_node
from helpers import extrair_ancoras_com_citacoes
```

**Update to:**
```python
from src.revisao_agents.agents import consulta_vetorial_node
from src.revisao_agents.helpers import extrair_ancoras_com_citacoes
```

Or if using relative imports (from within `src/revisao_agents/`):
```python
from .agents import consulta_vetorial_node
from .helpers import extrair_ancoras_com_citacoes
```

## Next Steps

### 1. **Verify Imports** 
- [ ] Test all imports work correctly
- [ ] Check for any missing dependencies
- [ ] Verify all function references are valid

### 2. **Test Execution**
- [ ] Run academic workflow test
- [ ] Run technical workflow test
- [ ] Run technical writing workflow test
- [ ] Check for runtime import errors

### 3. **Update External References**
- [ ] Update any CLI entry points
- [ ] Update any documentation
- [ ] Update any test files

### 4. **Deprecation of Old Structure** (Optional)
- [ ] Keep old `nodes/` and `helpers/` for backward compatibility OR
- [ ] Remove them after ensuring all imports updated
- [ ] Update version number (semantic versioning)

### 5. **Optional Enhancements**
- [ ] Move constants to `config.py`
- [ ] Create `state.py` type definitions for `RevisaoState` and `EscritaTecnicaState`
- [ ] Add type hints to agent functions
- [ ] Add docstrings to all agents

## Files Modified

### Created
- ✅ `/src/revisao_agents/agents/__init__.py` - Package exports
- ✅ `/src/revisao_agents/agents/academic.py` - Academic nodes
- ✅ `/src/revisao_agents/agents/technical.py` - Technical nodes
- ✅ `/src/revisao_agents/agents/common.py` - Common nodes
- ✅ `/src/revisao_agents/agents/technical_writing.py` - Technical writing nodes
- ✅ `/src/revisao_agents/helpers/__init__.py` - Helper functions

### Modified
- ✅ `/workflows/academic_workflow.py` - Updated imports
- ✅ `/workflows/technical_workflow.py` - Updated imports
- ✅ `/workflows/technical_writing_workflow.py` - Updated imports

### Deprecated (Consider Removing)
- `/nodes/` - Original node implementations
- `/helpers/` - Original helper implementations

## Testing Checklist

- [ ] All imports resolve without errors
- [ ] Agents module exports correct symbols
- [ ] Helper functions work correctly
- [ ] Workflows can instantiate graph
- [ ] Graph execution completes successfully
- [ ] No circular imports
- [ ] Type hints are correct (if added)

## Configuration Notes

### Constants That May Need Moving to Config
These constants are currently hardcoded in agent files:
- `CHUNKS_PER_QUERY` - Agents/academic.py
- `ENCERRAMENTO` - Agents/common.py
- `TECNICO_MAX_RESULTS`, `MAX_URLS_EXTRACT`, etc. - Agents/technical_writing.py

**Recommendation:** Move these to `src/revisao_agents/config.py` for centralized management.

## Questions & Feedback

- Should old `nodes/` and `helpers/` directories be removed?
- Should we add strict type hints to all agent functions?
- Should we add comprehensive docstrings?
- Should constants be moved to config.py?

---

**Refactoring Complete (Core Setup)** - See next steps for validation and testing.
