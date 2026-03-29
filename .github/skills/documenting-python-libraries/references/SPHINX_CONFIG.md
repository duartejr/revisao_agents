# Sphinx Configuration Reference

This reference expands the Sphinx setup in `SKILL.md`.

## Recommended Directory Layout

```text
docs/
|-- conf.py
|-- index.md
|-- api.md
|-- tutorials/
|   `-- getting-started.md
`-- _static/
```

## Install Dependencies

```bash
pip install sphinx furo myst-parser sphinx-copybutton
```

Optional:

```bash
pip install sphinx-autodoc-typehints sphinx-design sphinxext-opengraph
```

## `conf.py` Baseline

```python
from __future__ import annotations

import os
import sys
from datetime import datetime

# Ensure package import works during doc builds
sys.path.insert(0, os.path.abspath("../src"))

project = "Your Package"
author = "Your Team"
current_year = datetime.now().year
copyright = f"{current_year}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_copybutton",
]

autosummary_generate = True
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]
```

## API Reference Pattern

Create `docs/api.md`:

````md
# API Reference

```{eval-rst}
.. automodule:: your_package
   :members:
   :undoc-members:
   :show-inheritance:
```
````

## `index.md` Pattern

````md
# Your Package

Short summary.

```{toctree}
:maxdepth: 2
:caption: Contents

api
tutorials/getting-started
```
````

## Build and Validate

```bash
sphinx-build -b html docs docs/_build/html
```

Use warnings as errors for CI:

```bash
sphinx-build -W -b html docs docs/_build/html
```

## Common Failure Modes

1. Import errors during autodoc.
Fix: ensure package path is in `sys.path` and dependencies are installed.

2. Google docstrings not parsed correctly.
Fix: verify `sphinx.ext.napoleon` is enabled and docstrings follow Google format.

3. Markdown directives not recognized.
Fix: ensure `myst_parser` is installed and included in `extensions`.

4. Broken internal links.
Fix: run `sphinx-build -b linkcheck docs docs/_build/linkcheck` in CI.

## CI Recommendations

- Build docs on pull requests.
- Fail on warnings (`-W`) to prevent doc regressions.
- Keep one docs Python version aligned with production support matrix.
