---
name: documenting-python-libraries
description: "Creates comprehensive Python library documentation including Google-style docstrings, Sphinx setup, API references, tutorials, and ReadTheDocs configuration. Use when writing docstrings, setting up Sphinx documentation, or creating user guides for Python libraries."
---

# Python Library Documentation

## When to Use

- Add or improve docs for a Python library before release.
- Standardize docstrings and API references for maintainability.
- Set up Sphinx + Read the Docs from scratch.
- Improve onboarding with quick start and tutorial content.

## Workflow

1. Define documentation scope and audience.
2. Document public API with Google-style docstrings.
3. Set up Sphinx with Napoleon and Markdown support.
4. Build README quick start and narrative user guides.
5. Configure Read the Docs and validate build output.

## Decision Points

1. If library users are primarily developers, prioritize API reference depth first.
2. If onboarding friction is high, prioritize README quick start and tutorial flow first.
3. If docs build fails on CI/RTD, reduce optional extensions and re-enable incrementally.
4. If examples drift from code behavior, convert examples into tested snippets.

## Docstring Style (Google)

```python
def encode(latitude: float, longitude: float, *, precision: int = 12) -> str:
    """Encode geographic coordinates to a quadtree string.

    Args:
        latitude: The latitude in degrees (-90 to 90).
        longitude: The longitude in degrees (-180 to 180).
        precision: Number of characters in output. Defaults to 12.

    Returns:
        A string representing the encoded location.

    Raises:
        ValidationError: If coordinates are out of valid range.

    Example:
        >>> encode(37.7749, -122.4194)
        '9q8yy9h7wr3z'
    """
```

## Sphinx Quick Setup

```bash
# Install
pip install sphinx furo myst-parser sphinx-copybutton

# Initialize
sphinx-quickstart docs/
```

**conf.py essentials:**

```python
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Google docstrings
    'myst_parser',          # Markdown support
]
html_theme = 'furo'
```

## pyproject.toml Dependencies

```toml
[project.optional-dependencies]
docs = [
    "sphinx>=7.0",
    "furo>=2024.0",
    "myst-parser>=2.0",
]
```

## README Template

```markdown
# Package Name

[![PyPI](https://badge.fury.io/py/package.svg)](https://pypi.org/project/package/)

Short description of what it does.

## Installation

pip install package

## Quick Start

from package import function
result = function(args)

## Documentation

Full docs at [package.readthedocs.io](https://package.readthedocs.io/)
```

## ReadTheDocs (.readthedocs.yaml)

```yaml
version: 2
build:
  os: ubuntu-22.04
  tools:
    python: "3.11"
sphinx:
  configuration: docs/conf.py
python:
  install:
    - method: pip
      path: .
      extra_requirements: [docs]
```

For detailed setup, see:
- **[SPHINX_CONFIG.md](./references/SPHINX_CONFIG.md)** - Full Sphinx configuration
- **[TUTORIALS.md](./references/TUTORIALS.md)** - Tutorial writing guide

## Checklist

```text
README:
- [ ] Clear project description
- [ ] Installation instructions
- [ ] Quick start example
- [ ] Link to full documentation

API Docs:
- [ ] All public functions documented
- [ ] Args, Returns, Raises sections
- [ ] Examples in docstrings
- [ ] Type hints included
```

## Learn More

This skill is based on the [Documentation](https://mcginniscommawill.com/guides/python-library-development/#documentation-your-librarys-ambassador) section of the [Guide to Developing High-Quality Python Libraries](https://mcginniscommawill.com/guides/python-library-development/) by [Will McGinnis](https://mcginniscommawill.com/).
