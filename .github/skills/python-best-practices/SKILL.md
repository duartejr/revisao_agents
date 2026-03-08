---
name: python-best-practices
description: Guide developers to follow Python best practices, write clean, readable code, use English naming and documentation, and structure projects effectively. Activated when users ask about improving code quality, refactoring, style guidelines, or writing better Python.
---

# Python Best Practices & Clean Code Skill

This skill helps you write high-quality Python code by following established best practices. It emphasizes **clarity, readability, and maintainability** – using English for names, comments, and documentation. It also covers proper project structure, testing, and AI prompt writing.

## When to Use This Skill

Activate this skill when the user mentions:
- ✨ "Best practices" or "clean code"
- 🔧 "Refactor this" or "improve this code"
- 📝 "Make this more readable" or "use proper naming"
- 🧪 "How should I test this?" or "unit tests"
- 🏗️ "Project structure" or "organize my code"
- 💬 "Write a better prompt" or "how to ask Copilot"
- 🐍 General Python development questions

## Core Principles

1. **Readability counts** – Code is written once, read many times.
2. **Explicit is better than implicit** – Clear intent over clever tricks.
3. **Simple is better than complex** – Favor straightforward solutions.
4. **Consistency** – Follow a uniform style throughout the project.
5. **Documentation** – Explain *why*, not just *what*.
6. **Testability** – Write code that is easy to test.

## Code Style & Naming (PEP 8)

### Naming Conventions (Always in English)
| Element | Convention | Example |
|---------|------------|---------|
| Variables, functions, methods | `snake_case` | `user_count`, `calculate_total()` |
| Classes | `PascalCase` | `CustomerRepository`, `PDFGenerator` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_RETRIES`, `DEFAULT_TIMEOUT` |
| Private attributes/methods | Prefix `_` | `_internal_cache`, `_validate_input()` |
| "Protected" (subclasses) | Single `_` | `_base_method()` |
| Magic methods | `__dunder__` | `__init__`, `__len__` |

### What to Avoid
- ❌ Cryptic abbreviations: `usr_cnt`, `calc_tot`
- ❌ Single letters except trivial loops: `for i in range(10)` is fine, but `x` as a variable name is not.
- ❌ Non-English names: `dameDatos`, `berechneSumme` (unless the domain demands it, but prefer English for broader collaboration)

### Code Layout
- Indent with **4 spaces** (never tabs)
- Maximum line length: **79 characters** (docstrings/comments: 72)
- Separate top-level functions/classes with **two blank lines**
- Separate methods inside a class with **one blank line**
- Imports:
  - One import per line
  - Group in order: standard library, third-party, local modules
  - Avoid `from module import *`

**When the user shows code, you should:**
- Point out naming violations
- Suggest more descriptive names
- Show how to reorganize imports

## Documentation (in English)

### Comments
- Use comments to explain **why**, not **what** (the code shows what)
- Keep comments up-to-date
- Use complete sentences, proper punctuation

**Bad:**
```python
# increment i
i += 1
```

**Good:**
```python
# Move to the next item in the queue (FIFO order)
i += 1
```

### Docstrings (PEP 257)
Every module, class, function, and method should have a docstring.

**Function docstring template:**
```python
def fetch_data(api_url, timeout=30):
    """Fetch data from a REST API endpoint.

    Args:
        api_url (str): Full URL of the API endpoint.
        timeout (int): Request timeout in seconds. Defaults to 30.

    Returns:
        dict: Parsed JSON response.

    Raises:
        requests.Timeout: If the request times out.
        requests.HTTPError: If the HTTP status is not 200.
    """
```

**Class docstring:**
```python
class DataProcessor:
    """Process and clean raw data from various sources.

    Attributes:
        source (str): Data source identifier.
        config (dict): Configuration parameters.
    """
```

### Inline Comments
Use sparingly, but when needed:
```python
# BAD: Comment just repeats code
x = x + 1  # increment x by 1

# GOOD: Explains non-obvious logic
# We add 1 because the index is 0-based but our IDs are 1-based
x = x + 1
```

## Type Hints (PEP 484)
Use type hints for better readability and tooling support.

**Before:**
```python
def process_items(items):
    result = []
    for item in items:
        result.append(item * 2)
    return result
```

**After:**
```python
from typing import List

def process_items(items: List[int]) -> List[int]:
    """Double each integer in the input list."""
    return [item * 2 for item in items]
```

**Common hints:**
- `str`, `int`, `float`, `bool`
- `List[str]`, `Dict[str, int]`, `Tuple[float, float]`
- `Optional[str]` = `Union[str, None]`
- `Any` for dynamic types (use sparingly)
- Use `from __future__ import annotations` for forward references

## Writing Tests

### Principles
- Write tests early (TDD when appropriate)
- Each test should verify one behavior
- Test both success and failure paths
- Use descriptive test names: `test_method_returns_correct_value_when_input_valid()`

### Example with pytest
```python
# test_calculator.py
import pytest
from calculator import add

def test_add_positive_numbers():
    assert add(2, 3) == 5

def test_add_negative_numbers():
    assert add(-1, -1) == -2

def test_add_mixed_numbers():
    assert add(-5, 5) == 0
```

**When the user asks about testing:**
- Suggest which framework (pytest, unittest) and why
- Show how to structure tests
- Recommend test coverage tools

## Project Structure

A typical Python project layout:
```
my_project/
├── .gitignore
├── README.md
├── requirements.txt          # or pyproject.toml + poetry.lock
├── setup.py / pyproject.toml
├── my_project/               # main package
│   ├── __init__.py
│   ├── module1.py
│   ├── module2.py
│   └── subpackage/
│       ├── __init__.py
│       └── ...
├── tests/
│   ├── __init__.py
│   ├── test_module1.py
│   └── test_module2.py
└── docs/                     # documentation (optional)
```

**Virtual environment & dependency management:**
- Use `venv` or `conda` for isolation
- Pin dependencies (e.g., `requirements.txt` with versions)
- Use `pip freeze > requirements.txt` to export
- Consider `poetry` or `pipenv` for more advanced dependency management

## Version Control (Git)

### Commit Messages (in English)
Follow the [Conventional Commits](https://www.conventionalcommits.org/) style:
```
<type>(<scope>): <subject>

<body>
```

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Example:**
```
feat(api): add timeout parameter to fetch_data

Allow callers to specify a timeout to avoid hanging on slow responses.
Closes #42
```

### Branching
- Use feature branches: `feature/implement-login`
- Keep commits focused and atomic
- Write clear commit messages

## Error Handling

### Use Exceptions, Not Return Codes
**Bad:**
```python
def divide(a, b):
    if b == 0:
        return None
    return a / b
```

**Good:**
```python
def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
```

### Catching Exceptions
- Be specific about exception types
- Avoid bare `except:`
- Log or handle appropriately

```python
try:
    result = divide(10, 0)
except ValueError as e:
    logger.error(f"Invalid input: {e}")
    raise  # re-raise if cannot handle
```

### Use Context Managers (`with`) for Resources
```python
with open("file.txt", "r") as f:
    content = f.read()
# file automatically closed
```

## Logging Instead of Print

**Bad:**
```python
print(f"User {user_id} logged in")
```

**Good:**
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("User %s logged in", user_id)
```

**Log levels:** DEBUG, INFO, WARNING, ERROR, CRITICAL

## Pythonic Idioms

### List Comprehensions & Generators
**Bad:**
```python
squares = []
for x in range(10):
    squares.append(x**2)
```

**Good:**
```python
squares = [x**2 for x in range(10)]
```

Use generator expressions for large data: `(x**2 for x in range(10))`

### Contextual `with`
Already covered.

### Use `enumerate` Instead of Manual Index
**Bad:**
```python
i = 0
for item in items:
    print(i, item)
    i += 1
```

**Good:**
```python
for i, item in enumerate(items):
    print(i, item)
```

### Use `zip` to Iterate in Parallel
```python
names = ["Alice", "Bob"]
scores = [85, 92]
for name, score in zip(names, scores):
    print(f"{name}: {score}")
```

### Use `dict.get()` with Default
```python
value = my_dict.get("key", default_value)
```

### Use `collections` for Common Data Structures
- `defaultdict`, `Counter`, `deque`, `namedtuple`

## Writing Prompts in English

When the user asks about writing better prompts (e.g., for Copilot or other AI):

- **Be specific and clear** – State exactly what you want, including context and constraints.
- **Use full English sentences** – Avoid fragmented phrases.
- **Provide examples** – Show input and expected output.
- **Specify the role** – "You are a Python expert reviewing code..."
- **Include details** – Libraries, versions, edge cases.

**Example of a good prompt:**
> "Write a Python function that takes a list of integers and returns a new list containing only the even numbers, sorted in descending order. Include type hints and a docstring."

**What to avoid:**
> "even nums list desc"

## General Advice for the AI

When the user asks for help with code:
1. **First, assess the current code** for adherence to best practices.
2. **Provide specific, actionable feedback** – point out lines or patterns.
3. **Show examples** of both bad and good versions.
4. **Explain *why*** the suggestion is better (readability, maintainability, etc.).
5. **Offer to refactor** if the user shares a code snippet.
6. **Encourage the user** to adopt these habits gradually.

## Quick Reference Checklist for Users

- [ ] Names are descriptive and in English
- [ ] Functions/classes have docstrings
- [ ] Type hints are used
- [ ] Imports are grouped and clean
- [ ] Code is formatted according to PEP 8 (can use `black`)
- [ ] No print statements; logging is used
- [ ] Exceptions are handled properly
- [ ] Tests exist for core functionality
- [ ] Project has a clear structure
- [ ] Git commits have meaningful messages

---

**Version:** 1.0.0  
**Author:** Adapted for GitHub Copilot Skills