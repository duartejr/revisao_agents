---
description: "Use when the user asks to commit, create/update a pull request, or prepare code for PR review. Run the Python Review custom agent first, apply its findings, and iterate until approval or explicit stop."
name: "Commit and PR Review Loop"
---
# Commit and PR Review Loop

When the user asks to commit changes or create/update a pull request, follow this workflow before finalizing:

1. Run the `Python Review` custom agent against the pending change set.
2. Read all findings carefully, prioritizing high and medium severity items.
3. Apply the recommended fixes in code and tests.
4. Re-run relevant validations (for example `pytest` or focused checks).
5. Run the `Python Review` custom agent again.
6. Repeat until one of the following is true:
   - The review is effectively approved (no unresolved high/medium findings), or
   - The user explicitly asks to stop.

Additional requirements during this loop:

- Enforce documentation quality based on the `documenting-python-libraries` skill, especially Google-style docstrings for public APIs.
- Ensure developer-facing code artifacts are in English (identifiers, comments, docstrings), except intentionally localized user-facing copy.
- Treat low-severity findings as optional: address them when practical, but do not block commit/PR finalization on low-only findings.
- If a recommendation is not applicable or cannot be implemented safely, explain why and propose the safest alternative before committing or opening/updating the PR.
