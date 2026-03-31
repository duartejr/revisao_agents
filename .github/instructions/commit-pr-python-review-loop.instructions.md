---
description: "Use when the user asks to commit, create/update a pull request, or prepare code for PR review. Run the Python Review custom agent first, then run the Python Documentation Follow-up custom agent, apply their findings, and iterate until approval or explicit stop."
name: "Commit and PR Review Loop"
---
# Commit and PR Review Loop

When the user asks to commit changes or create/update a pull request, follow this workflow before finalizing:

1. Run the `Python Review` custom agent against the pending change set.
2. Immediately after that review pass, run the `Python Documentation Follow-up` custom agent against the same pending change set.
3. Read all findings carefully, prioritizing high and medium severity review items and any concrete documentation gaps identified by the documentation agent.
4. Apply the recommended fixes in code, tests, docstrings, `README.md`, and `docs/` as needed.
5. Re-run relevant validations (for example `pytest` or focused checks).
6. Run the `Python Review` custom agent again.
7. Run the `Python Documentation Follow-up` custom agent again after each new review pass.
8. Repeat until one of the following is true:
   - The review is effectively approved (no unresolved high/medium findings), or
   - The user explicitly asks to stop.

Additional requirements during this loop:

- Enforce documentation quality based on the `documenting-python-libraries` skill, especially Google-style docstrings for public APIs.
- Ensure the documentation pass checks whether `README.md` or files under `docs/` must be updated to reflect the change.
- Ensure developer-facing code artifacts are in English (identifiers, comments, docstrings), except intentionally localized user-facing copy.
- Treat low-severity findings as optional: address them when practical, but do not block commit/PR finalization on low-only findings.
- If a recommendation is not applicable or cannot be implemented safely, explain why and propose the safest alternative before committing or opening/updating the PR.
