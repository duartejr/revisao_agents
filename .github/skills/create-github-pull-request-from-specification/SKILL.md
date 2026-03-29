---
name: create-github-pull-request-from-specification
description: 'Create or update a GitHub pull request from .github/pull_request_template.md. Use when user asks to create PR from spec/template, verify existing PR for branch, fill title/body from template, mark ready for review, and self-assign.'
argument-hint: 'targetBranch=<branch-name>'
---

# Create GitHub Pull Request From Specification

Create or update a GitHub Pull Request using `${workspaceFolder}/.github/pull_request_template.md` as the source of truth.

## When To Use

Use this skill when the user asks to:
- create a PR from a specification/template
- open PR for current branch and fill template sections
- check if PR already exists before creating a new one
- update PR title/body based on `.github/pull_request_template.md`
- move draft PR to ready for review
- assign PR to the current user

## Inputs

- `targetBranch`: base branch for the PR (for example `main`, `dev`)
- `headBranch`: current branch (use `git branch --show-current`)

## Required Source

- Template file: `${workspaceFolder}/.github/pull_request_template.md`

If template file is missing, stop and ask the user whether to:
1. create a minimal template now, or
2. proceed with a standard PR body.

## Procedure

1. Analyze template requirements.
- Read `.github/pull_request_template.md`.
- Extract mandatory sections (context, changes, testing, risks, checklist, links).

2. Detect existing open PR for current branch.
- Check open PR where `head == current branch`.
- If one exists, update that PR instead of creating a new one.

3. Create draft PR only when none exists.
- Use base `targetBranch` and head `current branch`.
- Start as draft when possible.

4. Gather changed content for summary.
- Inspect diff (`git diff target...head` or PR diff).
- Build concise bullets describing behavior changes, not only file names.

5. Update PR title and body from template.
- Fill all template sections with concrete data from code changes and tests.
- Keep title specific and action-oriented.

6. Mark PR ready for review.
- Convert draft -> ready after body/title are complete.

7. Assign PR to PR author.
- Resolve current authenticated user.
- Assign PR to that user.

8. Return PR URL to user.

## Branching Logic

- Existing PR found:
  - skip creation
  - update title/body
  - set ready-for-review if still draft
  - assign user

- No PR found:
  - create draft PR
  - update title/body
  - set ready-for-review
  - assign user

- Internet/API unavailable:
  - provide exact commands user can run later
  - include prepared title/body text

## Tool Preference

Prefer dedicated PR tools when available (e.g., get/list/create/update PR).

If unavailable, fallback to GitHub CLI:
- `gh pr list --head <branch> --state open`
- `gh pr create --base <targetBranch> --head <branch> --draft --title "..." --body "..."`
- `gh pr edit <number> --title "..." --body "..."`
- `gh pr ready <number>`
- `gh pr edit <number> --add-assignee @me`

## Completion Checks

Before finishing, verify:
- exactly one open PR for current branch
- title reflects specification scope
- body sections from template are filled with non-placeholder content
- PR is ready for review (not draft)
- PR assigned to creator
- PR URL returned to user

## Output Contract

Always return:
- PR URL
- base/head branches
- whether PR was created or updated
- quick checklist of completed steps (template filled, ready, assigned)
