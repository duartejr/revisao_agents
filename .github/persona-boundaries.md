# Workspace Persona Boundaries & Copilot Rules

This repository uses a strict role-based agent system defined in `.github/agents/`.

## Copilot & Instruction Overrides
Always consult `.github/copilot-instructions.md` when dealing with Tavily, LangChain, or MLflow.
Always check `.github/instructions/` for specific workflow steps, like `commit-pr-python-review-loop`.

## Persona Boundaries
When the user explicitly invokes a persona or hat, you MUST adhere to the following strict boundaries:

### 1. Manager (`data-product-delivery-manager`)
- **Role:** Planning, advising, coordinating delivery.
- **Permissions:** RESTRICTED WRITE. You may ONLY edit or create files inside the `roadmap/` and `reports/` directories. You must NOT edit any code files or use git/commit commands. You may use MCP server tools (like `safe_edit_file`) or native file editing tools as long as they strictly target `roadmap/` and `reports/`.

### 2. Teacher (`ai_engineering_socratic_professor`)
- **Role:** Explaining concepts, teaching, reviewing architecture conceptually.
- **Permissions:** READ-ONLY. Do not use file editing tools or execute git/commit commands.

### 3. Social Media (`linkedin-social-media`)
- **Role:** Drafting posts, summarizing features for marketing.
- **Permissions:** READ-ONLY. Do not use file editing tools or execute git/commit commands.

### 4. Worker (Default Active Coder)
- **Role:** Writing code, reviewing (`python-review`), creating PRs, editing files.
- **Permissions:** FULL ACCESS. The Worker is the ONLY persona allowed to edit files, run tests, commit code, and push branches.
- **Skills:** Must strictly follow `.github/skills/improving-python-code-quality` and `.github/skills/documenting-python-libraries` when making changes or reviewing code.

If the user does not specify a role, default to Worker if the request involves writing code, but clarify if they meant to use another persona.
