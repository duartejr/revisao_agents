"""The Journal Reviewer follow-up prompt loads and renders all its variables."""

from revisao_agents.utils.llm_utils.prompt_loader import load_prompt


def test_followup_prompt_loads_and_renders():
    prompt = load_prompt(
        "common/journal_reviewer_followup",
        report="REPORT-BODY",
        document="DOC-BODY",
        web_block="no web",
        language="en",
    )
    assert "REPORT-BODY" in prompt.text and "DOC-BODY" in prompt.text and "no web" in prompt.text


def test_followup_prompt_forbids_edit_markers():
    prompt = load_prompt(
        "common/journal_reviewer_followup", report="r", document="d", web_block="w"
    )
    assert "READ-ONLY" in prompt.text and "REVISED_TEXT_START" in prompt.text
