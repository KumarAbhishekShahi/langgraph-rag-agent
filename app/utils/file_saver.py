"""
file_saver.py
=============
Saves every agent output to a timestamped Markdown file in outputs/.

Every time the agent produces a result (analysis, Gherkin, code),
it is automatically saved as:
    outputs/20260401_003015_analyze.md
    outputs/20260401_003045_gherkin.md
    outputs/20260401_003120_code.md

This gives you a full audit trail of every run without any manual step.

Usage in agents:
    from app.utils.file_saver import save_output

    saved_path = save_output(
        intent="analyze",
        user_input="Add OAuth2 SSO...",
        answer="1. Feasibility\n..."
    )
    print(f"Saved to: {saved_path}")

For code agent (pass issue_key too):
    saved_path = save_output(
        intent="code",
        user_input="Generate code for PROJ-123",
        answer="...",
        issue_key="PROJ-123"
    )
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Output folder — auto-created on first save
OUTPUTS_DIR = Path("./outputs")

# Title lookup per intent
_TITLES = {
    "ingest":  "Ingestion Summary",
    "analyze": "Requirement Analysis",
    "gherkin": "Gherkin Issue",
    "code":    "Code Generation",
}


def save_output(
    intent:     str,
    user_input: str,
    answer:     str,
    issue_key:  Optional[str] = None,
    jira_result: Optional[dict] = None,
) -> str:
    """
    Save an agent output to a timestamped Markdown file.

    Args:
        intent      : "ingest" | "analyze" | "gherkin" | "code"
        user_input  : the user's original requirement or feature request
        answer      : the full LLM-generated answer text
        issue_key   : Jira issue key (code mode only, optional)
        jira_result : Jira creation result dict (gherkin mode only, optional)

    Returns:
        The absolute path of the saved file as a string.
        Returns empty string if saving fails (non-fatal).
    """
    try:
        OUTPUTS_DIR.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"{timestamp}_{intent}.md"
        filepath  = OUTPUTS_DIR / filename

        title = _TITLES.get(intent, "Agent Output")
        now   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            f"# {title}",
            f"",
            f"**Generated:** {now}",
            f"**Mode:** `{intent}`",
        ]

        if issue_key:
            lines.append(f"**Jira Issue:** `{issue_key}`")

        if jira_result and jira_result.get("success"):
            lines.append(
                f"**Jira Created:** [{jira_result['issue_key']}]"
                f"({jira_result['issue_url']})"
            )

        if user_input:
            lines += ["", "## Input", "", user_input]

        lines += ["", "## Output", "", answer]

        filepath.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Output saved → {filepath}")
        return str(filepath)

    except Exception as e:
        logger.error(f"Could not save output to file: {e}")
        return ""
