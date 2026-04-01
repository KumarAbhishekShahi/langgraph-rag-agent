"""
state.py
========
Shared LangGraph state definition.

AgentState is passed through every node in the graph.
Each node reads what it needs and writes its outputs back.

Fields
------
user_input          : raw text from the user (requirement, feature request, etc.)
intent              : routing key — "ingest" | "analyze" | "gherkin" | "code"
issue_key           : Jira issue key e.g. "PROJ-123" (code agent only)
issue_details       : full Jira issue dict fetched by code_agent
retrieved_context   : list of text chunks from ChromaDB
final_answer        : final LLM-generated output text
error               : error message if any node failed, else None
jira_issue_created  : result dict from jira_create_tool (gherkin agent only)
conversation_history: list of past turns {role, content} for session memory
"""

from typing import TypedDict, Optional, List


class ConversationTurn(TypedDict):
    role:    str   # "user" or "assistant"
    content: str   # text of the turn (truncated to 500 chars for storage)


class AgentState(TypedDict):
    # ── Core ──────────────────────────────────────────────────────────────────
    user_input:  str
    intent:      str                 # "ingest" | "analyze" | "gherkin" | "code"
    issue_key:   Optional[str]       # Jira issue key (code mode)

    # ── Agent outputs ─────────────────────────────────────────────────────────
    issue_details:    Optional[dict] # raw Jira issue fields (code agent)
    retrieved_context: List[str]     # ChromaDB chunks used as context
    final_answer:      str           # LLM-generated response

    # ── Error ─────────────────────────────────────────────────────────────────
    error: Optional[str]             # set if any step fails; None on success

    # ── Gherkin → Jira creation result ────────────────────────────────────────
    jira_issue_created: Optional[dict]
    # Structure when set:
    # {
    #   "success":   True | False,
    #   "issue_key": "PROJ-148",
    #   "issue_url": "https://company.atlassian.net/browse/PROJ-148",
    #   "summary":   "Payment Failure Email Notification",
    #   "error":     None | "reason string"
    # }

    # ── Session memory (Step 14) ───────────────────────────────────────────────
    conversation_history: List[ConversationTurn]
