"""
graph.py
========
Builds the LangGraph StateGraph that wires all agent nodes together.

Flow
----
                         ┌──────────────┐
                         │  START NODE  │
                         │  (intent     │
                         │   routing)   │
                         └──────┬───────┘
              ┌─────────────────┼───────────────────┐
              │                 │                   │
       "ingest"          "analyze"             "gherkin"
              │                 │                   │
     ┌────────▼───┐    ┌────────▼───┐    ┌──────────▼────┐
     │ ingest_node│    │ analysis_  │    │ gherkin_node  │
     │            │    │ node       │    │ (generate +   │
     └────────────┘    └────────────┘    │  create Jira) │
              │                 │        └───────────────┘
              │                 │                │
              └─────────────────┼────────────────┘
                                │
                        "code" ─┘
                                │
                       ┌────────▼───┐
                       │ code_node  │
                       │ (Jira +    │
                       │  codebase) │
                       └────────────┘
                                │
                    ┌───────────▼──────────┐
                    │    output_node       │
                    │ (print / save output)│
                    └──────────────────────┘
                                │
                              END

Intent routing is done in route_intent() based on state["intent"].
"""

from langgraph.graph import StateGraph, END

from app.state import AgentState
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Import agent nodes ────────────────────────────────────────────────────────
# Imported here (not at top of file) to avoid circular imports and
# allow the graph to be built before all agents are fully initialised.

def _ingest_node(state: AgentState) -> AgentState:
    from app.agents.ingest_agent import ingest_node
    return ingest_node(state)

def _analysis_node(state: AgentState) -> AgentState:
    from app.agents.requirement_agent import analysis_node
    return analysis_node(state)

def _gherkin_node(state: AgentState) -> AgentState:
    from app.agents.gherkin_agent import gherkin_node
    return gherkin_node(state)

def _code_node(state: AgentState) -> AgentState:
    from app.agents.code_agent import code_node
    return code_node(state)

def _output_node(state: AgentState) -> AgentState:
    return output_node(state)


# ── Intent router ─────────────────────────────────────────────────────────────

def route_intent(state: AgentState) -> str:
    """
    Read state["intent"] and return the next node name.
    Called by LangGraph as a conditional edge from START.
    """
    intent = state.get("intent", "").strip().lower()
    logger.info(f"Routing intent: '{intent}'")

    if intent == "ingest":
        return "ingest_node"
    elif intent == "analyze":
        return "analysis_node"
    elif intent == "gherkin":
        return "gherkin_node"
    elif intent == "code":
        return "code_node"
    else:
        logger.warning(f"Unknown intent '{intent}'. Defaulting to analysis_node.")
        return "analysis_node"


# ── Output node ───────────────────────────────────────────────────────────────

def output_node(state: AgentState) -> AgentState:
    """
    Terminal node: prints final answer and any Jira creation results.
    Streaming agents (Step 11) print while running, so this node
    only handles error display and Jira creation confirmation.
    """
    error = state.get("error")

    if error:
        print("\n" + "=" * 60)
        print("  ERROR")
        print("=" * 60)
        print(f"\n{error}\n")
        print("Tips:")
        print("  - Check your .env settings")
        print("  - Run: ollama list  (confirm model is available)")
        print("  - Run: ollama serve (if Ollama is not running)")
        return state

    # If answer wasn't already printed by streaming, print it now
    final_answer = state.get("final_answer", "")
    if final_answer:
        print("\n" + "=" * 60)
        print("  RESULT")
        print("=" * 60)
        print(f"\n{final_answer}\n")

    # Show Jira issue creation result (gherkin agent)
    jira_result = state.get("jira_issue_created")
    if jira_result:
        if jira_result.get("success"):
            print("\n" + "-" * 60)
            print(f"  ✅ Jira Issue Created: {jira_result['issue_key']}")
            print(f"     URL    : {jira_result['issue_url']}")
            print(f"     Summary: {jira_result['summary']}")
            print("-" * 60)
        else:
            print("\n" + "-" * 60)
            print(f"  ⚠️  Jira Issue NOT created: {jira_result.get('error', 'unknown error')}")
            print("-" * 60)

    return state


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph():
    """
    Assemble and compile the LangGraph StateGraph.
    Returns a compiled graph ready to call with .invoke(state).
    """
    graph = StateGraph(AgentState)

    # Register all nodes
    graph.add_node("ingest_node",   _ingest_node)
    graph.add_node("analysis_node", _analysis_node)
    graph.add_node("gherkin_node",  _gherkin_node)
    graph.add_node("code_node",     _code_node)
    graph.add_node("output_node",   _output_node)

    # Entry point: route based on intent
    graph.set_entry_point("ingest_node")   # default; overridden by conditional edge below
    graph.add_conditional_edges(
        "__start__",
        route_intent,
        {
            "ingest_node":   "ingest_node",
            "analysis_node": "analysis_node",
            "gherkin_node":  "gherkin_node",
            "code_node":     "code_node",
        }
    )

    # All agent nodes → output_node → END
    for node in ["ingest_node", "analysis_node", "gherkin_node", "code_node"]:
        graph.add_edge(node, "output_node")

    graph.add_edge("output_node", END)

    return graph.compile()
