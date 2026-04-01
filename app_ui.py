"""
app_ui.py
=========
Streamlit web UI for the LangGraph Enterprise RAG Agent.

Run:
    streamlit run app_ui.py

Features:
  - Sidebar: choose mode, set Jira project key, run ingestion
  - Main panel: text input + run button
  - Streaming-style output (renders final answer with st.markdown)
  - Collapsible "Retrieved Context" expander to show RAG chunks
  - Jira issue link shown as clickable badge after Gherkin mode
  - Conversation history in sidebar (last 5 turns)
  - Dark/light mode follows Streamlit theme

Requirements:
    pip install streamlit
"""

import streamlit as st
from app.state import AgentState
from app.graph import build_graph
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LangGraph RAG Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "history"      not in st.session_state: st.session_state.history      = []
if "last_context" not in st.session_state: st.session_state.last_context = []
if "graph"        not in st.session_state: st.session_state.graph        = build_graph()

graph = st.session_state.graph


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 RAG Agent")
    st.caption("LangGraph · Jira · Confluence · ChromaDB")
    st.divider()

    mode = st.selectbox(
        "Mode",
        options=["analyze", "gherkin", "code"],
        format_func=lambda x: {
            "analyze": "📋 Analyze Requirement",
            "gherkin": "📝 Generate Gherkin + Jira",
            "code":    "💻 Generate Code",
        }[x]
    )

    st.divider()
    st.subheader("⚡ Quick Actions")

    if st.button("🔄 Run Ingestion", use_container_width=True):
        with st.spinner("Ingesting all sources..."):
            state: AgentState = {
                "user_input":           "",
                "intent":               "ingest",
                "retrieved_context":    [],
                "final_answer":         "",
                "conversation_history": st.session_state.history,
                "jira_issue_created":   None,
                "error":                None,
            }
            result = graph.invoke(state)
            st.session_state.history = result.get("conversation_history", [])
        st.success("✅ Ingestion complete!")
        with st.expander("Ingestion Summary"):
            st.text(result.get("final_answer", ""))

    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history      = []
        st.session_state.last_context = []
        st.info("History cleared.")

    st.divider()
    st.subheader("📜 Recent History")
    if st.session_state.history:
        for turn in st.session_state.history[-6:]:
            role  = turn.get("role", "?")
            icon  = "🧑" if role == "user" else "🤖"
            text  = turn.get("content", "")[:100]
            st.caption(f"{icon} **{role.title()}:** {text}...")
    else:
        st.caption("No history yet.")


# ── Main panel ────────────────────────────────────────────────────────────────
st.title("🤖 LangGraph Enterprise RAG Agent")
st.caption("Powered by Ollama · ChromaDB · LangGraph · Atlassian")

mode_labels = {
    "analyze": ("📋 Analyze Requirement",
                 "Describe the requirement or user story to analyze"),
    "gherkin": ("📝 Generate Gherkin Issue",
                 "Describe the feature — Gherkin will be generated and posted to Jira"),
    "code":    ("💻 Generate Code",
                 "Enter a Jira issue key (e.g. PROJ-148) or describe what to implement"),
}
label, placeholder = mode_labels[mode]
st.subheader(label)

user_input = st.text_area(
    "Your input",
    placeholder=placeholder,
    height=140,
    label_visibility="collapsed",
)

run_btn = st.button("▶ Run", type="primary", use_container_width=False)

if run_btn and user_input.strip():
    with st.spinner(f"Running {mode} agent..."):
        state: AgentState = {
            "user_input":           user_input,
            "intent":               mode,
            "retrieved_context":    [],
            "final_answer":         "",
            "conversation_history": st.session_state.history,
            "jira_issue_created":   None,
            "error":                None,
        }
        result = graph.invoke(state)

    st.session_state.history      = result.get("conversation_history", [])
    st.session_state.last_context = result.get("retrieved_context", [])

    # ── Error ─────────────────────────────────────────────────────────────
    if result.get("error"):
        st.error(f"❌ {result['error']}")

    else:
        # ── Jira badge ────────────────────────────────────────────────────
        jira = result.get("jira_issue_created")
        if jira and jira.get("success"):
            st.success(
                f"✅ Jira issue created: "
                f"[{jira['issue_key']}]({jira['issue_url']})"
            )

        # ── Main output ───────────────────────────────────────────────────
        st.divider()
        st.markdown(result.get("final_answer", ""), unsafe_allow_html=False)

        # ── Retrieved context expander ────────────────────────────────────
        if st.session_state.last_context:
            with st.expander(
                f"🔍 Retrieved Context ({len(st.session_state.last_context)} chunks)"
            ):
                for i, chunk in enumerate(st.session_state.last_context):
                    st.caption(f"**Chunk {i+1}**")
                    st.code(chunk[:600], language="text")
                    st.divider()

elif run_btn and not user_input.strip():
    st.warning("⚠️ Please enter some input before running.")
