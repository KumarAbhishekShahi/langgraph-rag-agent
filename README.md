# 🤖 LangGraph Enterprise RAG Agent

> **Intelligent Requirement Analysis · Gherkin Generation · Code Synthesis**
> Powered by LangGraph · ChromaDB · Ollama · Atlassian APIs

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://langchain-ai.github.io/langgraph)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-orange.svg)](https://www.trychroma.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

1. [What Is This Project?](#-what-is-this-project)
2. [What Problem Does It Solve?](#-what-problem-does-it-solve)
3. [Why This Use Case Matters](#-why-this-use-case-matters)
4. [Architecture Overview](#️-architecture-overview)
5. [Project Structure](#-project-structure)
6. [Prerequisites](#-prerequisites)
7. [Installation & Setup](#-installation--setup)
8. [Configuration](#️-configuration)
9. [Running the Project](#-running-the-project)
10. [Using the CLI](#-using-the-cli)
11. [Using the Web UI](#-using-the-web-ui)
12. [Running the Scheduler](#-running-the-scheduler)
13. [Testing](#-testing)
14. [Sample Data](#-sample-data)
15. [Troubleshooting](#-troubleshooting)
16. [FAQ](#-faq)

---

## 🎯 What Is This Project?

The **LangGraph Enterprise RAG Agent** is a production-grade AI system that connects to your organisation's existing knowledge sources — Jira, Confluence, KB articles, local files, and source code repositories — and provides four intelligent capabilities through a unified agentic interface:

| Mode | What It Does |
|------|-------------|
| **Ingest** | Loads all knowledge sources into a local ChromaDB vector store |
| **Analyze** | Deep-analyzes a new requirement across 15 structured dimensions |
| **Gherkin** | Generates a complete Gherkin BDD specification and auto-creates a Jira Story |
| **Code** | Generates implementation code grounded in both KB documentation AND existing source code |

The system runs **fully offline by default** using Ollama (no cloud API costs, no data leaving your machine). It can be switched to OpenAI or Anthropic by changing a single environment variable.

---

## 🔥 What Problem Does It Solve?

### The Enterprise Knowledge Fragmentation Problem

In most software engineering teams, knowledge is scattered across many systems:

```
Product requirements  →  Confluence / SharePoint
User stories          →  Jira / Azure DevOps
Technical patterns    →  Internal wikis / KB articles
Architecture guides   →  PDFs / Word documents
Existing source code  →  Git repositories
```

When a developer receives a new requirement, they must **manually**:

1. Search Jira for similar past stories
2. Hunt through Confluence for design patterns
3. Dig through KB articles for integration guides
4. Browse the codebase for existing implementations
5. Write analysis, acceptance criteria, and test scenarios from scratch
6. Repeat for every new feature — wasting 2–4 hours per story

This project **automates that entire workflow** in under 60 seconds.

### Concrete Pain Points Eliminated

| Manual Process | Automated by This Agent |
|----------------|------------------------|
| Searching Jira for similar stories | Semantic MMR retrieval across all indexed issues |
| Checking Confluence for design docs | Automatic Confluence page indexing and retrieval |
| Writing BDD test scenarios from scratch | AI-generated Gherkin with full Given/When/Then |
| Creating Jira Stories manually | Auto-POST to Jira via Atlassian API |
| Finding which source files to modify | Codebase-filtered RAG retrieval by file/class |
| Doing 15-point feasibility analysis | Structured 15-section requirement analysis |
| Re-doing the same search every sprint | Scheduled background re-ingestion |

---

## 💡 Why This Use Case Matters

### 1. RAG + Enterprise Data = Massive Leverage

Generic LLMs (GPT-4, Llama 3) have no knowledge of your company's specific:
- Domain terminology and abbreviations
- Past architectural decisions and trade-offs
- Existing integrations and API contracts
- Compliance constraints and security policies

By grounding every response in **your actual company data**, the agent produces analysis that a generic LLM simply cannot — without expensive fine-tuning.

### 2. Agentic Workflow Over Simple Q&A

This is not a chatbot. It is a **multi-node LangGraph agent** that:
- Routes intent automatically (ingest / analyze / gherkin / code)
- Executes multi-step pipelines per intent
- Calls external APIs (Jira, Confluence) as tool nodes
- Maintains conversation history across turns
- Saves every output to a timestamped Markdown file

### 3. Local-First, Privacy-Safe

All vector embeddings are generated locally using `all-MiniLM-L6-v2` from HuggingFace — no data is sent to any external service unless you explicitly configure OpenAI or Anthropic. This makes it suitable for:
- Regulated industries (BFSI, Healthcare, Government)
- IP-sensitive software companies
- Air-gapped enterprise environments (with local Ollama)

### 4. Production Patterns, Not Prototypes

The codebase implements patterns from real production AI systems:
- **MMR retrieval** (Maximal Marginal Relevance) — avoids returning 5 identical chunks
- **Dual-context code generation** — separate KB retrieval + codebase-filtered retrieval
- **Incremental ingestion** — adds only new documents without full rebuilds
- **DeepSeek-R1 think-block stripping** — cleans `<think>` tags from reasoning models
- **APScheduler background jobs** — keeps the knowledge base fresh automatically

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    User Interface                        │
│            CLI (main.py)  │  Streamlit (app_ui.py)       │
└──────────────────┬───────────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────────┐
│              LangGraph StateGraph (app/graph.py)         │
│                                                          │
│  route_intent() ──► ingest_node                          │
│                 ──► analysis_node                        │
│                 ──► gherkin_node ──► jira_create_tool    │
│                 ──► code_node                            │
│                         │                               │
│                   output_node ──► AgentState             │
└──────────────────┬───────────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────────┐
│                    RAG Pipeline                          │
│                                                          │
│  Sources                Embedding         Storage        │
│  ─────────────          ──────────────    ───────────    │
│  Jira issues     ──►    all-MiniLM-L6-v2  ChromaDB       │
│  Confluence pages──►    (local, offline)  (persisted     │
│  KB articles     ──►                      to disk)       │
│  PDF/CSV/HTML    ──►                                     │
│  Source code     ──►                                     │
│                                                          │
│  Retrieval: MMR search (relevance + diversity)           │
└──────────────────┬───────────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────────┐
│                    LLM Backend                           │
│   Ollama (default) │ OpenAI │ Anthropic                  │
│   llama3.2         │ gpt-4o │ claude-3-5-sonnet          │
└──────────────────────────────────────────────────────────┘
```

### Data Flow

```
1. INGEST:  Sources → Loaders → Text Splitter → Embedder → ChromaDB
2. QUERY:   User Input → MMR Retrieval → Prompt Builder → LLM → Output
3. GHERKIN: Query → Gherkin LLM → Jira API → Story Created
4. CODE:    Issue Key → Dual RAG (KB + Codebase) → Code LLM → Output
```

---

## 📁 Project Structure

```
langgraph-rag-agent/
│
├── main.py                    ← CLI entry point (interactive + --ingest --demo)
├── app_ui.py                  ← Streamlit web UI
├── scheduler.py               ← Background auto-ingestion (APScheduler)
├── config.py                  ← All settings via environment variables
├── requirements.txt           ← Python dependencies
├── .env.example               ← Environment variable template
├── .gitignore
│
├── app/
│   ├── state.py               ← AgentState TypedDict (LangGraph state schema)
│   ├── graph.py               ← LangGraph StateGraph — all nodes wired
│   ├── prompts.py             ← LangChain PromptTemplates (ANALYSIS/GHERKIN/CODE)
│   │
│   ├── agents/
│   │   ├── ingest_agent.py    ← Loads all sources, builds ChromaDB
│   │   ├── requirement_agent.py ← 15-section RAG requirement analysis
│   │   ├── gherkin_agent.py   ← Gherkin generation + Jira auto-create
│   │   └── code_agent.py      ← Dual-RAG code generation
│   │
│   ├── rag/
│   │   ├── embedder.py        ← HuggingFace all-MiniLM-L6-v2 (cached, offline)
│   │   ├── vectorstore.py     ← ChromaDB build / load / incremental add
│   │   ├── retriever.py       ← MMR + codebase-filtered search
│   │   └── loader.py          ← Orchestrates all 5 source loaders
│   │
│   ├── tools/
│   │   ├── jira_tool.py       ← Jira JQL fetch → LangChain Documents
│   │   ├── confluence_tool.py ← Confluence space fetch → Documents
│   │   ├── kb_tool.py         ← Local KB articles (MD/HTML/TXT) → Documents
│   │   ├── file_tool.py       ← PDF, CSV, HTML, TXT, MD → Documents
│   │   ├── codebase_tool.py   ← Source code scanner → Documents
│   │   └── jira_create_tool.py ← POST new Jira Story from Gherkin
│   │
│   └── utils/
│       ├── logger.py          ← Centralised logging + strip_think_block()
│       └── file_saver.py      ← Auto-saves every output to outputs/
│
├── kb_articles/               ← Your KB docs go here (MD, HTML, TXT)
│   ├── oauth2-sso-guide.md
│   ├── payment-gateway-integration.md
│   ├── api-design-standards.html
│   └── notification-service.txt
│
├── sample_data/               ← Sample files for demo/testing
│   ├── sample.txt             ← Product requirements document
│   ├── products.csv           ← Product catalogue CSV
│   └── faq.html               ← FAQ page (HTML)
│
├── outputs/                   ← Auto-saved agent outputs (gitignored)
│
└── tests/
    ├── test_loader.py         ← 12 unit tests for loaders and utils
    └── test_retriever.py      ← 4 mock-based retriever tests
```

---

## ⚙️ Prerequisites

Before you begin, ensure you have the following installed on your machine:

### Required

| Dependency | Version | Why Needed | Install |
|------------|---------|------------|---------|
| **Python** | 3.11+ | Runtime | [python.org](https://python.org) |
| **Ollama** | Latest | Local LLM inference | [ollama.com](https://ollama.com) |
| **Git** | Any | Clone / version control | [git-scm.com](https://git-scm.com) |

### Optional (for full enterprise features)

| Dependency | Why Needed |
|------------|------------|
| Jira Cloud account | Ingest Jira issues + auto-create stories |
| Confluence Cloud account | Ingest Confluence pages |
| OpenAI API key | Use GPT-4o instead of Ollama |
| Anthropic API key | Use Claude instead of Ollama |

> ✅ **Demo mode works with zero API keys** — just Python + Ollama.

---

## 🛠️ Installation & Setup

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/langgraph-rag-agent.git
cd langgraph-rag-agent
```

### Step 2 — Create a Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

> **Note:** Always activate the virtual environment before running any `python` or `pip` command. You should see `(.venv)` in your terminal prompt.

### Step 3 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** First install downloads ~500 MB (PyTorch + sentence-transformers + chromadb). Subsequent installs use cache. If you are on a slow connection, run `pip install -r requirements.txt --timeout 120`.

### Step 4 — Install and Start Ollama

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download from https://ollama.com/download/windows

# Pull the default model (llama3.2 — 2 GB download)
ollama pull llama3.2

# Start the Ollama server (keep this terminal open)
ollama serve
```

> **Note:** Ollama must be running (`ollama serve`) before you start the agent. It listens on `http://localhost:11434` by default. You can verify it is running by visiting that URL in your browser — you should see `Ollama is running`.

#### Alternative Smaller Models (for low-RAM machines)

```bash
ollama pull llama3.2:1b     # 1.3 GB — good for 8 GB RAM machines
ollama pull phi3:mini        # 2.3 GB — Microsoft Phi-3, fast on CPU
ollama pull mistral          # 4.1 GB — good general-purpose model
```

Update `LLM_MODEL` in `.env` to match whichever model you pull.

### Step 5 — Configure Environment Variables

```bash
# Copy the template
cp .env.example .env

# Edit .env with your settings
# On Windows: notepad .env
# On macOS/Linux: nano .env  or  code .env
```

See the full [Configuration](#️-configuration) section below.

---

## ⚙️ Configuration

Open `.env` and configure the following. All settings have sensible defaults — the only required field for offline use is `LLM_MODEL`.

### LLM Backend

```env
# Which LLM to use: "ollama" (default, free), "openai", or "anthropic"
LLM_BACKEND=ollama

# Ollama settings (used when LLM_BACKEND=ollama)
LLM_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434

# OpenAI settings (used when LLM_BACKEND=openai)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o

# Anthropic settings (used when LLM_BACKEND=anthropic)
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```

### RAG / Embedding Settings

```env
# HuggingFace embedding model (runs locally, no API key)
EMBEDDING_MODEL=all-MiniLM-L6-v2

# ChromaDB storage location (auto-created on first ingest)
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=enterprise_kb

# Chunk size for text splitting (tokens, approx)
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# MMR retrieval settings
RETRIEVAL_K=5            # chunks returned per query
RETRIEVAL_FETCH_K=20     # candidates considered before MMR filtering
RETRIEVAL_LAMBDA_MULT=0.5  # 0.0 = max diversity, 1.0 = max relevance
```

### Jira Integration (Optional)

```env
JIRA_URL=https://yourcompany.atlassian.net
JIRA_USERNAME=your.email@company.com
JIRA_API_TOKEN=your-api-token-here
JIRA_PROJECT_KEY=MYPROJ
JIRA_JQL=project=MYPROJ AND updated>=-30d ORDER BY updated DESC
JIRA_MAX_RESULTS=100
```

> **Getting your Jira API token:**
> 1. Go to https://id.atlassian.com/manage-profile/security
> 2. Click **Create API token**
> 3. Copy the token and paste it as `JIRA_API_TOKEN`

### Confluence Integration (Optional)

```env
CONFLUENCE_URL=https://yourcompany.atlassian.net
CONFLUENCE_USERNAME=your.email@company.com
CONFLUENCE_API_TOKEN=your-api-token-here
CONFLUENCE_SPACE_KEY=MYSPACE
CONFLUENCE_MAX_PAGES=50
```

> **Finding your Confluence space key:**
> Navigate to your Confluence space. The key is visible in the URL:
> `https://yourcompany.atlassian.net/wiki/spaces/MYSPACE/` → key is `MYSPACE`

### Local Data Sources

```env
# Folder for KB articles (MD, HTML, TXT files)
KB_ARTICLES_PATH=./kb_articles

# Folder for other documents (PDF, CSV, HTML, TXT, MD)
SAMPLE_DATA_PATH=./sample_data

# Path to your source code repository (leave empty to skip)
CODEBASE_PATH=
# Example: CODEBASE_PATH=C:/projects/my-spring-app/src
```

### Scheduler

```env
# How often the background ingestion runs (hours)
SCHEDULER_INTERVAL_HOURS=6
```

### Logging

```env
# Log level: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL=INFO
```

---

## 🚀 Running the Project

There are three ways to run the agent. Start with **Option A** for your first run.

### Option A — Demo Mode (No API Keys, Quickest Start)

This runs a full ingestion on the sample KB articles and performs a demonstration analysis. No Jira or Confluence credentials needed.

```bash
# Make sure Ollama is running in another terminal: ollama serve

python main.py --demo
```

Expected output:
```
╔══════════════════════════════════════════════════════════╗
║    LangGraph Enterprise RAG Agent  v1.0                 ║
╚══════════════════════════════════════════════════════════╝

[DEMO MODE] Running ingestion on sample KB articles...

[SOURCE 1/5] Jira Issues
[SKIP] Jira not configured — skipping
...
[SOURCE 4/5] Local Files (PDF, HTML, TXT, CSV)
[INFO] Found 3 local files in: ./sample_data
[OK] Loaded 3 local files

[STEP] Generating embeddings and storing in ChromaDB...
[OK] ChromaDB vectorstore saved to: ./chroma_db

[DEMO MODE] Analyzing sample requirement...
[STEP 3] Generating analysis (model: llama3.2)...
...
[DEMO] ✅ Demo complete. Check outputs/ folder for saved results.
```

### Option B — Interactive CLI

```bash
python main.py
```

The numbered menu will appear:
```
┌─ MENU ──────────────────────────────────────────────────┐
│  1  Ingest all sources → build ChromaDB                 │
│  2  Analyze requirement                                  │
│  3  Generate Gherkin + create Jira Story                │
│  4  Generate code for a Jira issue                      │
│  5  Show conversation history                           │
│  0  Exit                                                │
└─────────────────────────────────────────────────────────┘
  Choose [0-5]:
```

**Recommended first-run sequence:**
1. Choose `1` → run ingestion (builds ChromaDB from your sources)
2. Choose `2` → paste a requirement and see the analysis
3. Choose `3` → generate a Gherkin specification
4. Choose `4` → generate code (provide a Jira issue key)

### Option C — Ingestion Only (for CI/CD pipelines)

```bash
python main.py --ingest
```

This runs ingestion and exits immediately. Useful for scheduled jobs in CI/CD.

---

## 💻 Using the CLI

### Ingestion (Option 1)

```
Choose [0-5]: 1
```

- Connects to all configured sources (Jira, Confluence, KB, files, codebase)
- Unconfigured sources are automatically skipped (no error)
- Creates `./chroma_db/` folder with all vectors

### Requirement Analysis (Option 2)

```
Choose [0-5]: 2

Enter the requirement to analyze:
(Press Enter twice or type '.' to submit)

> We need to add OAuth2 Single Sign-On (SSO) to our customer portal.
> Users should be able to log in with their Microsoft Azure AD account.
> The system must support PKCE flow, refresh tokens, and auto-logout after 8 hours.
>
```

The agent will produce a **15-section analysis** including:
- Feasibility assessment
- Dependencies and integrations
- Risks and mitigations
- Acceptance criteria
- Story point estimate
- Similar past issues from your Jira

### Gherkin Generation (Option 3)

```
Choose [0-5]: 3

Enter the feature request for Gherkin generation:
> Customer wants to pay via UPI on the checkout page.
.
```

Output includes:
- Complete Gherkin BDD spec (Feature + multiple Scenarios)
- Acceptance criteria
- Jira Story creation result (if Jira is configured)
- Direct URL to the new Jira issue

### Code Generation (Option 4)

```
Choose [0-5]: 4

Enter a Jira issue key (e.g. PROJ-148) or describe what to code:
> PROJ-148
.
```

> **Tip:** Providing an actual Jira issue key (e.g. `PROJ-148`) gives much better results because the agent fetches the full issue details from the vectorstore. You can also describe what to implement in free text.

The agent uses **dual retrieval**:
1. KB + Confluence context (for patterns, API contracts)
2. Codebase context (filtered to `source_type=codebase` — finds the most relevant existing source files)

---

## 🌐 Using the Web UI

```bash
streamlit run app_ui.py
```

Open your browser at `http://localhost:8501`.

### UI Features

- **Sidebar:** Select mode (Analyze / Gherkin / Code), run ingestion, clear history
- **Main panel:** Text area for input + Run button
- **Output:** Rendered Markdown with Jira badge (if issue created)
- **Context expander:** Shows the exact RAG chunks used for each response
- **History panel:** Last 6 conversation turns

> **Note:** The Streamlit UI does not stream tokens word-by-word (Streamlit renders on completion). For streaming output, use the CLI.

---

## ⏰ Running the Scheduler

The scheduler keeps ChromaDB fresh by running ingestion automatically in the background.

```bash
# Start scheduler (ingests every 6 hours)
python scheduler.py

# Run one ingestion immediately, then schedule every 6 hours
python scheduler.py --run-now

# Override interval to every 2 hours
python scheduler.py --interval 2

# Override interval to every 30 minutes (testing)
python scheduler.py --interval 0.5
```

Expected output:
```
╔══════════════════════════════════════════════════════════╗
║  LangGraph RAG Agent — Background Scheduler             ║
╚══════════════════════════════════════════════════════════╝
  Interval  : every 6.0 hour(s)
  Log file  : ingestion_log.txt
  Press Ctrl+C to stop.

[INFO] Running immediate ingestion before scheduling...
```

All scheduler runs are logged to `ingestion_log.txt` in the project root.

---

## 🧪 Testing

### Run All Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run with short tracebacks (cleaner for CI)
pytest tests/ -v --tb=short

# Run a specific test file
pytest tests/test_loader.py -v

# Run a specific test class
pytest tests/test_loader.py::TestKbTool -v

# Run a specific test
pytest tests/test_loader.py::TestKbTool::test_loads_markdown_files -v
```

### Test Coverage

```bash
pip install pytest-cov
pytest tests/ --cov=app --cov-report=term-missing
```

### What Is Tested

| Test File | Test Class | Tests |
|-----------|-----------|-------|
| `test_loader.py` | `TestKbTool` | MD loading, HTML stripping, empty file skip, missing path |
| `test_loader.py` | `TestFileTool` | TXT loading, CSV loading, missing path |
| `test_loader.py` | `TestLoggerUtils` | format_history, strip_think_block |
| `test_loader.py` | `TestJiraCreateHelpers` | feature name extraction, label detection, graceful skip |
| `test_retriever.py` | `TestRetriever` | Formatted output, empty query, no-docs fallback, multi-chunk |

> **Note:** Tests use `tmp_path` fixtures and mock vectorstores — no real Jira/Confluence/ChromaDB connection is needed to run the tests.

### Expected Test Output

```
tests/test_loader.py::TestKbTool::test_loads_markdown_files PASSED
tests/test_loader.py::TestKbTool::test_loads_html_files PASSED
tests/test_loader.py::TestKbTool::test_skips_empty_files PASSED
tests/test_loader.py::TestKbTool::test_returns_empty_if_path_missing PASSED
tests/test_loader.py::TestFileTool::test_loads_txt_file PASSED
tests/test_loader.py::TestFileTool::test_loads_csv_file PASSED
tests/test_loader.py::TestFileTool::test_returns_empty_if_path_missing PASSED
tests/test_loader.py::TestLoggerUtils::test_format_history_empty PASSED
tests/test_loader.py::TestLoggerUtils::test_format_history_recent_turns PASSED
tests/test_loader.py::TestLoggerUtils::test_strip_think_block PASSED
tests/test_loader.py::TestLoggerUtils::test_strip_think_block_no_tag PASSED
tests/test_loader.py::TestJiraCreateHelpers::test_extract_feature_name PASSED
tests/test_loader.py::TestJiraCreateHelpers::test_extract_labels_auth PASSED
tests/test_loader.py::TestJiraCreateHelpers::test_extract_labels_payment PASSED
tests/test_loader.py::TestJiraCreateHelpers::test_jira_create_skipped_if_not_configured PASSED
tests/test_retriever.py::TestRetriever::test_returns_formatted_strings PASSED
tests/test_retriever.py::TestRetriever::test_empty_query_returns_fallback PASSED
tests/test_retriever.py::TestRetriever::test_no_docs_returns_fallback PASSED
tests/test_retriever.py::TestRetriever::test_multiple_chunks_returned PASSED

19 passed in 3.42s
```

---

## 📄 Sample Data

The `kb_articles/` and `sample_data/` folders contain ready-to-use sample files for immediate testing without any external integrations.

### KB Articles (`kb_articles/`)

| File | Content |
|------|---------|
| `oauth2-sso-guide.md` | OAuth2/OIDC integration guide with Spring Boot config examples |
| `payment-gateway-integration.md` | Stripe/Razorpay patterns, PCI-DSS requirements, refund rules |
| `api-design-standards.html` | REST API design standards, HTTP methods, pagination, rate limiting |
| `notification-service.txt` | Email/SMS/Push notification service integration guide |

### Sample Data (`sample_data/`)

| File | Content |
|------|---------|
| `sample.txt` | Enterprise platform PRD — auth, orders, billing, reporting requirements |
| `products.csv` | 10-row product catalogue with pricing and supplier data |
| `faq.html` | Customer FAQ covering accounts, payments, API, security |

### Adding Your Own Files

```bash
# Add KB articles (any of: .md .html .txt)
cp your-architecture-doc.md kb_articles/
cp your-integration-guide.html kb_articles/

# Add documents (PDF, CSV, HTML, TXT, MD)
cp requirements-v2.pdf sample_data/
cp product-catalogue.csv sample_data/

# Re-run ingestion to index new files
python main.py --ingest
```

---

## 🔧 Troubleshooting

### `ModuleNotFoundError: No module named 'langchain.schema'`

```bash
# langchain.schema was removed in LangChain 0.3+
# Fix: install langchain-core and langchain-chroma
pip install langchain-core langchain-chroma --upgrade
```

### `ModuleNotFoundError: No module named 'langchain_chroma'`

```bash
pip install langchain-chroma
```

### `ConnectionRefusedError` or `Ollama not running`

```bash
# Make sure Ollama server is started
ollama serve

# Verify it is listening
curl http://localhost:11434
# Expected: "Ollama is running"

# Check your model is pulled
ollama list
```

### `RuntimeError: ChromaDB not found`

```bash
# You must run ingestion before querying
python main.py --ingest
# or
# Choose option 1 in the interactive menu
```

### `Jira ingestion failed: 401 Unauthorized`

- Double-check `JIRA_USERNAME` is your **email address** (not username)
- Regenerate your Jira API token at https://id.atlassian.com/manage-profile/security
- Ensure `JIRA_URL` does **not** have a trailing slash

### Embeddings very slow on first run

The first run downloads the `all-MiniLM-L6-v2` model (~80 MB) and generates embeddings. On CPU this can take 1–3 minutes for large datasets. Subsequent runs use the cached model and pre-built ChromaDB — they are instant.

### Out of memory during embedding

Reduce the batch size by setting a smaller dataset first. Or use a lighter embedding model:
```env
EMBEDDING_MODEL=all-MiniLM-L2-v2   # smaller, faster, slightly less accurate
```

### `AttributeError: 'NoneType' object has no attribute 'invoke'`

The LangGraph was built before all imports resolved. Restart the Python process:
```bash
# Close and re-activate venv, then re-run
deactivate && source .venv/bin/activate
python main.py
```

---

## ❓ FAQ

**Q: Do I need a GPU to run this?**
No. The default embedding model (`all-MiniLM-L6-v2`) and Ollama's `llama3.2:1b` both run comfortably on CPU with 8 GB RAM. A GPU will make it faster but is not required.

**Q: Can I use this with Azure OpenAI instead of standard OpenAI?**
Not yet out-of-the-box, but you can modify `_get_llm()` in any agent file to use `AzureChatOpenAI` from `langchain_openai`.

**Q: My company uses GitHub Issues instead of Jira. Can I ingest those?**
The current codebase supports Jira only. To add GitHub Issues, create `app/tools/github_tool.py` following the same `fetch_X() → List[Document]` pattern as the other tools, and add it to `app/rag/loader.py`.

**Q: How do I add a new KB article folder?**
Add files to `kb_articles/` and re-run ingestion. Sub-folders are scanned recursively. Supported formats: `.md`, `.html`, `.htm`, `.txt`.

**Q: Can I index a private GitHub repository as codebase?**
Yes. Clone the repo locally first, then set `CODEBASE_PATH=/path/to/cloned/repo` in `.env`. The codebase tool scans the local filesystem — no GitHub API needed.

**Q: What happens if Jira is down during ingestion?**
Each source loader is isolated. If Jira fails, it logs a warning and returns an empty list. All other sources (Confluence, KB, files, codebase) continue unaffected.

**Q: How do I clear and rebuild the vector store?**
```bash
rm -rf ./chroma_db
python main.py --ingest
```

**Q: Can I run multiple agents in parallel?**
The current design is single-threaded per invocation. For parallel processing, run multiple independent Python processes with separate ChromaDB paths.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [LangGraph](https://langchain-ai.github.io/langgraph) — agent orchestration framework
- [ChromaDB](https://www.trychroma.com) — local vector database
- [Ollama](https://ollama.com) — local LLM inference engine
- [sentence-transformers](https://sbert.net) — `all-MiniLM-L6-v2` embeddings
- [Atlassian Python API](https://atlassian-python-api.readthedocs.io) — Jira/Confluence client
- [Streamlit](https://streamlit.io) — rapid web UI framework
