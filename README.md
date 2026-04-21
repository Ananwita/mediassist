# MediAssist AI — Intelligent Patient Query Assistant

**Domain:** Hospital Patient Services  
**Hospital:** MediCare General Hospital, Bhubaneswar  
**Course:** Agentic AI 2026 — Capstone Project  

---

## Problem Statement

MediCare General Hospital receives 200+ helpline calls daily. 80% ask the same five questions.
MediAssist is a 24/7 LangGraph-powered AI assistant that answers from a verified knowledge base,
remembers the conversation, uses live tools for time-sensitive queries, and never fabricates information.

---

## Architecture

```
User question
     ↓
[memory_node]    → sliding window · extract patient name
     ↓
[router_node]    → retrieve / tool / skip / emergency
     ↓
[retrieval_node / tool_node / skip_node]
     ↓
[answer_node]    → grounded LLM response
     ↓
[eval_node]      → faithfulness 0.0–1.0 · retry if < 0.7
     ↓
[save_node]      → append to messages → END
```

---

## Tech Stack

| Component       | Library                        |
|-----------------|-------------------------------|
| Agent framework | LangGraph (StateGraph)         |
| LLM             | Groq — llama-3.3-70b-versatile |
| Vector DB       | ChromaDB                       |
| Embeddings      | SentenceTransformers           |
| UI              | Streamlit                      |
| Evaluation      | RAGAS                          |

---

## Project Structure

```
mediassist/
├── config.py              # All constants (model, thresholds, helpline numbers)
├── agent/
│   ├── knowledge_base.py  # 12 hospital documents
│   ├── rag.py             # ChromaDB pipeline
│   ├── state.py           # MediAssistState TypedDict
│   ├── nodes.py           # All 8 node functions
│   └── graph.py           # StateGraph assembly + compile
├── ui/
│   └── app.py             # Streamlit UI
├── tests/
│   ├── test_nodes.py      # Isolation tests for each node
│   ├── test_graph.py      # End-to-end + memory tests
│   └── test_redteam.py    # Red-team adversarial tests
├── docs/
│   └── mediassist.md   # Submission documentation
├── scripts/
│   └── run_evaluation.py  # RAGAS baseline evaluation
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/mediassist.git
cd mediassist

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Launch the app
streamlit run ui/app.py
```

---

## Key Design Decisions

- **State-first design:** `MediAssistState` TypedDict was defined before any node was written.
- **Retrieval verification gate:** `rag.verify_retrieval()` must pass before graph compilation.
- **Emergency fast-path:** Emergency queries bypass RAG and go directly to `answer_node`.
- **Anti-hallucination contract:** `answer_node` is prompted to say "I don't know" and provide the helpline if context is missing.
- **Faithfulness retry loop:** `eval_node` scores answers; below 0.7 triggers a retry with a stricter system prompt.


