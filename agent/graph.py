import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from functools import partial

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import MediAssistState
from agent.rag import RAGPipeline
from agent.nodes import (
    memory_node,
    router_node,
    retrieval_node,
    skip_node,
    tool_node,
    answer_node,
    eval_node,
    save_node,
)
from config import FAITHFULNESS_THRESHOLD, MAX_EVAL_RETRIES


# ── Conditional edge functions ────────────────────────────────────────────────

def route_decision(state: MediAssistState) -> str:
    """
    Reads state.route (set by router_node) and returns the next node name.
    Emergency route skips retrieval and goes directly to answer_node.
    """
    route = state.get("route", "retrieve")
    if route == "emergency":
        return "answer"
    if route == "tool":
        return "tool"
    if route == "skip":
        return "skip"
    return "retrieve"


def eval_decision(state: MediAssistState) -> str:
    """
    Reads faithfulness score and retry count (both set by eval_node).
    Returns 'answer' to retry, or 'save' to accept and end.
    """
    score = state.get("faithfulness", 1.0)
    retries = state.get("eval_retries", 0)

    if score < FAITHFULNESS_THRESHOLD and retries < MAX_EVAL_RETRIES:
        return "answer"
    return "save"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph() -> tuple:
    """
    Assembles and compiles the MediAssist LangGraph StateGraph.

    Steps:
    1. Initialise and verify the RAG pipeline (gate — must pass before compile).
    2. Add all 8 nodes to the graph.
    3. Set memory_node as the entry point.
    4. Wire all fixed edges.
    5. Wire conditional edges for routing and eval retry loop.
    6. Compile with MemorySaver for thread-level conversation persistence.

    Returns:
        (compiled_app, rag_pipeline)
    """

    # ── Step 1: RAG pipeline (must verify before graph compile) ───────────────
    print("[Graph] Initialising RAG pipeline...")
    rag = RAGPipeline()
    rag.verify_retrieval()

    # ── Step 2: Instantiate the graph ─────────────────────────────────────────
    graph = StateGraph(MediAssistState)

    # ── Step 3: Add all nodes ─────────────────────────────────────────────────
    graph.add_node("memory",   memory_node)
    graph.add_node("router",   router_node)
    graph.add_node("retrieve", partial(retrieval_node, rag=rag))
    graph.add_node("tool",     tool_node)
    graph.add_node("skip",     skip_node)
    graph.add_node("answer_node",   answer_node)
    graph.add_node("eval",     eval_node)
    graph.add_node("save",     save_node)

    # ── Step 4: Set entry point ────────────────────────────────────────────────
    graph.set_entry_point("memory")

    # ── Step 5: Fixed edges ───────────────────────────────────────────────────
    graph.add_edge("memory",   "router")
    graph.add_edge("retrieve", "answer_node")
    graph.add_edge("tool",     "answer_node")
    graph.add_edge("skip",     "answer_node")
    graph.add_edge("answer_node",   "eval")
    graph.add_edge("save",     END)

    # ── Step 6: Conditional edges ─────────────────────────────────────────────
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "retrieve": "retrieve",
            "tool":     "tool",
            "skip":     "skip",
            "answer":   "answer_node",
        },
    )

    graph.add_conditional_edges(
        "eval",
        eval_decision,
        {
            "answer": "answer_node",
            "save":   "save",
        },
    )

    # ── Step 7: Compile with MemorySaver ──────────────────────────────────────
    checkpointer = MemorySaver()
    compiled_app = graph.compile(checkpointer=checkpointer)

    print("[Graph] Graph compiled successfully.\n")
    return compiled_app, rag


# ── Convenience ask() helper ──────────────────────────────────────────────────

def ask(app, question: str, thread_id: str) -> dict:
    """
    Invoke the compiled graph for a single question.

    Args:
        app       : compiled LangGraph app returned by build_graph()
        question  : patient's question string
        thread_id : unique session identifier — same thread_id preserves memory

    Returns:
        Full result dict from app.invoke() — keys: answer, route, faithfulness,
        eval_retries, sources, messages, patient_name, etc.
    """
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: MediAssistState = {
        "question":     question
    }

    result = app.invoke(initial_state, config=config)
    return result
