import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import uuid
from unittest.mock import MagicMock, patch

from config import EMERGENCY_NUMBER, HELPLINE


# ── Helpers ───────────────────────────────────────────────────────────────────

def fresh_thread() -> str:
    return str(uuid.uuid4())


def run(label: str, passed: bool, detail: str = ""):
    marker = "PASS" if passed else "FAIL"
    line = f"  [{marker}]  {label}"
    if detail:
        line += f" — {detail}"
    print(line)
    return passed


def make_mock_rag(context: str = "", sources: list = None):
    """Return a mock RAGPipeline that returns preset context and sources."""
    mock_rag = MagicMock()
    mock_rag.retrieve.return_value = (
        context or "[OPD Timings]\nGeneral OPD: 8:00 AM to 2:00 PM.",
        sources or ["OPD Timings"],
    )
    mock_rag.verify_retrieval.return_value = True
    return mock_rag


def make_mock_llm(responses: list):
    """
    Return a mock LLM that returns responses in sequence.
    Each call to .invoke() pops the next string from responses.
    """
    mock_llm = MagicMock()
    side_effects = [MagicMock(content=r) for r in responses]
    mock_llm.invoke.side_effect = side_effects
    return mock_llm


def build_test_graph(llm_responses: list, rag=None):
    """
    Build the compiled graph with mocked LLM and RAGPipeline.
    Returns (compiled_app, rag_instance).
    """
    mock_rag = rag or make_mock_rag()
    mock_llm = make_mock_llm(llm_responses)

    with patch("agent.nodes.llm", mock_llm), \
         patch("agent.rag.RAGPipeline", return_value=mock_rag), \
         patch("agent.graph.RAGPipeline", return_value=mock_rag):
        from agent.graph import build_graph
        app, _ = build_graph()

    return app, mock_rag, mock_llm


# ── Test 1: End-to-end OPD query ──────────────────────────────────────────────

def test_end_to_end_opd():
    print("\nTest 1 — End-to-end: OPD timings query")
    results = []

    mock_rag = make_mock_rag(
        context="[OPD Timings]\nGeneral OPD: 8:00 AM to 2:00 PM.",
        sources=["OPD Timings"],
    )

    llm_responses = [
        "retrieve",                                     # router_node
        "General OPD is open from 8:00 AM to 2:00 PM.",  # answer_node
        "0.95",                                         # eval_node
    ]

    mock_llm = make_mock_llm(llm_responses)

    with patch("agent.nodes.llm", mock_llm), \
         patch("agent.rag.RAGPipeline", return_value=mock_rag), \
         patch("agent.graph.RAGPipeline", return_value=mock_rag):
        from agent.graph import build_graph, ask
        app, _ = build_graph()
        result = ask(app, "What are the OPD timings?", fresh_thread())

    results.append(run(
        "answer is non-empty",
        len(result.get("answer", "")) > 0,
        f"answer: {result.get('answer', '')[:60]}",
    ))
    results.append(run(
        "route is 'retrieve'",
        result.get("route") == "retrieve",
        f"got: {result.get('route')}",
    ))
    results.append(run(
        "sources returned",
        len(result.get("sources", [])) > 0,
        f"got: {result.get('sources')}",
    ))
    results.append(run(
        "faithfulness score present",
        isinstance(result.get("faithfulness"), float),
        f"got: {result.get('faithfulness')}",
    ))
    results.append(run(
        "messages contains user and assistant turns",
        len(result.get("messages", [])) >= 2,
        f"got: {len(result.get('messages', []))} messages",
    ))

    return results


# ── Test 2: Emergency query ───────────────────────────────────────────────────

def test_emergency_query():
    print("\nTest 2 — Emergency query: bypass retrieval")
    results = []

    mock_rag = make_mock_rag()
    # Emergency fast-path: router never calls LLM, answer_node never calls LLM
    # Only eval_node may call LLM (with empty retrieved, it auto-scores 1.0 and skips)
    llm_responses = []  # no LLM calls expected at all

    mock_llm = make_mock_llm(llm_responses)

    with patch("agent.nodes.llm", mock_llm), \
         patch("agent.rag.RAGPipeline", return_value=mock_rag), \
         patch("agent.graph.RAGPipeline", return_value=mock_rag):
        from agent.graph import build_graph, ask
        app, _ = build_graph()
        result = ask(app, "I am having a heart attack", fresh_thread())

    results.append(run(
        "route is 'emergency'",
        result.get("route") == "emergency",
        f"got: {result.get('route')}",
    ))
    results.append(run(
        "emergency number in answer",
        EMERGENCY_NUMBER in result.get("answer", ""),
        f"answer: {result.get('answer', '')[:80]}",
    ))
    results.append(run(
        "rag.retrieve NOT called",
        mock_rag.retrieve.call_count == 0,
        f"retrieve call count: {mock_rag.retrieve.call_count}",
    ))
    results.append(run(
        "LLM NOT called (emergency fast-path)",
        mock_llm.invoke.call_count == 0,
        f"LLM call count: {mock_llm.invoke.call_count}",
    ))

    # Additional emergency keywords
    for keyword in ["chest pain", "call ambulance", "unconscious patient"]:
        mock_rag2 = make_mock_rag()
        mock_llm2 = make_mock_llm([])
        with patch("agent.nodes.llm", mock_llm2), \
             patch("agent.rag.RAGPipeline", return_value=mock_rag2), \
             patch("agent.graph.RAGPipeline", return_value=mock_rag2):
            from agent.graph import build_graph, ask
            app2, _ = build_graph()
            r2 = ask(app2, keyword, fresh_thread())
        results.append(run(
            f"'{keyword}' → emergency route",
            r2.get("route") == "emergency",
        ))

    return results


# ── Test 3: Tool query ────────────────────────────────────────────────────────

def test_tool_query():
    print("\nTest 3 — Tool query: date/time")
    results = []

    mock_rag = make_mock_rag()

    llm_responses = [
        "tool",                                             # router_node
        "The OPD is currently open as it is a weekday.",   # answer_node
        "1.0",                                             # eval_node
    ]
    mock_llm = make_mock_llm(llm_responses)

    with patch("agent.nodes.llm", mock_llm), \
         patch("agent.rag.RAGPipeline", return_value=mock_rag), \
         patch("agent.graph.RAGPipeline", return_value=mock_rag):
        from agent.graph import build_graph, ask
        app, _ = build_graph()
        result = ask(app, "Is the OPD open right now?", fresh_thread())

    results.append(run(
        "route is 'tool'",
        result.get("route") == "tool",
        f"got: {result.get('route')}",
    ))
    results.append(run(
        "tool_result is non-empty string",
        isinstance(result.get("tool_result"), str) and len(result.get("tool_result", "")) > 0,
        f"tool_result: {result.get('tool_result', '')[:60]}",
    ))
    results.append(run(
        "tool_result contains date/time information",
        any(w in result.get("tool_result", "").lower() for w in [
            "date", "time", "monday", "tuesday", "wednesday",
            "thursday", "friday", "saturday", "sunday",
        ]),
        f"tool_result: {result.get('tool_result', '')[:60]}",
    ))
    results.append(run(
        "rag.retrieve NOT called for tool route",
        mock_rag.retrieve.call_count == 0,
        f"retrieve call count: {mock_rag.retrieve.call_count}",
    ))
    results.append(run(
        "answer is non-empty",
        len(result.get("answer", "")) > 0,
    ))

    return results


# ── Test 4: Retry loop ────────────────────────────────────────────────────────

def test_retry_loop():
    print("\nTest 4 — Eval retry loop: low faithfulness triggers retry")
    results = []

    mock_rag = make_mock_rag(
        context="[OPD Timings]\nGeneral OPD: 8:00 AM to 2:00 PM.",
        sources=["OPD Timings"],
    )

    llm_responses = [
        "retrieve",                                          # router_node
        "The OPD opens at 8 AM and closes at some time.",   # answer_node — attempt 1
        "0.45",                                              # eval_node — FAIL (below 0.7)
        "OPD is open 8:00 AM to 2:00 PM per our records.",  # answer_node — retry
        "0.92",                                              # eval_node — PASS
    ]
    mock_llm = make_mock_llm(llm_responses)

    with patch("agent.nodes.llm", mock_llm), \
         patch("agent.rag.RAGPipeline", return_value=mock_rag), \
         patch("agent.graph.RAGPipeline", return_value=mock_rag):
        from agent.graph import build_graph, ask
        app, _ = build_graph()
        result = ask(app, "What are the OPD timings?", fresh_thread())

    results.append(run(
        "final faithfulness is 0.92 (retry passed)",
        result.get("faithfulness") == 0.92,
        f"got: {result.get('faithfulness')}",
    ))
    results.append(run(
        "eval_retries is 2 (two eval cycles ran)",
        result.get("eval_retries") == 2,
        f"got: {result.get('eval_retries')}",
    ))
    results.append(run(
        "answer is non-empty after retry",
        len(result.get("answer", "")) > 0,
        f"answer: {result.get('answer', '')[:60]}",
    ))
    results.append(run(
        "answer_node was called twice (original + retry)",
        mock_llm.invoke.call_count == 5,  # router + answer1 + eval1 + answer2 + eval2
        f"total LLM calls: {mock_llm.invoke.call_count}",
    ))

    return results


# ── Test 5: Multi-turn memory ─────────────────────────────────────────────────

def test_multi_turn_memory():
    print("\nTest 5 — Multi-turn memory: patient name persists across turns")
    results = []

    THREAD = fresh_thread()

    mock_rag = make_mock_rag(
        context="[OPD Timings]\nGeneral OPD: 8:00 AM to 2:00 PM.",
        sources=["OPD Timings"],
    )

    # Turn 1: patient introduces name
    llm_responses_t1 = [
        "skip",                          # router — greeting
        "Hello Meera! How can I help?",  # answer
        "1.0",                           # eval (skip route → auto 1.0, but we provide anyway)
    ]

    # Turn 2: factual question (no name in question)
    llm_responses_t2 = [
        "retrieve",                                            # router
        "Meera, the OPD is open from 8:00 AM to 2:00 PM.",   # answer — must use name
        "0.90",                                                # eval
    ]

    # Turn 3: follow-up without any context re-stated
    llm_responses_t3 = [
        "retrieve",                                              # router
        "Meera, the evening OPD runs from 5:00 PM to 8:00 PM.", # answer
        "0.88",                                                  # eval
    ]

    with patch("agent.rag.RAGPipeline", return_value=mock_rag), \
         patch("agent.graph.RAGPipeline", return_value=mock_rag):
        from agent.graph import build_graph, ask
        app, _ = build_graph()

        # Turn 1
        mock_llm_t1 = make_mock_llm(llm_responses_t1)
        with patch("agent.nodes.llm", mock_llm_t1):
            r1 = ask(app, "Hello, my name is Meera.", THREAD)

        results.append(run(
            "Turn 1 — patient_name extracted as 'Meera'",
            r1.get("patient_name") == "Meera",
            f"got: {r1.get('patient_name')}",
        ))
        results.append(run(
            "Turn 1 — messages contains 2 entries (user + assistant)",
            len(r1.get("messages", [])) >= 2,
            f"got: {len(r1.get('messages', []))} messages",
        ))

        # Turn 2
        mock_llm_t2 = make_mock_llm(llm_responses_t2)
        with patch("agent.nodes.llm", mock_llm_t2):
            r2 = ask(app, "What are the OPD timings?", THREAD)

        results.append(run(
            "Turn 2 — patient_name still 'Meera' (memory persisted)",
            r2.get("patient_name") == "Meera",
            f"got: {r2.get('patient_name')}",
        ))
        results.append(run(
            "Turn 2 — messages accumulating across turns",
            len(r2.get("messages", [])) >= 4,
            f"got: {len(r2.get('messages', []))} messages",
        ))

        # Turn 3
        mock_llm_t3 = make_mock_llm(llm_responses_t3)
        with patch("agent.nodes.llm", mock_llm_t3):
            r3 = ask(app, "What about the evening OPD?", THREAD)

        results.append(run(
            "Turn 3 — patient_name still 'Meera' after 3 turns",
            r3.get("patient_name") == "Meera",
            f"got: {r3.get('patient_name')}",
        ))
        results.append(run(
            "Turn 3 — messages contain at least 6 entries (3 user + 3 assistant)",
            len(r3.get("messages", [])) >= 6,
            f"got: {len(r3.get('messages', []))} messages",
        ))
        results.append(run(
            "Turn 3 — same thread_id preserves full conversation",
            any(
                m.get("content", "") == "Hello, my name is Meera."
                for m in r3.get("messages", [])
            ),
        ))

    return results


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all():
    print("=" * 55)
    print("MEDIASSIST — GRAPH END-TO-END TESTS")
    print("=" * 55)

    all_results = []
    all_results += test_end_to_end_opd()
    all_results += test_emergency_query()
    all_results += test_tool_query()
    all_results += test_retry_loop()
    all_results += test_multi_turn_memory()

    total  = len(all_results)
    passed = sum(all_results)
    failed = total - passed

    print()
    print("=" * 55)
    print(f"RESULTS  {passed}/{total} passed   {failed} failed")
    print("=" * 55)

    if failed > 0:
        print("Graph tests failed. Fix before running red-team tests.")
    else:
        print("All graph tests passed. Proceed to red-team testing.")

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
