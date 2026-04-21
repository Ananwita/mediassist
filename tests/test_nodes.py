import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch

from agent.state import MediAssistState


# ── Helpers ───────────────────────────────────────────────────────────────────

def base_state(**overrides) -> MediAssistState:
    """Return a fully populated state with safe defaults."""
    state: MediAssistState = {
        "question":     "What are the OPD timings?",
        "messages":     [],
        "patient_name": None,
        "route":        "retrieve",
        "retrieved":    "",
        "sources":      [],
        "tool_result":  "",
        "answer":       "",
        "faithfulness": 1.0,
        "eval_retries": 0,
    }
    state.update(overrides)
    return state


def run(label: str, passed: bool, detail: str = ""):
    marker = "PASS" if passed else "FAIL"
    line = f"  [{marker}]  {label}"
    if detail:
        line += f" — {detail}"
    print(line)
    return passed


# ── memory_node ───────────────────────────────────────────────────────────────

def test_memory_node():
    from agent.nodes import memory_node
    print("\nmemory_node")
    results = []

    # 1. Appends question to messages
    state = base_state(question="Hello", messages=[])
    out = memory_node(state)
    results.append(run(
        "appends question to messages",
        any(m["content"] == "Hello" for m in out["messages"]),
    ))

    # 2. Extracts patient name
    state = base_state(question="My name is Priya")
    out = memory_node(state)
    results.append(run(
        "extracts patient name",
        out["patient_name"] == "Priya",
        f"got: {out['patient_name']}",
    ))

    # 3. Capitalises patient name
    state = base_state(question="my name is ravi")
    out = memory_node(state)
    results.append(run(
        "capitalises patient name",
        out["patient_name"] == "Ravi",
        f"got: {out['patient_name']}",
    ))

    # 4. Does not overwrite existing name if no new name in question
    state = base_state(question="What are OPD timings?", patient_name="Arjun")
    out = memory_node(state)
    results.append(run(
        "preserves existing name when no new name in question",
        out["patient_name"] == "Arjun",
        f"got: {out['patient_name']}",
    ))

    # 5. Sliding window — keeps only last 6 messages
    old_msgs = [{"role": "user", "content": f"msg{i}"} for i in range(10)]
    state = base_state(question="New question", messages=old_msgs)
    out = memory_node(state)
    results.append(run(
        "sliding window keeps last 6 + new = 7 messages",
        len(out["messages"]) <= 7,
        f"got: {len(out['messages'])} messages",
    ))

    # 6. Returns required keys
    state = base_state()
    out = memory_node(state)
    results.append(run(
        "returns messages and patient_name keys",
        "messages" in out and "patient_name" in out,
    ))

    return results


# ── router_node ───────────────────────────────────────────────────────────────

def test_router_node():
    print("\nrouter_node")
    results = []

    # Mock the LLM to avoid API calls
    mock_llm = MagicMock()

    # 1. Emergency keyword triggers emergency route without LLM call
    with patch("agent.nodes.llm", mock_llm):
        from agent.nodes import router_node
        state = base_state(question="I am having a heart attack")
        out = router_node(state)
        results.append(run(
            "emergency keyword → route=emergency (no LLM call)",
            out["route"] == "emergency",
            f"got: {out['route']}",
        ))
        results.append(run(
            "emergency route does not call LLM",
            mock_llm.invoke.call_count == 0,
        ))

    # 2. LLM returns 'retrieve'
    mock_llm.reset_mock()
    mock_llm.invoke.return_value.content = "retrieve"
    with patch("agent.nodes.llm", mock_llm):
        state = base_state(question="What is the cardiology fee?")
        out = router_node(state)
        results.append(run(
            "LLM returns 'retrieve' → route=retrieve",
            out["route"] == "retrieve",
            f"got: {out['route']}",
        ))

    # 3. LLM returns 'tool'
    mock_llm.invoke.return_value.content = "tool"
    with patch("agent.nodes.llm", mock_llm):
        state = base_state(question="Is the OPD open right now?")
        out = router_node(state)
        results.append(run(
            "LLM returns 'tool' → route=tool",
            out["route"] == "tool",
            f"got: {out['route']}",
        ))

    # 4. LLM returns 'skip'
    mock_llm.invoke.return_value.content = "skip"
    with patch("agent.nodes.llm", mock_llm):
        state = base_state(question="Hello there!")
        out = router_node(state)
        results.append(run(
            "LLM returns 'skip' → route=skip",
            out["route"] == "skip",
            f"got: {out['route']}",
        ))

    # 5. LLM returns unexpected string → falls back to 'retrieve'
    mock_llm.invoke.return_value.content = "banana"
    with patch("agent.nodes.llm", mock_llm):
        state = base_state(question="Some question")
        out = router_node(state)
        results.append(run(
            "unexpected LLM output falls back to 'retrieve'",
            out["route"] == "retrieve",
            f"got: {out['route']}",
        ))

    # 6. All other emergency keywords
    emergency_qs = ["chest pain", "call ambulance", "he is unconscious", "severe bleeding"]
    with patch("agent.nodes.llm", mock_llm):
        for eq in emergency_qs:
            out = router_node(base_state(question=eq))
            results.append(run(
                f"'{eq}' → emergency",
                out["route"] == "emergency",
                f"got: {out['route']}",
            ))

    return results


# ── retrieval_node ────────────────────────────────────────────────────────────

def test_retrieval_node():
    from agent.nodes import retrieval_node
    print("\nretrieval_node")
    results = []

    # Mock RAGPipeline
    mock_rag = MagicMock()
    mock_rag.retrieve.return_value = (
        "[OPD Timings]\nGeneral OPD: 8:00 AM to 2:00 PM.",
        ["OPD Timings"],
    )

    # 1. Returns retrieved context
    state = base_state(question="What are the OPD timings?")
    out = retrieval_node(state, rag=mock_rag)
    results.append(run(
        "returns non-empty retrieved context",
        len(out["retrieved"]) > 0,
    ))

    # 2. Returns sources list
    results.append(run(
        "returns sources list",
        out["sources"] == ["OPD Timings"],
        f"got: {out['sources']}",
    ))

    # 3. Passes question to rag.retrieve
    results.append(run(
        "calls rag.retrieve with the question",
        mock_rag.retrieve.call_args[0][0] == "What are the OPD timings?",
    ))

    # 4. Handles empty retrieval gracefully
    mock_rag.retrieve.return_value = ("", [])
    state = base_state(question="Random unknown query")
    out = retrieval_node(state, rag=mock_rag)
    results.append(run(
        "handles empty retrieval — retrieved='' sources=[]",
        out["retrieved"] == "" and out["sources"] == [],
    ))

    return results


# ── skip_node ─────────────────────────────────────────────────────────────────

def test_skip_node():
    from agent.nodes import skip_node
    print("\nskip_node")
    results = []

    state = base_state()
    out = skip_node(state)

    results.append(run("returns retrieved=''",  out["retrieved"] == ""))
    results.append(run("returns sources=[]",    out["sources"] == []))
    results.append(run("returns exactly 2 keys", set(out.keys()) == {"retrieved", "sources"}))

    return results


# ── tool_node ─────────────────────────────────────────────────────────────────

def test_tool_node():
    from agent.nodes import tool_node
    print("\ntool_node")
    results = []

    # 1. Returns a string
    state = base_state(question="What is today's date?")
    out = tool_node(state)
    results.append(run(
        "returns tool_result as string",
        isinstance(out["tool_result"], str),
    ))

    # 2. Contains date/time information
    results.append(run(
        "tool_result contains date/time info",
        any(word in out["tool_result"].lower() for word in ["date", "time", "monday", "tuesday",
            "wednesday", "thursday", "friday", "saturday", "sunday"]),
        f"got: {out['tool_result'][:80]}",
    ))

    # 3. Never raises — even on weird input
    state = base_state(question="!@#$%^&*()")
    try:
        out = tool_node(state)
        results.append(run("never raises on strange input", True))
    except Exception as e:
        results.append(run("never raises on strange input", False, str(e)))

    # 4. Returns only tool_result key
    state = base_state(question="What time is it?")
    out = tool_node(state)
    results.append(run(
        "returns tool_result key",
        "tool_result" in out,
    ))

    return results


# ── answer_node ───────────────────────────────────────────────────────────────

def test_answer_node():
    print("\nanswer_node")
    results = []

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "General OPD is open from 8:00 AM to 2:00 PM."

    # 1. Emergency route bypasses LLM entirely
    with patch("agent.nodes.llm", mock_llm):
        from agent.nodes import answer_node
        state = base_state(route="emergency")
        mock_llm.reset_mock()
        out = answer_node(state)
        results.append(run(
            "emergency route does not call LLM",
            mock_llm.invoke.call_count == 0,
        ))
        results.append(run(
            "emergency answer contains emergency number",
            "040-9999-0000" in out["answer"],
            f"got: {out['answer'][:80]}",
        ))

    # 2. Normal retrieve route calls LLM
    mock_llm.reset_mock()
    mock_llm.invoke.return_value.content = "OPD is open 8 AM to 2 PM."
    with patch("agent.nodes.llm", mock_llm):
        state = base_state(
            route="retrieve",
            retrieved="[OPD Timings]\nGeneral OPD: 8:00 AM to 2:00 PM.",
            sources=["OPD Timings"],
        )
        out = answer_node(state)
        results.append(run(
            "retrieve route calls LLM",
            mock_llm.invoke.call_count == 1,
        ))
        results.append(run(
            "returns non-empty answer",
            len(out["answer"]) > 0,
        ))

    # 3. Patient name appears in system prompt
    mock_llm.reset_mock()
    mock_llm.invoke.return_value.content = "Hello Meera, OPD is 8 AM to 2 PM."
    with patch("agent.nodes.llm", mock_llm):
        state = base_state(
            route="retrieve",
            patient_name="Meera",
            retrieved="[OPD Timings]\nGeneral OPD: 8:00 AM to 2:00 PM.",
        )
        answer_node(state)
        call_args = str(mock_llm.invoke.call_args)
        results.append(run(
            "patient name injected into LLM call",
            "Meera" in call_args,
            f"name in prompt: {'Meera' in call_args}",
        ))

    # 4. Retry instruction added when eval_retries >= 1
    mock_llm.reset_mock()
    mock_llm.invoke.return_value.content = "I don't have that information."
    with patch("agent.nodes.llm", mock_llm):
        state = base_state(
            route="retrieve",
            retrieved="[OPD Timings]\nGeneral OPD: 8:00 AM to 2:00 PM.",
            eval_retries=1,
        )
        answer_node(state)
        call_args = str(mock_llm.invoke.call_args)
        results.append(run(
            "retry instruction added when eval_retries=1",
            "RETRY" in call_args or "retry" in call_args.lower() or "threshold" in call_args.lower(),
        ))

    # 5. Returns 'answer' key
    mock_llm.invoke.return_value.content = "Test answer."
    with patch("agent.nodes.llm", mock_llm):
        state = base_state(route="retrieve", retrieved="some context")
        out = answer_node(state)
        results.append(run("returns answer key", "answer" in out))

    return results


# ── eval_node ─────────────────────────────────────────────────────────────────

def test_eval_node():
    print("\neval_node")
    results = []

    mock_llm = MagicMock()

    # 1. Skips eval when retrieved is empty — returns 1.0
    with patch("agent.nodes.llm", mock_llm):
        from agent.nodes import eval_node
        state = base_state(retrieved="", answer="Hello!", eval_retries=0)
        out = eval_node(state)
        results.append(run(
            "empty retrieved → faithfulness=1.0 (skip eval)",
            out["faithfulness"] == 1.0,
            f"got: {out['faithfulness']}",
        ))
        results.append(run(
            "empty retrieved → LLM not called",
            mock_llm.invoke.call_count == 0,
        ))

    # 2. Returns faithfulness score from LLM
    mock_llm.reset_mock()
    mock_llm.invoke.return_value.content = "0.85"
    with patch("agent.nodes.llm", mock_llm):
        state = base_state(
            retrieved="[OPD Timings]\nGeneral OPD: 8 AM to 2 PM.",
            answer="OPD is open from 8 AM to 2 PM.",
            eval_retries=0,
        )
        out = eval_node(state)
        results.append(run(
            "faithfulness score parsed from LLM",
            out["faithfulness"] == 0.85,
            f"got: {out['faithfulness']}",
        ))

    # 3. Score clamped to [0.0, 1.0]
    mock_llm.invoke.return_value.content = "1.5"
    with patch("agent.nodes.llm", mock_llm):
        state = base_state(retrieved="context", answer="answer", eval_retries=0)
        out = eval_node(state)
        results.append(run(
            "score > 1.0 clamped to 1.0",
            out["faithfulness"] == 1.0,
            f"got: {out['faithfulness']}",
        ))

    mock_llm.invoke.return_value.content = "-0.3"
    with patch("agent.nodes.llm", mock_llm):
        state = base_state(retrieved="context", answer="answer", eval_retries=0)
        out = eval_node(state)
        results.append(run(
            "score < 0.0 clamped to 0.0",
            out["faithfulness"] == 0.0,
            f"got: {out['faithfulness']}",
        ))

    # 4. eval_retries incremented by 1
    mock_llm.invoke.return_value.content = "0.9"
    with patch("agent.nodes.llm", mock_llm):
        state = base_state(retrieved="context", answer="answer", eval_retries=0)
        out = eval_node(state)
        results.append(run(
            "eval_retries incremented from 0 to 1",
            out["eval_retries"] == 1,
            f"got: {out['eval_retries']}",
        ))

    # 5. LLM failure defaults to 1.0 (safe)
    mock_llm.invoke.side_effect = Exception("API timeout")
    with patch("agent.nodes.llm", mock_llm):
        state = base_state(retrieved="context", answer="answer", eval_retries=0)
        out = eval_node(state)
        results.append(run(
            "LLM failure → defaults to faithfulness=1.0",
            out["faithfulness"] == 1.0,
            f"got: {out['faithfulness']}",
        ))
    mock_llm.invoke.side_effect = None

    return results


# ── save_node ─────────────────────────────────────────────────────────────────

def test_save_node():
    from agent.nodes import save_node
    print("\nsave_node")
    results = []

    # 1. Appends assistant answer to messages
    state = base_state(
        messages=[{"role": "user", "content": "What are OPD timings?"}],
        answer="OPD is open from 8 AM to 2 PM.",
    )
    out = save_node(state)
    last = out["messages"][-1]
    results.append(run(
        "appends assistant message",
        last["role"] == "assistant" and last["content"] == "OPD is open from 8 AM to 2 PM.",
        f"got role={last['role']}",
    ))

    # 2. Total message count increases by 1
    results.append(run(
        "message count increases by 1",
        len(out["messages"]) == 2,
        f"got: {len(out['messages'])}",
    ))

    # 3. Works on empty messages list
    state = base_state(messages=[], answer="Hello!")
    out = save_node(state)
    results.append(run(
        "works on empty messages list",
        len(out["messages"]) == 1,
    ))

    # 4. Returns messages key
    results.append(run("returns messages key", "messages" in out))

    return results


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all():
    print("=" * 55)
    print("MEDIASSIST — NODE ISOLATION TESTS")
    print("=" * 55)

    all_results = []
    all_results += test_memory_node()
    all_results += test_router_node()
    all_results += test_retrieval_node()
    all_results += test_skip_node()
    all_results += test_tool_node()
    all_results += test_answer_node()
    all_results += test_eval_node()
    all_results += test_save_node()

    total  = len(all_results)
    passed = sum(all_results)
    failed = total - passed

    print()
    print("=" * 55)
    print(f"RESULTS  {passed}/{total} passed   {failed} failed")
    print("=" * 55)

    if failed > 0:
        print("Some tests failed. Fix before assembling the graph.")
    else:
        print("All node tests passed. Safe to proceed to graph assembly.")

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
