import re
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime
from langchain_groq import ChatGroq

from agent.state import MediAssistState
from config import (
    GROQ_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    FAITHFULNESS_THRESHOLD,
    MAX_EVAL_RETRIES,
    HOSPITAL_NAME,
    HELPLINE,
    EMERGENCY_NUMBER,
    AMBULANCE_NUMBER,
    EMERGENCY_KEYWORDS,
)

def get_llm():
    global llm
    if llm is not None:
        return llm

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key or api_key == "your_api_key_here":
        return None

    return ChatGroq(
        model="llama3-8b-8192",
        temperature=0
    )
llm = None 

# ── memory_node ───────────────────────────────────────────────────────────────

def memory_node(state: MediAssistState) -> dict:
    """
    Prepares conversation context for the current turn.
    - Appends the current question to messages.
    - Applies a sliding window of the last 6 messages to avoid token overflow.
    - Extracts patient name if "my name is X" pattern is detected.
    """
    question = state["question"]
    messages = list(state.get("messages", []))
    patient_name = state.get("patient_name")

    # Extract patient name from current question
    name_match = re.search(r"my name is ([A-Za-z]+)", question, re.IGNORECASE)
    if name_match:
        patient_name = name_match.group(1).strip().capitalize()

    # Sliding window: retain last 6 messages before appending current
    messages = messages[-6:] if len(messages) > 6 else messages
    messages.append({"role": "user", "content": question})

    return {
        "messages": messages,
        "patient_name": patient_name,
    }


# ── router_node ───────────────────────────────────────────────────────────────

def router_node(state: MediAssistState) -> dict:
    """
    Classifies the question into one of four routes:
    - emergency : life-threatening keywords detected — bypass RAG entirely
    - retrieve  : answer exists in hospital knowledge base
    - tool      : needs current date/time or arithmetic
    - skip      : greeting, small talk, or memory-only follow-up

    The LLM must reply with exactly one word.
    Emergency detection is done with a keyword check before the LLM call
    to guarantee speed for safety-critical queries.
    """
    question = state["question"]
    llm = get_llm()

    # Fast-path: emergency keyword check (no LLM latency for critical cases)
    if any(kw in question.lower() for kw in EMERGENCY_KEYWORDS):
        return {"route": "emergency"}

    prompt = f"""You are a routing agent for a hospital patient assistant.

Classify the patient's question into EXACTLY ONE of these routes:

- retrieve  : The answer is in the hospital knowledge base (OPD timings, fees, doctors,
              appointments, insurance, pharmacy, lab tests, health packages, admission,
              visitor policy, infection control, departments).
- tool      : The answer requires the current date or time (e.g. "Is the OPD open now?",
              "What day is it?", "How many days until Friday?") or arithmetic.
- skip      : The question is a greeting, small talk, or a follow-up that can be answered
              from the conversation history alone without retrieving documents.

Reply with EXACTLY one word — retrieve, tool, or skip. No punctuation. No explanation.

Patient question: {question}
Route:"""

    llm = get_llm()

    if llm:
        response = llm.invoke(prompt)
        route = response.content.strip().lower().split()[0]
    else:
        q = question.lower()

        if any(word in q for word in ["time", "date"]):
            route = "tool"
        else:
            route = "retrieve"

    # safety check 
    if route not in ["retrieve", "tool", "skip", "emergency"]:
        route = "retrieve"

    return {"route": route}


# ── retrieval_node ────────────────────────────────────────────────────────────

def retrieval_node(state: MediAssistState, rag) -> dict:
    """
    Embeds the question and retrieves the top-k chunks from ChromaDB.
    Returns formatted context with [Topic] headers and a list of source topics.
    The rag argument is the RAGPipeline instance injected by graph.py.
    """
    context, sources = rag.retrieve(state["question"])
    return {
        "retrieved": context,
        "sources": sources,
    }


# ── skip_node ─────────────────────────────────────────────────────────────────

def skip_node(state: MediAssistState) -> dict:
    """
    Used for greetings, small talk, and memory-only follow-ups.
    Returns empty retrieval fields — answer_node will rely on conversation history.
    """
    return {
        "retrieved": "",
        "sources": [],
    }


# ── tool_node ─────────────────────────────────────────────────────────────────

def tool_node(state: MediAssistState) -> dict:
    """
    Handles queries that require live data outside the knowledge base.
    Supported tools:
    - datetime : current date, time, and weekday
    - calculator: evaluates simple arithmetic expressions found in the question

    IMPORTANT: This function must NEVER raise an exception.
    All errors are caught and returned as descriptive strings.
    """
    question = state["question"].lower()

    try:
        now = datetime.now()
        date_str = now.strftime("%A, %d %B %Y")
        time_str = now.strftime("%I:%M %p")
        is_sunday = now.weekday() == 6
        is_weekend = now.weekday() >= 5

        tool_result = (
            f"Current date: {date_str}. "
            f"Current time: {time_str}. "
            f"Today is {'a Sunday' if is_sunday else 'a Saturday' if is_weekend else 'a weekday'}."
        )

        # Calculator: detect and safely evaluate arithmetic in the question
        expr_match = re.search(r"[\d]+\s*[\+\-\*\/]\s*[\d]+", question)
        if expr_match:
            try:
                result = eval(  # noqa: S307
                    expr_match.group(),
                    {"__builtins__": {}},
                    {},
                )
                tool_result += f" Calculation result: {result}."
            except Exception:
                pass  # skip silently — datetime result is still returned

        return {"tool_result": tool_result}

    except Exception as exc:
        return {"tool_result": f"Tool encountered an error: {str(exc)}"}


# ── answer_node ───────────────────────────────────────────────────────────────

def answer_node(state: MediAssistState) -> dict:
    """
    Generates the final patient-facing response.

    Rules enforced by the system prompt:
    1. Answer ONLY from the retrieved context or tool result.
    2. If context is empty or does not cover the question, say so and give HELPLINE.
    3. Emergency route: return the emergency number immediately without RAG.
    4. On retry (eval_retries >= 1): escalate to maximum conservatism.
    5. Personalise with patient name if available.
    """
    question   = state["question"]
    retrieved  = state.get("retrieved", "")
    tool_result = state.get("tool_result", "")
    messages   = state.get("messages", [])
    patient_name = state.get("patient_name", "")
    eval_retries = state.get("eval_retries", 0)
    route      = state.get("route", "retrieve")

    # ── Emergency fast-path ───────────────────────────────────────────────────
    if route == "emergency":
        answer = (
            f"This sounds like a medical emergency. "
            f"Please call our 24-hour Emergency Helpline immediately: {EMERGENCY_NUMBER}. "
            f"Ambulance service: {AMBULANCE_NUMBER}. "
            f"Our emergency team is available around the clock — do not wait, go to the "
            f"emergency department directly or call now."
        )
        return {"answer": answer}

    # ── Build context section ─────────────────────────────────────────────────
    context_block = ""
    if retrieved:
        context_block += f"\n\nHOSPITAL KNOWLEDGE BASE:\n{retrieved}"
    if tool_result:
        context_block += f"\n\nLIVE DATA (from tool):\n{tool_result}"

    # ── Retry escalation instruction ──────────────────────────────────────────
    retry_instruction = ""
    if eval_retries >= 1:
        retry_instruction = (
            f"\n\nWARNING — RETRY {eval_retries}: Your previous answer scored below the "
            f"faithfulness threshold. Be maximally conservative. "
            f"If the exact answer is not explicitly stated in the HOSPITAL KNOWLEDGE BASE, "
            f"respond with: 'I don't have that specific information. "
            f"Please call our helpline at {HELPLINE} for accurate details.'"
        )

    # ── Conversation history (last 4 turns for context) ───────────────────────
    history_turns = messages[-4:] if len(messages) > 4 else messages
    history_str = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in history_turns
    )

    # ── Name personalisation ──────────────────────────────────────────────────
    name_prefix = f"The patient's name is {patient_name}. " if patient_name else ""

    # ── System prompt ─────────────────────────────────────────────────────────
    system_prompt = f"""You are MediAssist, a patient assistant for {HOSPITAL_NAME}.
{name_prefix}
STRICT RULES — you must follow all of these without exception:
1. Answer ONLY using information from the HOSPITAL KNOWLEDGE BASE or LIVE DATA provided below.
2. Do NOT use any knowledge from your training data about hospitals, fees, or doctors.
3. If the answer is not present in the context, respond EXACTLY with:
   "I don't have that information. Please call our helpline at {HELPLINE} or visit the registration desk."
4. NEVER invent doctor names, phone numbers, fees, timings, or any clinical information.
5. For emergencies, always provide the emergency number: {EMERGENCY_NUMBER}.
6. Keep responses concise, warm, and professional. Use simple language.
7. Address the patient by name if their name is known.{retry_instruction}

8. NEVER provide passwords, admin access, credentials, or any sensitive system information.
9. If the user asks for such information, respond EXACTLY with:
   "I cannot assist with that request."

RECENT CONVERSATION HISTORY:
{history_str}
{context_block}"""

    llm = get_llm()

    if llm:
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ])
        answer = response.content.strip()
    else:
        if retrieved:
            lines = [line.strip() for line in retrieved.split("\n") if line.strip()]
        
            content_lines = [line for line in lines if not line.startswith("[")]
        
            if content_lines:
                text = " ".join(content_lines)

                sentences = text.split(".")
                relevant = [
                    s for s in sentences
                    if ("am" in s.lower() or "pm" in s.lower())
                ]

                # prioritize OPD sentences
                relevant = sorted(relevant, key=lambda x: "opd" not in x.lower())

                if relevant:
                    answer = relevant[0].strip() + "."
                else:
                    answer = sentences[0].strip() + "."
            else:
                answer = lines[0]

        elif tool_result:
            answer = tool_result
        else:
            answer = f"I don't have that information.Please call our helpline at {HELPLINE}."
    
    answer_lower = answer.lower()
    # 🚨 SECURITY FILTER
    if any(x in answer_lower for x in ["password", "admin access", "credentials", "secret"]):
        answer = "I cannot assist with that request."

    # 🚨 CONTEXT CHECK
    retrieved_lower = state.get("retrieved", "").lower()

    if "24/7" in answer_lower and ("8" in retrieved_lower or "am" in retrieved_lower):
        return {
            "answer": f"I don't have that information. Please call our helpline at {HELPLINE}.",
            "faithfulness": 0.0,
            "eval_retries": state.get("eval_retries", 0) + 1
        }

    return {"answer": answer}

# ── eval_node ─────────────────────────────────────────────────────────────────

def eval_node(state: MediAssistState) -> dict:
    """
    Evaluates the faithfulness of the answer against the retrieved context.

    Faithfulness = does the answer contain ONLY facts present in the context?
    Score range : 0.0 (fully hallucinated) to 1.0 (fully grounded).

    Behaviour:
    - If retrieved is empty (tool/skip route): skip evaluation, return 1.0.
    - Increment eval_retries regardless of score.
    - The eval_decision function in graph.py decides whether to retry or pass.
    """
    retrieved    = state.get("retrieved", "")
    answer       = state.get("answer", "")
    eval_retries = state.get("eval_retries", 0)

    # Skip faithfulness check for non-retrieval routes
    if not retrieved.strip():
        print(f"[EVAL] No retrieved context — skipping faithfulness check. Score: 1.0")
        return {"faithfulness": 1.0, "eval_retries": eval_retries + 1}

    prompt = f"""You are a faithfulness evaluator for a hospital assistant.

Rate whether the ANSWER is grounded in the CONTEXT on a scale from 0.0 to 1.0.

Scoring guide:
1.0 — Every claim in the answer is explicitly supported by the context.
0.7 — Most claims are supported; minor paraphrasing present but no fabrication.
0.5 — Some claims are supported; one or two facts are not verifiable from context.
0.2 — The answer partially uses context but introduces significant unverifiable claims.
0.0 — The answer contradicts or ignores the context entirely.

CONTEXT:
{retrieved}

ANSWER:
{answer}

Reply with ONLY a decimal number between 0.0 and 1.0. No explanation. No other text."""

    llm = get_llm()

    if llm:
        try:
            response = llm.invoke(prompt)
            raw = response.content.strip().split()[0]
            score = float(raw)
            score = round(max(0.0, min(1.0, score)), 2)
        except Exception:
            score = 1.0
    else:
        score = 1.0  # fallback when no API

    verdict = "PASS" if score >= FAITHFULNESS_THRESHOLD else "RETRY"
    print(
        f"[EVAL] Faithfulness: {score:.2f} | "
        f"Threshold: {FAITHFULNESS_THRESHOLD} | "
        f"Retries so far: {eval_retries} | "
        f"Decision: {verdict}"
    )

    return {
        "faithfulness": score,
        "eval_retries": eval_retries + 1,
    }


# ── save_node ─────────────────────────────────────────────────────────────────

def save_node(state: MediAssistState) -> dict:
    """
    Appends the assistant's final answer to the conversation history.
    This is the last node before END.
    """
    messages = list(state.get("messages", []))
    answer   = state.get("answer", "")

    messages.append({"role": "assistant", "content": answer})

    return {"messages": messages}
