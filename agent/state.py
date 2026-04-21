from typing import TypedDict, List, Optional


class MediAssistState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    question: str                  # Current user question

    # ── Conversation memory ───────────────────────────────────────────────────
    messages: List[dict]           # Full conversation history — {"role": ..., "content": ...}
    patient_name: Optional[str]    # Extracted from "my name is X"; persists across turns

    # ── Routing ───────────────────────────────────────────────────────────────
    route: str                     # "retrieve" | "tool" | "skip" | "emergency"

    # ── Retrieval outputs ─────────────────────────────────────────────────────
    retrieved: str                 # Formatted context string with [Topic] headers
    sources: List[str]             # Topic names of retrieved chunks — used for citations

    # ── Tool outputs ──────────────────────────────────────────────────────────
    tool_result: str               # String result from tool_node — always a string, never raises

    # ── Answer and evaluation ─────────────────────────────────────────────────
    answer: str                    # Final assistant response
    faithfulness: float            # Score 0.0–1.0 from eval_node
    eval_retries: int              # Retry counter — capped at MAX_EVAL_RETRIES (2)
