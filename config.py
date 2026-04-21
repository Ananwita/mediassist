import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
LLM_MODEL           = "llama-3.3-70b-versatile"
LLM_TEMPERATURE     = 0

# ── RAG ──────────────────────────────────────────────────────────────────────
EMBED_MODEL         = "all-MiniLM-L6-v2"
CHROMA_COLLECTION   = "medicare_kb"
RETRIEVAL_TOP_K     = 3

# ── Evaluation ───────────────────────────────────────────────────────────────
FAITHFULNESS_THRESHOLD  = 0.7
MAX_EVAL_RETRIES        = 2

# ── Hospital constants ────────────────────────────────────────────────────────
HOSPITAL_NAME       = "MediCare General Hospital, Bhubaneswar"
HELPLINE            = "0674 7111000"
EMERGENCY_NUMBER    = "0674 2725228"
AMBULANCE_NUMBER    = "1800 309 8888"

EMERGENCY_KEYWORDS  = [
    "emergency", "chest pain", "heart attack", "stroke",
    "ambulance", "accident", "unconscious", "bleeding",
    "breathless", "seizure", "fainted", "critical",
]
