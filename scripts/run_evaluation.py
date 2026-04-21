import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import uuid

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

from agent.graph import build_graph, ask
from config import GROQ_API_KEY, LLM_MODEL, EMBED_MODEL


# ── QA test cases ─────────────────────────────────────────────────────────────

QA_PAIRS = [
    {
        "question": "What are the OPD timings on weekdays?",
        "ground_truth": (
            "General OPD runs from 8:00 AM to 2:00 PM. "
            "Evening OPD runs from 5:00 PM to 8:00 PM on weekdays."
        ),
    },
    {
        "question": "How much does a cardiologist consultation cost?",
        "ground_truth": "A cardiologist consultation at MediCare General Hospital costs Rs. 800.",
    },
    {
        "question": "What is the emergency helpline number?",
        "ground_truth": (
            "The 24-hour emergency helpline number is 040-9999-0000. "
            "The ambulance service number is 040-9999-1111."
        ),
    },
    {
        "question": "How can I book an appointment at MediCare?",
        "ground_truth": (
            "Appointments can be booked online at medicarehyd.in/appointments, "
            "by calling 040-2345-6789 Monday to Saturday 8 AM to 6 PM, "
            "or in person at the registration counter."
        ),
    },
    {
        "question": "What does the Basic Health Check package include?",
        "ground_truth": (
            "The Basic Health Check package costs Rs. 1,499 and includes CBC, "
            "fasting blood sugar, lipid profile, urine analysis, BMI, "
            "blood pressure assessment, and a physician consultation."
        ),
    },
]


# ── Collect answers from the live graph ───────────────────────────────────────

def collect_results(app) -> dict:
    data = {
        "question":     [],
        "answer":       [],
        "contexts":     [],
        "ground_truth": [],
    }

    print("Running agent on all 5 QA pairs...\n")

    for i, pair in enumerate(QA_PAIRS, 1):
        thread_id = f"eval_{uuid.uuid4().hex[:8]}"
        result = ask(app, pair["question"], thread_id)

        answer   = result.get("answer", "")
        retrieved = result.get("retrieved", "")

        print(f"  [{i}/5] Q : {pair['question']}")
        print(f"         A : {answer[:90]}{'...' if len(answer) > 90 else ''}")
        print(f"         Route      : {result.get('route', 'unknown')}")
        print(f"         Sources    : {result.get('sources', [])}")
        print(f"         Faithfulness (eval_node): {result.get('faithfulness', 'n/a')}")
        print()

        data["question"].append(pair["question"])
        data["answer"].append(answer)
        data["contexts"].append([retrieved] if retrieved else ["No context retrieved."])
        data["ground_truth"].append(pair["ground_truth"])

    return data


# ── Run RAGAS evaluation ──────────────────────────────────────────────────────

def run_evaluation():
    print("=" * 60)
    print("MEDIASSIST — RAGAS BASELINE EVALUATION")
    print("=" * 60)
    print()

    # Build the agent graph (includes RAG init + verify_retrieval gate)
    print("Building graph...")
    app, _ = build_graph()
    print()

    # Collect agent answers
    data = collect_results(app)
    dataset = Dataset.from_dict(data)

    # Configure RAGAS evaluator LLM and embeddings
    evaluator_llm = LangchainLLMWrapper(
        ChatGroq(model=LLM_MODEL, temperature=0, api_key=GROQ_API_KEY)
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    )

    print("Running RAGAS evaluation...")
    print()

    scores = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    # ── Print results ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("RAGAS BASELINE SCORES")
    print("=" * 60)

    score_dict = scores.to_pandas()[
        ["faithfulness", "answer_relevancy", "context_precision"]
    ].to_dict(orient="list")

    metric_labels = {
        "faithfulness":       "Faithfulness        (0–1, higher = more grounded)",
        "answer_relevancy":   "Answer relevancy    (0–1, higher = more on-topic)",
        "context_precision":  "Context precision   (0–1, higher = less noise in retrieval)",
    }

    averages = {}
    for metric, label in metric_labels.items():
        values = score_dict.get(metric, [])
        values = [v for v in values if v is not None]
        avg = round(sum(values) / len(values), 4) if values else 0.0
        averages[metric] = avg
        print(f"  {label}")
        for i, v in enumerate(values, 1):
            bar = "█" * int((v or 0) * 20)
            print(f"    Q{i}: {v:.4f}  {bar}")
        print(f"    AVG: {avg:.4f}")
        print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for metric, avg in averages.items():
        threshold = 0.70
        status = "PASS" if avg >= threshold else "BELOW THRESHOLD"
        print(f"  {metric:<22} avg={avg:.4f}   [{status}]")

    overall = round(sum(averages.values()) / len(averages), 4)
    print(f"\n  Overall average score : {overall:.4f}")
    print("=" * 60)

    return scores


if __name__ == "__main__":
    run_evaluation()
