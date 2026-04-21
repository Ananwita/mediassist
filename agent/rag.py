import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chromadb
from sentence_transformers import SentenceTransformer

from agent.knowledge_base import DOCUMENTS
from config import EMBED_MODEL, CHROMA_COLLECTION, RETRIEVAL_TOP_K


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for MediAssist.

    Responsibilities:
    - Embed all 12 hospital documents at startup using SentenceTransformer.
    - Store embeddings in a ChromaDB in-memory collection.
    - Expose retrieve(query, top_k) → (formatted_context, sources).
    - Expose verify_retrieval() → must be called and pass before graph compile.
    """

    def __init__(self):
        print("[RAG] Loading embedding model...")
        self.embedder = SentenceTransformer(EMBED_MODEL)

        print("[RAG] Initialising ChromaDB in-memory collection...")
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

        self._load_documents()
        print(f"[RAG] Knowledge base loaded — {len(DOCUMENTS)} documents ready.\n")

    # ── Private ───────────────────────────────────────────────────────────────

    def _load_documents(self):
        """Embed all documents and add them to the ChromaDB collection."""
        texts = [doc["text"] for doc in DOCUMENTS]
        ids = [doc["id"] for doc in DOCUMENTS]
        metadatas = [{"topic": doc["topic"]} for doc in DOCUMENTS]

        embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )

    # ── Public ────────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> tuple[str, list[str]]:
        """
        Retrieve the top-k most relevant document chunks for a query.

        Args:
            query:  The patient's question.
            top_k:  Number of chunks to retrieve (default: 3 from config).

        Returns:
            (context, sources)
            context: Formatted string with [Topic] headers, ready for the LLM prompt.
            sources: List of topic names for citations and debug display.
        """
        query_embedding = self.embedder.encode([query], show_progress_bar=False).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        if not documents:
            return "", []

        context_parts = []
        sources = []

        for doc_text, meta in zip(documents, metadatas):
            topic = meta.get("topic", "Unknown")
            context_parts.append(f"[{topic}]\n{doc_text}")
            sources.append(topic)

        formatted_context = "\n\n".join(context_parts)
        return formatted_context, sources

    def verify_retrieval(self) -> bool:
        """
        Run 3 sanity-check queries against the knowledge base.
        Prints results to stdout.
        Must be called before graph.compile() — a broken KB cannot be
        fixed by improving the LLM prompt.

        Returns:
            True if all queries return at least one relevant source.
        Raises:
            RuntimeError if any query returns no results.
        """
        test_queries = [
            ("What are the OPD timings?",            "OPD Timings"),
            ("How do I book an appointment?",         "Appointment Booking"),
            ("What is the emergency helpline number?","Emergency Services"),
        ]

        print("=" * 55)
        print("RAG RETRIEVAL VERIFICATION")
        print("=" * 55)

        all_passed = True

        for query, expected_topic in test_queries:
            context, sources = self.retrieve(query, top_k=3)

            if not sources:
                print(f"  FAIL  '{query}'")
                print(f"        Expected topic: {expected_topic}")
                print(f"        Got: no results\n")
                all_passed = False
                continue

            hit = expected_topic in sources
            status = "PASS" if hit else "WARN"
            print(f"  {status}  '{query}'")
            print(f"        Sources returned : {sources}")
            if not hit:
                print(f"        Expected topic  : {expected_topic} (not in top-{RETRIEVAL_TOP_K})")
                all_passed = False
            print()

        print("=" * 55)
        if all_passed:
            print("Verification PASSED — safe to compile the graph.\n")
        else:
            raise RuntimeError(
                "Retrieval verification FAILED. "
                "Fix the knowledge base or embedding setup before proceeding."
            )

        return True
