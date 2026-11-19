"""Traditional (Naive) single-shot RAG pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

DEFAULT_CHAT_MODEL = "gemini-3.5-flash-lite"


@dataclass
class TraditionalRagResult:
    question: str
    answer: str
    retrieved_sources: list[str]
    retrieved_distances: list[float]
    retrieval_calls: int
    llm_calls: int
    elapsed_s: float


def extract_text(content: Any) -> str:
    """Normalize LangChain message content (string or content blocks) to plain text."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()


def ask_traditional_rag(
    question: str,
    store: Chroma,
    llm: Any | None = None,
    k: int = 2,
    model_name: str = DEFAULT_CHAT_MODEL,
) -> TraditionalRagResult:
    """Always retrieve top-k chunks once, then make one LLM call."""
    chat = llm or ChatGoogleGenerativeAI(model=model_name, temperature=0)

    t0 = time.perf_counter()
    hits = store.similarity_search_with_score(question, k=k)
    context = "\n\n".join(
        f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc, _ in hits
    )
    messages = [
        (
            "system",
            "Answer concisely using only the provided context for internal service facts. "
            "If the context does not contain the answer, state that it is not in the provided context.",
        ),
        ("human", f"Context:\n{context}\n\nQuestion: {question}"),
    ]
    response = chat.invoke(messages)
    elapsed_s = time.perf_counter() - t0

    return TraditionalRagResult(
        question=question,
        answer=extract_text(response.content),
        retrieved_sources=[str(doc.metadata.get("source", "unknown")) for doc, _ in hits],
        retrieved_distances=[round(float(dist), 4) for _, dist in hits],
        retrieval_calls=1,
        llm_calls=1,
        elapsed_s=elapsed_s,
    )
