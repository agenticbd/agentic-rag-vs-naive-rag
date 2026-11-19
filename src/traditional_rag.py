"""Traditional (Naive) single-shot RAG pipeline with token usage tracking."""

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
    input_tokens: int
    output_tokens: int
    total_tokens: int
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


def extract_token_usage(message: Any) -> tuple[int, int, int]:
    """Return (input_tokens, output_tokens, total_tokens) from a LangChain AIMessage."""
    usage = getattr(message, "usage_metadata", None) or {}
    if not isinstance(usage, dict):
        return 0, 0, 0
    in_tok = int(usage.get("input_tokens", 0) or 0)
    out_tok = int(usage.get("output_tokens", 0) or 0)
    tot_tok = int(usage.get("total_tokens", in_tok + out_tok) or (in_tok + out_tok))
    return in_tok, out_tok, tot_tok


def ask_traditional_rag(
    question: str,
    store: Chroma,
    llm: Any | None = None,
    k: int = 2,
    model_name: str = DEFAULT_CHAT_MODEL,
) -> TraditionalRagResult:
    """Always retrieve top-k chunks once, then make one LLM call."""
    chat = llm or ChatGoogleGenerativeAI(model=model_name, timeout=45, max_retries=3)

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
    in_tok, out_tok, tot_tok = extract_token_usage(response)

    return TraditionalRagResult(
        question=question,
        answer=extract_text(response.content),
        retrieved_sources=[str(doc.metadata.get("source", "unknown")) for doc, _ in hits],
        retrieved_distances=[round(float(dist), 4) for _, dist in hits],
        retrieval_calls=1,
        llm_calls=1,
        input_tokens=in_tok,
        output_tokens=out_tok,
        total_tokens=tot_tok,
        elapsed_s=elapsed_s,
    )
