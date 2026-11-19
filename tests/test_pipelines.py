"""Offline deterministic tests for Traditional RAG and Agentic RAG helpers."""

from __future__ import annotations

import hashlib

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agentic_rag import create_search_tool, extract_tool_steps
from src.corpus import build_vectorstore, get_sample_documents, split_documents
from src.traditional_rag import ask_traditional_rag, extract_text


class DeterministicHashEmbeddings(Embeddings):
    """Lightweight offline token-hash embeddings for deterministic tests."""

    def __init__(self, dims: int = 32) -> None:
        self.dims = dims

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:2], "big") % self.dims
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class StubChatModel:
    """Minimal chat model stub for testing Traditional RAG control flow offline."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        return AIMessage(content=self.reply)


def test_corpus_and_chunking():
    docs = get_sample_documents()
    assert len(docs) == 5
    chunks = split_documents(docs)
    assert len(chunks) == 5
    assert all("source" in c.metadata for c in chunks)


def test_traditional_rag_always_retrieves(tmp_path):
    store = build_vectorstore(
        persist_directory=tmp_path / "chroma",
        collection_name="test_trad",
        embedding_function=DeterministicHashEmbeddings(),
        rebuild=True,
    )
    stub = StubChatModel("120 req/min")
    res = ask_traditional_rag("What is the rate limit?", store, llm=stub, k=2)

    assert res.answer == "120 req/min"
    assert res.retrieval_calls == 1
    assert res.llm_calls == 1
    assert len(res.retrieved_sources) == 2
    assert stub.last_messages is not None


def test_search_tool_and_step_extraction(tmp_path):
    store = build_vectorstore(
        persist_directory=tmp_path / "chroma",
        collection_name="test_agent",
        embedding_function=DeterministicHashEmbeddings(),
        rebuild=True,
    )
    search_tool = create_search_tool(store, k=2)
    out = search_tool.invoke({"query": "Async Billing Pipeline cooldown"})
    assert "source=" in out
    assert "distance=" in out

    assert extract_text([{"type": "text", "text": "90 seconds"}]) == "90 seconds"
    messages = [
        HumanMessage(content="What is the circuit-breaker cooldown for Project Zephyr?"),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "call_1", "name": "search_docs", "args": {"query": "Project Zephyr"}}
            ],
        ),
        ToolMessage(
            content="Project Zephyr is the internal codename for the Async Billing Pipeline.",
            tool_call_id="call_1",
        ),
        AIMessage(content="90 seconds"),
    ]
    steps = extract_tool_steps(messages)
    assert len(steps) == 1
    assert steps[0].tool_name == "search_docs"
    assert steps[0].args == {"query": "Project Zephyr"}
    assert "Async Billing Pipeline" in steps[0].result_preview
