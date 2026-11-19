"""Offline deterministic tests for Traditional RAG and Agentic RAG helpers."""

from __future__ import annotations

import hashlib

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agentic_rag import ask_agentic_rag, create_rag_tools, extract_tool_steps
from src.benchmark import is_correct
from src.corpus import build_vectorstore, get_sample_documents, list_document_sources, split_documents
from src.traditional_rag import ask_traditional_rag, extract_text, extract_token_usage


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
    """Minimal chat model stub with usage_metadata for testing Traditional RAG offline."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        return AIMessage(
            content=self.reply,
            usage_metadata={"input_tokens": 110, "output_tokens": 12, "total_tokens": 122},
        )


class StubAgent:
    """Minimal agent stub verifying recursion_limit config and token aggregation."""

    def __init__(self) -> None:
        self.last_config = None

    def invoke(self, payload, config=None):
        self.last_config = config
        return {
            "messages": [
                HumanMessage(content=payload["messages"][0][1]),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "call_1", "name": "search_docs", "args": {"query": "Project Zephyr"}}
                    ],
                    usage_metadata={"input_tokens": 150, "output_tokens": 20, "total_tokens": 170},
                ),
                ToolMessage(
                    content="[source=architecture_codenames.md distance=0.3100]\nProject Zephyr -> Async Billing Pipeline",
                    tool_call_id="call_1",
                ),
                AIMessage(
                    content="90 seconds",
                    usage_metadata={"input_tokens": 240, "output_tokens": 10, "total_tokens": 250},
                ),
            ]
        }


def test_corpus_and_chunking():
    docs = get_sample_documents()
    assert len(docs) == 12
    sources = list_document_sources(docs)
    assert "architecture_codenames.md" in sources
    assert "webhook_relay_limits.md" in sources
    assert "database_failover.md" in sources
    chunks = split_documents(docs)
    assert len(chunks) == 12
    assert all("source" in c.metadata for c in chunks)


def test_traditional_rag_always_retrieves_and_counts_tokens(tmp_path):
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
    assert res.total_tokens == 122
    assert len(res.retrieved_sources) == 2
    assert stub.last_messages is not None


def test_rag_tools_and_agent_metrics(tmp_path):
    store = build_vectorstore(
        persist_directory=tmp_path / "chroma",
        collection_name="test_agent",
        embedding_function=DeterministicHashEmbeddings(),
        rebuild=True,
    )
    list_tool, search_tool = create_rag_tools(store, k=2)
    sources_str = list_tool.invoke({})
    assert "architecture_codenames.md" in sources_str
    assert "webhook_relay_limits.md" in sources_str

    out = search_tool.invoke({"query": "Async Billing Pipeline cooldown"})
    assert "source=" in out
    assert "distance=" in out

    assert extract_text([{"type": "text", "text": "90 seconds"}]) == "90 seconds"
    assert extract_token_usage(AIMessage(content="ok")) == (0, 0, 0)

    stub_agent = StubAgent()
    res = ask_agentic_rag("What is the cooldown for Project Zephyr?", stub_agent, recursion_limit=10)
    assert stub_agent.last_config == {"recursion_limit": 10}
    assert res.answer == "90 seconds"
    assert res.retrieval_calls == 1
    assert res.tool_calls == 1
    assert res.llm_calls == 2
    assert res.total_tokens == 420
    assert res.steps[0].sources == ["architecture_codenames.md"]

    steps = extract_tool_steps(stub_agent.invoke({"messages": [("user", "q")]})["messages"])
    assert len(steps) == 1
    assert is_correct("Project Meridian is longer by 60 seconds.", (("meridian", "webhook relay"), ("60",)))
