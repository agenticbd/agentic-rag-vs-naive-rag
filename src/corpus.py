"""Sample 5-document engineering corpus and ChromaDB builder."""

from __future__ import annotations

import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"


def get_sample_documents() -> list[Document]:
    """Return a 5-document internal engineering knowledge base."""
    return [
        Document(
            page_content=(
                "Internal Service Codename Directory.\n"
                "Project Zephyr is the internal codename for the Async Billing Pipeline.\n"
                "Project Borealis is the internal codename for the Session Token Gateway.\n"
                "Project Meridian is the internal codename for the Webhook Relay Worker."
            ),
            metadata={"source": "architecture_codenames.md"},
        ),
        Document(
            page_content=(
                "Async Billing Pipeline Runtime Guardrails.\n"
                "Default tenant rate limit: 120 req/min.\n"
                "Burst allowance: 30 requests over 10 seconds.\n"
                "Circuit-breaker cooldown: 90 seconds after 5 consecutive upstream 503 responses."
            ),
            metadata={"source": "billing_rate_limits.md"},
        ),
        Document(
            page_content=(
                "Session Token Gateway Guardrails.\n"
                "Default client rate limit: 600 req/min.\n"
                "Circuit-breaker cooldown: 45 seconds when the identity provider times out.\n"
                "Access token TTL: 15 minutes."
            ),
            metadata={"source": "auth_service.md"},
        ),
        Document(
            page_content=(
                "On-Call Circuit-Breaker Triage Runbook.\n"
                "When a circuit-breaker cooldown alert fires for Project Zephyr, "
                "check the #billing-alerts channel and verify upstream payment gateway status. "
                "Numeric cooldown settings live in the service-specific guardrail docs, not here."
            ),
            metadata={"source": "incident_runbook.md"},
        ),
        Document(
            page_content=(
                "Production Deployment Policy.\n"
                "All backend services roll out via a 10% canary for 15 minutes before full promotion. "
                "Rollback automatically if 5xx error rates exceed 1.5%."
            ),
            metadata={"source": "deploy_policy.md"},
        ),
    ]


def split_documents(
    docs: list[Document],
    chunk_size: int = 360,
    chunk_overlap: int = 40,
) -> list[Document]:
    """Split documents into chunks while preserving source metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


def build_vectorstore(
    persist_directory: Path | str = "chroma_db",
    collection_name: str = "engineering_kb",
    embedding_function: Embeddings | None = None,
    rebuild: bool = True,
) -> Chroma:
    """Create (or rebuild) a local ChromaDB vector store for the sample corpus."""
    persist_path = Path(persist_directory)
    if rebuild and persist_path.exists():
        shutil.rmtree(persist_path)

    embeddings = embedding_function or GoogleGenerativeAIEmbeddings(
        model=DEFAULT_EMBEDDING_MODEL
    )
    chunks = split_documents(get_sample_documents())
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_path),
    )
