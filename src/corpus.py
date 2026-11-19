"""12-document internal engineering handbook and ChromaDB vector store builder."""

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
    """Return a 12-document internal engineering knowledge base."""
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
                "Webhook Relay Worker Runtime Guardrails.\n"
                "Default outbound rate limit: 240 req/min per endpoint.\n"
                "Maximum delivery retries: 5 attempts with exponential backoff.\n"
                "Circuit-breaker cooldown: 150 seconds after 8 consecutive delivery timeouts."
            ),
            metadata={"source": "webhook_relay_limits.md"},
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
                "When a circuit-breaker cooldown alert fires for Project Zephyr or Project Meridian, "
                "check #platform-alerts and verify upstream gateway health. "
                "Numeric cooldown durations live in each service's guardrail doc, not in this runbook."
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
        Document(
            page_content=(
                "Payments Settlement SLA.\n"
                "Monthly availability target: 99.95%.\n"
                "p95 settlement API latency budget: 400 ms.\n"
                "Chargeback dispute response window: 72 hours."
            ),
            metadata={"source": "payments_sla.md"},
        ),
        Document(
            page_content=(
                "Redis Cache Policy.\n"
                "Pricing lookup cache TTL: 300 seconds.\n"
                "Session metadata cache TTL: 60 seconds.\n"
                "Eviction policy across shared clusters: allkeys-lru."
            ),
            metadata={"source": "cache_policy.md"},
        ),
        Document(
            page_content=(
                "PostgreSQL Failover Architecture.\n"
                "Primary databases run in active-standby mode across two availability zones.\n"
                "Target RPO: 5 seconds. Target RTO: 45 seconds. Read-replica lag alert: 2 seconds.\n"
                "Database credentials are managed exclusively in Vault and never stored in engineering docs."
            ),
            metadata={"source": "database_failover.md"},
        ),
        Document(
            page_content=(
                "Observability and Alerting Standards.\n"
                "High-severity alerts route to PagerDuty within 60 seconds.\n"
                "Error-budget burn alerts trigger at 2% burn within 1 hour.\n"
                "Structured application logs are retained for 30 days."
            ),
            metadata={"source": "observability_alerts.md"},
        ),
        Document(
            page_content=(
                "Staging Environment Configuration.\n"
                "Staging runs with synthetic load capped at 50 req/min.\n"
                "Nightly database anonymization and seed jobs run at 02:00 UTC.\n"
                "Feature flags default to opt-in per pull request."
            ),
            metadata={"source": "staging_config.md"},
        ),
        Document(
            page_content=(
                "Public API Versioning Policy.\n"
                "Currently supported major API versions: v2 and v3.\n"
                "Minimum deprecation notice window before retirement: 180 days.\n"
                "Deprecated endpoints must emit RFC 8594 Sunset headers."
            ),
            metadata={"source": "api_versioning.md"},
        ),
    ]


def list_document_sources(docs: list[Document] | None = None) -> list[str]:
    """Return sorted source filenames present in the engineering handbook."""
    source_docs = docs if docs is not None else get_sample_documents()
    return sorted({str(doc.metadata.get("source", "unknown")) for doc in source_docs})


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
    """Create (or rebuild) a local ChromaDB vector store for the 12-doc corpus."""
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
