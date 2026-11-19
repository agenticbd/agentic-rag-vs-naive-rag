"""Benchmark comparing Traditional RAG vs. Agentic RAG across 3 query archetypes."""

from __future__ import annotations

import os
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from src.agentic_rag import ask_agentic_rag, build_agentic_rag
from src.corpus import DEFAULT_EMBEDDING_MODEL, build_vectorstore
from src.traditional_rag import DEFAULT_CHAT_MODEL, ask_traditional_rag


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    label: str
    question: str
    expected_substrings: tuple[str, ...]


CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        case_id="single_hop",
        label="Single-hop lookup",
        question="What is the default rate limit for the Async Billing Pipeline?",
        expected_substrings=("120",),
    ),
    BenchmarkCase(
        case_id="multi_hop",
        label="Multi-hop indirect",
        question="What is the circuit-breaker cooldown for Project Zephyr?",
        expected_substrings=("90",),
    ),
    BenchmarkCase(
        case_id="no_retrieval",
        label="No-retrieval Python",
        question="Convert 90 seconds to minutes as a Python expression.",
        expected_substrings=("90 / 60", "1.5"),
    ),
)


def is_correct(answer: str, expected_substrings: tuple[str, ...]) -> bool:
    lower = answer.lower()
    return any(sub.lower() in lower for sub in expected_substrings)


def run_benchmark(trials: int = 3, k: int = 2) -> int:
    dotenv_file = os.getenv("DOTENV_FILE")
    if dotenv_file:
        load_dotenv(dotenv_file)
    elif Path(".env").is_file():
        load_dotenv(Path(".env"))
    if not os.getenv("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY is not set; skipping live Gemini benchmark.", file=sys.stderr)
        return 2

    print(f"Model: {DEFAULT_CHAT_MODEL} | Embeddings: {DEFAULT_EMBEDDING_MODEL} | k={k} | trials={trials}")

    with tempfile.TemporaryDirectory(prefix="rag-bench-") as tmpdir:
        t_index0 = time.perf_counter()
        store = build_vectorstore(
            persist_directory=Path(tmpdir) / "chroma_db",
            collection_name="bench_kb",
            rebuild=True,
        )
        index_s = time.perf_counter() - t_index0
        doc_count = store._collection.count()
        print(f"Cold ChromaDB index: {doc_count} chunks in {index_s:.2f}s")
        print("-" * 78)

        llm = ChatGoogleGenerativeAI(model=DEFAULT_CHAT_MODEL, temperature=0)
        agent = build_agentic_rag(store, model_name=DEFAULT_CHAT_MODEL, k=k)

        trad_passes = 0
        agent_passes = 0

        for case in CASES:
            trad_times: list[float] = []
            trad_ok = 0
            trad_sources_sample: list[str] = []
            trad_answer_sample = ""

            agent_times: list[float] = []
            agent_ok = 0
            agent_retrievals: list[int] = []
            agent_llm_calls: list[int] = []
            agent_steps_sample = []
            agent_answer_sample = ""

            for trial in range(1, trials + 1):
                tr = ask_traditional_rag(case.question, store, llm=llm, k=k)
                trad_times.append(tr.elapsed_s)
                if is_correct(tr.answer, case.expected_substrings):
                    trad_ok += 1
                if trial == 1:
                    trad_sources_sample = tr.retrieved_sources
                    trad_answer_sample = tr.answer

                ar = ask_agentic_rag(case.question, agent)
                agent_times.append(ar.elapsed_s)
                agent_retrievals.append(ar.retrieval_calls)
                agent_llm_calls.append(ar.llm_calls)
                if is_correct(ar.answer, case.expected_substrings):
                    agent_ok += 1
                if trial == 1:
                    agent_steps_sample = ar.steps
                    agent_answer_sample = ar.answer

            if trad_ok == trials:
                trad_passes += 1
            if agent_ok == trials:
                agent_passes += 1

            print(f"[{case.case_id}] {case.label}: {case.question}")
            print(
                f"  Traditional RAG: pass={trad_ok}/{trials} | retrievals=1 | llm_calls=1 | "
                f"median={statistics.median(trad_times):.2f}s "
                f"(range {min(trad_times):.2f}s-{max(trad_times):.2f}s) | "
                f"sources={trad_sources_sample}"
            )
            print(f"    trial1 answer: {trad_answer_sample}")
            print(
                f"  Agentic RAG:     pass={agent_ok}/{trials} | "
                f"retrievals={agent_retrievals} | llm_calls={agent_llm_calls} | "
                f"median={statistics.median(agent_times):.2f}s "
                f"(range {min(agent_times):.2f}s-{max(agent_times):.2f}s)"
            )
            print(f"    trial1 answer: {agent_answer_sample}")
            for idx, step in enumerate(agent_steps_sample, start=1):
                print(f"    trial1 step {idx}: {step.tool_name}({step.args}) -> {step.result_preview}")
            print("-" * 78)

        print(
            f"Summary (all {trials} trials passing): "
            f"Traditional RAG={trad_passes}/{len(CASES)} | Agentic RAG={agent_passes}/{len(CASES)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_benchmark())
