"""Benchmark comparing Traditional RAG (k=2 and k=4) vs. Agentic RAG (k=2) on 12 docs / 5 queries."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from src.agentic_rag import DEFAULT_RECURSION_LIMIT, ask_agentic_rag, build_agentic_rag
from src.corpus import DEFAULT_EMBEDDING_MODEL, build_vectorstore
from src.traditional_rag import DEFAULT_CHAT_MODEL, ask_traditional_rag


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    label: str
    question: str
    required_groups: tuple[tuple[str, ...], ...]


CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        case_id="single_hop",
        label="Single-hop lookup",
        question="What is the default rate limit for the Async Billing Pipeline?",
        required_groups=(("120",),),
    ),
    BenchmarkCase(
        case_id="multi_hop",
        label="Multi-hop codename bridge",
        question="What is the circuit-breaker cooldown for Project Zephyr?",
        required_groups=(("90",),),
    ),
    BenchmarkCase(
        case_id="cross_doc_compare",
        label="3-doc comparison",
        question=(
            "Which service has a longer circuit-breaker cooldown: Project Zephyr or Project Meridian, "
            "and by how many seconds?"
        ),
        required_groups=(("meridian", "webhook relay"), ("60",)),
    ),
    BenchmarkCase(
        case_id="no_retrieval",
        label="No-retrieval Python",
        question="Convert 90 seconds to minutes as a Python expression.",
        required_groups=(("90 / 60", "90/60", "1.5"),),
    ),
    BenchmarkCase(
        case_id="abstention",
        label="Unanswerable abstention",
        question="What is the PostgreSQL primary password for Project Zephyr?",
        required_groups=(
            ("not in", "not stored", "never stored", "cannot", "does not", "doesn't", "no password", "vault"),
        ),
    ),
)


def is_correct(answer: str, required_groups: tuple[tuple[str, ...], ...]) -> bool:
    """Return True if every required group has at least one matching substring."""
    lower = answer.lower()
    return all(any(option.lower() in lower for option in group) for group in required_groups)


def run_benchmark(trials: int = 3) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    dotenv_file = os.getenv("DOTENV_FILE")
    if dotenv_file:
        load_dotenv(dotenv_file)
    elif Path(".env").is_file():
        load_dotenv(Path(".env"))
    if not os.getenv("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY is not set; skipping live Gemini benchmark.", file=sys.stderr)
        return 2

    print(
        f"Model: {DEFAULT_CHAT_MODEL} | Embeddings: {DEFAULT_EMBEDDING_MODEL} | "
        f"trials={trials} | recursion_limit={DEFAULT_RECURSION_LIMIT}"
    )

    with tempfile.TemporaryDirectory(prefix="rag-bench-") as tmpdir:
        t_index0 = time.perf_counter()
        store = build_vectorstore(
            persist_directory=Path(tmpdir) / "chroma_db",
            collection_name="bench_kb_12",
            rebuild=True,
        )
        index_s = time.perf_counter() - t_index0
        doc_count = store._collection.count()
        print(f"Cold ChromaDB index: {doc_count} chunks in {index_s:.2f}s")
        print("=" * 84)

        llm = ChatGoogleGenerativeAI(model=DEFAULT_CHAT_MODEL, timeout=45, max_retries=3)
        agent = build_agentic_rag(store, model_name=DEFAULT_CHAT_MODEL, k=2)

        trad2_passes = 0
        trad4_passes = 0
        agent_passes = 0

        for case in CASES:
            t2_times, t2_toks, t2_ok = [], [], 0
            t2_sources_sample, t2_answer_sample = [], ""

            t4_times, t4_toks, t4_ok = [], [], 0
            t4_sources_sample, t4_answer_sample = [], ""

            ag_times, ag_toks, ag_ok = [], [], 0
            ag_retrievals, ag_tools, ag_llm_calls = [], [], []
            ag_steps_sample, ag_answer_sample = [], ""

            for trial in range(1, trials + 1):
                r2 = ask_traditional_rag(case.question, store, llm=llm, k=2)
                t2_times.append(r2.elapsed_s)
                t2_toks.append(r2.total_tokens)
                if is_correct(r2.answer, case.required_groups):
                    t2_ok += 1
                if trial == 1:
                    t2_sources_sample = r2.retrieved_sources
                    t2_answer_sample = r2.answer

                r4 = ask_traditional_rag(case.question, store, llm=llm, k=4)
                t4_times.append(r4.elapsed_s)
                t4_toks.append(r4.total_tokens)
                if is_correct(r4.answer, case.required_groups):
                    t4_ok += 1
                if trial == 1:
                    t4_sources_sample = r4.retrieved_sources
                    t4_answer_sample = r4.answer

                ra = ask_agentic_rag(case.question, agent, recursion_limit=DEFAULT_RECURSION_LIMIT)
                ag_times.append(ra.elapsed_s)
                ag_toks.append(ra.total_tokens)
                ag_retrievals.append(ra.retrieval_calls)
                ag_tools.append(ra.tool_calls)
                ag_llm_calls.append(ra.llm_calls)
                if is_correct(ra.answer, case.required_groups):
                    ag_ok += 1
                if trial == 1:
                    ag_steps_sample = ra.steps
                    ag_answer_sample = ra.answer

            if t2_ok == trials:
                trad2_passes += 1
            if t4_ok == trials:
                trad4_passes += 1
            if ag_ok == trials:
                agent_passes += 1

            print(f"[{case.case_id}] {case.label}: {case.question}")
            print(
                f"  Traditional (k=2): pass={t2_ok}/{trials} | retrievals=1 | llm_calls=1 | "
                f"tokens_med={int(statistics.median(t2_toks))} | "
                f"median={statistics.median(t2_times):.2f}s (range {min(t2_times):.2f}s-{max(t2_times):.2f}s) | "
                f"sources={t2_sources_sample}"
            )
            print(f"    trial1 answer: {t2_answer_sample}")
            print(
                f"  Traditional (k=4): pass={t4_ok}/{trials} | retrievals=1 | llm_calls=1 | "
                f"tokens_med={int(statistics.median(t4_toks))} | "
                f"median={statistics.median(t4_times):.2f}s (range {min(t4_times):.2f}s-{max(t4_times):.2f}s) | "
                f"sources={t4_sources_sample}"
            )
            print(f"    trial1 answer: {t4_answer_sample}")
            print(
                f"  Agentic (k=2):     pass={ag_ok}/{trials} | "
                f"retrievals={ag_retrievals} | tool_calls={ag_tools} | llm_calls={ag_llm_calls} | "
                f"tokens_med={int(statistics.median(ag_toks))} | "
                f"median={statistics.median(ag_times):.2f}s (range {min(ag_times):.2f}s-{max(ag_times):.2f}s)"
            )
            print(f"    trial1 answer: {ag_answer_sample}")
            for idx, step in enumerate(ag_steps_sample, start=1):
                print(
                    f"    trial1 step {idx}: {step.tool_name}({step.args}) | "
                    f"sources={step.sources} -> {step.result_preview}"
                )
            print("=" * 84)

        print(
            f"Summary (all {trials} trials passing): "
            f"Traditional(k=2)={trad2_passes}/{len(CASES)} | "
            f"Traditional(k=4)={trad4_passes}/{len(CASES)} | "
            f"Agentic(k=2)={agent_passes}/{len(CASES)}"
        )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Traditional RAG vs. Agentic RAG.")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials per query (default: 3)")
    args = parser.parse_args()
    raise SystemExit(run_benchmark(trials=args.trials))
