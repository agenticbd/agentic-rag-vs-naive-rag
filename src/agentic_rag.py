"""Agentic RAG pipeline built with LangChain create_agent, 2 tools, and recursion_limit."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.errors import GraphRecursionError

from src.traditional_rag import DEFAULT_CHAT_MODEL, extract_text, extract_token_usage

DEFAULT_RECURSION_LIMIT = 10
SOURCE_TAG_RE = re.compile(r"\[source=([^\s\]]+)")


@dataclass
class ToolStep:
    tool_name: str
    args: dict[str, Any]
    sources: list[str]
    result_preview: str


@dataclass
class AgenticRagResult:
    question: str
    answer: str
    steps: list[ToolStep]
    retrieval_calls: int
    tool_calls: int
    llm_calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    elapsed_s: float


def create_rag_tools(store: Chroma, k: int = 2) -> list[Any]:
    """Create discovery and vector-search tools over a ChromaDB store."""

    @tool
    def list_doc_sources() -> str:
        """List all document filenames available in the internal engineering knowledge base."""
        meta = store._collection.get(include=["metadatas"]).get("metadatas") or []
        sources = sorted({str(m.get("source", "unknown")) for m in meta if isinstance(m, dict)})
        return ", ".join(sources)

    @tool
    def search_docs(query: str) -> str:
        """Search internal engineering docs for a service name, codename, rate limit, or policy."""
        hits = store.similarity_search_with_score(query, k=k)
        lines = []
        for doc, distance in hits:
            src = doc.metadata.get("source", "unknown")
            lines.append(f"[source={src} distance={distance:.4f}]\n{doc.page_content}")
        return "\n---\n".join(lines)

    return [list_doc_sources, search_docs]


def build_agentic_rag(
    store: Chroma,
    model_name: str = DEFAULT_CHAT_MODEL,
    k: int = 2,
):
    """Build a tool-calling Agentic RAG graph over the ChromaDB store."""
    tools = create_rag_tools(store, k=k)
    llm = ChatGoogleGenerativeAI(model=model_name, timeout=45, max_retries=3)
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a concise engineering assistant. "
            "Use list_doc_sources or search_docs only when the question requires internal engineering facts; "
            "skip tools for general Python or math questions. "
            "If a retrieved doc maps a codename to a service name, call search_docs again using the resolved service name. "
            "If the requested secret or fact is not stored in the docs, state clearly that it is not in the engineering docs."
        ),
    )


def extract_tool_steps(messages: list[Any]) -> list[ToolStep]:
    """Extract ordered tool invocations, matched source files, and outputs from agent messages."""
    tool_outputs: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_outputs[msg.tool_call_id] = extract_text(msg.content)

    steps: list[ToolStep] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                call_id = call.get("id", "")
                raw_out = tool_outputs.get(call_id, "")
                sources = SOURCE_TAG_RE.findall(raw_out)
                preview = raw_out.replace("\n", " ")[:160]
                steps.append(
                    ToolStep(
                        tool_name=str(call.get("name", "")),
                        args=dict(call.get("args", {})),
                        sources=sources,
                        result_preview=preview,
                    )
                )
    return steps


def ask_agentic_rag(
    question: str,
    agent: Any,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
) -> AgenticRagResult:
    """Invoke the Agentic RAG graph with a recursion guard and record cost + latency metrics."""
    t0 = time.perf_counter()
    try:
        output = agent.invoke(
            {"messages": [("user", question)]},
            config={"recursion_limit": recursion_limit},
        )
    except GraphRecursionError:
        return AgenticRagResult(
            question=question,
            answer=f"[stopped: reached recursion_limit={recursion_limit}]",
            steps=[],
            retrieval_calls=0,
            tool_calls=0,
            llm_calls=recursion_limit,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            elapsed_s=time.perf_counter() - t0,
        )
    elapsed_s = time.perf_counter() - t0

    messages = output.get("messages", [])
    steps = extract_tool_steps(messages)
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    final_answer = extract_text(ai_messages[-1].content) if ai_messages else ""

    in_tok = out_tok = tot_tok = 0
    for msg in ai_messages:
        i_t, o_t, t_t = extract_token_usage(msg)
        in_tok += i_t
        out_tok += o_t
        tot_tok += t_t

    retrieval_calls = sum(1 for s in steps if s.tool_name == "search_docs")

    return AgenticRagResult(
        question=question,
        answer=final_answer,
        steps=steps,
        retrieval_calls=retrieval_calls,
        tool_calls=len(steps),
        llm_calls=len(ai_messages),
        input_tokens=in_tok,
        output_tokens=out_tok,
        total_tokens=tot_tok,
        elapsed_s=elapsed_s,
    )
