"""Agentic RAG pipeline built with LangChain create_agent and a retrieval tool."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from src.traditional_rag import DEFAULT_CHAT_MODEL, extract_text


@dataclass
class ToolStep:
    tool_name: str
    args: dict[str, Any]
    result_preview: str


@dataclass
class AgenticRagResult:
    question: str
    answer: str
    steps: list[ToolStep]
    retrieval_calls: int
    llm_calls: int
    elapsed_s: float


def create_search_tool(store: Chroma, k: int = 2):
    """Wrap vector-store similarity search as a callable agent tool."""

    @tool
    def search_docs(query: str) -> str:
        """Search internal engineering docs for a service name, codename, rate limit, or policy."""
        hits = store.similarity_search_with_score(query, k=k)
        lines = []
        for doc, distance in hits:
            src = doc.metadata.get("source", "unknown")
            lines.append(f"[source={src} distance={distance:.4f}]\n{doc.page_content}")
        return "\n---\n".join(lines)

    return search_docs


def build_agentic_rag(
    store: Chroma,
    model_name: str = DEFAULT_CHAT_MODEL,
    k: int = 2,
):
    """Build a tool-calling Agentic RAG graph over the ChromaDB store."""
    search_tool = create_search_tool(store, k=k)
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
    return create_agent(
        model=llm,
        tools=[search_tool],
        system_prompt=(
            "You are a concise engineering assistant. "
            "Use search_docs only when the question needs facts from internal engineering docs; "
            "skip it for general Python or math questions. "
            "If a retrieved doc maps a codename to another service name, call search_docs again "
            "using that resolved service name before answering."
        ),
    )


def extract_tool_steps(messages: list[Any]) -> list[ToolStep]:
    """Extract ordered tool invocations and their matching outputs from agent messages."""
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
                preview = raw_out.replace("\n", " ")[:180]
                steps.append(
                    ToolStep(
                        tool_name=str(call.get("name", "")),
                        args=dict(call.get("args", {})),
                        result_preview=preview,
                    )
                )
    return steps


def ask_agentic_rag(question: str, agent: Any) -> AgenticRagResult:
    """Invoke the Agentic RAG graph and record retrieval calls, LLM calls, and latency."""
    t0 = time.perf_counter()
    output = agent.invoke({"messages": [("user", question)]})
    elapsed_s = time.perf_counter() - t0

    messages = output.get("messages", [])
    steps = extract_tool_steps(messages)
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    final_answer = extract_text(ai_messages[-1].content) if ai_messages else ""

    return AgenticRagResult(
        question=question,
        answer=final_answer,
        steps=steps,
        retrieval_calls=len(steps),
        llm_calls=len(ai_messages),
        elapsed_s=elapsed_s,
    )
