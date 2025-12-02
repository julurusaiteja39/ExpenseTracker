from typing import List, TypedDict, Dict

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from .config import OPENAI_MODEL
from .storage import get_vectorstore


class SpendingAnalysis(BaseModel):
    bullet_points: List[str] = Field(
        ...,
        description="3-5 concise observations about the user's spending that relate to the question.",
        min_items=3,
        max_items=5,
    )


class AdvisorAnswer(BaseModel):
    response: str = Field(..., description="Direct answer to the user's finance question.")
    tips: List[str] = Field(
        ...,
        description="2-3 short, actionable money management tips tailored to the user's pattern.",
        min_items=2,
        max_items=3,
    )


class AdvisorState(TypedDict, total=False):
    question: str
    retrieved_context: str
    analysis_points: List[str]
    answer: str
    tips: List[str]
    categories_in_context: List[str]
    category_totals: Dict[str, float]


def retrieve_node(state: AdvisorState) -> AdvisorState:
    question = state["question"]
    vs = get_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": 8})
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    categories: List[str] = []
    totals: Dict[str, float] = {}
    seen_tx_ids = set()
    for doc in docs:
        meta = doc.metadata or {}
        cat = meta.get("category") or "unknown"
        if cat not in categories:
            categories.append(cat)

        tx_id = meta.get("id")
        if tx_id:
            if tx_id in seen_tx_ids:
                continue
            seen_tx_ids.add(tx_id)

        amt = meta.get("amount")
        currency = (meta.get("currency") or "").strip() or "USD"
        if isinstance(amt, (int, float)):
            key = f"{cat}|{currency}"
            totals[key] = totals.get(key, 0.0) + float(amt)

    return {
        "retrieved_context": context,
        "categories_in_context": categories,
        "category_totals": totals,
    }


def analyze_node(state: AdvisorState) -> AdvisorState:
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)
    structured_llm = llm.with_structured_output(SpendingAnalysis)
    question = state["question"]
    context = state.get("retrieved_context", "")
    categories = state.get("categories_in_context", []) or ["unknown"]
    category_totals = state.get("category_totals", {}) or {}

    totals_lines = []
    for key, val in category_totals.items():
        if "|" in key:
            cat, cur = key.split("|", 1)
        else:
            cat, cur = key, ""
        totals_lines.append(f"- {cat} ({cur or 'unknown currency'}): {val:.2f}")
    totals_text = "\n".join(totals_lines) if totals_lines else "No totals computed from metadata."

    prompt = f"""You are a helpful personal finance analyst.

User question:
{question}

Relevant past transactions:
{context}

Categories present (from metadata): {", ".join(categories)}
Totals by category and currency (use these numbers exactly when summarizing):
{totals_text}

Briefly (3-5 bullet points) analyze the user's spending. You MUST mention every category present in the retrieved transactions (including �?oother�?? or unknown categories) and include their amounts.

"""
    resp = structured_llm.invoke(prompt)
    return {"analysis_points": resp.bullet_points}


def answer_node(state: AdvisorState) -> AdvisorState:
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.3)
    structured_llm = llm.with_structured_output(AdvisorAnswer)
    question = state["question"]
    context = state.get("retrieved_context", "")
    analysis_points = state.get("analysis_points", [])
    categories = state.get("categories_in_context", []) or []
    category_totals = state.get("category_totals", {}) or {}

    totals_lines = []
    for key, val in category_totals.items():
        if "|" in key:
            cat, cur = key.split("|", 1)
        else:
            cat, cur = key, ""
        totals_lines.append(f"- {cat} ({cur or 'unknown currency'}): {val:.2f}")
    totals_text = "\n".join(totals_lines) if totals_lines else "No totals computed from metadata."
    analysis_text = (
        "\n".join(f"- {point}" for point in analysis_points) if analysis_points else "No structured analysis available."
    )

    prompt = f"""You are a friendly personal finance assistant.

User question:
{question}

Relevant past transactions:
{context}

Categories present (from metadata): {", ".join(categories) if categories else "none"}
Totals by category and currency (use these numbers when useful):
{totals_text}

Analysis of spending:
{analysis_text}

Now answer the user's question in clear, simple language.
If it makes sense, include concrete numbers (like total amount spent per category, approximate monthly spending, etc.).
Also include 2-3 short, practical tips for better money management based on their pattern.
"""
    resp = structured_llm.invoke(prompt)
    tips_text = "\n".join(f"- {tip}" for tip in resp.tips)
    answer_text = resp.response
    if tips_text:
        answer_text = f"{resp.response}\n\nTips:\n{tips_text}"
    return {"answer": answer_text, "tips": resp.tips}


def build_workflow():
    graph = StateGraph(AdvisorState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "analyze")
    graph.add_edge("analyze", "answer")
    graph.add_edge("answer", END)

    return graph
