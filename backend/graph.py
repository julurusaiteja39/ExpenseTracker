from typing import List, TypedDict

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


def retrieve_node(state: AdvisorState) -> AdvisorState:
    question = state["question"]
    vs = get_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    return {"retrieved_context": context}


def analyze_node(state: AdvisorState) -> AdvisorState:
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)
    structured_llm = llm.with_structured_output(SpendingAnalysis)
    question = state["question"]
    context = state.get("retrieved_context", "")

    prompt = f"""You are a helpful personal finance analyst.

User question:
{question}

Relevant past transactions:
{context}

Briefly (3-5 bullet points) analyze the user's spending. You MUST mention every category present in the retrieved transactions (including “other” or unknown categories) and include their amounts.

"""
    resp = structured_llm.invoke(prompt)
    return {"analysis_points": resp.bullet_points}


def answer_node(state: AdvisorState) -> AdvisorState:
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.3)
    structured_llm = llm.with_structured_output(AdvisorAnswer)
    question = state["question"]
    context = state.get("retrieved_context", "")
    analysis_points = state.get("analysis_points", [])
    analysis_text = (
        "\n".join(f"- {point}" for point in analysis_points) if analysis_points else "No structured analysis available."
    )

    prompt = f"""You are a friendly personal finance assistant.

User question:
{question}

Relevant past transactions:
{context}

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
