from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from .config import OPENAI_MODEL
from .storage import get_vectorstore


class AdvisorState(TypedDict, total=False):
    question: str
    retrieved_context: str
    analysis: str
    answer: str


def retrieve_node(state: AdvisorState) -> AdvisorState:
    question = state["question"]
    vs = get_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    return {"retrieved_context": context}


def analyze_node(state: AdvisorState) -> AdvisorState:
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)
    question = state["question"]
    context = state.get("retrieved_context", "")

    prompt = f"""You are a helpful personal finance analyst.

User question:
{question}

Relevant past transactions:
{context}

Briefly (3-5 bullet points) analyze this user's recent spending pattern that is relevant to the question.
"""
    resp = llm.invoke(prompt)
    return {"analysis": resp.content}


def answer_node(state: AdvisorState) -> AdvisorState:
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.3)
    question = state["question"]
    context = state.get("retrieved_context", "")
    analysis = state.get("analysis", "")

    prompt = f"""You are a friendly personal finance assistant.

User question:
{question}

Relevant past transactions:
{context}

Analysis of spending:
{analysis}

Now answer the user's question in clear, simple language.
If it makes sense, include concrete numbers (like total amount spent per category, approximate monthly spending, etc.).
Also include 2-3 short, practical tips for better money management based on their pattern.
"""
    resp = llm.invoke(prompt)
    return {"answer": resp.content}


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