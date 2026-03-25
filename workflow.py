from typing import TypedDict
from langgraph.graph import StateGraph, END
from agents import *

class RAGState(TypedDict):
    query: str
    retriever: object
    retrieved_docs: list
    reranked_docs: list
    answer: str
    sources: list
    recommendations: list
    session_id: str


def build_graph():

    graph = StateGraph(RAGState)

    graph.add_node("planner", planner_agent)
    graph.add_node("retrieval", retrieval_agent)
    graph.add_node("reranker", reranker_agent)
    graph.add_node("answer", answer_agent)
    graph.add_node("recommendation", recommendation_agent)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "retrieval")
    graph.add_edge("retrieval", "reranker")
    graph.add_edge("reranker", "answer")
    graph.add_edge("answer", "recommendation")
    graph.add_edge("recommendation", END)

    return graph.compile()
