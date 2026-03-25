from llm import generate_answer
from reranker import rerank


def planner_agent(state):
    return state


def retrieval_agent(state):

    docs = state["retriever"].search(state["query"])

    if not docs:
        return {**state, "retrieved_docs": []}

    return {**state, "retrieved_docs": docs}


def reranker_agent(state):

    docs = state.get("retrieved_docs", [])

    if not docs:
        return {**state, "reranked_docs": []}

    return {**state, "reranked_docs": rerank(state["query"], docs)}


def answer_agent(state):

    docs = state.get("reranked_docs", [])

    if not docs:
        return {**state, "answer": "No relevant info found", "sources": []}

    context = "\n".join([d.page_content for d in docs])

    return {
        **state,
        "answer": generate_answer(state["query"], context),
        "sources": [d.metadata for d in docs]
    }


def recommendation_agent(state):

    docs = state.get("reranked_docs", [])

    return {
        **state,
        "recommendations": [d.page_content[:200] for d in docs[:3]]
    }
