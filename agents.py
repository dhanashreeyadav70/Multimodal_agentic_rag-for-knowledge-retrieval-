from llm import generate_answer
from reranker import rerank


def planner_agent(state):
    return state


# def retrieval_agent(state):

#     docs = state["retriever"].search(state["query"])

#     if not docs:
#         return {**state, "retrieved_docs": []}

#     return {**state, "retrieved_docs": docs}
def retrieval_agent(state):

    docs = state["retriever"].search(state["query"])

    # ✅ If video present, skip retrieval logic
    if any(d.metadata.get("type") == "video" for d in docs):
        return {**state, "retrieved_docs": docs}

    return {**state, "retrieved_docs": docs or []}

def reranker_agent(state):

    docs = state.get("retrieved_docs", [])

    if not docs:
        return {**state, "reranked_docs": []}

    return {**state, "reranked_docs": rerank(state["query"], docs)}


from memory import get_memory, add_to_memory

def answer_agent(state):

    docs = state.get("reranked_docs", [])

    if not docs:
        return {**state, "answer": "No relevant info found", "sources": []}

    context = "\n".join([d.page_content for d in docs])

    session_id = state.get("session_id", "default")

    memory = get_memory(session_id)

    answer = generate_answer(state["query"], context, memory)

    add_to_memory(session_id, state["query"], answer)

    return {
        **state,
        "answer": answer,
        "sources": [d.metadata for d in docs]
    }

def recommendation_agent(state):

    docs = state.get("reranked_docs", [])

    return {
        **state,
        "recommendations": [d.page_content[:200] for d in docs[:3]]
    }
