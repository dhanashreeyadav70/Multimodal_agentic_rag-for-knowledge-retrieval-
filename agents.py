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


def answer_agent(state):

    docs = state.get("reranked_docs", [])

    if not docs:
        return {
            **state,
            "answer": "No relevant information found.",
            "sources": []
        }

    # ✅ VIDEO
    if any(d.metadata.get("type") == "video" for d in docs):
        return {
            **state,
            "answer": "📹 Video uploaded.\n\n"
                      "Please upload transcript/audio/screenshots for analysis.",
            "sources": []
        }

    # ✅ IMAGE WITH NO TEXT
    if any(d.page_content == "NO_TEXT_IN_IMAGE" for d in docs):
        return {
            **state,
            "answer": "🖼️ Image uploaded.\n\n"
                      "No readable text detected in the image.\n\n"
                      "👉 You can:\n"
                      "• Upload clearer image\n"
                      "• Provide description\n"
                      "• Ask visual-related question",
            "sources": []
        }

    # ✅ OCR FAILED
    if any(d.page_content == "OCR_FAILED" for d in docs):
        return {
            **state,
            "answer": "🖼️ Image uploaded.\n\n"
                      "Text extraction failed.\n"
                      "Try uploading a clearer image.",
            "sources": []
        }

    # ✅ NORMAL FLOW
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
