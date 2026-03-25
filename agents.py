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

    # ✅ DETECT VIDEO USING METADATA (ROBUST FIX)
    if any(d.metadata.get("type") == "video" for d in docs):
        return {
            **state,
            "answer": "📹 Video uploaded successfully.\n\n"
                      "Currently, video analysis is not supported in this deployment.\n\n"
                      "👉 To analyze the video, please upload ONE of the following:\n"
                      "• Transcript (.txt / .srt)\n"
                      "• Audio (.mp3 / .wav)\n"
                      "• Screenshots/images\n\n"
                      "Once provided, I can summarize and extract insights.",
            "sources": []
        }

    # ✅ AUDIO (OPTIONAL)
    if any(d.metadata.get("type") == "audio" for d in docs):
        return {
            **state,
            "answer": "🎵 Audio uploaded.\n\n"
                      "Transcription is not enabled in this deployment.\n"
                      "Please upload transcript for analysis.",
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
