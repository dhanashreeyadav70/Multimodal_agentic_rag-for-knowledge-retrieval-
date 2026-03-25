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
        return {
            **state,
            "answer": "No relevant information found.",
            "sources": []
        }

    # ✅ VIDEO HANDLING (NO LLM CALL)
    if any(d.page_content == "VIDEO_FILE" for d in docs):
        return {
            **state,
            "answer": "📹 Video uploaded successfully.\n\n"
                      "Currently, video analysis is not supported.\n\n"
                      "👉 To proceed, please upload:\n"
                      "- Transcript (.txt/.srt)\n"
                      "- Audio (.mp3/.wav)\n"
                      "- Screenshots/images\n\n"
                      "Then I can summarize and analyze it.",
            "sources": []
        }

    # ✅ AUDIO HANDLING
    if any(d.page_content == "AUDIO_FILE" for d in docs):
        return {
            **state,
            "answer": "🎵 Audio uploaded.\n\nTranscription not enabled.\n"
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
