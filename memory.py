from collections import defaultdict

# Store chat history per session
chat_memory = defaultdict(list)


def add_to_memory(session_id, query, answer):

    chat_memory[session_id].append({
        "query": query,
        "answer": answer
    })


def get_memory(session_id, k=5):

    history = chat_memory.get(session_id, [])

    # last k interactions
    history = history[-k:]

    memory_text = ""

    for item in history:
        memory_text += f"""
User: {item['query']}
Assistant: {item['answer']}
"""

    return memory_text
