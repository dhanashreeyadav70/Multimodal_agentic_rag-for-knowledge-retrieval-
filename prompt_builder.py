# def build_dynamic_prompt(query, context):

#     query = str(query)
#     context = str(context)

#     if "file" in query.lower() or "dataset" in query.lower():
#         task = "Explain what this dataset contains."

#     elif "compare" in query.lower():
#         task = "Compare clearly."

#     elif "recommend" in query.lower():
#         task = "Give recommendations."

#     elif "summary" in query.lower():
#         task = "Provide summary."

#     else:
#         task = "Answer clearly."

#     return f"""
# You are an enterprise AI assistant. Do not hallucinate.

# Task: {task}

# Context:
# {context}

# Question:
# {query}

# Answer:
# """

def build_dynamic_prompt(query, context, memory=""):

    query = str(query)
    context = str(context)
    memory = str(memory)

    return f"""
You are an enterprise AI assistant.

Guidelines::
1. Conversation history
2. Retrieved context

3. Use the provided context as the primary source of truth.
4. If relevant information is partially missing, use your general knowledge to enhance the answer, but clearly distinguish it.
5. Do NOT fabricate facts. If unsure, mention limitations.
6. Provide clear, concise, and structured responses.
7. Highlight key insights, policies, or findings.
8. Where applicable, include recommendations or next steps.

DO NOT hallucinate.

---------------------
Conversation History:
{memory}

---------------------
Context:
{context}

---------------------
User Question:
{query}

---------------------
Answer clearly and contextually:
"""
