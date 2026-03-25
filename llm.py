import os
from groq import Groq
from prompt_builder import build_dynamic_prompt

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_answer(query, context):

    prompt = build_dynamic_prompt(query, context)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content


def refine_query(query):

    query = str(query)

    if "file" in query.lower():
        return "Describe the dataset and its contents"

    return query
