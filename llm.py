import os
from groq import Groq
from prompt_builder import build_dynamic_prompt

# ✅ STREAMLIT SECRET SUPPORT
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not set")

client = Groq(api_key=api_key)


def generate_answer(query, context, memory=""):

    prompt = build_dynamic_prompt(query, context, memory)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content
