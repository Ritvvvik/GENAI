# pip install -q -U google-genai
from dotenv import load_dotenv
from google import genai
import os

load_dotenv(override=True)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def chat_with_gemini(query):
    response = client.models.generate_content(
        model=os.getenv("GEMINI_CHAT_MODEL"),
        contents=query,
    )
    return response.text


print(chat_with_gemini("Write a haiku about recursion in programming."))
print(chat_with_gemini("What is recursion in programming."))
print(chat_with_gemini("What are the production level usecases of generative ai."))
