from dotenv import load_dotenv
from google import genai
import os
import logfire

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


@logfire.instrument("Calling gemini model", extract_args=True)
def chat_with_gemini(query):
    response = client.models.generate_content(
        model=os.environ["GEMINI_CHAT_MODEL"],
        contents=query,
    )
    return response.text


print(chat_with_gemini("Write a haiku about recursion in programming."))
print(chat_with_gemini("What is recursion in programming."))
print(chat_with_gemini("What are the production level usecases of generative ai."))
