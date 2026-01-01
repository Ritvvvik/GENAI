# pip install mistralai
from dotenv import load_dotenv
from mistralai import Mistral
import os

load_dotenv(override=True)

client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))


def chat_with_mistral(query):
    completion = client.chat.complete(
        model=os.getenv("MISTRAL_CHAT_MODEL"),
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ],
    )
    return completion.choices[0].message.content


print(chat_with_mistral("Write a haiku about recursion in programming."))
print(chat_with_mistral("What is recursion in programming."))
