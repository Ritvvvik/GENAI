from dotenv import load_dotenv
from openai import OpenAI
import os
import logfire

load_dotenv(override=True)

logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.instrument_openai()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def chat_with_openai(query):
    completion = client.chat.completions.create(
        model=os.getenv("OPENAI_CHAT_MODEL"),
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ],
    )
    return completion.choices[0].message.content


# print(chat_with_openai("What is recursion in programming."))
print(chat_with_openai("Give me short description about india"))
