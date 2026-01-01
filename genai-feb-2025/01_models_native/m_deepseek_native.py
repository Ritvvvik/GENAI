# pip install deepseek
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv(override=True)

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
)


def chat_with_deepseek(query):
    completion = client.chat.completions.create(
        model=os.getenv("DEEPSEEK_CHAT_MODEL"),
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ],
    )
    return completion.choices[0].message.content


print(chat_with_deepseek("Write a haiku about recursion in programming."))
print(chat_with_deepseek("What is recursion in programming."))
