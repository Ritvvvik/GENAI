# pip install openai
# pip install python-dotenv
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv(override=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

completion = client.chat.completions.create(
    model=os.getenv("OPENAI_CHAT_MODEL"),
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about recursion in programming."},
    ],
)
print(completion.choices[0].message)
print(completion.choices[0].message.content)

completion = client.chat.completions.create(
    model=os.getenv("OPENAI_CHAT_MODEL"),
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is recursion in programming."},
    ],
)
print(completion.choices[0].message)
print(completion.choices[0].message.content)
