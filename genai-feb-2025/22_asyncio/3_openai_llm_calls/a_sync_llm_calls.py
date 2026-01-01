from dotenv import load_dotenv
from openai import OpenAI
import logfire
import os
import time

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.instrument_openai()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def openai_call(query):
    completion = client.chat.completions.create(
        model=os.getenv("OPENAI_CHAT_MODEL"),
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ],
    )
    return completion.choices[0].message.content


def approach1() -> None:
    for _ in range(3):
        res = openai_call("what is the capital of india")
        print(res)


start_time = time.perf_counter()
approach1()
end_time = time.perf_counter()
print(f"Approach1: {end_time - start_time:.2f} seconds")
