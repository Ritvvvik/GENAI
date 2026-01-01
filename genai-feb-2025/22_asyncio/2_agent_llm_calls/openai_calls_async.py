# pip install openai
from dotenv import load_dotenv
from openai import AsyncOpenAI
import os
import asyncio
import time

load_dotenv(override=True)

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def openai_call(query):
    completion = await client.chat.completions.create(
        model=os.getenv("OPENAI_CHAT_MODEL"),
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ],
    )
    return completion.choices[0].message.content


async def approach1():
    for _ in range(3):
        res = await openai_call("Give me haiku about generative ai")
        print(res)


async def approach2():
    res = await asyncio.gather(
        *[openai_call("Give me haiku about generative ai") for _ in range(3)]
    )
    print(res)


start_time = time.perf_counter()
asyncio.run(approach1())
end_time = time.perf_counter()
print(f"Approach1: {end_time - start_time:.2f} seconds")

start_time = time.perf_counter()
asyncio.run(approach2())
end_time = time.perf_counter()
print(f"Approach2: {end_time - start_time:.2f} seconds")
