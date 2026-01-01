import asyncio
import time


async def dummy3(value: int) -> int:
    await asyncio.sleep(1)
    return value * 10


async def approach3() -> None:
    tasks = [dummy3(i) for i in range(10)]
    values = await asyncio.gather(*tasks)
    print(sum(values))


start_time = time.perf_counter()
asyncio.run(approach3())
end_time = time.perf_counter()
print(f"Approach3: {end_time - start_time:.2f} seconds")
