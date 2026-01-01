import asyncio
import time


async def dummy2(value: int) -> int:
    await asyncio.sleep(1)
    return value * 10


async def approach2() -> None:
    tot_count = 0
    for i in range(10):
        res = await dummy2(i)
        tot_count += res
    print(tot_count)


start_time = time.perf_counter()
asyncio.run(approach2())
end_time = time.perf_counter()
print(f"Approach2: {end_time - start_time:.2f} seconds")
