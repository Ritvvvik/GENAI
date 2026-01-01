import time


def dummy1(value: int) -> int:
    time.sleep(1)
    return value * 10


def approach1() -> None:
    tot_count = 0
    for i in range(10):
        tot_count += dummy1(i)
    print(tot_count)


start_time = time.perf_counter()
approach1()
end_time = time.perf_counter()
print(f"Approach1: {end_time - start_time:.2f} seconds")
