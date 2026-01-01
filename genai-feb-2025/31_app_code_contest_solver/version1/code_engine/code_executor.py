from enum import Enum
import multiprocessing
import queue
import subprocess
import sys
import time
import traceback


class ExecOutcome(Enum):
    PASSED = "PASSED"  # code executes and output matches expected output
    WRONG_ANSWER = (
        "WRONG_ANSWER"  # code executes and output does NOT matches expected output
    )
    TIME_LIMIT_EXCEEDED = "TIME_LIMIT_EXCEEDED"  # code executes and didn't exit in time, output is ignored in this case
    RUNTIME_ERROR = "RUNTIME_ERROR"  # code failed to execute (crashed)
    COMPILATION_ERROR = "COMPILATION_ERROR"  # code failed to compile
    MEMORY_LIMIT_EXCEEDED = (
        "MEMORY_LIMIT_EXCEEDED"  # code exceeded memory limit during execution
    )


class CodeExecutor:
    def __init__(self):
        multiprocessing.set_start_method("spawn", force=True)

    def exec_program(self, q, program, input_data, expected_output, timeout):
        try:
            start_time = time.time()
            process = subprocess.Popen(
                [sys.executable, "-c", program],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = process.communicate(input=input_data, timeout=timeout)
            if time.time() - start_time > timeout:
                raise TimeoutError("Execution timed out.")
            if process.returncode != 0:
                q.put(f"failed: {stderr}")
            else:
                if stdout.strip() == expected_output.strip():
                    q.put("passed")
                else:
                    q.put(f"wrong answer. Expected '{expected_output}', got '{stdout}'")
        except subprocess.TimeoutExpired:
            process.kill()
            q.put("timed out")
        except Exception:
            q.put(f"failed: {traceback.format_exc()}")

    def check_correctness(
        self, program: str, input_data: str, expected_output: str, timeout: float
    ) -> str:
        q = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=self.exec_program,
            args=(q, program, input_data, expected_output, timeout),
        )
        process.start()
        process.join(timeout=timeout + 1)
        if process.is_alive():
            process.terminate()
            process.join()
            result = "timed out"
        else:
            try:
                result = q.get_nowait()
            except queue.Empty:
                result = "no result returned"
        return result

    def evaluate(self, src_uid: str, unittests: list[dict], code: str) -> bool:
        results = []
        for test_case in unittests:
            input_data = test_case["input"]
            print(input_data)
            expected_output = test_case["output"][0]
            print(expected_output)
            test_result = self.check_correctness(code, input_data, expected_output, 2)
            results.append(test_result)
        for result in results:
            print(result)
        return True
        

    def evaluateWithFeedback(
        self, src_uid: str, unittests: list[dict], code: str
    ) -> tuple[bool, str]:
        
        return False, ""

