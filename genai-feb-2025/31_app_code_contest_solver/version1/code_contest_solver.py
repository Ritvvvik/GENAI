import os
import time
import logfire
from utils.parser import *
from config.config_reader import *
from code_engine.code_executor import *
from llm.prompt_engine import *
from agents.code_generator_agent import *


class CodeContestSolverDirect:
    def __init__(self):
        self.code_executor = CodeExecutor()
        logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
        time.sleep(2)
        logfire.instrument_pydantic_ai()
        logfire.instrument_openai()

    async def generate(self, problem: Problem) -> tuple[bool, str, str]:
        is_solved = False
        cur_iter = 1
        while cur_iter <= settings.max_iter and not is_solved:
            logfire.info(f"Iteration: {cur_iter}")
            logfire.info("Generating Code")
            code_generator_agent_prompt = get_code_generator_agent_prompt(problem)
            code_generator_agent_result = await code_generator_agent.run(
                code_generator_agent_prompt
            )
            code = parse_response(code_generator_agent_result.output)
            logfire.info(f"code:{code}")

            logfire.info("Executing Code...")
            is_solved = self.code_executor.evaluate(
                problem.src_uid, problem.unit_tests, code
            )
            logfire.info(f"exec result:{is_solved}")
            cur_iter += 1
        return (is_solved, "", code)
