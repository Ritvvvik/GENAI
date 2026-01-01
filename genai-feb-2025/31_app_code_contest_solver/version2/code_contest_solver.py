import os
import time
import logfire
from utils.parser import *
from config.config_reader import *
from code_engine.code_executor import *
from llm.prompt_engine import *
from agents.code_generator_agent import *
from agents.code_improver_agent import *


class CodeContestSolverReflectionZS:
    def __init__(self):
        self.code_executor = CodeExecutor()
        logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
        time.sleep(2)
        logfire.instrument_pydantic_ai()
        logfire.instrument_openai()

    async def generate(self, problem: Problem) -> tuple[bool, str, str]:
        code_generator_agent_prompt = get_code_generator_agent_prompt(problem)
        code_generator_agent_result = await code_generator_agent.run(
            code_generator_agent_prompt
        )
        code = parse_response(code_generator_agent_result.output)
        logfire.info(f"Seed Code:\n{code}")

        is_solved = False
        cur_iter = 1
        while cur_iter <= settings.max_iter:
            (is_solved, feedback) = self.code_executor.evaluateWithFeedback(
                problem.src_uid, problem.unit_tests, code
            )
            if is_solved:
                break
            logfire.info(f"Improvement Iteration: {cur_iter}")
            logfire.info(f"is_solved:{is_solved}")
            logfire.info(f"Feedback:\n{feedback}")
            code_improver_agent_prompt = get_code_improver_agent_prompt(
                problem, feedback, code
            )
            code_improver_agent_result = await code_improver_agent.run(
                code_improver_agent_prompt
            )
            code = parse_response(code_improver_agent_result.output)
            logfire.info(f"Modified Code:\n{code}")
            cur_iter += 1
        return (is_solved, "", code)
