import os
import time
import logfire
from agents.retriever_agent import *
from agents.plan_generator_agent import *
from agents.plan_verifier_agent import *
from agents.code_generator_agent import *
from agents.code_improver_agent import *
from llm.prompt_engine import *
from code_engine.code_executor import *
from config.config_reader import settings
from utils.problem import Problem
from utils.parser import parse_response


class CodeContestSolverMapCoder:
    def __init__(self):
        self.code_executor = CodeExecutor()
        logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
        time.sleep(2)
        logfire.instrument_pydantic_ai()
        logfire.instrument_openai()

    async def retrieve_similar_problems(
        self, problem: Problem
    ) -> tuple[list[ProblemInfo], str]:
        retriever_agent_prompt = get_retriever_agent_prompt(problem)
        retriever_agent_result = await retriever_agent.run(retriever_agent_prompt)
        for i, p in enumerate(retriever_agent_result.output.similar_problems):
            logfire.info(f"Similar Problem {i + 1}")
            logfire.info(f"Description:{p.description}")
            logfire.info(f"Code:{p.code}")
            logfire.info(f"Planning:{p.plan}")
        logfire.info(
            f"Algorithm to solve problem:{retriever_agent_result.output.algorithm_technique}"
        )
        return (
            retriever_agent_result.output.similar_problems,
            retriever_agent_result.output.algorithm_technique,
        )

    async def generate_plans_with_verified_scores(
        self,
        problem: Problem,
        similar_problems: list[ProblemInfo],
        algorithm_technique: str,
    ) -> list[tuple[str, float, ProblemInfo]]:
        plans = []
        for _, similar_problem in enumerate(similar_problems):
            plan_generator_agent_prompt = get_plan_generator_agent_prompt(
                similar_problem,
                problem,
                algorithm_technique,
            )
            plan_generator_agent_result = await plan_generator_agent.run(
                plan_generator_agent_prompt
            )
            plan_verifier_agent_prompt = get_plan_verifier_agent_prompt(
                problem, plan_generator_agent_result.output
            )
            plan_verifier_agent_result = await plan_verifier_agent.run(
                plan_verifier_agent_prompt
            )
            plans.append(
                (
                    plan_generator_agent_result.output,
                    plan_verifier_agent_result.output.confidence_score,
                    similar_problem,
                )
            )
        plans.sort(key=lambda x: x[1], reverse=True)
        for i, plan in enumerate(plans):
            logfire.info(f"plan {i + 1}, confidence score: {plan[1]}")
            logfire.info(plan[0])
        return plans

    async def generate_code_with_feedback_loop(
        self, problem: Problem, plan: str
    ) -> tuple[bool, str, str]:
        code_generator_agent_prompt = get_code_generator_agent_prompt(problem, plan)
        code_generator_agent_result = await code_generator_agent.run(
            code_generator_agent_prompt
        )
        code = parse_response(code_generator_agent_result.output)
        logfire.info(f"Seed Code\n{code}")

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
                problem, code, feedback, plan
            )
            code_improver_agent_result = await code_improver_agent.run(
                code_improver_agent_prompt
            )
            code = parse_response(code_improver_agent_result.output.modified_code)
            plan = code_improver_agent_result.output.modified_plan
            logfire.info(
                f"Current Plan\n{code_improver_agent_result.output.current_plan}"
            )
            logfire.info(f"Modified Plan\n{plan}")
            logfire.info(f"Modified Code\n{code}")
            cur_iter += 1
        return (is_solved, plan, code)

    async def generate(self, problem: Problem) -> tuple[bool, str, str]:
        (similar_problems, algorithm_technique) = await self.retrieve_similar_problems(
            problem
        )
        sorted_plans_with_scores = await self.generate_plans_with_verified_scores(
            problem, similar_problems, algorithm_technique
        )
        final_code = ""
        final_plan = ""
        for scored_plan in sorted_plans_with_scores:
            (is_solved, final_plan, final_code) = (
                await self.generate_code_with_feedback_loop(problem, scored_plan[0])
            )
            if is_solved:
                break
        return (is_solved, final_plan, final_code)
