from agents.retriever_agent import ProblemInfo
from utils.problem import Problem


def prompt_for_problem(problem: Problem) -> str:
    return f"""
    # Problem Description:
    {problem.description}
    # Input Specification:
    {problem.input_spec}
    # Output Specification:
    {problem.output_spec}
    # Sample Inputs:
    {problem.sample_inputs}
    # Sample Outputs:
    {problem.sample_outputs}
    # Note:
    {problem.notes}
    # Take input from:
    {problem.input_from}
    # Give output to:
    {problem.output_to}
    """


def get_retriever_agent_prompt(problem: Problem) -> str:
    return prompt_for_problem(problem)


def get_plan_generator_agent_prompt(
    similar_problem: ProblemInfo, problem: Problem, algo_technique: str
) -> str:
    return f"""
    # Problem to be solved:
    {prompt_for_problem(problem)}
    # Similar Problem Description:
    {similar_problem.description}
    # Similar Problem Plan:
    {similar_problem.plan}
    # Relevant Algorithm technique to be used:
    {algo_technique}
    """


def get_plan_verifier_agent_prompt(problem: Problem, plan: str) -> str:
    return f"""
    # Problem to be solved:
    {prompt_for_problem(problem)}
    # Plan:
    {plan}
    """


def get_code_generator_agent_prompt(problem: Problem, plan: str) -> str:
    return f"""
        # Problem to be solved:
        {prompt_for_problem(problem)}
        # Plan:
        {plan}
        """


def get_code_improver_agent_prompt(
    problem: Problem, code: str, feedback: str, plan: str
) -> str:
    return f"""
        # Problem that you were solving:
        {prompt_for_problem(problem)}
        # Plan:
        {plan}
        # Buggy Code:
        {code}
        # Test Report:
        {feedback}
        """
