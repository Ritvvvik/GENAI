from utils.problem import *


def get_code_generator_agent_prompt(problem: Problem) -> str:
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
