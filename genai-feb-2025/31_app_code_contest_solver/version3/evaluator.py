import json
import os
import asyncio
from utils.jsonl import *
from config.config_reader import settings
from code_contest_solver import CodeContestSolverReflectionFS
from utils.problem import Problem


async def main():
    eval_data = read_jsonl(
        os.path.join(settings.file_paths.src_dir, settings.file_paths.eval_file)
    )
    solver = CodeContestSolverReflectionFS()
    results = []
    for p in eval_data:
        problem = Problem(
            src_uid=p["src_uid"],
            description=p["prob_desc_description"],
            input_spec=p["prob_desc_input_spec"],
            output_spec=p["prob_desc_output_spec"],
            sample_inputs=p["prob_desc_sample_inputs"],
            sample_outputs=p["prob_desc_sample_outputs"],
            notes=p["prob_desc_notes"],
            unit_tests=json.loads(p["hidden_unit_tests"]),
        )
        res = await solver.generate(problem)
        results.append(res)
    results_dir = settings.file_paths.results_file.split("/")[0]
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    write_jsonl(
        os.path.join(settings.file_paths.src_dir, settings.file_paths.results_file),
        results,
    )


if __name__ == "__main__":
    asyncio.run(main())
