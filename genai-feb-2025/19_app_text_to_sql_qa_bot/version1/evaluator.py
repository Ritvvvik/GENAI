import json
import os
from tqdm import tqdm
from rag import TextToSqlRAG
from config_reader import settings
from utils import *
import pandas as pd
from ragas import EvaluationDataset, SingleTurnSample
from ragas.metrics import LLMSQLEquivalence
from ragas import evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq


class TextToSqlEvaluator:
    def __init__(self):
        self.src_dir = settings.file_paths.src_dir
        self.eval_file = settings.file_paths.eval_file
        self.eval_file_with_response = settings.file_paths.eval_file_with_response
        self.results_dir = settings.file_paths.results_dir
        llm = ChatGroq(
            model=settings.judge_llm.name, api_key=settings.judge_llm.api_key
        )
        self.evaluator_llm = LangchainLLMWrapper(llm)
        self.db_file_path = os.path.join(self.src_dir, settings.file_paths.db_file)

    def generate_llm_responses(self):
        print(">>> Begin: Generating responses for evaluation questions...")
        with open(os.path.join(self.src_dir, self.eval_file), "r") as f:
            eval_data = json.load(f)

        rag = TextToSqlRAG()
        results = []
        for item in tqdm(eval_data, desc="Generating Answers Using RAG"):
            response = rag.execute(item["question"])
            retrieved_contexts = rag.retrieveContext(item["question"])
            tmp = {
                "id": item["id"],
                "user_input": item["question"],
                "retrieved_contexts": retrieved_contexts,
                "response": response,
                "reference": item["answer"],
            }
            results.append(tmp)

        with open(os.path.join(self.src_dir, self.eval_file_with_response), "w") as f:
            json.dump(results, f, indent=2)
        print(">>> End: Generating responses for evaluation questions...")

    def evaluate_llm_metrics(self):
        print(">>> Begin: Evaluating LLM metrics...")
        with open(os.path.join(self.src_dir, self.eval_file_with_response), "r") as f:
            eval_data = json.load(f)

        # build evaluation dataset
        samples = []
        for item in eval_data:
            sample = SingleTurnSample(
                user_input=item["user_input"],
                reference_contexts=item["retrieved_contexts"],
                response=item["response"],
                reference=item["reference"],
            )
            samples.append(sample)
        evaluation_dataset = EvaluationDataset(samples=samples)

        # define metrics
        sql_equivalence_metric = LLMSQLEquivalence(llm=self.evaluator_llm)

        # evaluate metrics
        result = evaluate(
            dataset=evaluation_dataset,
            metrics=[sql_equivalence_metric],
            run_config=RunConfig(max_workers=4, max_wait=60),
            show_progress=True,
        )

        # write results
        results_dir_path = os.path.join(self.src_dir, self.results_dir)
        if not os.path.exists(results_dir_path):
            os.makedirs(results_dir_path)

        print(result)
        df = result.to_pandas()
        df.to_csv(
            os.path.join(results_dir_path, "results_llm_metrics.csv"), index=False
        )
        print(">>> End: Evaluating LLM metrics...")

    def evaluate_non_llm_metrics(self):
        print(">>> Begin: Evaluating Non-LLM metrics...")
        with open(os.path.join(self.src_dir, self.eval_file_with_response), "r") as f:
            eval_data = json.load(f)

        ref_counts = []
        res_counts = []
        for i, item in enumerate(eval_data):
            ref_counts.append(execute_query(self.db_file_path, item["reference"]))
            res_counts.append(execute_query(self.db_file_path, item["response"]))

        results_dir_path = os.path.join(self.src_dir, self.results_dir)
        if not os.path.exists(results_dir_path):
            os.makedirs(results_dir_path)

        df = pd.json_normalize(eval_data)
        df["reference_counts"] = ref_counts
        df["response_counts"] = res_counts
        df.to_csv(
            os.path.join(results_dir_path, "results_non_llm_metrics.csv"), index=False
        )
        print(">>> End: Evaluating Non-LLM metrics...")


if __name__ == "__main__":
    evaluator = TextToSqlEvaluator()
    # evaluator.generate_llm_responses()
    # evaluator.evaluate_non_llm_metrics()
    evaluator.evaluate_llm_metrics()
