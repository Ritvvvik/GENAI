import json
import os
from tqdm import tqdm
from rag import DocumentationRAG
from config_reader import settings
from ragas import EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas import evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


class DocumentationQAEvaluator:
    def __init__(self):
        self.src_dir = settings.file_paths.src_dir
        self.eval_file = settings.file_paths.eval_file
        self.eval_file_with_response = settings.file_paths.eval_file_with_response
        self.results_dir = settings.file_paths.results_dir
        llm = ChatGroq(
            model=settings.judge_llm.name, api_key=settings.judge_llm.api_key
        )
        self.evaluator_llm = LangchainLLMWrapper(llm)
        embeddings_model = HuggingFaceEmbeddings(
            model_name=settings.embedder.model,
            encode_kwargs={"normalize_embeddings": settings.embedder.normalize},
            model_kwargs={"token": settings.embedder.token},
        )
        self.evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings_model)

    def generate_llm_responses(self):
        rag = DocumentationRAG()
        print(">>> Begin: Generating responses for evaluation questions...")
        with open(os.path.join(self.src_dir, self.eval_file), "r") as f:
            eval_data = json.load(f)

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

    def evaluate_retriever_metrics(self):
        pass

    def evaluate_generator_metrics(self):
        print(">>> Begin: Evaluating LLM metrics...")
        with open(os.path.join(self.src_dir, self.eval_file_with_response), "r") as f:
            eval_data = json.load(f)

        # eval_data = eval_data[:3]
        # build evaluation dataset
        samples = []
        for item in eval_data:
            sample = SingleTurnSample(
                user_input=item["user_input"],
                retrieved_contexts=item["retrieved_contexts"],
                response=item["response"],
                reference=item["reference"],
            )
            samples.append(sample)
        evaluation_dataset = EvaluationDataset(samples=samples)

        # define metrics
        faithfulness_metric = Faithfulness(llm=self.evaluator_llm)
        response_relevancy_metric = ResponseRelevancy(
            llm=self.evaluator_llm, embeddings=self.evaluator_embeddings
        )

        # evaluate metrics
        result = evaluate(
            dataset=evaluation_dataset,
            metrics=[
                faithfulness_metric,
                response_relevancy_metric,
            ],
            run_config=RunConfig(max_workers=4, max_wait=60),
            llm=self.evaluator_llm,
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


if __name__ == "__main__":
    evaluator = DocumentationQAEvaluator()
    # evaluator.generate_llm_responses()
    evaluator.evaluate_generator_metrics()
