# pip install ragas
from ragas import EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas import evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
import os
from langchain_groq import ChatGroq

# Sample 1
sample1 = SingleTurnSample(
    user_input="When was the first super bowl?",
    response="The first superbowl was held on Jan 15, 1967",
    retrieved_contexts=[
        "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
    ],
)

evaluation_dataset = EvaluationDataset(samples=[sample1])

llm = ChatGroq(model="gemma2-9b-it")
# llm = ChatOpenAI(model="gpt-4o-mini")
# llm = ChatOllama(model="mistral")
# pipe = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct")
# llm = HuggingFacePipeline(pipeline=pipe)
evaluator_llm = LangchainLLMWrapper(llm)

embeddings_model = HuggingFaceEmbeddings(
    model_name=os.getenv("HF_EMBEDDINGS_MODEL"),
    encode_kwargs={"normalize_embeddings": True},
    model_kwargs={"token": os.getenv("HUGGING_FACE_TOKEN")},
)
evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings_model)

faithfulness_metric = Faithfulness(llm=evaluator_llm)
response_relevancy_metric = ResponseRelevancy(
    llm=evaluator_llm, embeddings=evaluator_embeddings
)

run_config = RunConfig(max_workers=4, max_wait=60)
result = evaluate(
    dataset=evaluation_dataset,
    metrics=[faithfulness_metric, response_relevancy_metric],
    run_config=run_config,
)
print(result)
