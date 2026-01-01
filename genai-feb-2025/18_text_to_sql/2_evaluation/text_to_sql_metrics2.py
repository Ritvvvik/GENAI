# pip install ragas
from ragas import EvaluationDataset, SingleTurnSample
from ragas.metrics import LLMSQLEquivalence
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
    response="""
        SELECT pr.product_name, COUNT(oi.quantity)
        FROM order_items oi
        JOIN products pr ON oi.product_id = pr.product_id
        GROUP BY pr.product_name;
    """,
    reference="""
        SELECT p.product_name, COUNT(oi.quantity) AS total_quantity
        FROM order_items oi
        JOIN products p ON oi.product_id = p.product_id
        GROUP BY p.product_name;
    """,
    reference_contexts=[
        """
        Table order_items:
        - order_item_id: INT
        - order_id: INT
        - product_id: INT
        - quantity: INT
        """,
        """
        Table products:
        - product_id: INT
        - product_name: VARCHAR
        - price: DECIMAL
        """,
    ],
)


evaluation_dataset = EvaluationDataset(samples=[sample1])

# llm = ChatOpenAI(model="gpt-4o-mini")
# llm = ChatOllama(model="mistral")
# pipe = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct")
# llm = HuggingFacePipeline(pipeline=pipe)
llm = ChatGroq(model="gemma2-9b-it")
evaluator_llm = LangchainLLMWrapper(llm)

sql_equivalence_metric = LLMSQLEquivalence(llm=evaluator_llm)

result = evaluate(
    dataset=evaluation_dataset,
    metrics=[sql_equivalence_metric],
    run_config=RunConfig(max_workers=4, max_wait=60),
)
print(result)

df = result.to_pandas()
df
