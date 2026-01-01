from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
import os
import time
import logfire
from utils.parser import *
from config.config_reader import *
from code_engine.code_executor import *
from llm.prompt_engine import *
from agents.code_generator_agent import *
from agents.code_improver_agent import *
from ingestor.ingestor import KnowledgeIngestor


class CodeContestSolverReflectionFS:
    def __init__(self):
        self.vector_db_dir = os.path.join(
            settings.file_paths.src_dir, settings.file_paths.vector_db_dir
        )
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=settings.embedder.model,
            encode_kwargs={"normalize_embeddings": settings.embedder.normalize},
            model_kwargs={"token": settings.embedder.token},
        )
        self.reranker_model = HuggingFaceCrossEncoder(
            model_name=settings.reranker.model
        )
        if not os.path.exists(self.vector_db_dir):
            ingestor = KnowledgeIngestor()
            ingestor.ingest()
        else:
            self.vector_db = FAISS.load_local(
                folder_path=self.vector_db_dir,
                embeddings=self.embeddings_model,
                allow_dangerous_deserialization=True,
            )
        self.code_executor = CodeExecutor()
        logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
        time.sleep(2)
        logfire.instrument_pydantic_ai()
        logfire.instrument_openai()

    async def generate(self, problem: Problem) -> tuple[bool, str, str]:
        deps = Deps(
            self.vector_db,
            problem.description,
            self.reranker_model,
            settings.retriever.top_k,
            settings.reranker.top_k,
        )
        code_generator_agent_prompt = get_code_generator_agent_prompt(problem)
        code_generator_agent_result = await code_generator_agent.run(
            code_generator_agent_prompt, deps=deps
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
                code_improver_agent_prompt, deps=deps
            )
            code = parse_response(code_improver_agent_result.output)
            logfire.info(f"Modified Code:\n{code}")
            cur_iter += 1
        return (is_solved, "", code)
