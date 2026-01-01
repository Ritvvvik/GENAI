from pathlib import Path
from typing import List
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.messages import BinaryContent
import os
from dotenv import load_dotenv
from textwrap import dedent
import logfire

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.instrument_openai()

system_prompt = """
    You are smart vision agent with text extraction skills
"""

agent = Agent(
    model=GeminiModel(
        model_name=os.getenv("GEMINI_CHAT_MODEL"), api_key=os.getenv("GEMINI_API_KEY")
    ),
    system_prompt=dedent(system_prompt),
)

image_file_path = os.path.expanduser(
    "~/Documents/genai-training-pydanticai/data/vision_data/stv_jbs.png"
)
image_bytes = Path(image_file_path).read_bytes()
response = agent.run_sync(
    [
        "What's in the image?",
        BinaryContent(data=image_bytes, media_type="image/png"),
    ]
)
print(response.data)

image_file_path = os.path.expanduser(
    "~/Documents/genai-training-pydanticai/data/vision_data/grocery_test.png"
)
image_bytes = Path(image_file_path).read_bytes()
response = agent.run_sync(
    [
        "how much should I pay for this bill?",
        BinaryContent(data=image_bytes, media_type="image/png"),
    ]
)
print(response.data)
