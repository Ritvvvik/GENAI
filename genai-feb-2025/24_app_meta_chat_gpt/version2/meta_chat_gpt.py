import gradio as gr
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from config_reader import settings
from agent import gpt_agent
import logfire
import time


logfire.configure(token=settings.logfire.token)
time.sleep(1)
logfire.instrument_pydantic_ai()
logfire.instrument_openai()


def convert_gradio_history_to_pydantic_ai(history):
    history_pydantic_ai_format = []
    for msg in history:
        if msg["role"] == "user":
            tmp = ModelRequest(parts=[UserPromptPart(content=msg["content"])])
            history_pydantic_ai_format.append(tmp)
        elif msg["role"] == "assistant":
            tmp = ModelResponse(parts=[TextPart(content=msg["content"])])
            history_pydantic_ai_format.append(tmp)
    return history_pydantic_ai_format


async def respond(message, history):
    history_pydantic_ai_format = convert_gradio_history_to_pydantic_ai(history)
    async with gpt_agent.run_stream(
        message, message_history=history_pydantic_ai_format
    ) as result:
        async for text in result.stream_text():
            yield text


iface = gr.ChatInterface(
    respond,
    chatbot=gr.Chatbot(height=400, type="messages"),
    textbox=gr.Textbox(placeholder="Type your question", submit_btn=True),
    title="MetaChatGPT",
    description="Ask any question and get the answer.",
    flagging_mode="manual",
)

iface.launch()
