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


def respond(message, history):
    history_pydantic_ai_format = convert_gradio_history_to_pydantic_ai(history)
    gpt_response = gpt_agent.run_sync(
        message, message_history=history_pydantic_ai_format
    )
    return gpt_response.data


iface = gr.ChatInterface(
    respond,
    chatbot=gr.Chatbot(height=400, type="messages"),
    textbox=gr.Textbox(placeholder="Type your question", submit_btn=True),
    title="MetaChatGPT",
    description="Ask any question and get the answer.",
    flagging_mode="manual",
)

iface.launch()
