import gradio as gr
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
    ModelMessage,
)
from order_assistant import *

order_assistant = OrderAssistant()


def convert_gradio_history_to_pydantic_ai(history) -> list[ModelMessage]:
    history_pydantic_ai_format = []
    for msg in history:
        if msg["role"] == "user":
            tmp = ModelRequest(parts=[UserPromptPart(content=msg["content"])])
            history_pydantic_ai_format.append(tmp)
        elif msg["role"] == "assistant":
            tmp = ModelResponse(parts=[TextPart(content=msg["content"])])
            history_pydantic_ai_format.append(tmp)
    return history_pydantic_ai_format


def respond(message, history) -> str:
    history_pydantic_ai_format = convert_gradio_history_to_pydantic_ai(history)
    response = order_assistant.process(
        user_query=message, history=history_pydantic_ai_format
    )
    return response


iface = gr.ChatInterface(
    respond,
    chatbot=gr.Chatbot(height=400, type="messages"),
    textbox=gr.Textbox(placeholder="Type your query", submit_btn=True),
    title="Order Processing Assistant",
    description="Chat in natural language for any order related queries/updates",
    examples=[
        "What is the status of order 001?",
        "What is the status of order 002?",
        "please update the shipping address of order 001: 10 Fifth Avenue, LosAngels, California, 10005, USA",
        "please update the shipping address of order 002: 10 Fifth Avenue, LosAngels, California, 10005, USA",
        "Cancel my order 001",
        "Cancel my order 002",
        "Requesting a return of item shipped as part of order 002, reason being wrong size",
        "Escalate to human for my order 001 since it is not resolved",
        "Escalate to human for my order 001 since customer is unhappy",
    ],
)

iface.launch()
