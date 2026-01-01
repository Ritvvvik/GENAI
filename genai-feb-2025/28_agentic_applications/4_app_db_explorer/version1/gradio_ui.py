import gradio as gr
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
    ModelMessage,
)
from db_explorer import *

db_explorer = DBExplorer()


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


def respond(message, history, topk) -> str:
    history_pydantic_ai_format = convert_gradio_history_to_pydantic_ai(history)
    response = db_explorer.process(
        user_query=message,
        history=history_pydantic_ai_format,
        retriever_top_k=int(topk),
    )
    return response


iface = gr.ChatInterface(
    respond,
    chatbot=gr.Chatbot(height=400, type="messages"),
    textbox=gr.Textbox(placeholder="Type your query", submit_btn=True),
    additional_inputs=[
        gr.Slider(minimum=1, maximum=15, value=3, step=1, label="RetrieverTopK")
    ],
    title="Database Explorer",
    description="Chat in natural language for data exploration",
    examples=[
        ["How many albums are there?", 5],
        ["How many distinct genres are there?", 5],
        ["Which Employee has the Highest Total Number of Customers?", 5],
        ["Who are our top Customers according to Invoices?", 5],
        ["What is the most popular genre for Australia?", 5],
    ],
)

iface.launch()
