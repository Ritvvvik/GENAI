import gradio as gr
from rag import TextToSqlRAG

rag = TextToSqlRAG()


def chatbot(input_text):
    result = rag.execute(input_text)
    context = rag.retrieveContext(input_text)
    return context, result


print("Launching Gradio")

iface = gr.Interface(
    fn=chatbot,
    inputs=[gr.Textbox(label="Query")],
    examples=[
        "How many albums are there?",
        "Who are our top Customers according to Invoices?",
        "Which Employee has the Highest Total Number of Customers?",
        "How many Rock music listeners are there?",
        "What is the most popular genre for Australia?",
    ],
    title="TextToSQL QA Bot",
    description="This is a Text-to-sql question answering bot. It translates any natural language query to SQL.",
    outputs=[gr.Textbox(label="Context"), gr.Textbox(label="Response")],
    theme=gr.themes.Soft,
)

iface.launch(share=False)
