import gradio as gr
from rag import DocumentationRAG

rag = DocumentationRAG()


def chatbot(input_text, retriever_top_k):
    retriever_top_k = int(retriever_top_k)
    print(retriever_top_k)
    result = rag.execute(input_text, retriever_top_k)
    context = rag.retrieveContext(input_text, retriever_top_k)
    return context, result


print("Launching Gradio")

iface = gr.Interface(
    fn=chatbot,
    inputs=[
        gr.Textbox(label="Query"),
        gr.Slider(minimum=1, maximum=15, value=3, step=1, label="RetrieverTopK"),
    ],
    examples=[
        ["What are the supported llm models", "5"],
        ["What is the difference between claude 1 and 2", "5"],
        ["What are the rate limits of models?", "5"],
        ["What are the supported embedding models", "5"],
    ],
    title="Documentation QA Bot",
    description="Get answers for your questions on claude documentation.",
    outputs=[gr.Textbox(label="Context"), gr.Textbox(label="Response")],
)

iface.launch(share=False)
