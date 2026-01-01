import gradio as gr
from rag import DocumentationRAG

rag = DocumentationRAG()


def chatbot(input_text, retriever_top_k, reranker_top_k):
    retriever_top_k = int(retriever_top_k)
    reranker_top_k = int(reranker_top_k)
    result = rag.execute(input_text, retriever_top_k, reranker_top_k)
    context = rag.retrieveContext(input_text, retriever_top_k, reranker_top_k)
    return context, result


print("Launching Gradio")

iface = gr.Interface(
    fn=chatbot,
    inputs=[
        gr.Textbox(label="Query"),
        gr.Slider(minimum=1, maximum=15, value=10, step=1, label="RetrieverTopK"),
        gr.Slider(minimum=1, maximum=10, value=5, step=1, label="RerankerTopK"),
    ],
    examples=[
        ["What are the supported llm models", "5", "3"],
        ["What is the difference between claude 1 and 2", "5", "3"],
        ["What are the rate limits of models?", "5", "3"],
        ["What are the supported embedding models", "5", "3"],
    ],
    title="Documentation QA Bot",
    description="Get answers for your questions on claude documentation.",
    outputs=[gr.Textbox(label="Context"), gr.Textbox(label="Response")],
)

iface.launch(share=False)
