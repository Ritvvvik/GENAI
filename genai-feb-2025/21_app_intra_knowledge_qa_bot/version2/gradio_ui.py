import gradio as gr
from rag import IntraKnowledgeRAG

rag = IntraKnowledgeRAG()


def chatbot(input_text, retriever_top_k):
    retriever_top_k = int(retriever_top_k)
    result = rag.execute(input_text, retriever_top_k)
    context = rag.retrieveContextForUI(input_text, retriever_top_k)
    return context, result


print("Launching Gradio")

iface = gr.Interface(
    fn=chatbot,
    inputs=[
        gr.Textbox(label="Query"),
        gr.Slider(minimum=1, maximum=15, value=3, step=1, label="RetrieverTopK"),
    ],
    examples=[
        ["which navigation systems are created in 2024?", "2"],
        ["how much budget allocated for 2024?", "2"],
        ["major milestones achieved in 2024?", "2"],
        ["number of women employees in 2024?", "2"],
        [
            "number of women employees, inclusive of women in administration, in 2024?",
            "2",
        ],
        ["milestones achieved in 2025?", "2"],
    ],
    title="Intra Knowledge QA Bot",
    description="Ask questions on intra knowledge base and get the answers.",
    outputs=[gr.Textbox(label="Context"), gr.Textbox(label="Response")],
)

iface.launch(share=False)
