import gradio as gr
from classifier import TextClassifierRAG

classifier = TextClassifierRAG()


def chatbot(input_text, retriever_top_k):
    retriever_top_k = int(retriever_top_k)
    result = classifier.predict(input_text, retriever_top_k)
    context = classifier.retrieveContext(input_text, retriever_top_k)
    return context, result


print("Launching Gradio")

iface = gr.Interface(
    fn=chatbot,
    inputs=[
        gr.Textbox(label="Query"),
        gr.Slider(minimum=1, maximum=15, value=3, step=1, label="RetrieverTopK"),
    ],
    examples=[
        [
            "I'm confused about a charge on my recent auto insurance bill that's higher than my usual premium payment. Can you clarify what this extra fee is for and why it was added?",
            5,
        ],
        [
            "I came out to my car after shopping and found that another vehicle had hit it, but the driver was nowhere to be found. What should my next steps be in terms of filing an insurance claim for the damage?",
            5,
        ],
    ],
    title="Insurance Support Ticket Classifier Bot",
    description="Provide the ticket details and get the class of the ticket.",
    outputs=[gr.Textbox(label="Context"), gr.Textbox(label="Response")],
)

iface.launch(share=False)
