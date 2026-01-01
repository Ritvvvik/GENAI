import gradio as gr
from summarizer import TextSummarizer

summarizer = TextSummarizer()


async def chatbot(text):
    result = await summarizer.summarize(text)
    return result


print("Launching Gradio")

iface = gr.Interface(
    fn=chatbot,
    inputs=[gr.Textbox(label="Enter Text")],
    title="Text Summarizer Bot",
    description="Provide the text and get summary.",
    outputs=[gr.Textbox(label="Response")],
)

iface.launch(share=False)
