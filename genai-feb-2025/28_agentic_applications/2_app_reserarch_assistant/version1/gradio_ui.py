import gradio as gr
from deep_researcher import DeepReseacher

researcher = DeepReseacher()


async def chatbot(query):
    (short_summary, md_report) = await researcher.do_research(query)
    return short_summary, md_report, "follow_up_questions"


print("Launching Gradio")

iface = gr.Interface(
    fn=chatbot,
    inputs=[gr.Textbox(lines=2, label="Research Query/Topic")],
    title="Research Assistant",
    description="Enter your query to generate a report",
    outputs=[
        gr.Textbox(label="Short Summary"),
        gr.Textbox(label="Final Report"),
        gr.Textbox(label="FollowUp Questions"),
    ],
)

iface.launch(share=False)
