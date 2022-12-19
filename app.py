import gradio as gr

GRE_score = gr.inputs.Textbox(label="GRE Score")
CGPA = gr.inputs.Slider(minimum=0, maximum=10, step=0.01, label="CGPA")
University_rating = gr.inputs.Radio([5, 4, 3, 2, 1], label="University Rating")
SOP = gr.inputs.Slider(minimum=0, maximum=5, step=0.5)
LOR = gr.inputs.Slider(minimum=0, maximum=5, step=0.5)
Research = gr.inputs.Radio([1, 0], label="Research")
