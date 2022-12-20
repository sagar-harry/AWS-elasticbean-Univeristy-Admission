import gradio as gr
from src.models.predict_model import predict_value

GRE_score = gr.inputs.Slider(minimum=260, maximum=340, step=1, label="GRE Score")
CGPA = gr.inputs.Slider(minimum=0, maximum=10, step=0.01, label="CGPA")
University_rating = gr.inputs.Radio([5, 4, 3, 2, 1], label="University Rating")
SOP = gr.inputs.Slider(minimum=0, maximum=5, step=0.5, label="SOP")
LOR = gr.inputs.Slider(minimum=0, maximum=5, step=0.5, label="LOR")
Research = gr.inputs.Radio([1, 0], label="Research")

gr.Interface(predict_value, inputs=[GRE_score, CGPA, University_rating, SOP, LOR, Research],
        outputs="label",
        title="Chance of getting into university",
        live=True).launch()
        