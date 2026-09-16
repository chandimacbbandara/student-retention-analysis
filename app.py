"""
Student Retention Analysis — Gradio Application
Entry point for Hugging Face Spaces deployment.
"""

import gradio as gr
import pandas as pd


def analyse(file):
    """Placeholder analysis function."""
    if file is None:
        return "Please upload a dataset to begin analysis."
    df = pd.read_csv(file.name)
    return df.describe().to_string()


demo = gr.Interface(
    fn=analyse,
    inputs=gr.File(label="Upload Student Dataset (CSV)"),
    outputs=gr.Textbox(label="Analysis Output"),
    title="Student Retention Analysis",
    description="Upload a student dataset to generate a descriptive summary.",
)

if __name__ == "__main__":
    demo.launch()
