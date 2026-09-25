import gradio as gr

def dummy():
    return "Hello"

css = """
body {
    background-color: white !important;
}
"""
js = """
function() {
    // Force light theme
    setTimeout(() => {
        document.querySelector('html').classList.remove('dark');
        document.querySelector('html').style.colorScheme = 'light';
    }, 100);
}
"""

with gr.Blocks(js=js) as demo:
    gr.Markdown("# Hello")
    gr.Button("Test")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
