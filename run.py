import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    import gradio as gr
    from ui.gradio_app import build_ui
    demo = build_ui()
    demo.launch(theme=gr.themes.Soft(), ssr_mode=False)