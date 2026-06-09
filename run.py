import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    from ui.gradio_app import build_ui
    build_ui().launch(ssr_mode=False)