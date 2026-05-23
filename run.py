"""
Usage:
    python run.py              — launch Gradio UI
    python run.py batch        — run batch analysis on data/samples
    python run.py batch --images path/to/folder --output results
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

def main():
    import argparse

    parser = argparse.ArgumentParser(description="XAI_comp — CAM method comparison")
    subparsers = parser.add_subparsers(dest="command")

    batch_parser = subparsers.add_parser("batch", help="Run batch analysis on a folder of images")
    batch_parser.add_argument("--images", type=str, default="data/samples", help="Path to image folder")
    batch_parser.add_argument("--output", type=str, default="results", help="Output directory")
    batch_parser.add_argument("--models", type=str, nargs="*", default=None, help="Models to use")

    args = parser.parse_args()

    from app.models import get_device
    print(f"Device: {get_device()}")

    if args.command == "batch":
        from app.batch import run_batch
        run_batch(args.images, args.output, args.models)
    else:
        from ui.gradio_app import build_ui
        demo = build_ui()
        demo.launch()

if __name__ == "__main__":
    main()