"""
run_ui.py — Launch the Gradio web interface for the Revisão da Literatura Agent.

Usage:
    python run_ui.py                  # Launch locally on port 7860
    python run_ui.py --port 8080      # Custom port
    python run_ui.py --share          # Create a public Gradio link

The app connects to the same MongoDB and LLM backend used by the CLI.
Ensure that the .env file is configured before running.
"""

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Ensure src/ is on the path so both gradio_app and revisao_agents are found
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from gradio_app.app import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch the Revisão da Literatura Gradio UI"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port to serve on (default: 7860)"
    )
    parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio share link"
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" 🔬 Agente de Revisão da Literatura — UI Mode")
    print(f"    http://localhost:{args.port}")
    print("=" * 60)

    main(share=args.share, port=args.port)
