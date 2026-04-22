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
# Load .env FIRST — before any project import reads os.getenv() at module level
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv(override=False)  # does not override vars already set in the shell
except ImportError:
    pass  # python-dotenv not installed; rely on env vars being set externally

# ---------------------------------------------------------------------------
# Ensure src/ is on the path so both gradio_app and revisao_agents are found
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from gradio_app.app import main  # noqa: E402

try:
    from observability import initialize_experiments  # noqa: E402

    initialize_experiments()
except Exception as exc:  # noqa: BLE001
    import warnings

    warnings.warn(
        f"MLflow initialization failed — experiment tracking disabled: {exc}",
        stacklevel=1,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the Revisão da Literatura Gradio UI")
    parser.add_argument("--port", type=int, default=7860, help="Port to serve on (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    args = parser.parse_args()

    print("=" * 60)
    print(" 🔬 Agente de Revisão da Literatura — UI Mode")
    print(f"    http://localhost:{args.port}")
    print("=" * 60)

    main(share=args.share, port=args.port)
