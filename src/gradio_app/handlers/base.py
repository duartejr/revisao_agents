"""Low-level utilities shared across all Gradio event handlers.

Provides LLM provider management helpers and the ``_StdoutCapture`` /
``_StderrCapture`` context managers used to relay live agent output to
Gradio UI components.
"""

from __future__ import annotations

import glob
import logging
import os
import queue
import sys
import tempfile
from typing import Any

from revisao_agents.config import (
    get_runtime_config_summary,
    validate_provider,
)

# ═══════════════════════════════════════════════════════════════════════════
# UI Status & LLM Provider Management
# ═══════════════════════════════════════════════════════════════════════════


def list_llm_providers() -> list[str]:
    """Return supported LLM providers for UI selector, dynamically from the Factory.

    This ensures the UI always reflects the actual providers available in the system without hardcoding
    them in multiple places.

    Args:
        None

    Returns:
        A list of supported LLM provider names, capitalized for display in the UI.
    """
    from revisao_agents.utils.llm_utils.llm_providers import LLMProvider

    return [p.value for p in LLMProvider]


def get_current_llm_provider() -> str:
    """Return normalized current LLM provider from env.

    This function reads the LLM_PROVIDER environment variable, validates it against supported providers,
    and returns a normalized provider name. If the environment variable is not set or invalid, it defaults to "openai".

    Args:
        None

    Returns:
        A normalized LLM provider name.
    """
    try:
        provider = os.getenv("LLM_PROVIDER", "")
        return validate_provider(provider)
    except ValueError:
        return "openai"


def get_llm_provider_status() -> str:
    """Build concise status line for the global LLM selector.

    This function gathers information about the current LLM provider, model, and API key status to construct a
    user-friendly status message for display in the UI. It also checks for any configuration errors related to the
    provider.

    Args:
        None

    Returns:
        A status message string for the UI.
    """
    summary = get_runtime_config_summary()
    provider = summary["llm_provider"].capitalize()
    model = summary["llm_model"]
    key_ok = summary["llm_provider_key_present"]
    key_name = summary["llm_provider_key"]
    marker = "✅" if key_ok else "⚠️"
    key_msg = "key ok" if key_ok else f"missing {key_name}"
    provider_error = summary.get("llm_provider_error")

    if provider_error:
        return f"⚠️ Erro na Configuração: {provider_error}"

    return f"{marker} Provedor: {provider} | Modelo: {model} | {key_msg}"


def set_llm_provider(provider: str) -> tuple[str, str]:
    """Switch active provider globally for the current UI process.

    This function validates the requested provider, updates the environment variable, and returns the new provider
    along with an updated status message. If the provider is switched, it also resets the LLM_MODEL environment
    variable to ensure the new provider's default model is used.

    Args:
        provider: The name of the LLM provider to switch to.

    Returns:
        A tuple containing the normalized provider name and the status message.
    """
    normalized = validate_provider(provider)
    current = get_current_llm_provider()
    switched = normalized != current

    os.environ["LLM_PROVIDER"] = normalized

    if switched:
        os.environ["LLM_MODEL"] = ""

    status = get_llm_provider_status()
    if switched and "Model: <default>" in status:
        status = status + " (model reset to provider default)"
    return normalized, status


# ═══════════════════════════════════════════════════════════════════════════
# Live stdout capture classes
# ═══════════════════════════════════════════════════════════════════════════


class _StdoutCapture:
    """Context manager that redirects sys.stdout to a queue.

    This class captures writes to sys.stdout, sends complete lines to a queue for real-time UI updates, and also
    forwards all output to the original stdout to preserve normal console behavior.

    Args:
        q: A queue to which captured lines will be sent for UI display.

    Returns:
        An instance of _StdoutCapture.
    """

    def __init__(self, q: queue.Queue[str]) -> None:
        """Initialize the stdout capture context manager with a queue.

        Args:
            q: A queue to which captured lines will be sent for UI display.

        Returns:
            None
        """
        self._q = q
        self._buf = ""
        self._original: Any = None

    def __enter__(self) -> _StdoutCapture:
        """Redirect sys.stdout to this context manager, saving the original stdout.

        Returns:
            The _StdoutCapture instance itself, which will now capture writes to sys.stdout.
        """
        self._original = sys.stdout
        sys.stdout = self  # type: ignore[assignment]
        return self

    def __exit__(self, *_: Any) -> None:
        """Restore the original stdout and flush any remaining buffered output.

        Args:
            *_: Any additional arguments (ignored)

        Returns:
            None
        """
        if self._buf.strip():
            self._q.put(self._buf.rstrip())
            self._buf = ""
        sys.stdout = self._original

    def write(self, text: str) -> int:
        """Write text to the original stdout and capture complete lines for the queue.

        This method forwards all text to the original stdout and also buffers it to detect complete lines.
        When a newline is detected, the line is sent to the queue for UI updates.

        Args:
            text: The text to write to stdout.

        Returns:
            The number of characters written.
        """
        self._original.write(text)
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            stripped = line.rstrip()
            if stripped:
                self._q.put(stripped)
        return len(text)

    def flush(self) -> None:
        """Flush the original stdout.

        Returns:
            None
        """
        self._original.flush()

    @property
    def encoding(self) -> str:
        """Return the encoding of the original stdout, defaulting to 'utf-8' if not available.

        Returns:
            The encoding of the original stdout.
        """
        return getattr(self._original, "encoding", "utf-8")


class _StderrCapture:
    """Context manager that redirects sys.stderr to a queue.

    This class captures writes to sys.stderr, sends complete lines to a queue for real-time UI updates, and also
    forwards all output to the original stderr to preserve normal console behavior.

    Args:
        q: A queue to which captured lines will be sent for UI display.

    Returns:
        An instance of _StderrCapture.
    """

    def __init__(self, q: queue.Queue[str]):
        """Initialize the stderr capture context manager with a queue.

        Args:
            q: A queue to which captured lines will be sent for UI display.

        Returns:
            None
        """
        self._q = q
        self._buf = ""
        self._original: Any = None

    def __enter__(self) -> _StderrCapture:
        """Redirect sys.stderr to this context manager, saving the original stderr.

        Returns:
            The _StderrCapture instance itself, which will now capture writes to sys.stderr.
        """
        self._original = sys.stderr
        sys.stderr = self  # type: ignore[assignment]
        return self

    def __exit__(self, *_: Any) -> None:
        """Restore the original stderr and flush any remaining buffered output.

        Args:
            *_: Any additional arguments (ignored)

        Returns:
            None
        """
        if self._buf.strip():
            self._q.put(self._buf.rstrip())
            self._buf = ""
        sys.stderr = self._original

    def write(self, text: str) -> int:
        """Write text to the original stderr and capture complete lines for the queue.

        This method forwards all text to the original stderr and also buffers it to detect complete lines.
        When a newline is detected, the line is sent to the queue for UI updates.

        Args:
            text: The text to write to stderr.

        Returns:
            The number of characters written.
        """
        self._original.write(text)
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            stripped = line.rstrip()
            if stripped:
                self._q.put(stripped)
        return len(text)

    def flush(self) -> None:
        """Flush the original stderr."""
        self._original.flush()

    @property
    def encoding(self) -> str:
        """Return the encoding of the original stderr, defaulting to 'utf-8' if not available."""
        return getattr(self._original, "encoding", "utf-8")


class _QueueLogHandler(logging.Handler):
    """A logging handler that sends log records to a queue.

    This handler formats log records and sends them to a queue, which can be used for real-time UI updates of
    log messages.

    Args:
        q: A queue to which formatted log messages will be sent for UI display.

    Returns:
        An instance of _QueueLogHandler."""

    def __init__(self, q: queue.Queue[str]):
        """Initialize the logging handler with a queue.

        Args:
            q: A queue to which formatted log messages will be sent for UI display.

        Returns:
            None
        """
        super().__init__(level=logging.NOTSET)
        self._q = q

    def emit(self, record: logging.LogRecord) -> None:
        """Format the log record and send it to the queue.

        Args:
            record: The log record to be emitted.

        Returns:
            None
        """
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        if msg.strip():
            self._q.put(msg.rstrip())


class _LoggingCapture:
    """Context manager that adds a queue-backed logging handler to the root logger.

    This class captures log messages from the root logger, formats them, and sends them to a queue for real-time
    UI updates.

    Args:
        q: A queue to which formatted log messages will be sent for UI display.

    Returns:
        An instance of _LoggingCapture.
    """

    def __init__(self, q: queue.Queue[str]):
        """Initialize the logging capture context manager with a queue.

        Args:
            q: A queue to which log messages will be sent for UI display.

        Returns:
            None
        """
        self._q = q
        self._handler = _QueueLogHandler(q)
        self._logger = logging.getLogger()

    def __enter__(self) -> _LoggingCapture:
        """Enter the logging capture context.

        Returns:
            The logging capture context manager instance.
        """
        self._handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        self._logger.addHandler(self._handler)
        return self

    def __exit__(self, *_: Any) -> None:
        """Exit the logging capture context.

        Returns:
            None
        """
        self._logger.removeHandler(self._handler)


# ═══════════════════════════════════════════════════════════════════════════
# Shared internal helpers
# ═══════════════════════════════════════════════════════════════════════════


def _list_md(folder: str) -> list[str]:
    """List all .md files in the given folder.

    Args:
        folder: The path to the folder to search for .md files.

    Returns:
        A list of .md file paths in the given folder.
    """
    return glob.glob(os.path.join(folder, "*.md"))


def _find_newest_md(folder: str) -> str | None:
    """Find the newest .md file in the given folder.

    Args:
        folder: The path to the folder to search for .md files.

    Returns:
        The path to the newest .md file in the given folder, or None if no .md files are found.
    """
    files = _list_md(folder)
    return max(files, key=os.path.getmtime) if files else None


def _read_md(path: str | None) -> str:
    """Read the content of a markdown file.

    Args:
        path: The path to the markdown file.

    Returns:
        The content of the markdown file, or an empty string if the file does not exist or cannot be read.
    """
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _atomic_write(path: str, content: str) -> None:
    """Write content atomically to a file using a temporary file.

    Args:
        path: The path to the file to write.
        content: The content to write to the file.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, encoding="utf-8", dir=os.path.dirname(path) or "."
    ) as temp_file:
        temp_file.write(content)
        tmp_path = temp_file.name
    os.replace(tmp_path, path)


def _localized_text(language: str, pt_text: str, en_text: str) -> str:
    """Return text based on the detected language.

    Args:
        language: The detected language ("pt" or "en").
        pt_text: The text in Portuguese.
        en_text: The text in English.

    Returns:
        The text in the detected language.
    """
    return pt_text if language == "pt" else en_text


def _detect_user_language(user_text: str, fallback: str = "pt") -> str:
    """Detect if the user is writing in English or Portuguese based on simple markers.

    Args:
        user_text: The text input from the user to analyze for language detection.
        fallback: The default language to return if detection is inconclusive ("pt" or "en").

    Returns:
        The detected language code ("pt" or "en"), or the fallback if detection is inconclusive.
    """
    padded = f" {user_text.lower()} "
    pt_markers = [
        " seção ",
        " parágrafo ",
        " referências ",
        " referência ",
        " citação ",
        " fonte ",
        " internet ",
        " confirmar ",
        " confirme ",
        " cancelar ",
        " edição ",
        " achados ",
        " frase ",
        " trecho ",
        " artigos ",
        " mais ",
    ]
    en_markers = [
        " section ",
        " paragraph ",
        " references ",
        " reference ",
        " citation ",
        " source ",
        " internet ",
        " confirm ",
        " cancel ",
        " edit ",
        " findings ",
        " phrase ",
        " snippet ",
        " papers ",
        " more ",
    ]
    pt_score = sum(marker in padded for marker in pt_markers)
    en_score = sum(marker in padded for marker in en_markers)
    if en_score > pt_score:
        return "en"
    if pt_score > en_score:
        return "pt"
    return fallback
