"""Utility functions for timestamping and file I/O."""
import json
import os
from datetime import datetime, timezone

import config

MAX_JSON_SIZE = 1024 * 1024  # 1MB


def get_timestamp() -> str:
    """Return the current UTC timestamp as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def ensure_logs_dir():
    os.makedirs(config.LOGS_DIRECTORY, exist_ok=True)


def _run_counter_path() -> str:
    """Return the path of the run counter file."""
    ensure_logs_dir()
    return os.path.join(config.LOGS_DIRECTORY, "run_counter.txt")


def get_run_number() -> int:
    """Return the current run number stored on disk (0 if not found)."""
    path = _run_counter_path()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return int(fh.read().strip())
    except Exception:
        return 0


def increment_run_number() -> int:
    """Increment and persist the run counter, returning the new value."""
    run_no = get_run_number() + 1
    path = _run_counter_path()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(str(run_no))
    return run_no


def format_agent_id(run_no: int, index: int) -> str:
    """Return a unique agent identifier for the given run and index."""
    ts = get_timestamp().replace(":", "").replace("-", "")
    return f"{run_no}.{index}_{ts}"



def _wrap_text(text: str, width: int = 150) -> str:
    """Return text wrapped with newline every `width` characters."""
    lines = []
    for line in text.splitlines():
        while len(line) > width:
            lines.append(line[:width])
            line = line[width:]
        lines.append(line)
    return "\n".join(lines)


def append_improvement_log(
    run_no: int,
    prompt: str,
    method: str | None = None,
    conv_no: int | None = None,
    dataset_size: int | None = None,
) -> None:
    """Append the improved prompt to a persistent text log."""
    ensure_logs_dir()
    path = os.path.join(config.LOGS_DIRECTORY, "improved_prompts.txt")
    ts = get_timestamp()
    wrapped = _wrap_text(prompt, 150)
    method_part = f" method={method}" if method else ""
    conv_part = f" conv={conv_no}" if conv_no is not None else ""
    data_part = f" dataset={dataset_size}" if dataset_size is not None else ""
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(
            f"{ts} run={run_no}{conv_part}{data_part}{method_part} instructions=\"{wrapped}\"\n"
        )


def append_improver_instruction_log(run_no: int, prompt: str) -> None:
    """Append the prompt-improver instructions to a separate log."""
    ensure_logs_dir()
    path = os.path.join(config.LOGS_DIRECTORY, "improver_instructions.txt")
    ts = get_timestamp()
    wrapped = _wrap_text(prompt, 150)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"{ts} run={run_no} instructions=\"{wrapped}\"\n")


def append_wizard_score(
    run_no: int, conv_no: int, score: float, improved: bool = False
) -> None:
    """Append a wizard score entry to ``wizard_scores.csv``."""
    ensure_logs_dir()
    path = os.path.join(config.LOGS_DIRECTORY, "wizard_scores.csv")
    new_file = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as fh:
        if new_file:
            fh.write("timestamp,run,conversation,score,improved\n")
        fh.write(
            f"{get_timestamp()},{run_no},{conv_no},{score},{int(improved)}\n"
        )



def save_conversation_log(log_obj: dict, filename: str) -> None:
    """Save a conversation log as JSON under the logs directory."""
    ensure_logs_dir()
    path = os.path.join(config.LOGS_DIRECTORY, filename)
    with open(path, "w", encoding="utf-8") as f:
        # Use default=str so any non-serializable objects are converted to
        # strings rather than raising an exception.
        json.dump(log_obj, f, indent=config.JSON_INDENT, default=str)


def load_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def render_template(template_str: str, variables: dict) -> str:
    text = template_str
    for key, val in variables.items():
        text = text.replace(f"{{{{{key}}}}}", str(val))
    return text


def extract_json_array(text: str):
    """Return the first JSON array found in ``text``.

    The previous implementation relied on a regular expression that failed on
    nested structures like ``[{"a": [1, 2]}]``.  This version iteratively scans
    for ``[`` characters and attempts to decode a JSON array using
    ``json.JSONDecoder``.  It gracefully returns ``None`` if no valid array is
    found.
    """
    if len(text) > MAX_JSON_SIZE:
        return None

    decoder = json.JSONDecoder()
    index = 0
    while True:
        start = text.find("[", index)
        if start == -1:
            return None
        try:
            obj, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            # Found an opening bracket that does not start a JSON array; skip it
            index = start + 1
            continue
        if isinstance(obj, list):
            return obj
        # If it was valid JSON but not a list, search for the next '['
        index = start + 1


def extract_json_object(text: str):
    """Return the first JSON object or array found in ``text``.

    This extends :func:`extract_json_array` by handling either ``{`` or
    ``[`` as the start of the JSON value. The function scans for the next
    opening brace/bracket and attempts to decode JSON using
    ``json.JSONDecoder``. ``None`` is returned if no valid JSON object is
    found.
    """
    if len(text) > MAX_JSON_SIZE:
        return None

    decoder = json.JSONDecoder()
    index = 0
    while True:
        brace = text.find("{", index)
        bracket = text.find("[", index)
        candidates = [c for c in (brace, bracket) if c != -1]
        if not candidates:
            return None
        start = min(candidates)
        try:
            obj, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        return obj
