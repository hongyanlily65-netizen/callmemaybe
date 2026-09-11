import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError


def load_json_list(path: str | Path, label: str) -> list[Any]:
    file_path = Path(path)

    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Cannot read {label} file '{file_path}': {exc}") from exc

    if not content.strip():
        raise ValueError(f"The {label} file '{file_path}' is empty.")

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"The {label} file '{file_path}' is not valid JSON: {exc.msg} "
            f"at line {exc.lineno}, column {exc.colno}."
        ) from exc

    if not isinstance(data, list):
        raise ValueError(f"The {label} file '{file_path}' must contain a JSON list.")

    if not data:
        raise ValueError(f"The {label} file '{file_path}' must contain at least one item.")

    return data


def format_validation_error(exc: ValidationError) -> str:
    messages = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        message = error["msg"]
        if location:
            messages.append(f"{location}: {message}")
        else:
            messages.append(message)
    return "; ".join(messages)
