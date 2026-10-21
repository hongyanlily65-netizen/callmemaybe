import json
from pathlib import Path

from pydantic import BaseModel, ValidationError, field_validator


class PromptsError(ValueError):
    """Raised when the prompts file cannot be used."""


class Prompt(BaseModel):
    prompt: str

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt cannot be empty")
        return value


def get_prompts(path: str | Path) -> list[str]:
    """Load and validate prompts from a JSON file."""
    try:
        with Path(path).open("r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as exc:
        raise PromptsError(f"Prompts file not found: {path}.") from exc
    except PermissionError as exc:
        raise PromptsError(
            f"Not enough permissions to open prompts file: {path}."
        ) from exc
    except json.JSONDecodeError as exc:
        raise PromptsError(f"Prompts file contains invalid JSON: {path}.") from exc
    except OSError as exc:
        raise PromptsError(f"Unable to open prompts file: {path}.") from exc

    if not isinstance(data, list) or not data:
        raise PromptsError("Prompts file must contain a non-empty JSON list.")

    try:
        return [Prompt(prompt=item["prompt"]).prompt for item in data]
    except (KeyError, TypeError, ValidationError) as exc:
        raise PromptsError(
            "Prompts file does not match the expected JSON structure."
        ) from exc
