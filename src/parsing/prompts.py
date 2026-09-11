from pathlib import Path

from pydantic import BaseModel, ValidationError, field_validator

from .json_files import format_validation_error, load_json_list


class Prompt(BaseModel):
    prompt: str

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt cannot be empty")
        return value


def get_prompts(path: str | Path) -> list[str]:
    data = load_json_list(path, "input prompts")
    prompts = []
    for index, item in enumerate(data, start=1):
        try:
            prompts.append(Prompt.model_validate(item))
        except ValidationError as exc:
            details = format_validation_error(exc)
            raise ValueError(f"Invalid prompt item #{index}: {details}") from exc
    return [prompt.prompt for prompt in prompts]
