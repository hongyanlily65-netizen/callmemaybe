from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .json_files import format_validation_error, load_json_list


class FunctionParameter(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("parameter type cannot be empty")
        return value


class FunctionReturn(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("return type cannot be empty")
        return value


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    description: str
    parameters: dict[str, FunctionParameter] = Field(default_factory=dict)
    returns: FunctionReturn | None = None

    @field_validator("name", "description")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value cannot be empty")
        return value

    @field_validator("parameters")
    @classmethod
    def validate_parameters(
        cls,
        value: dict[str, FunctionParameter],
    ) -> dict[str, FunctionParameter]:
        for parameter_name in value:
            if not parameter_name.strip():
                raise ValueError("parameter name cannot be empty")
        return value


def get_function_definitions(path: str | Path) -> list[dict[str, Any]]:
    data = load_json_list(path, "function definitions")
    function_definitions = []
    for index, item in enumerate(data, start=1):
        try:
            function_definitions.append(FunctionDefinition.model_validate(item))
        except ValidationError as exc:
            details = format_validation_error(exc)
            raise ValueError(
                f"Invalid function definition item #{index}: {details}"
            ) from exc
    return [
        function_definition.model_dump()
        for function_definition in function_definitions
    ]
