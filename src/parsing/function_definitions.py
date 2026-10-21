import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError, model_validator



class FunctionsDefinitionError(ValueError):
    """Raised when the function definitions file cannot be used."""



class FunctionDefinition(BaseModel):
    """A function definition loaded from the input JSON file."""

    name: str
    description: str
    parameters: dict[str, dict[str, Any]]
    returns: dict[str, Any]
    full_definition: str

    @model_validator(mode="after")
    def validate_parameters(self) -> "FunctionDefinition":
        if not self.name.strip() or not self.description.strip():
            raise FunctionsDefinitionError(
                "Function name and description cannot be empty."
            )

        for parameter_name, parameter in self.parameters.items():
            if not parameter_name.strip() or not parameter.get("type"):
                raise FunctionsDefinitionError(
                    "Unsupported argument type "
                    f"for parameter {parameter_name} in function {self.name}."
                )
        return self


def get_function_definitions(path: str | Path) -> list[FunctionDefinition]:
    """Load and validate all function definitions from a JSON file."""
    try:
        with Path(path).open("r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as exc:
        raise FunctionsDefinitionError(
            f"Function definitions file not found: {path}."
        ) from exc
    except PermissionError as exc:
        raise FunctionsDefinitionError(
            f"Not enough permissions to open function definitions file: {path}."
        ) from exc
    except json.JSONDecodeError as exc:
        raise FunctionsDefinitionError(
            f"Function definitions file contains invalid JSON: {path}."
        ) from exc
    except OSError as exc:
        raise FunctionsDefinitionError(
            f"Unable to open function definitions file: {path}."
        ) from exc

    if not isinstance(data, list) or not data:
        raise FunctionsDefinitionError(
            "Function definitions file must contain a non-empty JSON list."
        )

    try:
        return [
            FunctionDefinition(
                name=function["name"],
                description=function["description"],
                parameters=function["parameters"],
                returns=function["returns"],
                full_definition=str(function),
            )
            for function in data
        ]
    except (KeyError, TypeError, ValidationError, FunctionsDefinitionError) as exc:
        raise FunctionsDefinitionError(
            "Function definitions file does not match the expected JSON structure."
        ) from exc
