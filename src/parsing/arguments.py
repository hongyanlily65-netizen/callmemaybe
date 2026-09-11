import argparse

from pydantic import BaseModel, ValidationError, field_validator

from .json_files import format_validation_error


DEFAULT_INPUT_PATH = "input/function_calling_tests.json"
DEFAULT_OUTPUT_PATH = "output/function_calling_results.json"
DEFAULT_FUNCTION_DEFINITIONS_PATH = "input/functions_definition.json"


class Arguments(BaseModel):
    input: str
    output: str
    function_definitions: str

    @field_validator("input", "output", "function_definitions")
    @classmethod
    def validate_path(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("path cannot be empty")
        return value


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help=(
            "Path to the input prompts JSON file. "
            f"Defaults to {DEFAULT_INPUT_PATH}."
        ),
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Path where the generated output JSON should be written. "
            f"Defaults to {DEFAULT_OUTPUT_PATH}."
        ),
    )
    parser.add_argument(
        "--function_definitions",
        default=DEFAULT_FUNCTION_DEFINITIONS_PATH,
        help=(
            "Path to the function definitions JSON file. "
            f"Defaults to {DEFAULT_FUNCTION_DEFINITIONS_PATH}."
        ),
    )
    return parser


def get_arguments() -> Arguments:
    parser = argparse.ArgumentParser(
        description="Run function-calling prompt processing with a local LLM."
    )
    add_arguments(parser)
    args = parser.parse_args()
    try:
        return Arguments(
            input=args.input,
            output=args.output,
            function_definitions=args.function_definitions,
        )
    except ValidationError as exc:
        parser.error(format_validation_error(exc))
