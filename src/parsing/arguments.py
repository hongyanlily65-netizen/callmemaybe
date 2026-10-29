import argparse

from pydantic import BaseModel


DEFAULT_INPUT_PATH = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT_PATH = "data/output/function_calling_results.json"
DEFAULT_FUNCTION_DEFINITIONS_PATH = "data/input/functions_definition.json"


class Arguments(BaseModel):
    input: str
    output: str
    function_definitions: str


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
        "--functions_definition",
        dest="function_definitions",
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
    return Arguments(
        input=args.input,
        output=args.output,
        function_definitions=args.function_definitions,
    )
