import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

from .model.PromptProcessor import PromptProcessor

FUNCTION_CALLING_TESTS_PATH = PROJECT_DIR / "input" / "function_calling_tests.json"
FUNCTION_DEFINITIONS_PATH = PROJECT_DIR / "input" / "functions_definition.json"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    function_calling_tests = load_json(FUNCTION_CALLING_TESTS_PATH)
    function_definitions = load_json(FUNCTION_DEFINITIONS_PATH)

    prompts = [item["prompt"] for item in function_calling_tests]
    processor = PromptProcessor(prompts, function_definitions)
    output = processor.process(function_calling_tests)

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
