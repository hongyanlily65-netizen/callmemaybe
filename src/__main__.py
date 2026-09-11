import json
from pathlib import Path
import sys

from .parsing.arguments import get_arguments


def main() -> None:
    args = get_arguments()

    from .model.PromptProcessor import PromptProcessor
    from .parsing.function_definitions import get_function_definitions
    from .parsing.prompts import get_prompts

    input_path = Path(args.input)
    definitions_path = Path(args.function_definitions)
    output_path = Path(args.output)

    try:
        prompts = get_prompts(input_path)
        function_definitions = get_function_definitions(definitions_path)
        processor = PromptProcessor(
            prompts=prompts,
            function_definitions=function_definitions,
        )
        output = processor.process()

        output_json = json.dumps(output, indent=2, ensure_ascii=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_json + "\n", encoding="utf-8")
        print(output_json)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
