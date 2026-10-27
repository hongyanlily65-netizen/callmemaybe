import unittest

from src.model.PromptProcessor import (
    PromptProcessor,
    parameter_value_matches_type,
)
from src.parsing.function_definitions import FunctionDefinition


class TokenIds:
    def __init__(self, ids: list[int]) -> None:
        self._ids = ids

    def tolist(self) -> list[list[int]]:
        return [self._ids]


class FakeLlm:
    """A character-level LLM that returns a predefined completion."""

    def __init__(self, output: str) -> None:
        self.output = output
        self.characters = sorted(set(output))
        self.char_to_id = {
            character: index for index, character in enumerate(self.characters)
        }
        self.id_to_char = {
            index: character for character, index in self.char_to_id.items()
        }
        self.output_index = 0

    def encode(self, text: str) -> TokenIds:
        ids = [
            self.char_to_id.get(character, 0)
            for character in text
        ]
        return TokenIds(ids)

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_char[token_id] for token_id in ids)

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        logits = [0.0] * len(self.characters)
        if self.output_index < len(self.output):
            character = self.output[self.output_index]
            logits[self.char_to_id[character]] = 10.0
        self.output_index += 1
        return logits


def processor_with(
    function: FunctionDefinition,
    output: str,
) -> PromptProcessor:
    processor = object.__new__(PromptProcessor)
    object.__setattr__(
        processor,
        "__dict__",
        {"prompts": [], "function_definitions": [function]},
    )
    object.__setattr__(
        processor,
        "__pydantic_private__",
        {"_llm": FakeLlm(output)},
    )
    return processor


class PromptProcessorTests(unittest.TestCase):
    def test_get_parameters_generates_one_complete_json_object(self) -> None:
        function = FunctionDefinition(
            name="fn_substitute_string_with_regex",
            description="Replace all matching text in a string.",
            parameters={
                "source_string": {"type": "string"},
                "regex": {"type": "string"},
                "replacement": {"type": "string"},
            },
            returns={"type": "string"},
            full_definition="",
        )
        output = 'Hello 34"[0-9]+"NUMBERS"'

        result = processor_with(function, output).get_parameters(
            function.name,
            'Replace all numbers in "Hello 34" with NUMBERS',
        )

        self.assertEqual(
            result,
            {
                "source_string": "Hello 34",
                "regex": "[0-9]+",
                "replacement": "NUMBERS",
            },
        )

    def test_string_values_must_not_be_empty(self) -> None:
        self.assertFalse(parameter_value_matches_type("", "string"))


if __name__ == "__main__":
    unittest.main()
