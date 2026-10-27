import unittest

from src.model.PromptProcessor import ParameterJsonState, PromptProcessor
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
        self.prompt_length: int | None = None

    def encode(self, text: str) -> TokenIds:
        ids = [
            self.char_to_id.get(character, 0)
            for character in text
        ]
        self.prompt_length = len(ids)
        return TokenIds(ids)

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_char[token_id] for token_id in ids)

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        assert self.prompt_length is not None
        output_index = len(input_ids) - self.prompt_length
        logits = [0.0] * len(self.characters)
        if output_index < len(self.output):
            logits[self.char_to_id[self.output[output_index]]] = 10.0
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
        output = (
            '{"source_string":"Hello 34","regex":"[0-9]+",'
            '"replacement":"NUMBERS"}'
        )

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

    def test_state_rejects_an_empty_string_value(self) -> None:
        state = ParameterJsonState(["name"], ["string"])

        self.assertIsNone(state.try_append('{"name":""}'))


if __name__ == "__main__":
    unittest.main()
