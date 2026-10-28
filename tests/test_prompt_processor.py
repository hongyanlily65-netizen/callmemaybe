import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.model.LLMModel import load_token_text_by_id
from src.model.PromptProcessor import (
    PromptProcessor,
    convert_parameter_value,
    get_constrained_parameter_value_token,
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

    def get_token_text(self, token_id: int) -> str:
        return self.id_to_char[token_id]

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        logits = [0.0] * len(self.characters)
        if self.output_index < len(self.output):
            character = self.output[self.output_index]
            logits[self.char_to_id[character]] = 10.0
        self.output_index += 1
        return logits


class VocabularyOnlyLlm(FakeLlm):
    def get_token_text(self, token_id: int) -> str:
        if token_id == 0:
            raise ValueError("Special token is not in the vocabulary.")
        return {1: "2", 2: '"'}[token_id]


def processor_with(
    function: FunctionDefinition,
    output: str,
    prompts: list[str] | None = None,
) -> PromptProcessor:
    processor = object.__new__(PromptProcessor)
    object.__setattr__(
        processor,
        "__dict__",
        {
            "prompts": prompts or [],
            "function_definitions": [function],
        },
    )
    object.__setattr__(
        processor,
        "__pydantic_private__",
        {"_llm": FakeLlm(output)},
    )
    return processor


class PromptProcessorTests(unittest.TestCase):
    def test_tokens_missing_from_vocabulary_are_skipped(self) -> None:
        token_id, value = get_constrained_parameter_value_token(
            logits=[10.0, 9.0, 8.0],
            value="",
            parameter_type="number",
            llm=VocabularyOnlyLlm(""),
        )

        self.assertEqual(token_id, 1)
        self.assertEqual(value, "2")

    def test_vocabulary_is_reversed_to_token_id_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            vocabulary_path = Path(temporary_directory) / "vocab.json"
            vocabulary_path.write_text(
                json.dumps({"hello": 10, " world": 11}),
                encoding="utf-8",
            )

            result = load_token_text_by_id(vocabulary_path)

        self.assertEqual(result, {10: "hello", 11: " world"})

    def test_process_uses_required_output_keys(self) -> None:
        function = FunctionDefinition(
            name="fn_add_numbers",
            description="Add two numbers.",
            parameters={
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            returns={"type": "number"},
            full_definition="",
        )
        prompt = "What is the sum of 2 and 3?"
        processor = processor_with(function, "", prompts=[prompt])

        with (
            patch.object(
                PromptProcessor,
                "get_function_name",
                return_value=function.name,
            ),
            patch.object(
                PromptProcessor,
                "get_parameters",
                return_value={"a": 2, "b": 3},
            ),
        ):
            result = processor.process()

        self.assertEqual(
            result,
            [
                {
                    "prompt": prompt,
                    "fn_name": "fn_add_numbers",
                    "args": {"a": 2, "b": 3},
                }
            ],
        )

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

    def test_parameter_values_use_declared_json_types(self) -> None:
        self.assertEqual(convert_parameter_value("2", "number"), 2)
        self.assertEqual(convert_parameter_value("2.5", "number"), 2.5)
        self.assertEqual(convert_parameter_value("7", "integer"), 7)
        self.assertIs(convert_parameter_value("true", "boolean"), True)
        self.assertEqual(convert_parameter_value("Ada", "string"), "Ada")


if __name__ == "__main__":
    unittest.main()
