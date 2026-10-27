import copy
import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, PrivateAttr

from .LLMProtocol import LLMProtocol
from ..parsing.function_definitions import FunctionDefinition


MAX_PARAMETER_VALUE_LENGTH = 64
MAX_PARAMETER_JSON_TOKENS = 256


class ParameterGenerationError(ValueError):
    """Raised when parameter JSON cannot be generated within the schema."""


@dataclass
class ParameterJsonState:
    """Checks a compact parameter object with fixed keys and simple types."""

    parameter_names: list[str]
    parameter_types: list[str]
    text: str = ""
    parameter_index: int = 0
    mode: str = "literal"
    literal: str = ""
    literal_index: int = 0
    current_value: str = ""
    complete: bool = False

    def __post_init__(self) -> None:
        if len(self.parameter_names) != len(self.parameter_types):
            raise ValueError(
                "Parameter names and types must have the same length."
            )
        if not self.parameter_names:
            self._set_literal("{}")
        else:
            self._set_literal("{" + self._parameter_prefix(0))

    def try_append(self, token_text: str) -> "ParameterJsonState | None":
        """Return a state only when token_text preserves the expected JSON."""
        if not token_text:
            return None

        candidate = copy.deepcopy(self)
        for char in token_text:
            if not candidate._append_character(char):
                return None
        return candidate

    def _append_character(self, char: str) -> bool:
        if self.complete:
            return False

        if self.mode == "literal":
            if char != self.literal[self.literal_index]:
                return False
            self.text += char
            self.literal_index += 1
            if self.literal_index == len(self.literal):
                if self.literal in {"}", "{}"}:
                    self.complete = True
                else:
                    self.mode = "string"
                    self.literal = ""
                    self.literal_index = 0
            return True

        if char == '"':
            if not self._value_matches_type():
                return False
            self.text += char
            self.parameter_index += 1
            self.current_value = ""
            self._set_literal_after_value()
            return True

        self.text += char
        self.current_value += char
        return self._value_is_not_too_long()

    def _set_literal(self, literal: str) -> None:
        self.mode = "literal"
        self.literal = literal
        self.literal_index = 0

    def _set_literal_after_value(self) -> None:
        if self.parameter_index == len(self.parameter_names):
            self._set_literal("}")
            return
        self._set_literal("," + self._parameter_prefix(self.parameter_index))

    def _parameter_prefix(self, index: int) -> str:
        return f'{json.dumps(self.parameter_names[index])}:"'

    def _value_is_not_too_long(self) -> bool:
        return len(self.current_value) <= MAX_PARAMETER_VALUE_LENGTH

    def _value_matches_type(self) -> bool:
        value_type = self.parameter_types[self.parameter_index]
        value = self.current_value

        if value_type == "string":
            return bool(value)
        if value_type == "integer":
            return bool(re.fullmatch(r"-?(0|[1-9][0-9]*)", value))
        if value_type == "number":
            return bool(
                re.fullmatch(
                    r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?",
                    value,
                )
            )
        return False


class PromptProcessor(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompts: list[str]
    function_definitions: list[FunctionDefinition]

    _llm: LLMProtocol = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        from .LLMModel import LLMModel

        self._llm = LLMModel()

    def process(self) -> list[dict[str, Any]]:
        output = []
        for prompt in self.prompts:
            prompt_output: dict[str, Any] = {}
            prompt_output["prompt"] = prompt

            fn_name = self.get_function_name(prompt)
            prompt_output["name"] = fn_name

            params = self.get_parameters(fn_name, prompt)
            prompt_output["parameters"] = params
            output.append(prompt_output)
        return output

    def get_function_name(self, prompt: str) -> str:
        prompt_message = (
            f"Functions={self.function_definitions}: "
            "Select function name. "
            f"To solve the following problem: {prompt}. "
            'function='
        )
        generated_name_tokens: list[int] = []
        possible_output_tokens = [
            self._llm.encode(fd.name).tolist()[0]
            for fd in self.function_definitions
        ]
        max_name_token_length = max(
            len(tokens) for tokens in possible_output_tokens
        )

        while len(generated_name_tokens) < max_name_token_length:
            prompt_tokenized = self._llm.encode(prompt_message)
            logits = self._llm.get_logits_from_input_ids(
                prompt_tokenized.tolist()[0]
            )
            next_token_id = get_next_token_id(
                logits,
                generated_name_tokens,
                possible_output_tokens,
            )
            generated_name_tokens.append(next_token_id)
            prompt_message += self._llm.decode([next_token_id])

            matched_function = None
            has_longer_candidate = False
            for fd, tokens in zip(
                self.function_definitions,
                possible_output_tokens,
            ):
                if tokens == generated_name_tokens:
                    matched_function = fd.name
                elif (
                    tokens[:len(generated_name_tokens)]
                    == generated_name_tokens
                ):
                    has_longer_candidate = True

            if matched_function is not None and not has_longer_candidate:
                return matched_function

        generated_name = self._llm.decode(generated_name_tokens).strip('"')
        raise ValueError(
            "Unable to match generated function name to function definitions: "
            f"{generated_name!r}"
        )

    def get_parameters(
        self,
        function_name: str,
        prompt: str,
    ) -> dict[str, str]:
        function = self._get_function_definition(function_name)
        parameter_names = list(function.parameters)
        parameter_types = [
            function.parameters[name]["type"] for name in parameter_names
        ]
        prompt_message = self._build_parameter_prompt(function, prompt)
        prompt_tokens: list[int] = self._llm.encode(prompt_message).tolist()[0]
        state = ParameterJsonState(parameter_names, parameter_types)

        for _ in range(MAX_PARAMETER_JSON_TOKENS):
            logits = self._llm.get_logits_from_input_ids(prompt_tokens)
            next_token_id, state = get_constrained_json_token(
                logits,
                state,
                self._llm,
            )
            prompt_tokens.append(next_token_id)
            if state.complete:
                return self._validate_parameter_json(state.text, function)

        raise ParameterGenerationError(
            "Parameter JSON generation exceeded the token limit."
        )

    def _get_function_definition(
        self,
        function_name: str,
    ) -> FunctionDefinition:
        for function in self.function_definitions:
            if function.name == function_name:
                return function
        raise ValueError(
            f"Unknown function name: {function_name!r}"
        )

    def _build_parameter_prompt(
        self,
        function: FunctionDefinition,
        prompt: str,
    ) -> str:
        schema = {
            "name": function.name,
            "description": function.description,
            "parameters": function.parameters,
        }
        key_order = ", ".join(function.parameters)
        return (
            f"User request: {prompt}\n"
            "Selected function schema: "
            f"{json.dumps(schema, ensure_ascii=False)}\n"
            "Extract the original arguments that must be passed to this "
            "function. "
            "Do not execute the function and do not return its result.\n"
            "Return only one compact JSON object. Use this exact key order: "
            f"{key_order}. All values must be JSON strings; numeric strings "
            "must "
            "match their declared numeric type. JSON="
        )

    def _validate_parameter_json(
        self,
        parameter_json: str,
        function: FunctionDefinition,
    ) -> dict[str, str]:
        try:
            result = json.loads(parameter_json)
        except json.JSONDecodeError as exc:
            raise ParameterGenerationError(
                "Constrained generation produced invalid parameter JSON."
            ) from exc

        expected_keys = list(function.parameters)
        if list(result) != expected_keys:
            raise ParameterGenerationError(
                "Parameter JSON contains unexpected keys."
            )
        if not all(isinstance(value, str) for value in result.values()):
            raise ParameterGenerationError(
                "Parameter JSON values must be strings."
            )
        return result


def get_next_token_id(
    logits: list[float],
    generated_name_tokens: list[int],
    possible_output_tokens: list[list[int]],
) -> int:
    current_token_index = len(generated_name_tokens)
    possible_next_tokens = []

    for tokens in possible_output_tokens:
        if tokens[:current_token_index] != generated_name_tokens:
            continue
        if len(tokens) > current_token_index:
            possible_next_tokens.append(tokens[current_token_index])

    if not possible_next_tokens:
        raise ValueError("No possible next tokens found")

    return max(possible_next_tokens, key=lambda token_id: logits[token_id])


def get_constrained_json_token(
    logits: list[float],
    state: ParameterJsonState,
    llm: LLMProtocol,
) -> tuple[int, ParameterJsonState]:
    """Choose the highest-scoring token whose decoded text preserves JSON."""
    sorted_ids = sorted(
        range(len(logits)),
        key=lambda token_id: logits[token_id],
        reverse=True,
    )
    for token_id in sorted_ids:
        next_state = state.try_append(llm.decode([token_id]))
        if next_state is not None:
            return token_id, next_state
    raise ParameterGenerationError(
        "No token can continue the parameter JSON."
    )
