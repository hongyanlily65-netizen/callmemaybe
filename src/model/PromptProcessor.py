import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, PrivateAttr

from .LLMProtocol import LLMProtocol
from ..parsing.function_definitions import FunctionDefinition


MAX_PARAMETER_VALUE_LENGTH = 64
MAX_PARAMETER_VALUE_TOKENS = 64


class ParameterGenerationError(ValueError):
    """Raised when parameter JSON cannot be generated within the schema."""


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
        prompt_message = self._build_parameter_prompt(function, prompt)
        prompt_tokens: list[int] = self._llm.encode(prompt_message).tolist()[0]
        result: dict[str, str] = {}

        for index, (parameter_name, definition) in enumerate(
            function.parameters.items()
        ):
            prefix = "{" if index == 0 else ","
            prompt_tokens += self._llm.encode(
                f'{prefix}{json.dumps(parameter_name)}:"'
            ).tolist()[0]
            result[parameter_name] = self._generate_parameter_value(
                prompt_tokens,
                definition["type"],
            )
            prompt_tokens += self._llm.encode('"').tolist()[0]

        prompt_tokens += self._llm.encode("}").tolist()[0]
        return result

    def _generate_parameter_value(
        self,
        prompt_tokens: list[int],
        parameter_type: str,
    ) -> str:
        value = ""

        for _ in range(MAX_PARAMETER_VALUE_TOKENS):
            logits = self._llm.get_logits_from_input_ids(prompt_tokens)
            next_token_id, next_value = get_constrained_parameter_value_token(
                logits,
                value,
                parameter_type,
                self._llm,
            )
            if next_token_id is None:
                return value
            prompt_tokens.append(next_token_id)
            value = next_value

        raise ParameterGenerationError(
            "Parameter value generation exceeded the token limit."
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


def get_constrained_parameter_value_token(
    logits: list[float],
    value: str,
    parameter_type: str,
    llm: LLMProtocol,
) -> tuple[int | None, str]:
    """Choose a value token or use a quote token to finish a valid value."""
    sorted_ids = sorted(
        range(len(logits)),
        key=lambda token_id: logits[token_id],
        reverse=True,
    )
    for token_id in sorted_ids:
        token_text = llm.decode([token_id])
        if token_text == '"' and parameter_value_matches_type(
            value,
            parameter_type,
        ):
            return None, value
        if '"' in token_text:
            continue

        candidate = value + token_text
        if parameter_value_can_continue(candidate, parameter_type):
            return token_id, candidate
    raise ParameterGenerationError(
        "No token can continue the parameter value."
    )


def parameter_value_can_continue(value: str, parameter_type: str) -> bool:
    if len(value) > MAX_PARAMETER_VALUE_LENGTH:
        return False
    if parameter_type == "string":
        return True
    if parameter_type == "integer":
        return bool(re.fullmatch(r"-?[0-9]*", value))
    if parameter_type == "number":
        return bool(re.fullmatch(r"[-+0-9.eE]*", value))
    return False


def parameter_value_matches_type(value: str, parameter_type: str) -> bool:
    if parameter_type == "string":
        return bool(value)
    if parameter_type == "integer":
        return bool(re.fullmatch(r"-?(0|[1-9][0-9]*)", value))
    if parameter_type == "number":
        return bool(
            re.fullmatch(
                r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?",
                value,
            )
        )
    return False
