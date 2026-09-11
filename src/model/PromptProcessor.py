from typing import Any

from pydantic import BaseModel, ConfigDict, PrivateAttr

from .LLMProtocol import LLMProtocol


class PromptProcessor(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompts: list[str]
    function_definitions: list[dict[str, Any]]

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
            'function="'
        )
        generated_name_tokens: list[int] = []
        possible_output_tokens = [
            self._llm.encode(fd["name"]).tolist()[0]
            for fd in self.function_definitions
        ]
        max_name_token_length = max(len(tokens) for tokens in possible_output_tokens)

        while len(generated_name_tokens) < max_name_token_length:
            prompt_tokenized = self._llm.encode(prompt_message)
            logits = self._llm.get_logits_from_input_ids(prompt_tokenized.tolist()[0])
            next_token_id = get_next_token_id(
                logits,
                generated_name_tokens,
                possible_output_tokens,
            )
            generated_name_tokens.append(next_token_id)
            prompt_message += self._llm.decode([next_token_id])

            for fd, tokens in zip(self.function_definitions, possible_output_tokens):
                if tokens == generated_name_tokens:
                    return fd["name"]

        return self._llm.decode(generated_name_tokens).strip('"')

    def get_parameters(self, function_name: str, prompt: str) -> dict:
        params = {}
        for fd in self.function_definitions:
            if fd["name"] == function_name:
                params = fd.get("parameters", {})
                break

        params_format: str = " ".join(
            f'{key}="<value>"' for key in params.keys()
        )
        rule_context = (
            f'The function you should call is "{function_name}".\n'
            "Read the parameter description below and provide only the arguments needed for this function.\n"
            "Use the exact parameter names from the description. Do not include angle brackets, labels, "
            "explanations, or extra text.\n"
            "Parameter description:\n"
            f"{params_format}"
        )
        token_context: list[int] = self._llm.encode(rule_context).tolist()[0]

        token_prompt: list[int] = self._llm.encode(
            f'Input= {prompt} \nFunction= "{function_name}"\n'
        ).tolist()[0]
        result: dict[str, str] = {}

        for key in params.keys():
            param_type = params[key].get("type")
            param_mask = self.get_param_mask(param_type)
            param_buffer = ""
            token_prompt += self._llm.encode(f'{key}="').tolist()[0]

            while '"' not in param_buffer and len(param_buffer) < 64:
                logits = self._llm.get_logits_from_input_ids(token_context + token_prompt)
                next_token_id = get_constrained_token_id(logits, param_mask, self._llm)
                param_buffer += self._llm.decode([next_token_id])
                token_prompt += [next_token_id]

            result[key] = param_buffer.strip('"')
            token_prompt += self._llm.encode("\n").tolist()[0]

        return result

    def get_param_mask(self, param_type: str) -> str:
        if param_type == "string":
            return 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_ "\''
        if param_type in {"integer", "number"}:
            return '0123456789.-"'
        return '"'


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


def get_constrained_token_id(logits: list[float], mask: str, llm: LLMProtocol) -> int:
    allowed_chars: set[str] = set(mask)
    sorted_ids = sorted(range(len(logits)), key=lambda token_id: logits[token_id], reverse=True)
    for token_id in sorted_ids:
        if llm.decode([token_id]) in allowed_chars:
            return token_id
    return sorted_ids[0]
