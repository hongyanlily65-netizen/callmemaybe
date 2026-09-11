from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMProtocol(Protocol):
    def encode(self, text: str) -> Any:
        ...

    def decode(self, ids: Any | list[int]) -> str:
        ...

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        ...
