from typing import Any, cast

import torch
from pydantic import BaseModel, ConfigDict, PrivateAttr

from llm_sdk import Small_LLM_Model

from .LLMProtocol import LLMProtocol


class LLMModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = "Qwen/Qwen3-0.6B"
    device: str | None = None
    dtype: torch.dtype | None = None
    trust_remote_code: bool = True

    _model: LLMProtocol = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._model = cast(
            LLMProtocol,
            Small_LLM_Model(
                model_name=self.model_name,
                device=self.device,
                dtype=self.dtype,
                trust_remote_code=self.trust_remote_code,
            ),
        )

    def encode(self, text: str) -> torch.Tensor:
        return self._model.encode(text)

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        return self._model.decode(ids)

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        return self._model.get_logits_from_input_ids(input_ids)
