import json
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, PrivateAttr

from llm_sdk import Small_LLM_Model


class LLMModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = "Qwen/Qwen3-0.6B"
    device: str | None = None
    dtype: torch.dtype | None = None
    trust_remote_code: bool = True

    _model: Small_LLM_Model = PrivateAttr()
    _token_text_by_id: dict[int, str] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        self._model = Small_LLM_Model(
            model_name=self.model_name,
            device=self.device,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
        )
        vocabulary_path = self._model.get_path_to_vocab_file()
        self._token_text_by_id = load_token_text_by_id(vocabulary_path)

    def encode(self, text: str) -> torch.Tensor:
        return self._model.encode(text)

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        return self._model.decode(ids)

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        return self._model.get_logits_from_input_ids(input_ids)

    def get_token_text(self, token_id: int) -> str:
        try:
            return self._token_text_by_id[token_id]
        except KeyError as exc:
            raise ValueError(
                f"Token ID {token_id} is missing from the vocabulary."
            ) from exc


def load_token_text_by_id(vocabulary_path: str | Path) -> dict[int, str]:
    """Load a token-to-ID vocabulary JSON and reverse its mapping."""
    path = Path(vocabulary_path)
    try:
        with path.open("r", encoding="utf-8") as file:
            vocabulary = json.load(file)
    except FileNotFoundError as exc:
        raise ValueError(f"Vocabulary file not found: {path}.") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Vocabulary file contains invalid JSON: {path}."
        ) from exc
    except OSError as exc:
        raise ValueError(f"Unable to read vocabulary file: {path}.") from exc

    if not isinstance(vocabulary, dict) or not vocabulary:
        raise ValueError(
            "Vocabulary file must contain a non-empty token-to-ID object."
        )

    token_text_by_id: dict[int, str] = {}
    for token_text, token_id in vocabulary.items():
        if (
            not isinstance(token_text, str)
            or not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or token_id < 0
        ):
            raise ValueError(
                "Vocabulary entries must map strings to non-negative "
                "integer token IDs."
            )
        if token_id in token_text_by_id:
            raise ValueError(
                f"Vocabulary contains duplicate token ID {token_id}."
            )
        token_text_by_id[token_id] = token_text

    return token_text_by_id
