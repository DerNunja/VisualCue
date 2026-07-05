"""OpenAI-compatible VLM client for local LM Studio-style endpoints."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

from PIL import Image

DEFAULT_MAX_TOKENS = 16_000


class VLMTokenLimitExceeded(RuntimeError):
    """Raised when a VLM response is truncated by the configured token limit."""

    def __init__(self, message: str, usage: dict[str, int] | None = None) -> None:
        super().__init__(message)
        self.usage = usage


class VLMClient:
    """Small deterministic chat-completion wrapper."""

    def __init__(self, base_url: str, model: str, api_key: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.last_usage: dict[str, int] | None = None

    def complete(self, system: str, user_text: str, image: Image.Image | None = None) -> str:
        """Single deterministic chat completion (temperature=0); returns raw text."""

        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        if image is None:
            messages.append({"role": "user", "content": user_text})
        else:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": _image_data_url(image)}},
                    ],
                }
            )
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=self.max_tokens,
            messages=messages,
        )
        self.last_usage = _usage_to_dict(getattr(response, "usage", None))
        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "length":
            raise VLMTokenLimitExceeded(
                f"VLM response exceeded max_tokens={self.max_tokens}",
                usage=self.last_usage,
            )
        content = choice.message.content
        if isinstance(content, str):
            return content
        return "" if content is None else str(content)


def _image_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _usage_to_dict(usage: Any) -> dict[str, int] | None:
    if usage is None:
        return None
    result: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, key, None)
        if value is not None:
            result[key] = int(value)
    return result or None
