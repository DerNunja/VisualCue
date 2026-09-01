"""OpenAI-compatible VLM client for local LM Studio-style endpoints."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

from PIL import Image

DEFAULT_MAX_TOKENS = 4_096
DEFAULT_PARSE_ERROR_RETRIES = 2
NO_HIDDEN_REASONING_INSTRUCTION = (
    "Do not emit hidden reasoning, chain-of-thought, or special channel tokens such as "
    "<|channel>thought. Return only the requested JSON object."
)


class VLMTokenLimitExceeded(RuntimeError):
    """Raised when a VLM response is truncated by the configured token limit."""

    def __init__(self, message: str, usage: dict[str, int] | None = None) -> None:
        super().__init__(message)
        self.usage = usage


class VLMRequestError(RuntimeError):
    """Raised when the OpenAI-compatible VLM backend rejects or fails a request."""

    def __init__(self, message: str, usage: dict[str, int] | None = None) -> None:
        super().__init__(message)
        self.usage = usage


class VLMClient:
    """Small deterministic chat-completion wrapper."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        parse_error_retries: int = DEFAULT_PARSE_ERROR_RETRIES,
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.parse_error_retries = max(0, int(parse_error_retries))
        self.last_usage: dict[str, int] | None = None

    def complete(self, system: str, user_text: str, image: Image.Image | None = None) -> str:
        """Single deterministic chat completion (temperature=0); returns raw text."""

        self.last_usage = None
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
        response = self._create_with_retries(messages)
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

    def _create_with_retries(self, messages: list[dict[str, Any]]) -> Any:
        last_exc: Exception | None = None
        for attempt in range(self.parse_error_retries + 1):
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    max_tokens=self.max_tokens,
                    messages=_messages_for_attempt(messages, attempt),
                )
            except Exception as exc:
                last_exc = exc
                if not _is_thought_channel_parse_error(exc) or attempt >= self.parse_error_retries:
                    raise VLMRequestError(_short_error_message(exc), usage=self.last_usage) from exc
        if last_exc is None:
            raise VLMRequestError("VLM request failed", usage=self.last_usage)
        raise VLMRequestError(_short_error_message(last_exc), usage=self.last_usage) from last_exc


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


def _messages_for_attempt(messages: list[dict[str, Any]], attempt: int) -> list[dict[str, Any]]:
    if attempt == 0:
        return messages
    copied = [dict(message) for message in messages]
    system = str(copied[0].get("content", ""))
    copied[0]["content"] = f"{system}\n\n{NO_HIDDEN_REASONING_INSTRUCTION}"
    return copied


def _is_thought_channel_parse_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "failed to parse input" in text and "<|channel>thought" in text


def _short_error_message(exc: Exception) -> str:
    text = str(exc).replace("\n", " ")
    if len(text) > 500:
        return text[:497] + "..."
    return text
