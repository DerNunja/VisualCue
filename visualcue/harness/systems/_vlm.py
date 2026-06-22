"""OpenAI-compatible VLM client for local LM Studio-style endpoints."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

from PIL import Image


class VLMClient:
    """Small deterministic chat-completion wrapper."""

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

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
            messages=messages,
        )
        content = response.choices[0].message.content
        if isinstance(content, str):
            return content
        return "" if content is None else str(content)


def _image_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
