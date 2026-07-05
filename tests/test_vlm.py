from __future__ import annotations

import pytest

from visualcue.harness.systems._vlm import DEFAULT_MAX_TOKENS, VLMClient, VLMTokenLimitExceeded


class _FakeUsage:
    prompt_tokens = 10
    completion_tokens = 20
    total_tokens = 30


class _FakeMessage:
    content = '{"ok": true}'


class _FakeChoice:
    def __init__(self, finish_reason: str) -> None:
        self.finish_reason = finish_reason
        self.message = _FakeMessage()


class _FakeResponse:
    def __init__(self, finish_reason: str) -> None:
        self.choices = [_FakeChoice(finish_reason)]
        self.usage = _FakeUsage()


class _FakeCompletions:
    def __init__(self, finish_reason: str) -> None:
        self.finish_reason = finish_reason
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return _FakeResponse(self.finish_reason)


class _FakeChat:
    def __init__(self, finish_reason: str) -> None:
        self.completions = _FakeCompletions(finish_reason)


class _FakeClient:
    def __init__(self, finish_reason: str) -> None:
        self.chat = _FakeChat(finish_reason)


def test_vlm_client_sets_max_tokens_and_records_usage() -> None:
    assert DEFAULT_MAX_TOKENS == 4096

    client = object.__new__(VLMClient)
    client.client = _FakeClient("stop")
    client.model = "fake"
    client.max_tokens = 16000
    client.last_usage = None

    assert client.complete("system", "user") == '{"ok": true}'
    assert client.client.chat.completions.kwargs["max_tokens"] == 16000
    assert client.last_usage == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}


def test_vlm_client_raises_when_response_hits_token_limit() -> None:
    client = object.__new__(VLMClient)
    client.client = _FakeClient("length")
    client.model = "fake"
    client.max_tokens = 16000
    client.last_usage = None

    with pytest.raises(VLMTokenLimitExceeded) as exc_info:
        client.complete("system", "user")

    assert exc_info.value.usage == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
