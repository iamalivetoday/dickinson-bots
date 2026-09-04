from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from discord_bot.backends.anthropic_backend import AnthropicBackend
from discord_bot.backends.base import GenerationError, TranscriptMessage
from discord_bot.registry import Actor


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def make_client(content):
    client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(
        return_value=SimpleNamespace(content=content)
    )))
    return client


@pytest.mark.asyncio
async def test_generate_returns_concatenated_text_blocks():
    client = make_client([text_block("Hello, "), text_block("world.")])
    backend = AnthropicBackend(client)
    actor = Actor(id="opus-4.8", backend="anthropic", model="claude-opus-4-8")

    reply = await backend.generate(actor, "opus-4.8", [])
    assert reply == "Hello, world."


@pytest.mark.asyncio
async def test_generate_ignores_non_text_blocks():
    tool_block = SimpleNamespace(type="tool_use", text=None)
    client = make_client([tool_block, text_block("the actual reply")])
    backend = AnthropicBackend(client)
    actor = Actor(id="opus-4.8", backend="anthropic", model="claude-opus-4-8")

    reply = await backend.generate(actor, "opus-4.8", [])
    assert reply == "the actual reply"


@pytest.mark.asyncio
async def test_generate_passes_model_persona_and_generation_defaults():
    client = make_client([text_block("reply")])
    backend = AnthropicBackend(client)
    actor = Actor(
        id="weil-opus-4.8", backend="anthropic", persona="weil", model="claude-opus-4-8",
        generation={"max_tokens": 300, "temperature": 0.6},
    )
    transcript = [TranscriptMessage("user:1", "Madeleine", "What is grace?")]

    await backend.generate(actor, "weil-opus-4.8", transcript)

    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-opus-4-8"
    assert "Simone Weil" in kwargs["system"]
    assert kwargs["max_tokens"] == 300
    assert kwargs["temperature"] == 0.6
    assert kwargs["messages"] == [{"role": "user", "content": "Madeleine: What is grace?"}]


@pytest.mark.asyncio
async def test_generate_applies_generation_defaults_when_actor_has_none():
    client = make_client([text_block("reply")])
    backend = AnthropicBackend(client)
    actor = Actor(id="opus-4.8", backend="anthropic", model="claude-opus-4-8")

    await backend.generate(actor, "opus-4.8", [])
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["max_tokens"] == 512
    assert kwargs["temperature"] == 1.0


@pytest.mark.asyncio
async def test_missing_model_raises_generation_error():
    backend = AnthropicBackend(make_client([]))
    actor = Actor(id="broken", backend="anthropic", model=None)
    with pytest.raises(GenerationError, match="no anthropic model configured"):
        await backend.generate(actor, "broken", [])


@pytest.mark.asyncio
async def test_empty_text_response_raises_generation_error():
    client = make_client([SimpleNamespace(type="tool_use", text=None)])
    backend = AnthropicBackend(client)
    actor = Actor(id="opus-4.8", backend="anthropic", model="claude-opus-4-8")
    with pytest.raises(GenerationError, match="no text content"):
        await backend.generate(actor, "opus-4.8", [])
