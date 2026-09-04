"""Anthropic backend.

Wraps an `anthropic.AsyncAnthropic` client, injected rather than constructed
here — so tests (and this module) never touch the network or read
environment variables. See discord_bot/config.py for how the real client is
built at bot startup.
"""
from __future__ import annotations

from typing import Sequence

from .base import GenerationError, TranscriptMessage, system_prompt_for, to_role_messages


class AnthropicBackend:
    def __init__(self, client):
        self._client = client

    async def generate(
        self, actor, participant_id: str, transcript: Sequence[TranscriptMessage]
    ) -> str:
        if not actor.model:
            raise GenerationError(f"actor {actor.id!r} has no anthropic model configured")

        gen = actor.generation
        response = await self._client.messages.create(
            model=actor.model,
            system=system_prompt_for(actor),
            messages=to_role_messages(participant_id, transcript),
            max_tokens=gen.get("max_tokens", 512),
            temperature=gen.get("temperature", 1.0),
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()
        if not text:
            raise GenerationError(f"anthropic returned no text content for actor {actor.id!r}")
        return text
