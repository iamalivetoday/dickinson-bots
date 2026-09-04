"""Backend-neutral dispatch: `generate(actor_id, participant_id, transcript)`.

The router is the single thing the room orchestrator talks to. Adding a new
backend (or a new actor on an existing backend) never touches this file —
see discord_bot/backends/anthropic_backend.py landing after local.py without
a router change.
"""
from __future__ import annotations

from typing import Sequence

from discord_bot.registry import ActorRegistry

from .base import Backend, GenerationError, TranscriptMessage


class BackendRouter:
    def __init__(self, registry: ActorRegistry, backends: dict[str, Backend]):
        self._registry = registry
        self._backends = backends

    async def generate(
        self, actor_id: str, participant_id: str, transcript: Sequence[TranscriptMessage]
    ) -> str:
        actor = self._registry.get(actor_id)
        try:
            backend = self._backends[actor.backend]
        except KeyError:
            raise GenerationError(
                f"no backend registered for {actor.backend!r} (actor {actor_id!r})"
            ) from None
        return await backend.generate(actor, participant_id, transcript)
