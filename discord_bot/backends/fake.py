"""Deterministic fake backend — never touches a network or a local model.

Used by the unit test suite and by the live Discord acceptance test (per
its own config/actors and never against `the flesh door`'s real actors),
so behaviour is fully reproducible: the same transcript always produces the
same reply.
"""
from __future__ import annotations

from typing import Sequence

from .base import TranscriptMessage


class FakeBackend:
    def __init__(self, template: str = "[{actor}] turn {n}: re: {last}"):
        self._template = template

    async def generate(
        self, actor, participant_id: str, transcript: Sequence[TranscriptMessage]
    ) -> str:
        last = transcript[-1].content if transcript else "(nothing yet)"
        n = sum(1 for m in transcript if m.speaker_id == participant_id) + 1
        return self._template.format(actor=participant_id, n=n, last=last[:60])
