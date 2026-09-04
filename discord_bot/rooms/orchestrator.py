"""The room orchestrator: the only thing that ever asks a backend to
generate a message.

`take_turn()` runs *one* explicit, bounded turn: resolve who speaks next
(the turn policy), build their transcript, generate, persist, done. It
never schedules a further turn on its own — a discord.py webhook or bot
message being posted is never, by itself, a reason to generate again;
callers (slash commands, the on_message router) decide when the next
take_turn() happens. That's what keeps a debate of N rounds from becoming
an infinite loop.
"""
from __future__ import annotations

import asyncio
import random

from discord_bot.backends.base import TranscriptMessage
from discord_bot.backends.router import BackendRouter
from discord_bot.persistence.models import Message, display_names
from discord_bot.persistence.store import RoomStore

from .turn_policies import DebatePolicy, RoundRobinPolicy, Trigger, build_policy


class RoomClosedError(RuntimeError):
    def __init__(self, room_id: str):
        super().__init__(f"room {room_id!r} is closed")
        self.room_id = room_id


class RoomBoundsExceededError(RuntimeError):
    def __init__(self, room_id: str, reason: str):
        super().__init__(f"room {room_id!r} hit its {reason} limit")
        self.room_id = room_id
        self.reason = reason


class RoomFinishedError(RuntimeError):
    """The turn policy itself has nothing left to say (e.g. a debate's
    rounds are used up) — distinct from a numeric bound being hit."""

    def __init__(self, room_id: str):
        super().__init__(f"room {room_id!r}'s turn policy has concluded")
        self.room_id = room_id


class RoomTimeoutError(RuntimeError):
    def __init__(self, room_id: str, participant_id: str, timeout_seconds: float):
        super().__init__(
            f"{participant_id!r} did not respond within {timeout_seconds}s in room {room_id!r}"
        )
        self.room_id = room_id
        self.participant_id = participant_id


def _approx_tokens(text: str) -> int:
    """A cheap, backend-agnostic stand-in for a real tokenizer — good enough
    to enforce a room-level budget without pulling in a specific model's
    tokenizer for a room that may hold several different actors."""
    return len(text.split())


class RoomOrchestrator:
    def __init__(self, store: RoomStore, router: BackendRouter):
        self._store = store
        self._router = router

    async def take_turn(
        self, room_id: str, trigger: Trigger = Trigger(), *, rng: random.Random | None = None
    ) -> Message:
        room = await self._store.get_room(room_id)
        if room.status != "active":
            raise RoomClosedError(room_id)

        participants = await self._store.list_participants(room_id)
        messages = await self._store.list_messages(room_id)
        turn_state = await self._store.get_turn_state(room_id)

        await self._enforce_bounds(room, turn_state, messages)

        policy = build_policy(room.turn_policy, room.turn_policy_config)
        last_speaker_id = messages[-1].speaker_id if messages else None
        decision = policy.decide(
            participants=participants,
            cursor=turn_state.cursor,
            last_speaker_id=last_speaker_id,
            trigger=trigger,
            rng=rng or random.Random(),
        )
        if decision.done:
            await self._store.close_room(room_id)
            raise RoomFinishedError(room_id)

        participant = await self._store.get_participant(room_id, decision.participant_id)
        transcript = [
            TranscriptMessage(m.speaker_id, m.speaker_name, m.content) for m in messages
        ]

        coro = self._router.generate(participant.actor_id, participant.id, transcript)
        try:
            if room.timeout_seconds:
                reply = await asyncio.wait_for(coro, timeout=room.timeout_seconds)
            else:
                reply = await coro
        except asyncio.TimeoutError:
            raise RoomTimeoutError(room_id, participant.id, room.timeout_seconds) from None

        speaker_name = display_names(participants)[participant.id]
        message = await self._store.append_message(room_id, participant.id, speaker_name, reply)
        await self._store.increment_turn_count(room_id)
        await self._store.update_turn_state(
            room_id,
            next_participant_id=_preview_next(policy, participants, decision.next_cursor),
            cursor=decision.next_cursor,
        )
        return message

    async def _enforce_bounds(self, room, turn_state, messages: list[Message]) -> None:
        """Checked before generating: once a bound is already met, the room
        closes and this attempt is refused — the turn that hit the bound
        was still delivered; this stops the *next* one."""
        if room.max_turns is not None and turn_state.turn_count >= room.max_turns:
            await self._store.close_room(room.id)
            raise RoomBoundsExceededError(room.id, "max_turns")
        if room.max_messages is not None and len(messages) >= room.max_messages:
            await self._store.close_room(room.id)
            raise RoomBoundsExceededError(room.id, "max_messages")
        if room.max_tokens is not None:
            used = sum(_approx_tokens(m.content) for m in messages)
            if used >= room.max_tokens:
                await self._store.close_room(room.id)
                raise RoomBoundsExceededError(room.id, "max_tokens")


def _preview_next(policy, participants, next_cursor: dict) -> str | None:
    """Best-effort "who's up after this" for policies with a deterministic
    order — used by /room status and as /next's default target. Policies
    without a fixed order (manual, reply, moderated) always require an
    explicit or freshly-random choice, so there's nothing to preview."""
    if isinstance(policy, RoundRobinPolicy) and participants:
        order = [p.id for p in participants]
        return order[next_cursor.get("index", 0) % len(order)]
    if isinstance(policy, DebatePolicy):
        if next_cursor.get("round", 0) >= policy.rounds:
            return None
        return policy.order[next_cursor.get("position", 0)]
    return None
