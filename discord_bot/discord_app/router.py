"""Routes an incoming Discord message to the room bound to its channel, and
runs exactly one turn if it should. Deliberately decoupled from discord.py's
own types (see IncomingMessage) so this — the actual decision logic, and
the part most worth testing — never needs a mocked discord.Client: give it
a real RoomStore/RoomOrchestrator and a FakeBackend and it behaves exactly
as it will in production. discord_app/bot.py does the thin, untested glue
of turning a real discord.Message into an IncomingMessage and a
RoutingOutcome into an actual webhook post.
"""
from __future__ import annotations

from dataclasses import dataclass

from discord_bot.backends.base import GenerationError
from discord_bot.persistence.models import display_names
from discord_bot.persistence.store import RoomStore
from discord_bot.registry import ActorRegistry
from discord_bot.rooms.orchestrator import (
    RoomBoundsExceededError,
    RoomClosedError,
    RoomFinishedError,
    RoomOrchestrator,
    RoomTimeoutError,
)
from discord_bot.rooms.turn_policies import Trigger

_ROOM_ERRORS = (RoomClosedError, RoomBoundsExceededError, RoomFinishedError, RoomTimeoutError, GenerationError)


@dataclass(frozen=True)
class IncomingMessage:
    guild_id: str | None
    channel_id: str
    author_id: str
    author_display_name: str
    author_is_bot: bool  # true for the bot's own messages *and* any webhook message
    content: str


@dataclass(frozen=True)
class RoutingOutcome:
    posted: bool
    actor_id: str | None = None
    display_name: str | None = None
    avatar_url: str | None = None
    content: str | None = None
    error: str | None = None


class MessageRouter:
    def __init__(
        self,
        store: RoomStore,
        orchestrator: RoomOrchestrator,
        registry: ActorRegistry,
        *,
        allowed_guild_id: str | None = None,
        allowed_user_ids: frozenset[str] = frozenset(),
    ):
        self._store = store
        self._orchestrator = orchestrator
        self._registry = registry
        self._allowed_guild_id = allowed_guild_id
        self._allowed_user_ids = allowed_user_ids

    async def handle_message(self, incoming: IncomingMessage) -> RoutingOutcome:
        # Loop prevention: a bot's own messages and every webhook message
        # (i.e. every actor's own reply) must never be treated as a new
        # human turn — discord.py glue is expected to have already set
        # author_is_bot=True for both.
        if incoming.author_is_bot:
            return RoutingOutcome(posted=False)
        if self._allowed_guild_id and incoming.guild_id != self._allowed_guild_id:
            return RoutingOutcome(posted=False)
        if self._allowed_user_ids and incoming.author_id not in self._allowed_user_ids:
            return RoutingOutcome(posted=False)

        room = await self._store.get_room_by_channel(incoming.channel_id)
        if room is None:
            return RoutingOutcome(posted=False)  # not a room-bound channel — nothing to do

        await self._store.append_message(
            room.id, f"user:{incoming.author_id}", incoming.author_display_name, incoming.content
        )

        participants = await self._store.list_participants(room.id)
        target = _resolve_reply_target(participants, incoming.content)
        if target is None:
            return RoutingOutcome(posted=False, error="no participant was addressed")

        try:
            message = await self._orchestrator.take_turn(
                room.id, Trigger(requested_participant_id=target)
            )
        except _ROOM_ERRORS as exc:
            return RoutingOutcome(posted=False, error=str(exc))

        participant = await self._store.get_participant(room.id, target)
        actor = self._registry.get(participant.actor_id)
        names = display_names(participants)
        return RoutingOutcome(
            posted=True,
            actor_id=actor.id,
            display_name=names[target],
            avatar_url=actor.avatar,
            content=message.content,
        )


def _resolve_reply_target(participants, content: str) -> str | None:
    """A voice channel's room has exactly one actor — unambiguous. A
    multi-actor room (a rooms/ thread) needs an explicit @mention of a
    participant's id or bare actor id in the message."""
    active = [p for p in participants if p.active]
    if len(active) == 1:
        return active[0].id

    # Match against the *disambiguated* display name only — with duplicate
    # instances active, the bare actor id ("@opus-4.8") is ambiguous and
    # must not silently resolve to whichever instance happens to be first.
    names = display_names(active)
    lowered = content.lower()
    for participant in active:
        if f"@{names[participant.id].lower()}" in lowered:
            return participant.id
    return None
