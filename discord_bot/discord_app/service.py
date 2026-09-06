"""Command logic, independent of discord.py's command framework.

Every slash command in commands.py is a thin wrapper over a method here,
for the same reason MessageRouter is separate from bot.py: this is the part
worth testing, and it tests against a real store/orchestrator with a fake
backend rather than a mocked interaction tree.

Anything that posts to Discord is expressed as a returned value (text, or
an ActorMessage saying "post this as this actor"), never done here — so
the service stays free of Discord objects.
"""
from __future__ import annotations

from dataclasses import dataclass

from discord_bot.backends.base import GenerationError
from discord_bot.persistence.models import Participant, Room, display_names
from discord_bot.persistence.store import (
    ParticipantNotFoundError,
    RoomNotFoundError,
    RoomStore,
    TooManyInstancesError,
)
from discord_bot.registry import ActorRegistry
from discord_bot.rooms.orchestrator import (
    RoomBoundsExceededError,
    RoomClosedError,
    RoomFinishedError,
    RoomOrchestrator,
    RoomTimeoutError,
)
from discord_bot.rooms.turn_policies import Trigger, TurnPolicyError

TURN_POLICIES = ("manual", "reply", "round_robin", "moderated", "debate")

# A salon/debate is always bounded, both by its own rounds and by hard
# room-level ceilings, so a runaway can't outlive the command that started it.
MAX_ROUNDS = 8
MAX_ACTORS_PER_ROOM = 8
DEFAULT_TIMEOUT_SECONDS = 180


class ServiceError(RuntimeError):
    """A command failed for a reason worth showing the user verbatim."""


@dataclass(frozen=True)
class ActorMessage:
    """One line to post to Discord under an actor's own identity."""

    participant_id: str
    display_name: str
    avatar_url: str | None
    content: str


class RoomService:
    def __init__(self, store: RoomStore, orchestrator: RoomOrchestrator, registry: ActorRegistry):
        self._store = store
        self._orchestrator = orchestrator
        self._registry = registry

    # -- directory ----------------------------------------------------------

    def describe_actors(self) -> str:
        lines = ["**Configured actors**", ""]
        for actor in sorted(self._registry, key=lambda a: a.id):
            substrate = actor.model or actor.base or "?"
            persona = actor.persona or "— (base model, no persona)"
            adapter = " + LoRA" if actor.adapter else ""
            lines.append(f"• `{actor.id}` — {persona} · {actor.backend}:{substrate}{adapter}")
        lines += ["", f"_{len(self._registry)} actors. Talk to one in its #voices channel, "
                      "or convene several with `/salon` or `/debate`._"]
        return "\n".join(lines)

    def actor_ids(self) -> list[str]:
        return self._registry.ids()

    # -- rooms --------------------------------------------------------------

    async def create_room(
        self,
        title: str,
        turn_policy: str,
        *,
        guild_id: str | None = None,
        channel_id: str | None = None,
        turn_policy_config: dict | None = None,
        max_turns: int | None = None,
        timeout_seconds: int | None = DEFAULT_TIMEOUT_SECONDS,
    ) -> Room:
        if turn_policy not in TURN_POLICIES:
            raise ServiceError(f"unknown turn policy {turn_policy!r} (expected one of {', '.join(TURN_POLICIES)})")
        return await self._store.create_room(
            title, turn_policy,
            turn_policy_config=turn_policy_config,
            discord_guild_id=guild_id, discord_channel_id=channel_id,
            max_turns=max_turns, timeout_seconds=timeout_seconds,
        )

    async def add_actor(self, room_id: str, actor_id: str) -> Participant:
        if actor_id not in self._registry:
            raise ServiceError(f"no such actor: `{actor_id}` (see /models)")
        participants = await self._store.list_participants(room_id)
        if len(participants) >= MAX_ACTORS_PER_ROOM:
            raise ServiceError(f"a room holds at most {MAX_ACTORS_PER_ROOM} participants")
        try:
            return await self._store.add_participant(room_id, actor_id)
        except RoomNotFoundError:
            raise ServiceError(f"no such room: `{room_id}`") from None
        except TooManyInstancesError as exc:
            raise ServiceError(str(exc)) from None

    async def remove_participant(self, room_id: str, participant_id: str) -> None:
        try:
            await self._store.remove_participant(room_id, participant_id)
        except ParticipantNotFoundError:
            raise ServiceError(f"`{participant_id}` is not a participant in this room") from None

    async def describe_room(self, room_id: str) -> str:
        try:
            room = await self._store.get_room(room_id)
        except RoomNotFoundError:
            raise ServiceError(f"no such room: `{room_id}`") from None
        participants = await self._store.list_participants(room_id)
        state = await self._store.get_turn_state(room_id)
        messages = await self._store.list_messages(room_id)
        names = display_names(participants)

        lines = [
            f"**{room.title}** · `{room.id}`",
            f"status: **{room.status}** · policy: **{room.turn_policy}**",
            f"turns taken: **{state.turn_count}**"
            + (f" / {room.max_turns}" if room.max_turns else "")
            + f" · messages: **{len(messages)}**",
        ]
        if participants:
            lines.append("participants: " + ", ".join(f"`{names[p.id]}`" for p in participants))
        else:
            lines.append("participants: _none yet — add some with `/room add`_")
        if state.next_participant_id:
            lines.append(f"up next: `{names.get(state.next_participant_id, state.next_participant_id)}`")
        return "\n".join(lines)

    async def participant_ids(self, room_id: str) -> list[str]:
        return [p.id for p in await self._store.list_participants(room_id)]

    # -- turns --------------------------------------------------------------

    async def take_turn(self, room_id: str, participant_id: str | None = None) -> ActorMessage:
        try:
            message = await self._orchestrator.take_turn(
                room_id, Trigger(requested_participant_id=participant_id)
            )
        except (RoomClosedError, RoomFinishedError, RoomBoundsExceededError, RoomTimeoutError) as exc:
            raise ServiceError(str(exc)) from None
        except (TurnPolicyError, GenerationError) as exc:
            raise ServiceError(str(exc)) from None
        except RoomNotFoundError:
            raise ServiceError(f"no such room: `{room_id}`") from None

        participant = await self._store.get_participant(room_id, message.speaker_id)
        actor = self._registry.get(participant.actor_id)
        return ActorMessage(
            participant_id=participant.id,
            display_name=message.speaker_name,
            avatar_url=actor.avatar,
            content=message.content,
        )

    # -- multi-participant conversations -------------------------------------

    async def prepare_conversation(
        self,
        title: str,
        actor_ids: list[str],
        rounds: int,
        *,
        policy: str,
        topic: str,
        guild_id: str | None = None,
        channel_id: str | None = None,
    ) -> Room:
        """Create a room, seed it with the topic, and add every requested
        actor — the same actor more than once if it's named more than once.
        Bounded up front: rounds and participants are clamped, and the room
        gets a hard max_turns so it can't outrun its own schedule."""
        if not actor_ids:
            raise ServiceError("name at least one actor")
        if len(actor_ids) > MAX_ACTORS_PER_ROOM:
            raise ServiceError(f"at most {MAX_ACTORS_PER_ROOM} participants per room")
        if not 1 <= rounds <= MAX_ROUNDS:
            raise ServiceError(f"rounds must be between 1 and {MAX_ROUNDS}")
        for actor_id in actor_ids:
            if actor_id not in self._registry:
                raise ServiceError(f"no such actor: `{actor_id}` (see /models)")

        total_turns = rounds * len(actor_ids)
        room = await self._store.create_room(
            title, policy,
            discord_guild_id=guild_id, discord_channel_id=channel_id,
            max_turns=total_turns, timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        )
        participants = [await self._store.add_participant(room.id, a) for a in actor_ids]

        if policy == "debate":
            room = await self._reconfigure_debate(room.id, [p.id for p in participants], rounds)

        await self._store.append_message(
            room.id, "system", "the question before us", topic
        )
        return room

    async def _reconfigure_debate(self, room_id: str, order: list[str], rounds: int) -> Room:
        # DebatePolicy needs its fixed order/rounds in turn_policy_config; the
        # participant ids only exist after they're added, so it's set here.
        await self._store.set_turn_policy_config(room_id, {"order": order, "rounds": rounds})
        return await self._store.get_room(room_id)

    async def run_conversation(self, room_id: str, turns: int):
        """Run up to `turns` explicitly-scheduled turns, yielding each
        message as it's produced. Stops early — without raising — the moment
        the room closes, finishes, or hits a bound: every generated message
        is scheduled here, one at a time, so there's no path by which
        posting one triggers another."""
        for _ in range(turns):
            try:
                yield await self.take_turn(room_id)
            except ServiceError:
                return
