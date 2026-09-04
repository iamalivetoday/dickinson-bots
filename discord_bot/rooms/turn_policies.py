"""Turn policies decide *who speaks next* — nothing else. They never call a
backend, touch the store, or post to Discord; the orchestrator does all of
that around a policy's decision. This keeps every policy a small, pure,
easily-tested function of (participants, cursor, trigger).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Protocol

from discord_bot.persistence.models import Participant


class TurnPolicyError(ValueError):
    """The requested turn can't be resolved (bad/missing participant, no
    active participants, malformed policy config, ...)."""


@dataclass(frozen=True)
class Trigger:
    """Context for one "who speaks next" decision.

    `requested_participant_id` is the explicit choice behind `manual`
    (/next <participant>) and `reply` (an @-addressed participant) — every
    other policy computes its own next speaker and ignores it.
    """

    requested_participant_id: str | None = None


@dataclass(frozen=True)
class TurnDecision:
    participant_id: str | None
    next_cursor: dict[str, Any]
    done: bool = False  # True: this policy has nothing left to say (e.g. debate's rounds are used up)


class TurnPolicy(Protocol):
    name: str

    def decide(
        self,
        *,
        participants: list[Participant],
        cursor: dict[str, Any],
        last_speaker_id: str | None,
        trigger: Trigger,
        rng: random.Random,
    ) -> TurnDecision: ...


def _require_active(participant_id: str, participants: list[Participant]) -> None:
    if not any(p.id == participant_id for p in participants):
        raise TurnPolicyError(f"{participant_id!r} is not an active participant in this room")


def _require_participants(participants: list[Participant]) -> None:
    if not participants:
        raise TurnPolicyError("this room has no active participants")


class ManualPolicy:
    """I select the next participant, every time."""

    name = "manual"

    def decide(self, *, participants, cursor, last_speaker_id, trigger, rng) -> TurnDecision:
        if trigger.requested_participant_id is None:
            raise TurnPolicyError("manual turn policy requires an explicit participant (/next <participant>)")
        _require_active(trigger.requested_participant_id, participants)
        return TurnDecision(trigger.requested_participant_id, next_cursor=cursor)


class ReplyPolicy:
    """Whoever was addressed responds."""

    name = "reply"

    def decide(self, *, participants, cursor, last_speaker_id, trigger, rng) -> TurnDecision:
        if trigger.requested_participant_id is None:
            raise TurnPolicyError("reply turn policy requires an addressed participant")
        _require_active(trigger.requested_participant_id, participants)
        return TurnDecision(trigger.requested_participant_id, next_cursor=cursor)


class RoundRobinPolicy:
    """Active participants speak in join order, cycling forever."""

    name = "round_robin"

    def decide(self, *, participants, cursor, last_speaker_id, trigger, rng) -> TurnDecision:
        _require_participants(participants)
        order = [p.id for p in participants]
        index = cursor.get("index", 0) % len(order)
        chosen = order[index]
        next_cursor = {"index": (index + 1) % len(order)}
        return TurnDecision(chosen, next_cursor=next_cursor)


class ModeratedPolicy:
    """The conductor (this bot) picks the next speaker — anyone active
    except whoever spoke last, so the room doesn't stall on one voice."""

    name = "moderated"

    def decide(self, *, participants, cursor, last_speaker_id, trigger, rng) -> TurnDecision:
        _require_participants(participants)
        candidates = [p.id for p in participants if p.id != last_speaker_id] or [
            p.id for p in participants
        ]
        chosen = rng.choice(candidates)
        return TurnDecision(chosen, next_cursor=cursor)


class DebatePolicy:
    """Fixed positions, order, and rounds, set at room creation time
    (room.turn_policy_config = {"order": [participant_id, ...], "rounds": N}).
    Cycles through `order` exactly `rounds` times, then reports done.
    """

    name = "debate"

    def __init__(self, config: dict[str, Any]):
        order = config.get("order")
        rounds = config.get("rounds")
        if not order or not isinstance(order, list):
            raise TurnPolicyError("debate turn policy requires a non-empty 'order' list")
        if not isinstance(rounds, int) or rounds < 1:
            raise TurnPolicyError("debate turn policy requires a positive integer 'rounds'")
        self.order = order
        self.rounds = rounds

    def decide(self, *, participants, cursor, last_speaker_id, trigger, rng) -> TurnDecision:
        for participant_id in self.order:
            _require_active(participant_id, participants)

        round_ = cursor.get("round", 0)
        position = cursor.get("position", 0)
        if round_ >= self.rounds:
            return TurnDecision(None, next_cursor=cursor, done=True)

        chosen = self.order[position]
        next_position = position + 1
        next_round = round_
        if next_position >= len(self.order):
            next_position = 0
            next_round += 1
        return TurnDecision(chosen, next_cursor={"position": next_position, "round": next_round})


def build_policy(name: str, config: dict[str, Any] | None = None) -> TurnPolicy:
    config = config or {}
    if name == "debate":
        return DebatePolicy(config)
    try:
        return {
            "manual": ManualPolicy,
            "reply": ReplyPolicy,
            "round_robin": RoundRobinPolicy,
            "moderated": ModeratedPolicy,
        }[name]()
    except KeyError:
        raise TurnPolicyError(f"unknown turn policy: {name!r}") from None
