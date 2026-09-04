"""Read models returned by RoomStore — plain dataclasses, decoupled from
sqlite Row objects.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Room:
    id: str
    title: str
    turn_policy: str
    turn_policy_config: dict[str, Any]
    human_id: str | None
    status: str
    max_turns: int | None
    max_messages: int | None
    max_tokens: int | None
    timeout_seconds: int | None
    discord_guild_id: str | None
    discord_channel_id: str | None
    created_at: str

    @classmethod
    def from_row(cls, row) -> "Room":
        return cls(
            id=row["id"],
            title=row["title"],
            turn_policy=row["turn_policy"],
            turn_policy_config=json.loads(row["turn_policy_config"]),
            human_id=row["human_id"],
            status=row["status"],
            max_turns=row["max_turns"],
            max_messages=row["max_messages"],
            max_tokens=row["max_tokens"],
            timeout_seconds=row["timeout_seconds"],
            discord_guild_id=row["discord_guild_id"],
            discord_channel_id=row["discord_channel_id"],
            created_at=row["created_at"],
        )


@dataclass(frozen=True)
class Participant:
    id: str
    room_id: str
    actor_id: str
    instance_suffix: str
    joined_at: str
    active: bool

    @classmethod
    def from_row(cls, row) -> "Participant":
        return cls(
            id=row["id"],
            room_id=row["room_id"],
            actor_id=row["actor_id"],
            instance_suffix=row["instance_suffix"],
            joined_at=row["joined_at"],
            active=bool(row["active"]),
        )


@dataclass(frozen=True)
class Message:
    id: int
    room_id: str
    seq: int
    speaker_id: str
    speaker_name: str
    content: str
    discord_message_id: str | None
    created_at: str

    @classmethod
    def from_row(cls, row) -> "Message":
        return cls(
            id=row["id"],
            room_id=row["room_id"],
            seq=row["seq"],
            speaker_id=row["speaker_id"],
            speaker_name=row["speaker_name"],
            content=row["content"],
            discord_message_id=row["discord_message_id"],
            created_at=row["created_at"],
        )


@dataclass(frozen=True)
class TurnState:
    room_id: str
    turn_count: int
    next_participant_id: str | None
    cursor: dict[str, Any]
    updated_at: str

    @classmethod
    def from_row(cls, row) -> "TurnState":
        return cls(
            room_id=row["room_id"],
            turn_count=row["turn_count"],
            next_participant_id=row["next_participant_id"],
            cursor=json.loads(row["cursor_json"]),
            updated_at=row["updated_at"],
        )


def display_names(participants: list[Participant]) -> dict[str, str]:
    """Map participant id -> Discord-facing display name.

    An actor id is shown bare unless this room currently holds more than
    one *active* instance of it, in which case every active instance of
    that actor shows its instance suffix (e.g. "opus-4.8:a", "opus-4.8:b").
    This is purely a rendering rule — stored participant ids never change.
    """
    counts: dict[str, int] = {}
    for p in participants:
        if p.active:
            counts[p.actor_id] = counts.get(p.actor_id, 0) + 1

    names = {}
    for p in participants:
        if counts.get(p.actor_id, 0) > 1:
            names[p.id] = f"{p.actor_id}:{p.instance_suffix}"
        else:
            names[p.id] = p.actor_id
    return names
