"""RoomStore: async CRUD over rooms, participants, messages, and turn state.

This is the only thing in discord_bot that touches SQL — everything else
(turn policies, the orchestrator, Discord commands) works with the
dataclasses in models.py.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from . import db
from .models import Message, Participant, Room, TurnState, VoiceChannelBinding, WebhookBinding

_MAX_INSTANCES_PER_ACTOR = 26  # 'a'..'z' — plenty for any real room

_UNSET = object()  # distinguishes "leave unchanged" from an explicit None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_room_id() -> str:
    return uuid.uuid4().hex[:8]


class RoomNotFoundError(LookupError):
    pass


class ParticipantNotFoundError(LookupError):
    pass


class TooManyInstancesError(ValueError):
    pass


class RoomStore:
    def __init__(self, conn: aiosqlite.Connection):
        self._conn = conn

    @classmethod
    async def open(cls, path: str | Path) -> "RoomStore":
        return cls(await db.connect(path))

    async def close(self) -> None:
        await self._conn.close()

    # -- rooms ----------------------------------------------------------

    async def create_room(
        self,
        title: str,
        turn_policy: str,
        *,
        room_id: str | None = None,
        turn_policy_config: dict[str, Any] | None = None,
        human_id: str | None = None,
        max_turns: int | None = None,
        max_messages: int | None = None,
        max_tokens: int | None = None,
        timeout_seconds: int | None = None,
        discord_guild_id: str | None = None,
        discord_channel_id: str | None = None,
    ) -> Room:
        room_id = room_id or _new_room_id()
        await self._conn.execute(
            """INSERT INTO rooms
               (id, title, turn_policy, turn_policy_config, human_id, max_turns,
                max_messages, max_tokens, timeout_seconds, discord_guild_id,
                discord_channel_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                room_id, title, turn_policy, json.dumps(turn_policy_config or {}),
                human_id, max_turns, max_messages, max_tokens, timeout_seconds,
                discord_guild_id, discord_channel_id, _now(),
            ),
        )
        await self._conn.commit()
        return await self.get_room(room_id)

    async def get_room(self, room_id: str) -> Room:
        cur = await self._conn.execute("SELECT * FROM rooms WHERE id = ?", (room_id,))
        row = await cur.fetchone()
        if row is None:
            raise RoomNotFoundError(room_id)
        return Room.from_row(row)

    async def list_rooms(self, status: str | None = "active") -> list[Room]:
        if status is None:
            cur = await self._conn.execute("SELECT * FROM rooms ORDER BY created_at")
        else:
            cur = await self._conn.execute(
                "SELECT * FROM rooms WHERE status = ? ORDER BY created_at", (status,)
            )
        return [Room.from_row(row) for row in await cur.fetchall()]

    async def close_room(self, room_id: str) -> None:
        await self._require_room(room_id)
        await self._conn.execute("UPDATE rooms SET status = 'closed' WHERE id = ?", (room_id,))
        await self._conn.commit()

    async def bind_discord_channel(self, room_id: str, guild_id: str, channel_id: str) -> None:
        await self._require_room(room_id)
        await self._conn.execute(
            "UPDATE rooms SET discord_guild_id = ?, discord_channel_id = ? WHERE id = ?",
            (guild_id, channel_id, room_id),
        )
        await self._conn.commit()

    async def get_room_by_channel(self, channel_id: str) -> Room | None:
        cur = await self._conn.execute(
            "SELECT * FROM rooms WHERE discord_channel_id = ? AND status = 'active'", (channel_id,)
        )
        row = await cur.fetchone()
        return Room.from_row(row) if row else None

    # -- participants -----------------------------------------------------

    async def add_participant(self, room_id: str, actor_id: str) -> Participant:
        await self._require_room(room_id)
        cur = await self._conn.execute(
            "SELECT instance_suffix FROM participants WHERE room_id = ? AND actor_id = ?",
            (room_id, actor_id),
        )
        used = {row["instance_suffix"] for row in await cur.fetchall()}
        suffix = _next_suffix(used)

        participant_id = actor_id if suffix == "a" else f"{actor_id}:{suffix}"
        joined_at = _now()
        await self._conn.execute(
            """INSERT INTO participants (id, room_id, actor_id, instance_suffix, joined_at, active)
               VALUES (?, ?, ?, ?, ?, 1)""",
            (participant_id, room_id, actor_id, suffix, joined_at),
        )
        await self._conn.commit()
        return Participant(
            id=participant_id, room_id=room_id, actor_id=actor_id,
            instance_suffix=suffix, joined_at=joined_at, active=True,
        )

    async def remove_participant(self, room_id: str, participant_id: str) -> None:
        cur = await self._conn.execute(
            "UPDATE participants SET active = 0 WHERE room_id = ? AND id = ?",
            (room_id, participant_id),
        )
        await self._conn.commit()
        if cur.rowcount == 0:
            raise ParticipantNotFoundError(participant_id)

    async def list_participants(self, room_id: str, *, active_only: bool = True) -> list[Participant]:
        if active_only:
            cur = await self._conn.execute(
                "SELECT * FROM participants WHERE room_id = ? AND active = 1 ORDER BY joined_at",
                (room_id,),
            )
        else:
            cur = await self._conn.execute(
                "SELECT * FROM participants WHERE room_id = ? ORDER BY joined_at", (room_id,)
            )
        return [Participant.from_row(row) for row in await cur.fetchall()]

    async def get_participant(self, room_id: str, participant_id: str) -> Participant:
        cur = await self._conn.execute(
            "SELECT * FROM participants WHERE room_id = ? AND id = ?", (room_id, participant_id)
        )
        row = await cur.fetchone()
        if row is None:
            raise ParticipantNotFoundError(participant_id)
        return Participant.from_row(row)

    # -- messages -----------------------------------------------------------

    async def append_message(
        self,
        room_id: str,
        speaker_id: str,
        speaker_name: str,
        content: str,
        *,
        discord_message_id: str | None = None,
    ) -> Message:
        await self._require_room(room_id)
        cur = await self._conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq FROM messages WHERE room_id = ?",
            (room_id,),
        )
        seq = (await cur.fetchone())["next_seq"]
        created_at = _now()
        cur = await self._conn.execute(
            """INSERT INTO messages
               (room_id, seq, speaker_id, speaker_name, content, discord_message_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (room_id, seq, speaker_id, speaker_name, content, discord_message_id, created_at),
        )
        await self._conn.commit()
        return Message(
            id=cur.lastrowid, room_id=room_id, seq=seq, speaker_id=speaker_id,
            speaker_name=speaker_name, content=content,
            discord_message_id=discord_message_id, created_at=created_at,
        )

    async def list_messages(self, room_id: str) -> list[Message]:
        cur = await self._conn.execute(
            "SELECT * FROM messages WHERE room_id = ? ORDER BY seq", (room_id,)
        )
        return [Message.from_row(row) for row in await cur.fetchall()]

    # -- turn state -----------------------------------------------------------

    async def get_turn_state(self, room_id: str) -> TurnState:
        cur = await self._conn.execute("SELECT * FROM turn_state WHERE room_id = ?", (room_id,))
        row = await cur.fetchone()
        if row is None:
            await self._require_room(room_id)
            return await self._init_turn_state(room_id)
        return TurnState.from_row(row)

    async def _init_turn_state(self, room_id: str) -> TurnState:
        updated_at = _now()
        await self._conn.execute(
            """INSERT INTO turn_state (room_id, turn_count, next_participant_id, cursor_json, updated_at)
               VALUES (?, 0, NULL, '{}', ?)""",
            (room_id, updated_at),
        )
        await self._conn.commit()
        return TurnState(room_id=room_id, turn_count=0, next_participant_id=None, cursor={}, updated_at=updated_at)

    async def update_turn_state(
        self,
        room_id: str,
        *,
        turn_count: int | None = None,
        next_participant_id: Any = _UNSET,  # str, or None to clear, or _UNSET to leave unchanged
        cursor: dict[str, Any] | None = None,
    ) -> TurnState:
        current = await self.get_turn_state(room_id)
        new_turn_count = current.turn_count if turn_count is None else turn_count
        new_next = current.next_participant_id if next_participant_id is _UNSET else next_participant_id
        new_cursor = current.cursor if cursor is None else cursor
        updated_at = _now()
        await self._conn.execute(
            """UPDATE turn_state
               SET turn_count = ?, next_participant_id = ?, cursor_json = ?, updated_at = ?
               WHERE room_id = ?""",
            (new_turn_count, new_next, json.dumps(new_cursor), updated_at, room_id),
        )
        await self._conn.commit()
        return TurnState(
            room_id=room_id, turn_count=new_turn_count, next_participant_id=new_next,
            cursor=new_cursor, updated_at=updated_at,
        )

    async def increment_turn_count(self, room_id: str) -> int:
        state = await self.get_turn_state(room_id)
        new_count = state.turn_count + 1
        await self._conn.execute(
            "UPDATE turn_state SET turn_count = ?, updated_at = ? WHERE room_id = ?",
            (new_count, _now(), room_id),
        )
        await self._conn.commit()
        return new_count

    # -- webhook bindings -----------------------------------------------------

    async def get_webhook_binding(self, channel_id: str) -> WebhookBinding | None:
        cur = await self._conn.execute(
            "SELECT * FROM webhooks WHERE channel_id = ?", (channel_id,)
        )
        row = await cur.fetchone()
        return WebhookBinding.from_row(row) if row else None

    async def save_webhook_binding(self, channel_id: str, webhook_id: str, webhook_token: str) -> WebhookBinding:
        created_at = _now()
        await self._conn.execute(
            """INSERT INTO webhooks (channel_id, webhook_id, webhook_token, created_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT (channel_id) DO UPDATE SET webhook_id = excluded.webhook_id,
                   webhook_token = excluded.webhook_token""",
            (channel_id, webhook_id, webhook_token, created_at),
        )
        await self._conn.commit()
        return WebhookBinding(
            channel_id=channel_id, webhook_id=webhook_id, webhook_token=webhook_token,
            created_at=created_at,
        )

    # -- voice channel bindings -----------------------------------------------

    async def get_voice_channel(self, actor_id: str) -> VoiceChannelBinding | None:
        cur = await self._conn.execute(
            "SELECT * FROM voice_channels WHERE actor_id = ?", (actor_id,)
        )
        row = await cur.fetchone()
        return VoiceChannelBinding.from_row(row) if row else None

    async def save_voice_channel(self, actor_id: str, channel_id: str, room_id: str) -> VoiceChannelBinding:
        created_at = _now()
        await self._conn.execute(
            """INSERT INTO voice_channels (actor_id, channel_id, room_id, created_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT (actor_id) DO UPDATE SET channel_id = excluded.channel_id,
                   room_id = excluded.room_id""",
            (actor_id, channel_id, room_id, created_at),
        )
        await self._conn.commit()
        return VoiceChannelBinding(actor_id=actor_id, channel_id=channel_id, room_id=room_id, created_at=created_at)

    async def all_voice_channels(self) -> list[VoiceChannelBinding]:
        cur = await self._conn.execute("SELECT * FROM voice_channels ORDER BY actor_id")
        return [VoiceChannelBinding.from_row(row) for row in await cur.fetchall()]

    # -- internal -----------------------------------------------------------

    async def _require_room(self, room_id: str) -> None:
        cur = await self._conn.execute("SELECT 1 FROM rooms WHERE id = ?", (room_id,))
        if await cur.fetchone() is None:
            raise RoomNotFoundError(room_id)


def _next_suffix(used: set[str]) -> str:
    for i in range(_MAX_INSTANCES_PER_ACTOR):
        letter = chr(ord("a") + i)
        if letter not in used:
            return letter
    raise TooManyInstancesError(
        f"more than {_MAX_INSTANCES_PER_ACTOR} instances of one actor in a single room"
    )
