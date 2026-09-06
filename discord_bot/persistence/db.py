"""SQLite schema + connection bootstrap for rooms, participants, messages,
and turn state. One file per deployment (see .env.example's ROOM_DB_PATH);
never committed — see .gitignore.
"""
from __future__ import annotations

from pathlib import Path

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS rooms (
    id                  TEXT PRIMARY KEY,
    title               TEXT NOT NULL,
    turn_policy         TEXT NOT NULL,
    turn_policy_config  TEXT NOT NULL DEFAULT '{}',   -- JSON: policy-specific config (e.g. debate order/rounds)
    human_id            TEXT,                          -- discord user id of the optional human participant
    status              TEXT NOT NULL DEFAULT 'active', -- 'active' | 'closed'
    max_turns           INTEGER,
    max_messages        INTEGER,
    max_tokens          INTEGER,
    timeout_seconds      INTEGER,
    discord_guild_id    TEXT,
    discord_channel_id  TEXT,                          -- forum post / thread this room lives in
    created_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS participants (
    id                TEXT NOT NULL,                   -- e.g. "opus-4.8" or "opus-4.8:b" — see models.py
    room_id           TEXT NOT NULL REFERENCES rooms(id),
    actor_id          TEXT NOT NULL,                   -- discord_bot.registry actor id
    instance_suffix   TEXT NOT NULL,                   -- join order within (room, actor): 'a', 'b', 'c', ...
    joined_at         TEXT NOT NULL,
    active            INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (room_id, id)
);

CREATE TABLE IF NOT EXISTS messages (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    room_id             TEXT NOT NULL REFERENCES rooms(id),
    seq                 INTEGER NOT NULL,               -- monotonic per-room order
    speaker_id          TEXT NOT NULL,                  -- participant id, or "user:<discord_id>" for the human
    speaker_name        TEXT NOT NULL,
    content             TEXT NOT NULL,
    discord_message_id  TEXT,
    created_at          TEXT NOT NULL,
    UNIQUE (room_id, seq)
);

CREATE TABLE IF NOT EXISTS turn_state (
    room_id               TEXT PRIMARY KEY REFERENCES rooms(id),
    turn_count            INTEGER NOT NULL DEFAULT 0,
    next_participant_id   TEXT,
    cursor_json           TEXT NOT NULL DEFAULT '{}',   -- turn-policy-specific cursor (e.g. round-robin index)
    updated_at            TEXT NOT NULL
);

-- One webhook per Discord channel (a webhook posts under any username/
-- avatar per-message, so a channel never needs more than one). The token
-- is a bearer credential for that webhook — this table is gitignored
-- (see .env.example / docs) exactly like the rest of this database.
CREATE TABLE IF NOT EXISTS webhooks (
    channel_id     TEXT PRIMARY KEY,
    webhook_id     TEXT NOT NULL,
    webhook_token  TEXT NOT NULL,
    created_at     TEXT NOT NULL
);

-- Which Discord channel is each actor's standing 1:1 "voice channel"
-- (bound to a "reply"-policy room via room_id). Keyed by actor id rather
-- than channel name/id, so a channel that gets renamed or moved back and
-- forth by sync is still recognized as the same actor's — see
-- discord_app/channels.py.
CREATE TABLE IF NOT EXISTS voice_channels (
    actor_id    TEXT PRIMARY KEY,
    channel_id  TEXT NOT NULL,
    room_id     TEXT NOT NULL REFERENCES rooms(id),
    created_at  TEXT NOT NULL
);
"""


async def connect(path: str | Path) -> aiosqlite.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(path)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys = ON")
    await conn.executescript(SCHEMA)
    await conn.commit()
    return conn
