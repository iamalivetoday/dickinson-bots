"""Idempotent channel provisioning and sync for a guild.

`sync_guild` can be run any number of times: it creates whatever's missing
(#foyer, #models, the voices/rooms categories, one channel per actor),
renames an actor's channel back if it drifted, and archives (never
deletes) a channel for an actor no longer in the registry — so history is
never destroyed. Actor identity is tracked through the store's
voice_channels table, not by guessing from channel names, so a channel
surviving a rename is still recognized as the same actor's.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import discord

from discord_bot.persistence.store import RoomStore
from discord_bot.registry import ActorRegistry

FOYER_NAME = "foyer"
MODELS_NAME = "models"
VOICES_CATEGORY = "voices"
ROOMS_CATEGORY = "rooms"
ROOMS_CHANNEL = "rooms"
ARCHIVED_CATEGORY = "archived"

FOYER_TOPIC = "Instructions and commands — see /models for the actor directory."
MODELS_TOPIC = "The current actor directory — kept in sync by /sync-models."


def channel_name_for(actor_id: str) -> str:
    """A valid Discord channel name for an actor id: lowercase, only
    letters/digits/hyphen/underscore. "weil-opus-4.8" -> "weil-opus-4-8"."""
    name = re.sub(r"[^a-z0-9_-]+", "-", actor_id.lower())
    name = re.sub(r"-{2,}", "-", name).strip("-")
    return name or "actor"


@dataclass(frozen=True)
class SyncReport:
    created: list[str] = field(default_factory=list)
    renamed: list[str] = field(default_factory=list)
    archived: list[str] = field(default_factory=list)
    unarchived: list[str] = field(default_factory=list)


async def sync_guild(guild: discord.Guild, registry: ActorRegistry, store: RoomStore) -> SyncReport:
    await _ensure_text_channel(guild, FOYER_NAME, category=None, topic=FOYER_TOPIC)
    await _ensure_text_channel(guild, MODELS_NAME, category=None, topic=MODELS_TOPIC)
    voices_category = await _ensure_category(guild, VOICES_CATEGORY)
    rooms_category = await _ensure_category(guild, ROOMS_CATEGORY)
    await _ensure_text_channel(guild, ROOMS_CHANNEL, category=rooms_category)

    report = SyncReport()
    wanted_ids = {actor.id for actor in registry}

    for actor in registry:
        await _sync_actor_channel(guild, store, actor.id, voices_category, report)

    archived_category = None
    for binding in await store.all_voice_channels():
        if binding.actor_id in wanted_ids:
            continue
        channel = guild.get_channel(int(binding.channel_id))
        if channel is None:
            continue
        expected_archived_name = f"archived-{channel_name_for(binding.actor_id)}"
        if channel.name == expected_archived_name and channel.category == archived_category:
            continue
        archived_category = archived_category or await _ensure_category(guild, ARCHIVED_CATEGORY)
        await channel.edit(name=expected_archived_name, category=archived_category)
        report.archived.append(binding.actor_id)

    return report


async def _sync_actor_channel(guild, store, actor_id, voices_category, report: SyncReport) -> None:
    expected_name = channel_name_for(actor_id)
    binding = await store.get_voice_channel(actor_id)
    channel = guild.get_channel(int(binding.channel_id)) if binding else None

    if channel is None:
        channel = await guild.create_text_channel(expected_name, category=voices_category)
        room = await store.create_room(
            f"{actor_id} voice channel", "reply",
            discord_guild_id=str(guild.id), discord_channel_id=str(channel.id),
        )
        await store.add_participant(room.id, actor_id)
        await store.save_voice_channel(actor_id, str(channel.id), room.id)
        report.created.append(actor_id)
        return

    was_archived = channel.category != voices_category
    if channel.name != expected_name or channel.category != voices_category:
        await channel.edit(name=expected_name, category=voices_category)
        if was_archived:
            report.unarchived.append(actor_id)
        else:
            report.renamed.append(actor_id)


async def _ensure_category(guild: discord.Guild, name: str) -> discord.CategoryChannel:
    for category in guild.categories:
        if category.name == name:
            return category
    return await guild.create_category(name)


async def _ensure_text_channel(
    guild: discord.Guild, name: str, *, category: discord.CategoryChannel | None, topic: str | None = None
) -> discord.TextChannel:
    for channel in guild.text_channels:
        if channel.name == name and channel.category == category:
            return channel
    return await guild.create_text_channel(name, category=category, topic=topic or discord.utils.MISSING)
