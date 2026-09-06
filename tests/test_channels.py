import itertools

import discord
import pytest

from discord_bot.discord_app.channels import (
    ARCHIVED_CATEGORY,
    FOYER_NAME,
    MODELS_NAME,
    ROOMS_CATEGORY,
    ROOMS_CHANNEL,
    VOICES_CATEGORY,
    channel_name_for,
    sync_guild,
)
from discord_bot.persistence.store import RoomStore
from discord_bot.registry import ActorRegistry

_channel_ids = itertools.count(1000)
_category_ids = itertools.count(2000)


class FakeCategory:
    def __init__(self, name):
        self.name = name
        self.id = next(_category_ids)


class FakeChannel:
    def __init__(self, name, category=None):
        self.name = name
        self.category = category
        self.id = next(_channel_ids)
        self.topic = None

    async def edit(self, **kwargs):
        if "name" in kwargs:
            self.name = kwargs["name"]
        if "category" in kwargs:
            self.category = kwargs["category"]
        if "topic" in kwargs:
            self.topic = kwargs["topic"]


class FakeGuild:
    def __init__(self, guild_id=1):
        self.id = guild_id
        self.categories = []
        self.text_channels = []
        self._by_id = {}

    async def create_category(self, name, **kwargs):
        cat = FakeCategory(name)
        self.categories.append(cat)
        self._by_id[cat.id] = cat
        return cat

    async def create_text_channel(self, name, *, category=None, topic=discord.utils.MISSING, **kwargs):
        ch = FakeChannel(name, category=category)
        if topic is not discord.utils.MISSING:
            ch.topic = topic
        self.text_channels.append(ch)
        self._by_id[ch.id] = ch
        return ch

    def get_channel(self, channel_id):
        return self._by_id.get(channel_id)


ACTORS_YAML = """
actors:
  - id: weil
    backend: fake
  - id: hugo
    backend: fake
"""


@pytest.fixture
async def store(tmp_path):
    s = await RoomStore.open(tmp_path / "rooms.sqlite3")
    yield s
    await s.close()


@pytest.fixture
def registry(tmp_path):
    path = tmp_path / "actors.yaml"
    path.write_text(ACTORS_YAML)
    return ActorRegistry.load(path)


def by_name(channels, name):
    return next((c for c in channels if c.name == name), None)


def test_channel_name_for_sanitizes_to_a_valid_discord_name():
    assert channel_name_for("weil") == "weil"
    assert channel_name_for("weil-opus-4.8") == "weil-opus-4-8"
    assert channel_name_for("Weird__Name..here") == "weird__name-here"
    assert channel_name_for("...") == "actor"


@pytest.mark.asyncio
async def test_sync_creates_the_full_structure_from_scratch(store, registry):
    guild = FakeGuild()
    report = await sync_guild(guild, registry, store)

    assert {c.name for c in guild.categories} == {VOICES_CATEGORY, ROOMS_CATEGORY}
    foyer = by_name(guild.text_channels, FOYER_NAME)
    models = by_name(guild.text_channels, MODELS_NAME)
    rooms = by_name(guild.text_channels, ROOMS_CHANNEL)
    assert foyer and foyer.category is None
    assert models and models.category is None
    voices_cat = by_name(guild.categories, VOICES_CATEGORY)
    assert rooms.category.name == ROOMS_CATEGORY

    weil_channel = by_name(guild.text_channels, "weil")
    hugo_channel = by_name(guild.text_channels, "hugo")
    assert weil_channel.category is voices_cat
    assert hugo_channel.category is voices_cat
    assert report.created == ["weil", "hugo"]

    # each actor got a bound "reply" room with itself as the sole participant
    binding = await store.get_voice_channel("weil")
    room = await store.get_room(binding.room_id)
    assert room.turn_policy == "reply"
    assert room.discord_channel_id == str(weil_channel.id)
    participants = await store.list_participants(room.id)
    assert [p.actor_id for p in participants] == ["weil"]


@pytest.mark.asyncio
async def test_sync_is_idempotent(store, registry):
    guild = FakeGuild()
    await sync_guild(guild, registry, store)
    channel_count = len(guild.text_channels)
    category_count = len(guild.categories)

    report = await sync_guild(guild, registry, store)

    assert len(guild.text_channels) == channel_count
    assert len(guild.categories) == category_count
    assert report.created == []
    assert report.renamed == []
    assert report.archived == []


@pytest.mark.asyncio
async def test_sync_renames_a_drifted_channel_back_preserving_identity(store, registry):
    guild = FakeGuild()
    await sync_guild(guild, registry, store)
    weil_channel = by_name(guild.text_channels, "weil")
    await weil_channel.edit(name="oops-renamed-by-a-human")

    report = await sync_guild(guild, registry, store)

    assert weil_channel.name == "weil"  # same object, renamed back — history preserved
    assert report.renamed == ["weil"]


@pytest.mark.asyncio
async def test_sync_archives_a_channel_for_a_removed_actor_without_deleting_it(store, registry, tmp_path):
    guild = FakeGuild()
    await sync_guild(guild, registry, store)
    weil_channel = by_name(guild.text_channels, "weil")

    reduced_path = tmp_path / "reduced.yaml"
    reduced_path.write_text("actors:\n  - id: hugo\n    backend: fake\n")
    reduced_registry = ActorRegistry.load(reduced_path)

    report = await sync_guild(guild, reduced_registry, store)

    assert weil_channel.name == "archived-weil"
    assert weil_channel.category.name == ARCHIVED_CATEGORY
    assert report.archived == ["weil"]
    # it's still the same channel object — never deleted/recreated
    assert weil_channel in guild.text_channels


@pytest.mark.asyncio
async def test_sync_unarchives_a_returning_actor(store, registry, tmp_path):
    guild = FakeGuild()
    await sync_guild(guild, registry, store)
    weil_channel = by_name(guild.text_channels, "weil")

    reduced_path = tmp_path / "reduced.yaml"
    reduced_path.write_text("actors:\n  - id: hugo\n    backend: fake\n")
    reduced_registry = ActorRegistry.load(reduced_path)
    await sync_guild(guild, reduced_registry, store)
    assert weil_channel.name == "archived-weil"

    report = await sync_guild(guild, registry, store)  # weil is back

    assert weil_channel.name == "weil"
    voices_cat = by_name(guild.categories, VOICES_CATEGORY)
    assert weil_channel.category is voices_cat
    assert report.unarchived == ["weil"]


@pytest.mark.asyncio
async def test_sync_recreates_a_channel_deleted_out_from_under_it(store, registry):
    guild = FakeGuild()
    await sync_guild(guild, registry, store)
    weil_channel = by_name(guild.text_channels, "weil")
    guild.text_channels.remove(weil_channel)
    del guild._by_id[weil_channel.id]

    report = await sync_guild(guild, registry, store)

    assert "weil" in report.created
    assert by_name(guild.text_channels, "weil") is not None
