import pytest

from discord_bot.backends.fake import FakeBackend
from discord_bot.backends.router import BackendRouter
from discord_bot.discord_app.router import IncomingMessage, MessageRouter
from discord_bot.persistence.store import RoomStore
from discord_bot.registry import ActorRegistry
from discord_bot.rooms.orchestrator import RoomOrchestrator

ACTORS_YAML = """
actors:
  - id: weil
    backend: fake
    avatar: https://example.test/weil.png
  - id: opus-4.8
    backend: fake
"""


def incoming(**overrides):
    defaults = dict(
        guild_id="guild-1", channel_id="chan-1", author_id="user-1",
        author_display_name="Madeleine", author_is_bot=False, content="hello",
    )
    defaults.update(overrides)
    return IncomingMessage(**defaults)


@pytest.fixture
async def env(tmp_path):
    store = await RoomStore.open(tmp_path / "rooms.sqlite3")
    registry_path = tmp_path / "actors.yaml"
    registry_path.write_text(ACTORS_YAML)
    registry = ActorRegistry.load(registry_path)
    backend_router = BackendRouter(registry, {"fake": FakeBackend()})
    orchestrator = RoomOrchestrator(store, backend_router)
    router = MessageRouter(store, orchestrator, registry)
    yield store, router
    await store.close()


async def bind_voice_room(store, actor_id, channel_id="chan-1"):
    room = await store.create_room(f"{actor_id} voice channel", "reply")
    await store.add_participant(room.id, actor_id)
    await store.bind_discord_channel(room.id, "guild-1", channel_id)
    return room


@pytest.mark.asyncio
async def test_bot_and_webhook_messages_are_never_routed_loop_prevention(env):
    store, router = env
    await bind_voice_room(store, "weil")

    outcome = await router.handle_message(incoming(author_is_bot=True))

    assert outcome.posted is False
    assert (await store.list_messages((await store.list_rooms())[0].id)) == []


@pytest.mark.asyncio
async def test_message_in_an_unbound_channel_is_ignored(env):
    store, router = env
    outcome = await router.handle_message(incoming(channel_id="some-other-channel"))
    assert outcome.posted is False
    assert outcome.error is None


@pytest.mark.asyncio
async def test_guild_restriction(env):
    store, router = env
    await bind_voice_room(store, "weil")
    restricted = MessageRouter(
        store, router._orchestrator, router._registry, allowed_guild_id="only-this-guild"
    )
    outcome = await restricted.handle_message(incoming(guild_id="guild-1"))
    assert outcome.posted is False


@pytest.mark.asyncio
async def test_user_restriction(env):
    store, router = env
    await bind_voice_room(store, "weil")
    restricted = MessageRouter(
        store, router._orchestrator, router._registry, allowed_user_ids=frozenset({"someone-else"})
    )
    outcome = await restricted.handle_message(incoming(author_id="user-1"))
    assert outcome.posted is False


@pytest.mark.asyncio
async def test_voice_channel_message_routes_to_its_sole_actor(env):
    store, router = env
    room = await bind_voice_room(store, "weil")

    outcome = await router.handle_message(incoming(content="What is attention?"))

    assert outcome.posted is True
    assert outcome.actor_id == "weil"
    assert outcome.display_name == "weil"
    assert outcome.avatar_url == "https://example.test/weil.png"
    assert outcome.content

    messages = await store.list_messages(room.id)
    assert [m.speaker_id for m in messages] == ["user:user-1", "weil"]
    assert messages[0].content == "What is attention?"


@pytest.mark.asyncio
async def test_multi_actor_room_requires_an_explicit_mention(env):
    store, router = env
    room = await store.create_room("Debate room", "reply")
    await store.add_participant(room.id, "weil")
    await store.add_participant(room.id, "opus-4.8")
    await store.bind_discord_channel(room.id, "guild-1", "chan-1")

    unaddressed = await router.handle_message(incoming(content="thoughts, anyone?"))
    assert unaddressed.posted is False
    assert unaddressed.error == "no participant was addressed"

    addressed = await router.handle_message(incoming(content="@weil what do you think?"))
    assert addressed.posted is True
    assert addressed.actor_id == "weil"


@pytest.mark.asyncio
async def test_multi_actor_room_mention_uses_disambiguated_display_name(env):
    store, router = env
    room = await store.create_room("Duel room", "reply")
    a = await store.add_participant(room.id, "opus-4.8")
    b = await store.add_participant(room.id, "opus-4.8")
    await store.bind_discord_channel(room.id, "guild-1", "chan-1")

    outcome = await router.handle_message(incoming(content=f"@{b.id} your move"))
    assert outcome.posted is True
    assert outcome.display_name == "opus-4.8:b"

    messages = await store.list_messages(room.id)
    assert messages[-1].speaker_id == b.id


@pytest.mark.asyncio
async def test_orchestrator_errors_surface_without_crashing(env):
    store, router = env
    room = await store.create_room("weil voice channel", "reply", max_turns=1)
    await store.add_participant(room.id, "weil")
    await store.bind_discord_channel(room.id, "guild-1", "chan-1")

    first = await router.handle_message(incoming(content="hi"))
    assert first.posted is True

    second = await router.handle_message(incoming(content="again?"))
    assert second.posted is False
    assert "max_turns" in second.error


@pytest.mark.asyncio
async def test_messages_in_a_closed_rooms_channel_are_ignored_not_crashed(env):
    store, router = env
    room = await bind_voice_room(store, "weil")
    await store.close_room(room.id)

    outcome = await router.handle_message(incoming(content="hello?"))
    assert outcome.posted is False
