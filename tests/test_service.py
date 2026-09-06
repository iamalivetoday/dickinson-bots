import pytest

from discord_bot.backends.fake import FakeBackend
from discord_bot.backends.router import BackendRouter
from discord_bot.discord_app.service import (
    MAX_ACTORS_PER_ROOM,
    MAX_ROUNDS,
    RoomService,
    ServiceError,
)
from discord_bot.persistence.store import RoomStore
from discord_bot.registry import ActorRegistry
from discord_bot.rooms.orchestrator import RoomOrchestrator

ACTORS_YAML = """
actors:
  - id: weil
    persona: weil
    backend: fake
    avatar: https://example.test/weil.png
  - id: hugo
    persona: hugo
    backend: fake
  - id: opus-4.8
    backend: fake
"""


@pytest.fixture
async def service(tmp_path):
    store = await RoomStore.open(tmp_path / "rooms.sqlite3")
    path = tmp_path / "actors.yaml"
    path.write_text(ACTORS_YAML)
    registry = ActorRegistry.load(path)
    orchestrator = RoomOrchestrator(store, BackendRouter(registry, {"fake": FakeBackend()}))
    yield RoomService(store, orchestrator, registry), store
    await store.close()


# -- directory --------------------------------------------------------------

@pytest.mark.asyncio
async def test_describe_actors_lists_every_actor_and_flags_persona_less_ones(service):
    svc, _ = service
    text = svc.describe_actors()
    assert "`weil`" in text and "`hugo`" in text and "`opus-4.8`" in text
    assert "base model, no persona" in text  # opus-4.8 has no persona


@pytest.mark.asyncio
async def test_actor_ids_are_sorted(service):
    svc, _ = service
    assert svc.actor_ids() == ["hugo", "opus-4.8", "weil"]


# -- rooms ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_room_rejects_an_unknown_policy(service):
    svc, _ = service
    with pytest.raises(ServiceError, match="unknown turn policy"):
        await svc.create_room("Bad", "telepathy")


@pytest.mark.asyncio
async def test_add_actor_rejects_an_unknown_actor(service):
    svc, _ = service
    room = await svc.create_room("Salon", "round_robin")
    with pytest.raises(ServiceError, match="no such actor"):
        await svc.add_actor(room.id, "nietzsche")


@pytest.mark.asyncio
async def test_the_same_actor_can_be_added_twice_with_distinct_participant_ids(service):
    svc, _ = service
    room = await svc.create_room("Duel", "round_robin")
    a = await svc.add_actor(room.id, "opus-4.8")
    b = await svc.add_actor(room.id, "opus-4.8")
    assert a.id == "opus-4.8"
    assert b.id == "opus-4.8:b"


@pytest.mark.asyncio
async def test_room_participant_cap_is_enforced(service):
    svc, _ = service
    room = await svc.create_room("Crowd", "round_robin")
    for _ in range(MAX_ACTORS_PER_ROOM):
        await svc.add_actor(room.id, "weil")
    with pytest.raises(ServiceError, match="at most"):
        await svc.add_actor(room.id, "weil")


@pytest.mark.asyncio
async def test_remove_unknown_participant_raises_service_error(service):
    svc, _ = service
    room = await svc.create_room("Salon", "round_robin")
    with pytest.raises(ServiceError, match="not a participant"):
        await svc.remove_participant(room.id, "ghost")


@pytest.mark.asyncio
async def test_describe_room_shows_duplicate_instances_disambiguated(service):
    svc, _ = service
    room = await svc.create_room("Duel", "round_robin")
    await svc.add_actor(room.id, "opus-4.8")
    await svc.add_actor(room.id, "opus-4.8")

    text = await svc.describe_room(room.id)
    assert "`opus-4.8:a`" in text
    assert "`opus-4.8:b`" in text
    assert "round_robin" in text


@pytest.mark.asyncio
async def test_describe_unknown_room_raises(service):
    svc, _ = service
    with pytest.raises(ServiceError, match="no such room"):
        await svc.describe_room("nope")


# -- turns ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_take_turn_returns_an_actor_message_with_its_avatar(service):
    svc, _ = service
    room = await svc.create_room("Chat", "manual")
    await svc.add_actor(room.id, "weil")

    message = await svc.take_turn(room.id, "weil")
    assert message.participant_id == "weil"
    assert message.display_name == "weil"
    assert message.avatar_url == "https://example.test/weil.png"
    assert message.content


@pytest.mark.asyncio
async def test_take_turn_surfaces_a_policy_error_as_a_service_error(service):
    svc, _ = service
    room = await svc.create_room("Chat", "manual")
    await svc.add_actor(room.id, "weil")
    with pytest.raises(ServiceError, match="requires an explicit participant"):
        await svc.take_turn(room.id, None)  # manual policy needs a target


# -- salon / debate -----------------------------------------------------------

@pytest.mark.asyncio
async def test_prepare_conversation_seeds_the_topic_and_bounds_the_room(service):
    svc, store = service
    room = await svc.prepare_conversation(
        "Salon: suffering", ["weil", "hugo"], 2, policy="round_robin", topic="Is suffering necessary?"
    )
    assert room.max_turns == 4  # 2 rounds x 2 participants
    messages = await store.list_messages(room.id)
    assert messages[0].content == "Is suffering necessary?"
    assert [p.actor_id for p in await store.list_participants(room.id)] == ["weil", "hugo"]


@pytest.mark.asyncio
async def test_prepare_conversation_allows_the_same_actor_twice(service):
    svc, store = service
    room = await svc.prepare_conversation(
        "Two opuses", ["opus-4.8", "opus-4.8", "weil"], 1,
        policy="round_robin", topic="Do you two differ?",
    )
    ids = [p.id for p in await store.list_participants(room.id)]
    assert ids == ["opus-4.8", "opus-4.8:b", "weil"]


@pytest.mark.asyncio
async def test_prepare_conversation_validates_actors_rounds_and_count(service):
    svc, _ = service
    with pytest.raises(ServiceError, match="at least one actor"):
        await svc.prepare_conversation("x", [], 1, policy="round_robin", topic="t")
    with pytest.raises(ServiceError, match="no such actor"):
        await svc.prepare_conversation("x", ["nobody"], 1, policy="round_robin", topic="t")
    with pytest.raises(ServiceError, match="rounds must be between"):
        await svc.prepare_conversation("x", ["weil"], MAX_ROUNDS + 1, policy="round_robin", topic="t")
    with pytest.raises(ServiceError, match="at most"):
        await svc.prepare_conversation(
            "x", ["weil"] * (MAX_ACTORS_PER_ROOM + 1), 1, policy="round_robin", topic="t"
        )


@pytest.mark.asyncio
async def test_debate_gets_its_fixed_order_and_rounds_from_the_participants(service):
    svc, store = service
    room = await svc.prepare_conversation(
        "Debate: art", ["weil", "hugo"], 3, policy="debate", topic="Is art a duty?"
    )
    assert room.turn_policy == "debate"
    assert room.turn_policy_config == {"order": ["weil", "hugo"], "rounds": 3}


@pytest.mark.asyncio
async def test_run_conversation_yields_exactly_the_scheduled_turns_in_order(service):
    svc, store = service
    room = await svc.prepare_conversation(
        "Salon", ["weil", "hugo"], 2, policy="round_robin", topic="Why beauty?"
    )
    speakers = [m.participant_id async for m in svc.run_conversation(room.id, 4)]
    assert speakers == ["weil", "hugo", "weil", "hugo"]


@pytest.mark.asyncio
async def test_run_conversation_stops_early_at_the_rooms_bound_without_raising(service):
    svc, store = service
    room = await svc.prepare_conversation(
        "Salon", ["weil", "hugo"], 1, policy="round_robin", topic="Why beauty?"
    )  # max_turns == 2
    # ask for far more turns than the room allows
    produced = [m async for m in svc.run_conversation(room.id, 50)]
    assert len(produced) == 2
    assert (await store.get_room(room.id)).status == "closed"


@pytest.mark.asyncio
async def test_a_debate_concludes_on_its_own_after_its_rounds(service):
    svc, store = service
    room = await svc.prepare_conversation(
        "Debate", ["weil", "hugo"], 2, policy="debate", topic="Is art a duty?"
    )
    produced = [m.participant_id async for m in svc.run_conversation(room.id, 100)]
    assert produced == ["weil", "hugo", "weil", "hugo"]
    assert (await store.get_room(room.id)).status == "closed"
