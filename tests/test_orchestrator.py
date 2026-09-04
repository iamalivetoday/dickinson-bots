import asyncio

import pytest

from discord_bot.backends.fake import FakeBackend
from discord_bot.backends.router import BackendRouter
from discord_bot.registry import ActorRegistry
from discord_bot.persistence.store import RoomStore
from discord_bot.rooms.orchestrator import (
    RoomBoundsExceededError,
    RoomClosedError,
    RoomFinishedError,
    RoomOrchestrator,
    RoomTimeoutError,
)
from discord_bot.rooms.turn_policies import Trigger


ACTORS_YAML = """
actors:
  - id: weil
    backend: fake
  - id: hugo
    backend: fake
  - id: dostoevsky
    backend: fake
"""


class RecordingBackend(FakeBackend):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.calls = 0

    async def generate(self, actor, participant_id, transcript):
        self.calls += 1
        return await super().generate(actor, participant_id, transcript)


class SlowBackend:
    async def generate(self, actor, participant_id, transcript):
        await asyncio.sleep(0.2)
        return "too slow"


@pytest.fixture
async def env(tmp_path):
    store = await RoomStore.open(tmp_path / "rooms.sqlite3")
    registry_path = tmp_path / "actors.yaml"
    registry_path.write_text(ACTORS_YAML)
    registry = ActorRegistry.load(registry_path)
    backend = RecordingBackend()
    router = BackendRouter(registry, {"fake": backend})
    orchestrator = RoomOrchestrator(store, router)
    yield store, orchestrator, backend
    await store.close()


@pytest.mark.asyncio
async def test_manual_turn_persists_message_and_turn_count(env):
    store, orch, backend = env
    room = await store.create_room("Chat", "manual")
    await store.add_participant(room.id, "weil")

    message = await orch.take_turn(room.id, Trigger(requested_participant_id="weil"))

    assert message.speaker_id == "weil"
    assert message.speaker_name == "weil"
    assert backend.calls == 1
    assert (await store.get_turn_state(room.id)).turn_count == 1
    assert [m.content for m in await store.list_messages(room.id)] == [message.content]


@pytest.mark.asyncio
async def test_one_take_turn_call_generates_exactly_once_no_recursive_chaining(env):
    """Loop prevention: a single explicit take_turn() must never cascade
    into further generation on its own."""
    store, orch, backend = env
    room = await store.create_room("Salon", "round_robin")
    await store.add_participant(room.id, "weil")
    await store.add_participant(room.id, "hugo")

    await orch.take_turn(room.id)

    assert backend.calls == 1
    assert len(await store.list_messages(room.id)) == 1


@pytest.mark.asyncio
async def test_round_robin_cycles_across_separate_take_turn_calls(env):
    store, orch, backend = env
    room = await store.create_room("Salon", "round_robin")
    await store.add_participant(room.id, "weil")
    await store.add_participant(room.id, "hugo")
    await store.add_participant(room.id, "dostoevsky")

    speakers = [(await orch.take_turn(room.id)).speaker_id for _ in range(4)]
    assert speakers == ["weil", "hugo", "dostoevsky", "weil"]


@pytest.mark.asyncio
async def test_round_robin_state_resumes_after_a_fresh_orchestrator_reconnects(tmp_path):
    """Simulates a bot restart mid-round-robin: a brand new RoomOrchestrator
    (and RoomStore) over the same db file must pick up the cursor, not
    restart it."""
    db_path = tmp_path / "rooms.sqlite3"
    registry_path = tmp_path / "actors.yaml"
    registry_path.write_text(ACTORS_YAML)
    registry = ActorRegistry.load(registry_path)

    store1 = await RoomStore.open(db_path)
    orch1 = RoomOrchestrator(store1, BackendRouter(registry, {"fake": FakeBackend()}))
    room = await store1.create_room("Salon", "round_robin")
    await store1.add_participant(room.id, "weil")
    await store1.add_participant(room.id, "hugo")
    first = await orch1.take_turn(room.id)
    assert first.speaker_id == "weil"
    await store1.close()

    store2 = await RoomStore.open(db_path)
    try:
        orch2 = RoomOrchestrator(store2, BackendRouter(registry, {"fake": FakeBackend()}))
        second = await orch2.take_turn(room.id)
        assert second.speaker_id == "hugo"  # continues, doesn't restart at "weil"
    finally:
        await store2.close()


@pytest.mark.asyncio
async def test_moderated_never_immediately_repeats_a_speaker(env):
    import random

    store, orch, backend = env
    room = await store.create_room("Salon", "moderated")
    await store.add_participant(room.id, "weil")
    await store.add_participant(room.id, "hugo")
    await store.add_participant(room.id, "dostoevsky")

    rng = random.Random(7)
    speakers = []
    for _ in range(10):
        msg = await orch.take_turn(room.id, rng=rng)
        speakers.append(msg.speaker_id)
    for prev, nxt in zip(speakers, speakers[1:]):
        assert prev != nxt


@pytest.mark.asyncio
async def test_debate_runs_fixed_rounds_then_finishes_and_closes_the_room(env):
    store, orch, backend = env
    room = await store.create_room(
        "Debate", "debate", turn_policy_config={"order": ["weil", "hugo"], "rounds": 2}
    )
    await store.add_participant(room.id, "weil")
    await store.add_participant(room.id, "hugo")

    speakers = [(await orch.take_turn(room.id)).speaker_id for _ in range(4)]
    assert speakers == ["weil", "hugo", "weil", "hugo"]

    with pytest.raises(RoomFinishedError):
        await orch.take_turn(room.id)
    assert (await store.get_room(room.id)).status == "closed"

    with pytest.raises(RoomClosedError):
        await orch.take_turn(room.id)


@pytest.mark.asyncio
async def test_max_turns_bound_closes_the_room(env):
    store, orch, backend = env
    room = await store.create_room("Chat", "manual", max_turns=2)
    await store.add_participant(room.id, "weil")

    await orch.take_turn(room.id, Trigger(requested_participant_id="weil"))
    await orch.take_turn(room.id, Trigger(requested_participant_id="weil"))
    with pytest.raises(RoomBoundsExceededError, match="max_turns"):
        await orch.take_turn(room.id, Trigger(requested_participant_id="weil"))
    assert (await store.get_room(room.id)).status == "closed"


@pytest.mark.asyncio
async def test_max_messages_bound_closes_the_room(env):
    store, orch, backend = env
    room = await store.create_room("Chat", "manual", max_messages=1)
    await store.add_participant(room.id, "weil")

    await orch.take_turn(room.id, Trigger(requested_participant_id="weil"))
    with pytest.raises(RoomBoundsExceededError, match="max_messages"):
        await orch.take_turn(room.id, Trigger(requested_participant_id="weil"))


@pytest.mark.asyncio
async def test_max_tokens_bound_closes_the_room(env):
    store, orch, backend = env
    room = await store.create_room("Chat", "manual", max_tokens=3)
    await store.add_participant(room.id, "weil")

    await orch.take_turn(room.id, Trigger(requested_participant_id="weil"))  # several words -> over budget
    with pytest.raises(RoomBoundsExceededError, match="max_tokens"):
        await orch.take_turn(room.id, Trigger(requested_participant_id="weil"))


@pytest.mark.asyncio
async def test_generation_timeout_raises_and_does_not_advance_turn_state(tmp_path):
    store = await RoomStore.open(tmp_path / "rooms.sqlite3")
    registry_path = tmp_path / "actors.yaml"
    registry_path.write_text(ACTORS_YAML)
    registry = ActorRegistry.load(registry_path)
    router = BackendRouter(registry, {"fake": SlowBackend()})
    orch = RoomOrchestrator(store, router)

    room = await store.create_room("Chat", "manual", timeout_seconds=0.05)
    await store.add_participant(room.id, "weil")

    with pytest.raises(RoomTimeoutError):
        await orch.take_turn(room.id, Trigger(requested_participant_id="weil"))

    assert (await store.get_turn_state(room.id)).turn_count == 0
    assert await store.list_messages(room.id) == []
    await store.close()


@pytest.mark.asyncio
async def test_closed_room_rejects_further_turns(env):
    store, orch, backend = env
    room = await store.create_room("Chat", "manual")
    await store.close_room(room.id)
    with pytest.raises(RoomClosedError):
        await orch.take_turn(room.id, Trigger(requested_participant_id="weil"))
