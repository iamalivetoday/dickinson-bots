import pytest

from discord_bot.persistence.models import display_names
from discord_bot.persistence.store import (
    ParticipantNotFoundError,
    RoomNotFoundError,
    RoomStore,
    TooManyInstancesError,
)


@pytest.fixture
async def store(tmp_path):
    s = await RoomStore.open(tmp_path / "rooms.sqlite3")
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_create_and_get_room_roundtrip(store):
    room = await store.create_room(
        "The Salon", "moderated", max_turns=20, discord_guild_id="g1", discord_channel_id="c1"
    )
    fetched = await store.get_room(room.id)
    assert fetched == room
    assert fetched.title == "The Salon"
    assert fetched.turn_policy == "moderated"
    assert fetched.max_turns == 20
    assert fetched.status == "active"


@pytest.mark.asyncio
async def test_get_missing_room_raises(store):
    with pytest.raises(RoomNotFoundError):
        await store.get_room("nonexistent")


@pytest.mark.asyncio
async def test_list_rooms_filters_by_status(store):
    a = await store.create_room("A", "manual")
    b = await store.create_room("B", "manual")
    await store.close_room(a.id)

    assert [r.id for r in await store.list_rooms(status="active")] == [b.id]
    assert {r.id for r in await store.list_rooms(status="closed")} == {a.id}
    assert {r.id for r in await store.list_rooms(status=None)} == {a.id, b.id}


@pytest.mark.asyncio
async def test_bind_discord_channel(store):
    room = await store.create_room("A", "manual")
    await store.bind_discord_channel(room.id, "guild-1", "chan-1")
    fetched = await store.get_room(room.id)
    assert fetched.discord_guild_id == "guild-1"
    assert fetched.discord_channel_id == "chan-1"


@pytest.mark.asyncio
async def test_first_participant_gets_bare_id_second_gets_suffix(store):
    room = await store.create_room("Duel", "round_robin")
    a = await store.add_participant(room.id, "opus-4.8")
    b = await store.add_participant(room.id, "opus-4.8")

    assert a.id == "opus-4.8"
    assert a.instance_suffix == "a"
    assert b.id == "opus-4.8:b"
    assert b.instance_suffix == "b"


@pytest.mark.asyncio
async def test_display_names_only_suffix_when_duplicates_are_active(store):
    room = await store.create_room("Duel", "round_robin")
    a = await store.add_participant(room.id, "opus-4.8")

    solo = await store.list_participants(room.id)
    assert display_names(solo) == {"opus-4.8": "opus-4.8"}

    b = await store.add_participant(room.id, "opus-4.8")
    both = await store.list_participants(room.id)
    assert display_names(both) == {"opus-4.8": "opus-4.8:a", "opus-4.8:b": "opus-4.8:b"}


@pytest.mark.asyncio
async def test_different_actors_never_get_suffixes(store):
    room = await store.create_room("Salon", "round_robin")
    await store.add_participant(room.id, "weil")
    await store.add_participant(room.id, "hugo")
    names = display_names(await store.list_participants(room.id))
    assert names == {"weil": "weil", "hugo": "hugo"}


@pytest.mark.asyncio
async def test_remove_participant_is_soft_and_never_reuses_a_suffix(store):
    room = await store.create_room("Duel", "round_robin")
    a = await store.add_participant(room.id, "opus-4.8")
    await store.remove_participant(room.id, a.id)

    assert await store.list_participants(room.id) == []
    assert len(await store.list_participants(room.id, active_only=False)) == 1

    c = await store.add_participant(room.id, "opus-4.8")
    assert c.instance_suffix == "b"  # not 'a' again


@pytest.mark.asyncio
async def test_remove_unknown_participant_raises(store):
    room = await store.create_room("Duel", "round_robin")
    with pytest.raises(ParticipantNotFoundError):
        await store.remove_participant(room.id, "nobody")


@pytest.mark.asyncio
async def test_more_than_26_instances_of_one_actor_raises(store):
    room = await store.create_room("Crowd", "round_robin")
    for _ in range(26):
        await store.add_participant(room.id, "opus-4.8")
    with pytest.raises(TooManyInstancesError):
        await store.add_participant(room.id, "opus-4.8")


@pytest.mark.asyncio
async def test_messages_are_ordered_by_monotonic_seq(store):
    room = await store.create_room("Salon", "round_robin")
    m1 = await store.append_message(room.id, "user:1", "Madeleine", "hello")
    m2 = await store.append_message(room.id, "weil", "Simone Weil", "hello back")
    m3 = await store.append_message(room.id, "hugo", "Victor Hugo", "and me!")

    assert [m.seq for m in (m1, m2, m3)] == [1, 2, 3]
    fetched = await store.list_messages(room.id)
    assert [m.content for m in fetched] == ["hello", "hello back", "and me!"]


@pytest.mark.asyncio
async def test_message_can_carry_a_discord_message_id_binding(store):
    room = await store.create_room("Salon", "round_robin")
    msg = await store.append_message(
        room.id, "weil", "Simone Weil", "hi", discord_message_id="123456789"
    )
    assert (await store.list_messages(room.id))[0].discord_message_id == "123456789"
    assert msg.discord_message_id == "123456789"


@pytest.mark.asyncio
async def test_turn_state_defaults_and_updates(store):
    room = await store.create_room("Salon", "round_robin")
    state = await store.get_turn_state(room.id)
    assert state.turn_count == 0
    assert state.next_participant_id is None
    assert state.cursor == {}

    updated = await store.update_turn_state(
        room.id, next_participant_id="weil", cursor={"index": 2}
    )
    assert updated.next_participant_id == "weil"
    assert updated.cursor == {"index": 2}

    cleared = await store.update_turn_state(room.id, next_participant_id=None)
    assert cleared.next_participant_id is None
    assert cleared.cursor == {"index": 2}  # untouched when not passed


@pytest.mark.asyncio
async def test_increment_turn_count(store):
    room = await store.create_room("Salon", "round_robin")
    assert await store.increment_turn_count(room.id) == 1
    assert await store.increment_turn_count(room.id) == 2
    assert (await store.get_turn_state(room.id)).turn_count == 2


@pytest.mark.asyncio
async def test_data_survives_a_restart(tmp_path):
    """Simulates a bot restart: close the connection, reopen the same file,
    and confirm rooms/participants/messages/turn state are all still there.
    """
    path = tmp_path / "rooms.sqlite3"
    store1 = await RoomStore.open(path)
    room = await store1.create_room("Salon", "round_robin", max_turns=10)
    p = await store1.add_participant(room.id, "weil")
    await store1.append_message(room.id, p.id, "Simone Weil", "Attention is prayer.")
    await store1.increment_turn_count(room.id)
    await store1.close()

    store2 = await RoomStore.open(path)
    try:
        recovered_room = await store2.get_room(room.id)
        assert recovered_room.title == "Salon"
        assert recovered_room.max_turns == 10

        participants = await store2.list_participants(room.id)
        assert [p.id for p in participants] == ["weil"]

        messages = await store2.list_messages(room.id)
        assert [m.content for m in messages] == ["Attention is prayer."]

        state = await store2.get_turn_state(room.id)
        assert state.turn_count == 1
    finally:
        await store2.close()
