import pytest

from discord_bot.backends.base import GenerationError
from discord_bot.backends.fake import FakeBackend
from discord_bot.backends.router import BackendRouter
from discord_bot.registry import ActorRegistry


def registry(tmp_path, actors_yaml):
    path = tmp_path / "actors.yaml"
    path.write_text(actors_yaml)
    return ActorRegistry.load(path)


@pytest.mark.asyncio
async def test_router_dispatches_to_the_actors_configured_backend(tmp_path):
    reg = registry(tmp_path, """
actors:
  - id: a
    backend: fake
  - id: b
    backend: fake
""")
    calls = []

    class RecordingBackend(FakeBackend):
        async def generate(self, actor, participant_id, transcript):
            calls.append((actor.id, participant_id))
            return await super().generate(actor, participant_id, transcript)

    router = BackendRouter(reg, {"fake": RecordingBackend()})
    reply = await router.generate("a", "a", [])
    assert calls == [("a", "a")]
    assert reply


@pytest.mark.asyncio
async def test_router_raises_when_no_backend_is_registered_for_the_actors_backend(tmp_path):
    reg = registry(tmp_path, """
actors:
  - id: a
    backend: fake
""")
    router = BackendRouter(reg, {})  # no backends wired up
    with pytest.raises(GenerationError, match="no backend registered"):
        await router.generate("a", "a", [])


@pytest.mark.asyncio
async def test_router_raises_for_unknown_actor(tmp_path):
    reg = registry(tmp_path, """
actors:
  - id: a
    backend: fake
""")
    router = BackendRouter(reg, {"fake": FakeBackend()})
    with pytest.raises(KeyError):
        await router.generate("nonexistent", "nonexistent", [])


@pytest.mark.asyncio
async def test_adding_a_new_backend_requires_no_router_change(tmp_path):
    """The router only ever looks up `backends[actor.backend]` — proving a
    third backend can be wired in purely as data, no router edits."""
    reg = registry(tmp_path, """
actors:
  - id: a
    backend: made-up-backend
""")
    router = BackendRouter(reg, {"made-up-backend": FakeBackend()})
    reply = await router.generate("a", "a", [])
    assert reply
