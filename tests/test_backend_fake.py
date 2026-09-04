import pytest

from discord_bot.backends.base import TranscriptMessage
from discord_bot.backends.fake import FakeBackend


@pytest.mark.asyncio
async def test_fake_backend_is_deterministic_given_same_transcript():
    backend = FakeBackend()
    actor = object()  # FakeBackend never inspects the actor
    transcript = [TranscriptMessage("user:1", "Madeleine", "hello there")]

    reply1 = await backend.generate(actor, "weil", transcript)
    reply2 = await backend.generate(actor, "weil", transcript)
    assert reply1 == reply2


@pytest.mark.asyncio
async def test_fake_backend_counts_the_participants_own_prior_turns():
    backend = FakeBackend()
    transcript = [
        TranscriptMessage("weil", "Simone Weil", "first"),
        TranscriptMessage("hugo", "Victor Hugo", "interjection"),
        TranscriptMessage("weil", "Simone Weil", "second"),
    ]
    reply = await backend.generate(object(), "weil", transcript)
    assert "turn 3" in reply


@pytest.mark.asyncio
async def test_fake_backend_never_touches_network_or_disk(monkeypatch):
    import socket

    def _boom(*a, **kw):
        raise AssertionError("FakeBackend must not open sockets")

    monkeypatch.setattr(socket, "socket", _boom)
    backend = FakeBackend()
    await backend.generate(object(), "weil", [])
