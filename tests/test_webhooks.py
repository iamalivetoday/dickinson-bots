from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import discord
import pytest

from discord_bot.discord_app.webhooks import WEBHOOK_NAME, WebhookManager
from discord_bot.persistence.store import RoomStore


@pytest.fixture
async def store(tmp_path):
    s = await RoomStore.open(tmp_path / "rooms.sqlite3")
    yield s
    await s.close()


def fake_channel(channel_id, *, spec=discord.TextChannel):
    channel = Mock(spec=spec)
    channel.id = channel_id
    return channel


@pytest.mark.asyncio
async def test_get_or_create_reuses_a_cached_binding_without_recreating(store):
    await store.save_webhook_binding("111", "999", "tok-abc")
    channel = fake_channel(111)
    channel.create_webhook = AsyncMock()

    manager = WebhookManager(store, client=Mock())
    webhook = await manager.get_or_create(channel)

    channel.create_webhook.assert_not_called()
    assert isinstance(webhook, discord.Webhook)
    assert webhook.id == 999


@pytest.mark.asyncio
async def test_get_or_create_creates_and_persists_when_no_binding_exists(store):
    channel = fake_channel(222)
    channel.create_webhook = AsyncMock(return_value=SimpleNamespace(id=555, token="tok-xyz"))

    manager = WebhookManager(store, client=Mock())
    webhook = await manager.get_or_create(channel)

    channel.create_webhook.assert_awaited_once_with(name=WEBHOOK_NAME)
    assert webhook.id == 555
    binding = await store.get_webhook_binding("222")
    assert binding.webhook_id == "555"
    assert binding.webhook_token == "tok-xyz"


@pytest.mark.asyncio
async def test_get_or_create_for_a_thread_uses_its_parents_webhook(store):
    parent = fake_channel(333)
    parent.create_webhook = AsyncMock(return_value=SimpleNamespace(id=1, token="t"))
    thread = fake_channel(444, spec=discord.Thread)
    thread.parent = parent

    manager = WebhookManager(store, client=Mock())
    await manager.get_or_create(thread)

    parent.create_webhook.assert_awaited_once()
    assert await store.get_webhook_binding("444") is None
    assert (await store.get_webhook_binding("333")).webhook_id == "1"


@pytest.mark.asyncio
async def test_send_as_splits_content_and_posts_each_chunk_under_the_actor(store):
    manager = WebhookManager(store, client=Mock())
    fake_webhook = SimpleNamespace(send=AsyncMock(return_value="sent"))
    manager.get_or_create = AsyncMock(return_value=fake_webhook)

    long_text = "word " * 1000
    channel = fake_channel(1)
    result = await manager.send_as(
        channel, username="weil", avatar_url="https://x/y.png", content=long_text
    )

    assert len(result) > 1
    assert fake_webhook.send.call_count == len(result)
    for call in fake_webhook.send.call_args_list:
        assert call.kwargs["username"] == "weil"
        assert call.kwargs["avatar_url"] == "https://x/y.png"
        assert call.kwargs["wait"] is True
        assert call.kwargs["thread"] is discord.utils.MISSING


@pytest.mark.asyncio
async def test_send_as_targets_the_thread_when_posting_into_one(store):
    manager = WebhookManager(store, client=Mock())
    fake_webhook = SimpleNamespace(send=AsyncMock(return_value="sent"))
    manager.get_or_create = AsyncMock(return_value=fake_webhook)

    thread = fake_channel(9, spec=discord.Thread)
    await manager.send_as(thread, username="weil", avatar_url=None, content="hi")

    assert fake_webhook.send.call_args.kwargs["thread"] is thread
