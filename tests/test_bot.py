"""Wiring tests for SalonBot.on_message — everything it depends on
(message_router, webhooks) is mocked, since the actual logic is already
covered by test_router.py and test_webhooks.py. This just proves on_message
calls the right thing with the right arguments, and short-circuits on
webhook/self messages (loop prevention) before ever asking the router.
"""
from unittest.mock import AsyncMock, Mock

import discord
import pytest

from discord_bot.discord_app.bot import SalonBot
from discord_bot.discord_app.config import Settings
from discord_bot.discord_app.router import RoutingOutcome


def make_bot():
    settings = Settings(
        discord_bot_token="tok", anthropic_api_key=None,
        actors_config_path="config/actors.yaml", room_db_path="data/rooms.sqlite3",
        allowed_guild_id=None, allowed_user_ids=frozenset(),
        local_max_cached_bases=1, local_device="cpu",
    )
    bot = SalonBot(settings, registry=Mock(), store=Mock(), backend_router=Mock())
    bot.message_router = Mock()
    bot.webhooks = Mock()
    return bot


def fake_message(*, webhook_id=None, author_is_bot=False, is_self=False, content="hi"):
    message = Mock(spec=discord.Message)
    message.webhook_id = webhook_id
    message.guild = Mock(id=123)
    message.channel = Mock()
    message.channel.id = 456
    message.channel.typing = Mock(return_value=_NullAsyncCtx())
    message.author = Mock(id=789, bot=author_is_bot, display_name="Madeleine")
    message.content = content
    return message, is_self


class _NullAsyncCtx:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc):
        return False


def test_every_command_registers_on_the_tree():
    """Guards against a command silently failing to attach (a decorator
    typo, a bad annotation discord.py rejects at registration time)."""
    bot = make_bot()
    assert sorted(c.name for c in bot.tree.get_commands()) == [
        "chat", "debate", "models", "next", "room", "salon", "sync-models",
    ]
    room = next(c for c in bot.tree.get_commands() if c.name == "room")
    assert sorted(s.name for s in room.commands) == ["add", "create", "remove", "status"]


@pytest.mark.asyncio
async def test_webhook_messages_never_reach_the_router():
    bot = make_bot()
    bot.message_router.handle_message = AsyncMock()
    message, _ = fake_message(webhook_id=999)

    await bot.on_message(message)

    bot.message_router.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_the_bots_own_messages_never_reach_the_router(monkeypatch):
    bot = make_bot()
    bot.message_router.handle_message = AsyncMock()
    message, _ = fake_message()
    # discord.Client.user is a read-only property backed by connection
    # state — patch it at the class level to simulate this message being
    # the bot's own.
    monkeypatch.setattr(discord.Client, "user", property(lambda self: message.author))

    await bot.on_message(message)

    bot.message_router.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_a_posted_outcome_is_sent_through_the_actors_webhook():
    bot = make_bot()
    bot.message_router.handle_message = AsyncMock(
        return_value=RoutingOutcome(
            posted=True, actor_id="weil", display_name="weil",
            avatar_url="https://x/y.png", content="Attention is the rarest form of generosity.",
        )
    )
    bot.webhooks.send_as = AsyncMock()
    message, _ = fake_message(content="What is attention?")

    await bot.on_message(message)

    bot.message_router.handle_message.assert_awaited_once()
    incoming = bot.message_router.handle_message.call_args.args[0]
    assert incoming.channel_id == "456"
    assert incoming.author_id == "789"
    assert incoming.content == "What is attention?"

    bot.webhooks.send_as.assert_awaited_once_with(
        message.channel, username="weil", avatar_url="https://x/y.png",
        content="Attention is the rarest form of generosity.",
    )


@pytest.mark.asyncio
async def test_an_unposted_outcome_never_touches_the_webhook():
    bot = make_bot()
    bot.message_router.handle_message = AsyncMock(return_value=RoutingOutcome(posted=False))
    bot.webhooks.send_as = AsyncMock()
    message, _ = fake_message()

    await bot.on_message(message)

    bot.webhooks.send_as.assert_not_called()
