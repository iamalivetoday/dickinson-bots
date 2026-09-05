"""The Discord bot: thin glue between discord.py and the tested logic in
router.py / webhooks.py. Almost nothing here is unit tested beyond wiring,
because almost nothing here is logic — see discord_app/router.py for that.
"""
from __future__ import annotations

import logging

import discord

from discord_bot.backends.router import BackendRouter
from discord_bot.persistence.store import RoomStore
from discord_bot.registry import ActorRegistry
from discord_bot.rooms.orchestrator import RoomOrchestrator

from .config import Settings
from .router import IncomingMessage, MessageRouter
from .webhooks import WebhookManager

log = logging.getLogger(__name__)

INTENTS = discord.Intents.default()
INTENTS.message_content = True  # required to read plain messages in voice/room channels
INTENTS.guilds = True
INTENTS.members = True  # for @-mention -> display name resolution


class SalonBot(discord.Client):
    def __init__(
        self,
        settings: Settings,
        registry: ActorRegistry,
        store: RoomStore,
        backend_router: BackendRouter,
    ):
        super().__init__(intents=INTENTS)
        self.settings = settings
        self.registry = registry
        self.store = store
        self.backend_router = backend_router
        self.orchestrator = RoomOrchestrator(store, backend_router)
        self.webhooks = WebhookManager(store, self)
        self.message_router = MessageRouter(
            store, self.orchestrator, registry,
            allowed_guild_id=settings.allowed_guild_id,
            allowed_user_ids=settings.allowed_user_ids,
        )
        self.tree = discord.app_commands.CommandTree(self)

    async def setup_hook(self) -> None:
        if self.settings.allowed_guild_id:
            guild = discord.Object(id=int(self.settings.allowed_guild_id))
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
        else:
            await self.tree.sync()

    async def on_ready(self) -> None:
        log.info("logged in as %s (guild restriction: %s)", self.user, self.settings.allowed_guild_id)

    async def on_message(self, message: discord.Message) -> None:
        # Loop prevention: never treat a webhook post (every actor's own
        # reply) or the bot's own message as a new human turn.
        if message.webhook_id is not None or message.author == self.user:
            return

        incoming = IncomingMessage(
            guild_id=str(message.guild.id) if message.guild else None,
            channel_id=str(message.channel.id),
            author_id=str(message.author.id),
            author_display_name=message.author.display_name,
            author_is_bot=message.author.bot,
            content=message.content,
        )
        async with message.channel.typing():
            outcome = await self.message_router.handle_message(incoming)

        if outcome.posted:
            await self.webhooks.send_as(
                message.channel, username=outcome.display_name,
                avatar_url=outcome.avatar_url, content=outcome.content,
            )
        elif outcome.error:
            log.warning("room error in channel %s: %s", message.channel.id, outcome.error)
