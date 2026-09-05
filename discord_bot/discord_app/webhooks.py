"""Actor identities: every actor posts through one webhook per channel,
under its own username and avatar — never as the bot account.
"""
from __future__ import annotations

import discord

from discord_bot.persistence.store import RoomStore

from .splitting import split_message

WEBHOOK_NAME = "Salon Conductor"


class WebhookManager:
    def __init__(self, store: RoomStore, client: discord.Client):
        self._store = store
        self._client = client

    async def get_or_create(self, channel: discord.abc.GuildChannel) -> discord.Webhook:
        """One webhook per *parent* channel — a thread/forum post reuses its
        parent's webhook (Discord webhooks belong to a channel, not a
        thread) and is targeted per-send via `thread=`."""
        host = channel.parent if isinstance(channel, discord.Thread) else channel
        binding = await self._store.get_webhook_binding(str(host.id))
        if binding is not None:
            return discord.Webhook.partial(
                int(binding.webhook_id), binding.webhook_token, client=self._client
            )
        webhook = await host.create_webhook(name=WEBHOOK_NAME)
        await self._store.save_webhook_binding(str(host.id), str(webhook.id), webhook.token)
        return webhook

    async def send_as(
        self,
        channel: discord.abc.GuildChannel,
        *,
        username: str,
        avatar_url: str | None,
        content: str,
    ) -> list[discord.WebhookMessage]:
        webhook = await self.get_or_create(channel)
        thread = channel if isinstance(channel, discord.Thread) else discord.utils.MISSING
        sent = []
        for chunk in split_message(content):
            sent.append(
                await webhook.send(
                    content=chunk, username=username, avatar_url=avatar_url,
                    thread=thread, wait=True,
                )
            )
        return sent
