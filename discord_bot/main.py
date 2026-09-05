"""Entrypoint: `python -m discord_bot.main` (from the repo root, with the
project .venv active) starts the bot for real, against real Discord.
"""
from __future__ import annotations

import asyncio
import logging

from anthropic import AsyncAnthropic

from discord_bot.backends.anthropic_backend import AnthropicBackend
from discord_bot.backends.local import LocalBackend
from discord_bot.backends.router import BackendRouter
from discord_bot.persistence.store import RoomStore
from discord_bot.registry import ActorRegistry

from .discord_app.bot import SalonBot
from .discord_app.config import load_settings

logging.basicConfig(level=logging.INFO)


async def run() -> None:
    settings = load_settings()
    registry = ActorRegistry.load(settings.actors_config_path)
    store = await RoomStore.open(settings.room_db_path)

    backends = {
        "local": LocalBackend(
            max_cached_bases=settings.local_max_cached_bases, device=settings.local_device
        ),
    }
    if settings.anthropic_api_key:
        backends["anthropic"] = AnthropicBackend(AsyncAnthropic(api_key=settings.anthropic_api_key))

    backend_router = BackendRouter(registry, backends)
    bot = SalonBot(settings, registry, store, backend_router)
    try:
        await bot.start(settings.discord_bot_token)
    finally:
        await store.close()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
