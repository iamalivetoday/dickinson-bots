"""Settings loaded from the environment (and a .env file, for local runs).
See .env.example for every variable this reads.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    discord_bot_token: str
    anthropic_api_key: str | None
    actors_config_path: Path
    room_db_path: Path
    allowed_guild_id: str | None
    allowed_user_ids: frozenset[str]
    local_max_cached_bases: int
    local_device: str | None  # None -> backend picks (mps if available, else cpu)


def load_settings(env: dict | None = None, *, env_file: str | Path | None = ".env") -> Settings:
    """Read settings from `env` (defaults to os.environ), after loading
    `env_file` into the process environment if it exists (python-dotenv
    never overrides variables already set, so real env vars still win).
    """
    if env_file is not None:
        _load_dotenv_if_present(env_file)
    env = os.environ if env is None else env

    token = env.get("DISCORD_BOT_TOKEN")
    if not token:
        raise ConfigError("DISCORD_BOT_TOKEN is required")

    guild_id = env.get("DISCORD_GUILD_ID") or None
    allowed_users = frozenset(
        u.strip() for u in env.get("DISCORD_ALLOWED_USER_IDS", "").split(",") if u.strip()
    )

    return Settings(
        discord_bot_token=token,
        anthropic_api_key=env.get("ANTHROPIC_API_KEY") or None,
        actors_config_path=Path(env.get("ACTORS_CONFIG_PATH", REPO_ROOT / "config" / "actors.yaml")),
        room_db_path=Path(env.get("ROOM_DB_PATH", REPO_ROOT / "data" / "rooms.sqlite3")),
        allowed_guild_id=guild_id,
        allowed_user_ids=allowed_users,
        local_max_cached_bases=int(env.get("LOCAL_MAX_CACHED_BASES", "1")),
        local_device=env.get("LOCAL_DEVICE") or None,
    )


def _load_dotenv_if_present(env_file: str | Path) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    path = Path(env_file)
    if path.exists():
        load_dotenv(path)
