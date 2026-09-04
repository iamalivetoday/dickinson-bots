"""Declarative actor registry: persona x substrate.

An `Actor` is a configuration, not a running thing — see `discord_bot.rooms`
for how actors become live room participants. Actors are loaded once from
`config/actors.yaml` (or an override path, e.g. the fake-actor registry used
by tests and the live acceptance test).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "actors.yaml"

VALID_BACKENDS = {"local", "anthropic", "fake"}


class ActorConfigError(ValueError):
    """A malformed or ambiguous entry in an actor registry file."""


@dataclass(frozen=True)
class Actor:
    id: str
    backend: str
    persona: str | None = None
    base: str | None = None      # local backend: HF base model id
    adapter: str | None = None   # local backend: LoRA adapter dir, relative to repo root
    model: str | None = None     # anthropic backend: model id
    avatar: str | None = None    # webhook avatar image URL
    generation: dict[str, Any] = field(default_factory=dict)

    def adapter_path(self) -> Path | None:
        """Absolute path to this actor's LoRA adapter, or None if it has none."""
        if self.adapter is None:
            return None
        path = Path(self.adapter)
        return path if path.is_absolute() else REPO_ROOT / path


class ActorRegistry:
    """All configured actors, keyed by actor id."""

    def __init__(self, actors: dict[str, Actor]):
        self._actors = actors

    def __contains__(self, actor_id: str) -> bool:
        return actor_id in self._actors

    def __iter__(self):
        return iter(self._actors.values())

    def __len__(self) -> int:
        return len(self._actors)

    def get(self, actor_id: str) -> Actor:
        try:
            return self._actors[actor_id]
        except KeyError:
            raise KeyError(f"no such actor: {actor_id!r}") from None

    def ids(self) -> list[str]:
        return sorted(self._actors)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> "ActorRegistry":
        path = Path(path)
        raw = yaml.safe_load(path.read_text()) or {}
        defaults = raw.get("defaults", {})
        actors: dict[str, Actor] = {}
        for entry in raw.get("actors", []):
            actor = _build_actor(entry, defaults)
            if actor.id in actors:
                raise ActorConfigError(f"duplicate actor id in {path}: {actor.id!r}")
            actors[actor.id] = actor
        if not actors:
            raise ActorConfigError(f"no actors declared in {path}")
        return cls(actors)


def _build_actor(entry: dict[str, Any], defaults: dict[str, Any]) -> Actor:
    entry = dict(entry)
    actor_id = entry.pop("id", None)
    if not actor_id or not isinstance(actor_id, str):
        raise ActorConfigError(f"actor entry missing a string 'id': {entry!r}")

    backend = entry.pop("backend", None)
    if backend not in VALID_BACKENDS:
        raise ActorConfigError(
            f"actor {actor_id!r}: backend must be one of {sorted(VALID_BACKENDS)}, "
            f"got {backend!r}"
        )

    generation = {**defaults.get(backend, {}), **entry.pop("generation", {})}

    if backend == "local" and not entry.get("base"):
        raise ActorConfigError(f"actor {actor_id!r}: local backend requires 'base'")
    if backend == "anthropic" and not entry.get("model"):
        raise ActorConfigError(f"actor {actor_id!r}: anthropic backend requires 'model'")

    try:
        return Actor(id=actor_id, backend=backend, generation=generation, **entry)
    except TypeError as exc:
        raise ActorConfigError(f"actor {actor_id!r}: {exc}") from exc
