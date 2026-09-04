"""Backend-neutral inference interface.

Every backend (local transformers/PEFT, Anthropic, the deterministic fake
used by tests) implements the same `generate(actor, participant_id,
transcript)` coroutine and knows nothing about Discord, rooms, or turn
policies — see `discord_bot.rooms.orchestrator` for the caller.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    # scripts/ is a plain sibling package, not an installed one — reuse its
    # per-author system prompts instead of forking a second copy of them.
    sys.path.insert(0, str(REPO_ROOT))

from scripts.persona import system_for  # noqa: E402

GENERIC_SYSTEM_PROMPT = "You are {name}. Answer thoughtfully, in your own voice."


class GenerationError(RuntimeError):
    """A backend could not produce a reply."""


@dataclass(frozen=True)
class TranscriptMessage:
    """One line of room transcript, with an explicit speaker identity.

    `speaker_id` is a participant id (e.g. "weil", "opus-4.8:a", or
    "user:123456789012345678"); `speaker_name` is what should be shown to
    the model (and to Discord) for that speaker.
    """

    speaker_id: str
    speaker_name: str
    content: str


class Backend(Protocol):
    async def generate(
        self, actor, participant_id: str, transcript: Sequence[TranscriptMessage]
    ) -> str:
        """Produce actor's next line, in character, given the transcript so far."""
        ...


def system_prompt_for(actor) -> str:
    """The system prompt for an actor: its persona's voice, or a generic
    instruction for a persona-less base-model actor."""
    if actor.persona:
        return system_for(actor.persona)
    return GENERIC_SYSTEM_PROMPT.format(name=actor.id)


def to_role_messages(
    participant_id: str, transcript: Sequence[TranscriptMessage]
) -> list[dict]:
    """Render a transcript as chat-API role turns from one participant's
    point of view: its own prior lines are "assistant"; everyone else's
    (human or other actors) are "user", tagged with who said it so the
    model can tell speakers apart in a multi-party room.

    Consecutive same-role turns are collapsed into one message — multi-
    party transcripts routinely have several other participants speak in a
    row, and providers like Anthropic require strict user/assistant
    alternation.
    """
    rendered: list[dict] = []
    for msg in transcript:
        if msg.speaker_id == participant_id:
            role, content = "assistant", msg.content
        else:
            role, content = "user", f"{msg.speaker_name}: {msg.content}"
        if rendered and rendered[-1]["role"] == role:
            rendered[-1]["content"] += "\n\n" + content
        else:
            rendered.append({"role": role, "content": content})

    if not rendered or rendered[0]["role"] != "user":
        rendered.insert(0, {"role": "user", "content": "(the conversation begins)"})
    return rendered
