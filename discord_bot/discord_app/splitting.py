"""Split a reply into chunks that fit Discord's per-message character limit,
breaking on whitespace where possible instead of mid-word.
"""
from __future__ import annotations

DISCORD_MESSAGE_LIMIT = 2000


def split_message(text: str, limit: int = DISCORD_MESSAGE_LIMIT) -> list[str]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        split_at = _best_split_point(window)
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _best_split_point(window: str) -> int:
    """Prefer breaking at a paragraph break, then any whitespace, within
    `window`; fall back to a hard cut at the window's end if the chunk has
    no whitespace at all (e.g. one very long token)."""
    for sep in ("\n\n", "\n", " "):
        idx = window.rfind(sep)
        if idx > 0:
            return idx + len(sep)
    return len(window)
