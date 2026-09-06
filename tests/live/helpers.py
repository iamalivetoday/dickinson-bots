"""Small Playwright helpers for driving the real Discord web client."""
from __future__ import annotations

from pathlib import Path

MESSAGE_SELECTOR = '[id^="chat-messages-"]'


def shot(page, artifacts_dir: Path, name: str) -> Path:
    path = Path(artifacts_dir) / name
    page.screenshot(path=str(path))
    return path


def sidebar_channel_names(page) -> list[str]:
    return [
        el.inner_text().strip()
        for el in page.query_selector_all('[data-list-item-id^="channels___"]')
        if el.inner_text().strip()
    ]


def open_channel(page, name: str) -> bool:
    """Click the named channel in the sidebar. Returns False if absent."""
    for el in page.query_selector_all('[data-list-item-id^="channels___"]'):
        if el.inner_text().strip().splitlines()[0].strip() == name:
            el.click()
            page.wait_for_timeout(1500)
            return True
    return False


def message_texts(page, limit: int = 40) -> list[str]:
    els = page.query_selector_all(MESSAGE_SELECTOR)
    return [el.inner_text() for el in els[-limit:]]


def send_message(page, text: str) -> None:
    box = page.locator('[role="textbox"]').last
    box.click()
    box.type(text, delay=15)
    page.keyboard.press("Enter")
    page.wait_for_timeout(1200)


def run_slash_command(page, command: str, *, wait_ms: int = 4000) -> None:
    """Type a slash command into the message box and submit it.

    Discord's autocomplete popup intercepts the first Enter, so this sends
    Enter twice: once to accept the highlighted command, once to submit.
    """
    box = page.locator('[role="textbox"]').last
    box.click()
    box.type(command, delay=25)
    page.wait_for_timeout(1200)
    page.keyboard.press("Enter")
    page.wait_for_timeout(600)
    page.keyboard.press("Enter")
    page.wait_for_timeout(wait_ms)


def wait_for_text(page, needle: str, timeout_ms: int = 30000) -> bool:
    """Poll the visible transcript for `needle`."""
    step = 1000
    waited = 0
    while waited < timeout_ms:
        if any(needle in t for t in message_texts(page)):
            return True
        page.wait_for_timeout(step)
        waited += step
    return False
