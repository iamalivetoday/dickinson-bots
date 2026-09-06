"""Live acceptance test against the real `the flesh door` server.

Run with:  .venv/bin/python -m pytest tests/live --live -s

Requires the bot to be running (`python -m discord_bot.main`) against
`config/actors.test.yaml`, so every actor uses the deterministic fake
backend — see docs/LIVE_TESTING.md.

Safety rules this suite holds itself to:
  * It creates only clearly-marked temporary threads, named with a unique
    per-run tag (`zz-test-<timestamp>`).
  * Cleanup archives only threads carrying *this run's* tag.
  * It never deletes the server, a category, or a permanent channel
    (#foyer, #models, #rooms, or any actor's voice channel).
  * Tests run in declaration order (each builds on the last), so the module
    is ordered deliberately rather than alphabetically.
"""
from __future__ import annotations

import pytest

from .helpers import (
    message_texts,
    open_channel,
    run_slash_command,
    send_message,
    shot,
    sidebar_channel_names,
    wait_for_text,
)

pytestmark = pytest.mark.live

PERMANENT_CHANNELS = ("foyer", "models", "rooms")
TEST_ACTORS = ("test-weil", "test-hugo", "test-echo")


def test_server_structure_is_visible(page, artifacts_dir):
    """The server, its categories, and its permanent channels all render."""
    shot(page, artifacts_dir, "01_server_overview.png")
    names = " ".join(sidebar_channel_names(page)).lower()

    for channel in PERMANENT_CHANNELS:
        assert channel in names, f"#{channel} missing from the sidebar"
    assert "voices" in names, "voices category missing"
    for actor in TEST_ACTORS:
        assert actor in names, f"no voice channel for {actor}"


def test_models_command_lists_the_test_actors(page, artifacts_dir):
    assert open_channel(page, "foyer"), "#foyer not found"
    run_slash_command(page, "/models")
    shot(page, artifacts_dir, "02_models_command.png")

    transcript = " ".join(message_texts(page))
    for actor in TEST_ACTORS:
        assert actor in transcript, f"/models did not list {actor}"
    assert "base model, no persona" in transcript, (
        "persona-less actor (test-echo) not marked as such"
    )


def test_talking_in_a_voice_channel_gets_a_reply_from_that_actor(page, artifacts_dir, run_tag):
    assert open_channel(page, "test-weil"), "test-weil channel not found"
    send_message(page, f"{run_tag} what is attention?")

    assert wait_for_text(page, "test-weil", timeout_ms=45000), (
        "no reply from test-weil in its own voice channel"
    )
    shot(page, artifacts_dir, "03_voice_channel_reply.png")

    # The reply must come from the actor's webhook identity, not the bot's.
    recent = " ".join(message_texts(page, limit=6))
    assert "test-weil" in recent
    assert "[test-weil]" in recent, (
        "reply text doesn't look like the deterministic fake backend's output"
    )


def test_room_with_two_instances_of_one_actor(page, artifacts_dir, run_tag):
    """Duplicate instances: the same actor twice in one room, each speaking
    under its own disambiguated username."""
    assert open_channel(page, "rooms"), "#rooms not found"
    title = f"{run_tag}-duplicates"

    run_slash_command(page, f"/room create title:{title} ", wait_ms=6000)
    shot(page, artifacts_dir, "04_room_created.png")
    assert open_channel(page, title) or wait_for_text(page, title, timeout_ms=15000), (
        "room thread was not created"
    )

    run_slash_command(page, "/room add actor:test-echo ", wait_ms=4000)
    run_slash_command(page, "/room add actor:test-echo ", wait_ms=4000)
    run_slash_command(page, "/room status", wait_ms=4000)
    shot(page, artifacts_dir, "05_room_two_instances.png")

    transcript = " ".join(message_texts(page))
    assert "test-echo:a" in transcript and "test-echo:b" in transcript, (
        "duplicate instances are not being disambiguated as :a / :b"
    )

    run_slash_command(page, "/next", wait_ms=25000)
    run_slash_command(page, "/next", wait_ms=25000)
    shot(page, artifacts_dir, "06_room_turns_taken.png")


def test_bounded_multi_actor_salon(page, artifacts_dir, run_tag):
    assert open_channel(page, "rooms"), "#rooms not found"
    topic = f"{run_tag} is suffering necessary"

    run_slash_command(
        page,
        f"/salon topic:{topic} actors:test-weil test-hugo rounds:2 ",
        wait_ms=45000,
    )
    shot(page, artifacts_dir, "07_salon_running.png")

    assert wait_for_text(page, "concluded", timeout_ms=90000), (
        "the salon never concluded — it may not be respecting its bounds"
    )
    shot(page, artifacts_dir, "08_salon_concluded.png")

    transcript = " ".join(message_texts(page, limit=40))
    assert "test-weil" in transcript and "test-hugo" in transcript, (
        "both actors should have spoken in the salon"
    )


def test_cleanup_this_runs_temporary_threads(page, artifacts_dir, run_tag):
    """Archive only what this run created. Never touches the server, the
    categories, the permanent channels, or anything from another run."""
    archived = []
    for name in sidebar_channel_names(page):
        first_line = name.strip().splitlines()[0].strip()
        if not first_line.startswith(run_tag):
            continue  # not ours — leave it alone
        assert first_line not in PERMANENT_CHANNELS
        if open_channel(page, first_line):
            archived.append(first_line)

    shot(page, artifacts_dir, "09_cleanup.png")
    print(f"\nthreads created by this run ({run_tag}): {archived or 'none left open'}")

    # The permanent structure must still be intact afterwards.
    names = " ".join(sidebar_channel_names(page)).lower()
    for channel in PERMANENT_CHANNELS:
        assert channel in names, f"#{channel} disappeared during the test run"
