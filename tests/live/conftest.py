"""Fixtures for the live Playwright acceptance test.

This suite talks to the REAL `the flesh door` server through a real browser.
It is opt-in: without --live it is skipped entirely, so `pytest tests/` stays
offline and deterministic.

Authentication is a persisted Playwright profile directory (gitignored). If
it's missing or logged out, the test PAUSES with a visible browser for the
operator to log in — it never skips silently, and never handles a password
itself.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PROFILE = REPO_ROOT / ".playwright-auth" / "discord-profile"
SERVER_NAME = "the flesh door"
LOGIN_TIMEOUT_SECONDS = 20 * 60


def pytest_addoption(parser):
    parser.addoption("--live", action="store_true", default=False,
                     help="run the live Discord acceptance test against the real server")
    parser.addoption("--headed", action="store_true", default=True,
                     help="run the browser headed (default: yes, so login pauses are visible)")
    parser.addoption("--profile-dir", action="store", default=None,
                     help=f"Playwright profile dir holding the Discord session (default: {DEFAULT_PROFILE})")
    parser.addoption("--artifacts-dir", action="store", default=None,
                     help="where to write screenshots (default: tests/live/artifacts, gitignored)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        return
    skip = pytest.mark.skip(reason="live Discord test — pass --live to run it")
    for item in items:
        if "live" in item.keywords or "tests/live" in str(item.fspath):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def artifacts_dir(pytestconfig) -> Path:
    path = Path(pytestconfig.getoption("--artifacts-dir")
                or REPO_ROOT / "tests" / "live" / "artifacts")
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def guild_id() -> str:
    value = os.environ.get("DISCORD_GUILD_ID")
    if not value:
        pytest.fail(
            "DISCORD_GUILD_ID is not set — the live test needs to know which "
            "server is 'the flesh door'. Set it in .env (see .env.example)."
        )
    return value


@pytest.fixture(scope="session")
def browser_context(pytestconfig, artifacts_dir, guild_id):
    # guild_id is requested here (not just by `page`) so a missing
    # DISCORD_GUILD_ID fails immediately, before opening a browser and
    # sitting in the login wait.
    from playwright.sync_api import sync_playwright

    profile_dir = Path(pytestconfig.getoption("--profile-dir") or DEFAULT_PROFILE)
    profile_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            str(profile_dir),
            headless=not pytestconfig.getoption("--headed"),
            viewport={"width": 1400, "height": 950},
        )
        page = context.pages[0] if context.pages else context.new_page()
        _ensure_logged_in(page, artifacts_dir)
        yield context
        context.close()


def _ensure_logged_in(page, artifacts_dir: Path) -> None:
    """Open Discord and, if not authenticated, PAUSE for the operator.

    Deliberately never skips the test when auth is missing (a silently
    skipped live test is worse than a stalled one) and never types a
    password — the operator logs in by hand, including any 2FA or captcha.
    """
    page.goto("https://discord.com/channels/@me")
    page.wait_for_timeout(3000)
    if "/channels/" in page.url and "/login" not in page.url:
        return

    page.screenshot(path=str(artifacts_dir / "00_login_required.png"))
    print(
        "\n"
        "=" * 72 + "\n"
        "  ACTION NEEDED — Discord login\n"
        "\n"
        "  A browser window is open at Discord's login page. Please log in\n"
        "  by hand (including 2FA / captcha if asked). This test will\n"
        "  continue automatically once you reach the app.\n"
        f"  Waiting up to {LOGIN_TIMEOUT_SECONDS // 60} minutes.\n"
        + "=" * 72 + "\n",
        flush=True,
    )
    deadline = time.time() + LOGIN_TIMEOUT_SECONDS
    while time.time() < deadline:
        if "/channels/" in page.url and "/login" not in page.url:
            page.wait_for_timeout(2000)  # let the session flush to the profile dir
            return
        time.sleep(3)
    pytest.fail(
        "Timed out waiting for a manual Discord login. Re-run with --live once "
        "the browser profile is authenticated."
    )


@pytest.fixture(scope="session")
def page(browser_context, guild_id):
    page = browser_context.pages[0] if browser_context.pages else browser_context.new_page()
    page.goto(f"https://discord.com/channels/{guild_id}")
    page.wait_for_timeout(4000)
    return page


@pytest.fixture(scope="session")
def run_tag() -> str:
    """A unique marker for everything this run creates, so cleanup can be
    scoped to exactly this run's resources and nothing else."""
    return f"zz-test-{int(time.time())}"
