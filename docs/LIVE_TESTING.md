# Live Discord acceptance test

`tests/live/` drives a **real browser against the real `the flesh door`
server**, to verify what unit tests can't: that the server structure, the
webhook identities, the slash commands, and a bounded multi-actor salon all
actually work in Discord.

Everything else (`pytest tests/`) is offline and deterministic. This suite is
opt-in and never runs by accident.

## Prerequisites

1. **The offline suite passes**: `.venv/bin/python -m pytest tests/ -q`
2. **Playwright's browser is installed**: `.venv/bin/playwright install chromium`
3. **`.env` has `DISCORD_GUILD_ID`** for `the flesh door` (see
   [`.env.example`](../.env.example)). Without it the test **fails
   immediately** with an explanatory message — it does not skip.
4. **The bot is running against the test registry**, so every actor uses the
   deterministic fake backend (no network, no weights, no GPU — the same
   transcript always yields the same reply, which is what lets the test
   assert on exact text):

   ```bash
   ACTORS_CONFIG_PATH=config/actors.test.yaml \
   ROOM_DB_PATH=data/rooms.test.sqlite3 \
     .venv/bin/python -m discord_bot.main
   ```

   A separate database keeps test rooms out of your real transcripts.

5. **Channels provisioned once** for the test actors: run `/sync-models` in
   the server while the bot is running with the config above. That creates
   `#test-weil`, `#test-hugo`, and `#test-echo` under **voices**.

## Running it

```bash
.venv/bin/python -m pytest tests/live --live -s
```

`-s` matters: the login pause prints instructions to the terminal.

The browser runs **headed** by default so you can see and act on anything
human-only. Screenshots land in `tests/live/artifacts/` (gitignored — they
can show private conversations).

## Authentication

The session lives in a persisted Playwright profile at
`.playwright-auth/discord-profile` (gitignored). On the first run — or if it
has logged out — the test **opens the login page and pauses for up to 20
minutes** with a printed prompt, then continues by itself once you're in.

Deliberate properties:

- **It never skips when auth is missing.** A silently-skipped live test that
  reports success is worse than one that waits for you. Missing auth pauses;
  missing config fails.
- **It never handles a password.** You log in by hand, including 2FA,
  captcha, or an OAuth approval. No credential is ever typed by the test,
  stored in the repo, or printed.

## What it checks

| Test | What it proves |
|---|---|
| `test_server_structure_is_visible` | The server, the **voices** category, `#foyer`, `#models`, `#rooms`, and a channel per test actor all render |
| `test_models_command_lists_the_test_actors` | `/models` runs and lists every actor, marking the persona-less one |
| `test_talking_in_a_voice_channel_...` | A plain human message routes to that channel's actor, which replies under **its own webhook username**, not the bot's |
| `test_room_with_two_instances_of_one_actor` | `/room create` opens a thread; adding `test-echo` twice yields `test-echo:a` and `test-echo:b`; `/next` runs turns |
| `test_bounded_multi_actor_salon` | A two-actor, two-round `/salon` runs, both actors speak, and it **concludes on its own** instead of looping |
| `test_cleanup_this_runs_temporary_threads` | Cleanup touched only this run's resources, and the permanent structure is intact |

Tests share a session-scoped browser and run in declaration order — each
builds on the previous one's state.

## Safety

The suite is constrained so a bad run can't cost you anything:

- Everything it creates is named with a **unique per-run tag**
  (`zz-test-<timestamp>`), so it's obviously temporary and identifiable.
- Cleanup only ever touches threads carrying **this run's** tag — never
  another run's leftovers, and never anything it didn't create.
- It **never deletes** the server, a category, or a permanent channel, and
  the final test re-asserts `#foyer`, `#models`, and `#rooms` are all still
  there.
- Actors are `test-`-prefixed and fake-backed, so no real actor's channel or
  transcript is written to.

## Troubleshooting

**"DISCORD_GUILD_ID is not set"** — expected without config. Put the guild id
in `.env`.

**It pauses at a login page every run** — the profile isn't persisting. Check
`.playwright-auth/` is writable and that you're not passing a different
`--profile-dir`.

**A `/`-command test fails with nothing in the transcript** — the bot isn't
running, isn't in this guild, or your user id isn't in
`DISCORD_ALLOWED_USER_IDS`. Check the bot's log.

**Slash commands don't autocomplete** — commands sync per-guild at startup;
confirm `DISCORD_GUILD_ID` is set for the bot too, and that it was invited
with the `applications.commands` scope.

**Replies come from the bot instead of the actor** — the bot is missing
**Manage Webhooks**, or a cached webhook row is stale. See the recovery
section in [`DISCORD.md`](DISCORD.md).

**Leftover `zz-test-*` threads** — safe to delete by hand. Cleanup is scoped
to a single run, so an interrupted run can leave its own behind.
