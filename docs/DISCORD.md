# The Discord salon

The author bots as a Discord server: talk to one actor in its own channel,
or convene several in a room and watch them argue.

Same models as `scripts/chat.py` and `scripts/salon.py` — this is another
front end over the same adapters, not a replacement. The Gradio app and the
CLI scripts still work exactly as documented in [`TRAINING.md`](../TRAINING.md).

## The three abstractions

**Actor** — a persona × a substrate, declared in
[`config/actors.yaml`](../config/actors.yaml). `weil` is the Weil persona on
a local LoRA adapter; `weil-opus-4.8` is the same persona on Anthropic;
`opus-4.8` is a substrate with no persona at all. An actor is *config*, not
a running thing.

**Participant** — one occurrence of an actor in one room. A room can hold
the same actor more than once: the first instance is `opus-4.8`, the next
is `opus-4.8:b`. Discord shows the bare actor id as the username unless a
room actually holds duplicates, in which case both show their suffix
(`opus-4.8:a`, `opus-4.8:b`).

**Room** — an optional human, any number of participants, one ordered
transcript, and a turn policy. Rooms are rows in sqlite, so they survive a
restart; a room is bound to a Discord channel or thread.

Nothing about a combination is special-cased. You talking to `weil`,
`weil` talking to `weil-qwen3.5`, two `opus-4.8` instances arguing with
`weil`, or a four-way debate are all the same code path: a room, some
participants, a turn policy.

## Setup

### 1. Install

```bash
uv venv --python 3.12 .venv                                  # if you haven't
uv pip install --python .venv -r requirements-discord.txt
uv pip install --python .venv pytest pytest-asyncio playwright  # tests only
```

### 2. Create the Discord application

In the [Developer Portal](https://discord.com/developers/applications):

1. **New Application**, name it `Salon Conductor`.
2. **Bot** tab → under *Privileged Gateway Intents* enable **Message Content
   Intent**. Leave *Presence* and *Server Members* off — the bot doesn't use
   them and shouldn't be asking for them.
3. **Reset Token**, copy it.
4. **OAuth2 → URL Generator**: scopes `bot` and `applications.commands`;
   permissions **View Channels**, **Send Messages**, **Send Messages in
   Threads**, **Create Public Threads**, **Read Message History**, **Manage
   Channels** (to provision actor channels), **Manage Webhooks** (to give
   each actor its own name and avatar). Nothing else — no admin, no member
   management.
5. Open the generated URL and add the bot to your server.

> **"Missing Access" when creating the application** is an account
> restriction, not a permissions bug: Discord gates application creation
> behind a verified phone number. Add one under *User Settings → Account →
> Phone Number* and retry. Repeated attempts get rate-limited for a while
> ("The resource is being rate limited") — wait it out rather than retrying
> in a loop.

### 3. Configure

```bash
cp .env.example .env
```

Fill in `DISCORD_BOT_TOKEN`, and `DISCORD_GUILD_ID` /
`DISCORD_ALLOWED_USER_IDS` (right-click → *Copy ID*, with *Developer Mode*
on under Settings → Advanced). Setting the guild id also makes slash
commands appear immediately instead of after Discord's global-sync delay.

`ANTHROPIC_API_KEY` is only needed if you use an `anthropic`-backed actor.

Every field is documented in [`.env.example`](../.env.example). **Never
commit `.env`** — it's gitignored, along with the sqlite database, model
weights, and transcripts.

### 4. Run

```bash
.venv/bin/python -m discord_bot.main
```

Then in the server, run **`/sync-models`** once. That provisions:

- `#foyer` — instructions and commands
- `#models` — the actor directory
- a **voices** category — one channel per configured actor
- a **rooms** category with `#rooms` — multi-participant rooms live here as
  threads

## Using it

| Command | What it does |
|---|---|
| `/models` | List every configured actor, its persona, and its substrate |
| `/chat <actor>` | Point you at that actor's channel |
| `/room create <title> [policy]` | Open a room as a thread under `#rooms` |
| `/room add <actor>` | Add an actor — run it twice for two instances of one actor |
| `/room remove <participant>` | Drop a participant (autocompletes instance ids) |
| `/room status` | Participants, turn count, bounds, who's up next |
| `/next [participant]` | Run exactly one turn |
| `/salon <topic> <actors> <rounds>` | Round-robin conversation, bounded |
| `/debate <topic> <actors> <rounds>` | Fixed order, fixed rounds |
| `/sync-models` | Re-provision channels to match config (idempotent) |

`actors` in `/salon` and `/debate` is space-separated, and repeating one
gives you two independent instances of it:

```
/salon topic: Is suffering necessary? actors: weil dostoevsky opus-4.8 opus-4.8 rounds: 2
```

**Talking to one actor:** just send a normal message in its voice channel —
no command needed. In a multi-actor room, `@` the participant you're
addressing (`@opus-4.8:b`), since a bare actor id is ambiguous when
duplicates are present.

### Turn policies

| Policy | Who speaks next |
|---|---|
| `manual` | Whoever you name in `/next <participant>` |
| `reply` | The participant addressed (voice channels use this) |
| `round_robin` | Participants in join order, cycling |
| `moderated` | The bot chooses — anyone but the last speaker |
| `debate` | A fixed order for a fixed number of rounds, then it's over |

### Bounds

Every generated message is scheduled explicitly by the room orchestrator,
one turn at a time. Webhook and bot messages never trigger generation, so a
salon can't feed itself. On top of that, rooms carry hard ceilings —
`max_turns`, `max_messages`, `max_tokens`, and a per-turn `timeout_seconds`
— and a room closes the moment one is hit. `/salon` and `/debate` set
`max_turns` to `rounds × participants` up front.

## Onboarding a new actor

See [`ACTOR_ONBOARDING.md`](ACTOR_ONBOARDING.md).

## Operating notes

**Local models are big.** Actors sharing a `base` share one loaded copy in
memory — a LoRA adapter attaches to the already-loaded base rather than
loading a second 7B model. `LOCAL_MAX_CACHED_BASES` (default 1) caps how
many *distinct* bases stay resident; past that, the least-recently-used one
is evicted and MPS's allocator cache is flushed. Mixing actors on different
bases with a cap of 1 means a reload per switch — slow but correct. Raise it
only if you have the RAM.

**Generation never blocks the bot.** Local inference runs in a thread
executor, and a per-base lock serializes generation and adapter switching so
two participants sharing a base can't race `set_adapter`.

**First run downloads weights.** A 7B base is several GB from Hugging Face,
and adapters must already exist under `authors/<name>/chat_model/` (they're
gitignored — retrain per `TRAINING.md` if this is a fresh clone).

## Recovery

**The bot died mid-salon.** Restart it. Rooms, participants, transcripts,
and turn cursors are all in sqlite, so a round-robin resumes with the next
speaker rather than restarting the cycle. Run `/next` to continue, or
`/room status` to see where things stand.

**Slash commands don't appear.** Set `DISCORD_GUILD_ID` and restart —
global commands can take up to an hour to propagate; guild commands are
instant. Also confirm you invited the bot with the `applications.commands`
scope.

**The bot ignores messages.** Check, in order: **Message Content Intent** is
enabled in the Developer Portal; `DISCORD_GUILD_ID` matches the server;
your user id is in `DISCORD_ALLOWED_USER_IDS` (or it's empty); and the
channel is actually a room — a channel created by hand isn't, `/sync-models`
or `/room create` makes one.

**An actor posts under the wrong name, or as the bot.** Actor identities are
webhooks, one per channel, cached in the database. If a webhook was deleted
in Discord, delete that row (`sqlite3 data/rooms.sqlite3 "delete from
webhooks where channel_id = '<id>'"`) and the bot will recreate it. Confirm
the bot still has **Manage Webhooks**.

**A room is stuck / won't take another turn.** It probably hit a bound —
`/room status` shows the turn count against `max_turns`, and a closed room
refuses further turns by design. Start a fresh one.

**Channels got renamed or messy.** `/sync-models` is idempotent and
non-destructive: it renames drifted channels back, archives channels for
actors no longer in config (renaming them `archived-<actor>` and moving
them to an *archived* category), and un-archives an actor that returns. It
never deletes a channel, so conversation history is never lost.

**Starting over.** Stop the bot and delete `data/rooms.sqlite3`. That
discards all rooms and transcripts but leaves Discord untouched; the next
`/sync-models` re-adopts the existing channels.

## Testing

```bash
.venv/bin/python -m pytest tests/ -q          # unit + integration, no network
```

Covers actor config, duplicate instances, multiple local bases, concurrent
adapter switching, persistence and restart recovery, all five turn
policies, hard bounds, loop prevention, channel sync, and message
splitting. Discord, Anthropic, and local inference are all mocked or faked
— nothing touches a network or a GPU.

For the live browser test against a real server, see
[`LIVE_TESTING.md`](LIVE_TESTING.md).
