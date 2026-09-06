# Onboarding a new actor

An actor is a **persona × a substrate**. Adding one to an
already-supported backend is a change to
[`config/actors.yaml`](../config/actors.yaml) and nothing else — no code.

## Case 1: an existing persona on a new substrate

You already have the Weil persona. To also run it on Anthropic:

```yaml
  - id: weil-opus-4.8
    persona: weil          # reuses scripts/persona.py's Weil voice
    backend: anthropic
    model: claude-opus-4-8
```

Then `/sync-models` in Discord. That's the whole job: the actor appears in
`/models`, gets its own channel, and can be named in `/salon` alongside the
local `weil` — the two are independent participants with the same voice on
different substrates.

## Case 2: a new local adapter

Train it first (see [`TRAINING.md`](../TRAINING.md)):

```bash
.venv/bin/python scripts/prepare_data.py <name>
.venv/bin/python scripts/build_chat_data.py <name> --limit 600
caffeinate -is .venv/bin/python scripts/train_chat.py <name> \
    --model Qwen/Qwen2.5-7B-Instruct --batch-size 1 --grad-accum 16
```

Add the persona's voice to `SYSTEM` in
[`scripts/persona.py`](../scripts/persona.py) — the same dict the training
scripts use, so the model is conditioned identically at train and inference
time. Then declare the actor:

```yaml
  - id: <name>
    persona: <name>
    backend: local
    base: Qwen/Qwen2.5-7B-Instruct        # match what you trained on
    adapter: authors/<name>/chat_model
```

Keeping `base` identical to the other local actors matters: actors sharing a
base share one loaded copy of it in memory, and only the small adapter
differs. A different base means a second multi-GB model resident (or
evicted and reloaded per switch — see `LOCAL_MAX_CACHED_BASES`).

## Case 3: a persona-less base model

Omit `persona` to talk to a substrate with no voice imposed:

```yaml
  - id: opus-4.8
    backend: anthropic
    model: claude-opus-4-8

  - id: qwen4-base
    backend: local
    base: Qwen/Qwen2.5-7B-Instruct        # no adapter: line
```

A persona-less local actor sharing a base with adapter-bearing actors speaks
as the bare base model for its turns (the backend disables adapters for
that generation).

## Every field

| Field | Applies to | Meaning |
|---|---|---|
| `id` | all | Unique actor id; also the Discord username (a `:b`-style suffix is added only when a room holds duplicates) |
| `backend` | all | `local` or `anthropic` |
| `persona` | optional | Key into `scripts/persona.py`'s `SYSTEM`. Omit for a persona-less base-model actor |
| `base` | local | Hugging Face id of the base model |
| `adapter` | local, optional | LoRA adapter directory, relative to the repo root. Omit to talk to the bare base |
| `model` | anthropic | Anthropic model id |
| `avatar` | optional | Image URL for this actor's webhook avatar |
| `generation` | optional | Per-actor overrides merged over `defaults.<backend>` |

Generation defaults live under `defaults:` at the top of the file, per
backend; an actor's `generation:` block overrides individual keys:

```yaml
  - id: dickinson
    persona: dickinson
    backend: local
    base: Qwen/Qwen2.5-7B-Instruct
    adapter: authors/dickinson/chat_model
    generation:
      max_new_tokens: 80        # she's terse; everything else inherited
```

## Verifying

```bash
.venv/bin/python -m pytest tests/test_registry.py -q
.venv/bin/python -c "
from discord_bot.registry import ActorRegistry
r = ActorRegistry.load()
print(r.ids())
print(r.get('<your-new-id>'))
"
```

Then in Discord: `/sync-models`, `/models`, and send a message in the new
actor's channel.

## Adding a whole new backend

Only needed for a genuinely new substrate (a local vLLM server, another
API). It's a new module in `discord_bot/backends/` implementing one method:

```python
async def generate(self, actor, participant_id, transcript) -> str
```

Register it in the `backends` dict in
[`discord_bot/main.py`](../discord_bot/main.py) under the name actors will
use in their `backend:` field. Nothing else changes — the router looks the
backend up by name and dispatches, the registry doesn't validate backend
names, and `tests/test_backend_router.py` asserts a new backend needs no
router edits.

Use `discord_bot/backends/base.py`'s helpers so your backend behaves like
the others: `system_prompt_for(actor)` for the persona (or the persona-less
fallback), and `to_role_messages(participant_id, transcript)` to render the
shared transcript from that participant's point of view, with other
speakers tagged by name and consecutive same-role turns collapsed.
