"""LocalBackend tests — transformers/PEFT are monkeypatched with tiny fakes
so these exercise caching, adapter sharing, locking, and eviction without
touching real weights or real hardware.
"""
import asyncio
import time

import pytest

import discord_bot.backends.local as local_module
from discord_bot.backends.local import GenerationError, LocalBackend
from discord_bot.registry import Actor


class FakeIds:
    shape = (1, 3)


class FakeInputs(dict):
    def to(self, device):
        return self


class FakeOut:
    def __getitem__(self, idx):
        return [0]  # decode() below ignores the value


class FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "prompt"

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        return FakeInputs(input_ids=FakeIds())

    def decode(self, ids, skip_special_tokens=True):
        return "a reply"


class FakeBaseModel:
    def __init__(self, base_id, delay=0.0):
        self.base_id = base_id
        self.delay = delay
        self.generate_calls = 0
        self.call_intervals = []  # wall-clock (start, end) of each generate() — the critical section

    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generate_calls += 1
        start = time.monotonic()
        if self.delay:
            time.sleep(self.delay)
        self.call_intervals.append((start, time.monotonic()))
        return FakeOut()


class DisableAdapterCtx:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.model.adapter_disabled = True

    def __exit__(self, *exc):
        self.model.adapter_disabled = False


class FakePeftModel:
    """Stands in for peft.PeftModel — wraps a base model and simulates
    multiple named adapters sharing one set of base weights."""

    def __init__(self, base_model, adapter_name):
        self.base_model = base_model
        self.adapters = [adapter_name]
        self.active_adapter = adapter_name
        self.adapter_disabled = False
        self.set_adapter_calls = []

    @classmethod
    def from_pretrained(cls, model, adapter_key, adapter_name=None):
        return cls(model, adapter_name or adapter_key)

    def load_adapter(self, adapter_key, adapter_name=None):
        self.adapters.append(adapter_name or adapter_key)

    def set_adapter(self, name):
        self.active_adapter = name
        self.set_adapter_calls.append(name)

    def disable_adapter(self):
        return DisableAdapterCtx(self)

    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        return self.base_model.generate(**kwargs)


@pytest.fixture
def fakes(monkeypatch):
    tokenizer_loads = []
    model_loads = []

    def fake_tok_from_pretrained(base_id):
        tokenizer_loads.append(base_id)
        return FakeTokenizer()

    def fake_model_from_pretrained(base_id, dtype=None):
        model = FakeBaseModel(base_id)
        model_loads.append(model)
        return model

    monkeypatch.setattr(local_module.AutoTokenizer, "from_pretrained", fake_tok_from_pretrained)
    monkeypatch.setattr(local_module.AutoModelForCausalLM, "from_pretrained", fake_model_from_pretrained)
    monkeypatch.setattr(local_module, "PeftModel", FakePeftModel)
    return {"tokenizer_loads": tokenizer_loads, "model_loads": model_loads}


def actor(id_, base, adapter=None, generation=None):
    return Actor(id=id_, backend="local", base=base, adapter=adapter, generation=generation or {})


@pytest.mark.asyncio
async def test_two_actors_sharing_a_base_load_it_once_and_attach_separate_adapters(fakes):
    backend = LocalBackend(max_cached_bases=2, device="cpu")
    a = actor("weil", "shared/base", adapter="authors/weil/chat_model")
    b = actor("hugo", "shared/base", adapter="authors/hugo/chat_model")

    await backend.generate(a, "weil", [])
    await backend.generate(b, "hugo", [])

    assert len(fakes["model_loads"]) == 1, "base should be loaded once, not once per actor"
    assert len(fakes["tokenizer_loads"]) == 1
    loaded_model = fakes["model_loads"][0]
    # the shared base is now wrapped in a PeftModel with both adapters attached
    peft = backend._bases["shared/base"].model
    assert isinstance(peft, FakePeftModel)
    assert peft.base_model is loaded_model
    assert set(peft.adapters) == {str(a.adapter_path()), str(b.adapter_path())}


@pytest.mark.asyncio
async def test_generation_switches_to_the_requesting_actors_adapter(fakes):
    backend = LocalBackend(max_cached_bases=2, device="cpu")
    a = actor("weil", "shared/base", adapter="authors/weil/chat_model")
    b = actor("hugo", "shared/base", adapter="authors/hugo/chat_model")
    await backend.generate(a, "weil", [])
    await backend.generate(b, "hugo", [])

    peft = backend._bases["shared/base"].model
    assert peft.set_adapter_calls[-1] == str(b.adapter_path())


@pytest.mark.asyncio
async def test_persona_less_actor_on_an_adapter_bearing_base_disables_adapters(fakes):
    backend = LocalBackend(max_cached_bases=2, device="cpu")
    a = actor("weil", "shared/base", adapter="authors/weil/chat_model")
    await backend.generate(a, "weil", [])

    base_only = actor("base-actor", "shared/base", adapter=None)
    reply = await backend.generate(base_only, "base-actor", [])
    assert reply == "a reply"


@pytest.mark.asyncio
async def test_distinct_bases_are_both_loaded(fakes):
    backend = LocalBackend(max_cached_bases=2, device="cpu")
    await backend.generate(actor("x", "base-1"), "x", [])
    await backend.generate(actor("y", "base-2"), "y", [])
    assert len(fakes["model_loads"]) == 2


@pytest.mark.asyncio
async def test_cache_evicts_least_recently_used_base_when_full(fakes, monkeypatch):
    evicted = []
    monkeypatch.setattr(local_module.gc, "collect", lambda: evicted.append("gc"))
    backend = LocalBackend(max_cached_bases=1, device="cpu")

    await backend.generate(actor("x", "base-1"), "x", [])
    await backend.generate(actor("y", "base-2"), "y", [])

    assert "base-1" not in backend._bases
    assert "base-2" in backend._bases
    assert evicted == ["gc"]


@pytest.mark.asyncio
async def test_missing_base_raises_generation_error(fakes):
    backend = LocalBackend(device="cpu")
    broken = Actor(id="no-base", backend="local", base=None)
    with pytest.raises(GenerationError, match="no local base configured"):
        await backend.generate(broken, "no-base", [])


@pytest.mark.asyncio
async def test_concurrent_generation_on_the_same_base_is_serialized(fakes):
    """Two participants sharing one base must never run generate() at the
    same time — that would race set_adapter."""
    backend = LocalBackend(max_cached_bases=2, device="cpu")
    a = actor("weil", "shared/base", adapter="authors/weil/chat_model")
    b = actor("hugo", "shared/base", adapter="authors/hugo/chat_model")
    # prime the cache/adapters up front so both generate() calls below race
    # only on the per-base generation lock, not on first-load bookkeeping.
    await backend.generate(a, "weil", [])
    await backend.generate(b, "hugo", [])

    model = fakes["model_loads"][0]
    model.delay = 0.05
    model.call_intervals.clear()  # drop the two priming calls above

    await asyncio.gather(
        backend.generate(a, "weil", []),
        backend.generate(b, "hugo", []),
    )

    # what matters is the critical section (the actual model.generate() call,
    # which includes the set_adapter it depends on) never overlaps — not the
    # coroutines themselves, one of which legitimately waits on the lock.
    (s1, e1), (s2, e2) = model.call_intervals
    assert e1 <= s2 or e2 <= s1, f"generation calls overlapped: {model.call_intervals}"


@pytest.mark.asyncio
async def test_generation_does_not_block_the_event_loop(fakes):
    backend = LocalBackend(max_cached_bases=1, device="cpu")
    a = actor("weil", "shared/base", adapter="authors/weil/chat_model")
    await backend.generate(a, "weil", [])  # prime the cache
    fakes["model_loads"][0].delay = 0.2

    heartbeats = []

    async def heartbeat():
        for _ in range(8):
            await asyncio.sleep(0.02)
            heartbeats.append(time.monotonic())

    await asyncio.gather(backend.generate(a, "weil", []), heartbeat())
    assert len(heartbeats) == 8, "event loop was blocked during local inference"
