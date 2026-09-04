"""Local transformers/PEFT backend.

Bases are cached and shared: every actor whose `base` matches an
already-loaded base attaches its LoRA adapter to that *same* in-memory model
(`PeftModel.load_adapter`) instead of loading a second copy — cheap, since
only the small adapter weights differ. A per-base `asyncio.Lock` serializes
generation and adapter switches (`set_adapter`) on that base, so concurrent
requests can never race an adapter swap. Cache size is configurable; past
it, the least-recently-used base is evicted and its memory freed (relevant
on MPS, which does not free its allocator cache on its own). Generation
itself runs in a thread executor so it never blocks the Discord event loop.
"""
from __future__ import annotations

import asyncio
import gc
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Sequence

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from discord_bot.registry import Actor

from .base import GenerationError, TranscriptMessage, system_prompt_for, to_role_messages

log = logging.getLogger(__name__)


def default_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class _LoadedBase:
    base_id: str
    device: str
    tokenizer: object
    model: object  # AutoModelForCausalLM, or PeftModel once >=1 adapter is attached
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    loaded_adapters: set = field(default_factory=set)


class LocalBackend:
    def __init__(self, max_cached_bases: int = 1, device: str | None = None):
        if max_cached_bases < 1:
            raise ValueError("max_cached_bases must be >= 1")
        self._max_cached_bases = max_cached_bases
        self._device = device or default_device()
        self._bases: "OrderedDict[str, _LoadedBase]" = OrderedDict()
        self._cache_lock = asyncio.Lock()  # guards the base-cache bookkeeping itself

    async def generate(
        self, actor: Actor, participant_id: str, transcript: Sequence[TranscriptMessage]
    ) -> str:
        if not actor.base:
            raise GenerationError(f"actor {actor.id!r} has no local base configured")

        loaded = await self._get_or_load_base(actor)
        messages = [{"role": "system", "content": system_prompt_for(actor)}]
        messages += to_role_messages(participant_id, transcript)

        loop = asyncio.get_running_loop()
        async with loaded.lock:
            return await loop.run_in_executor(
                None, self._generate_sync, loaded, actor, messages
            )

    async def _get_or_load_base(self, actor: Actor) -> _LoadedBase:
        loop = asyncio.get_running_loop()
        async with self._cache_lock:
            loaded = self._bases.get(actor.base)
            if loaded is None:
                loaded = await loop.run_in_executor(None, self._load_base_sync, actor.base)
                self._bases[actor.base] = loaded
                await self._evict_if_needed_locked()
            else:
                self._bases.move_to_end(actor.base)  # mark most-recently-used

            adapter_path = actor.adapter_path()
            if adapter_path is not None:
                adapter_key = str(adapter_path)
                if adapter_key not in loaded.loaded_adapters:
                    async with loaded.lock:
                        await loop.run_in_executor(
                            None, self._attach_adapter_sync, loaded, adapter_key
                        )
            return loaded

    def _load_base_sync(self, base_id: str) -> _LoadedBase:
        log.info("loading base model %s on %s", base_id, self._device)
        dtype = torch.float32 if self._device == "mps" else torch.bfloat16
        tokenizer = AutoTokenizer.from_pretrained(base_id)
        model = AutoModelForCausalLM.from_pretrained(base_id, dtype=dtype)
        model = model.to(self._device).eval()
        return _LoadedBase(base_id=base_id, device=self._device, tokenizer=tokenizer, model=model)

    def _attach_adapter_sync(self, loaded: _LoadedBase, adapter_key: str) -> None:
        log.info("attaching adapter %s to base %s", adapter_key, loaded.base_id)
        if isinstance(loaded.model, PeftModel):
            loaded.model.load_adapter(adapter_key, adapter_name=adapter_key)
        else:
            loaded.model = PeftModel.from_pretrained(
                loaded.model, adapter_key, adapter_name=adapter_key
            )
        loaded.loaded_adapters.add(adapter_key)

    def _generate_sync(self, loaded: _LoadedBase, actor: Actor, messages: list[dict]) -> str:
        model, tok = loaded.model, loaded.tokenizer
        adapter_path = actor.adapter_path()

        if isinstance(model, PeftModel) and adapter_path is not None:
            model.set_adapter(str(adapter_path))
            return self._run_generate(model, tok, loaded.device, actor, messages)
        if isinstance(model, PeftModel):
            # persona-less actor sharing a base that other actors have loaded
            # adapters onto — speak as the bare base model for this turn.
            with model.disable_adapter():
                return self._run_generate(model, tok, loaded.device, actor, messages)
        return self._run_generate(model, tok, loaded.device, actor, messages)

    def _run_generate(self, model, tok, device: str, actor: Actor, messages: list[dict]) -> str:
        gen = actor.generation
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=gen.get("max_new_tokens", 220),
                min_new_tokens=gen.get("min_new_tokens", 1),
                do_sample=True,
                temperature=gen.get("temperature", 0.7),
                top_p=gen.get("top_p", 0.9),
                repetition_penalty=gen.get("repetition_penalty", 1.2),
                pad_token_id=tok.eos_token_id,
            )
        reply = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return reply.strip()

    async def _evict_if_needed_locked(self) -> None:
        """Caller must already hold self._cache_lock."""
        while len(self._bases) > self._max_cached_bases:
            base_id, victim = self._bases.popitem(last=False)  # least-recently-used
            log.info("evicting base %s from cache", base_id)
            del victim.model
            gc.collect()
            if self._device == "mps":
                torch.mps.empty_cache()
