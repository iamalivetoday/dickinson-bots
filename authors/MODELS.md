# Trained models — live status

The adapter weights are gitignored (binaries), so this file is the record of what
each bot is currently running. `chat.py` / `salon.py` / `app.py` read the base from
each adapter's `BASE_MODEL.txt`. 1.5B adapters are backed up at
`authors/<author>/chat_model_1.5b/` (revert = copy that back over `chat_model/`).

| Author | Base model | Corpus | SFT ex. | Train loss | Updated |
|--------|-----------|-------:|--------:|-----------:|---------|
| Weil | Qwen2.5-7B-Instruct | ~517k words | 600 | 2.625 | 2026-07-06 |
| Dickinson | Qwen2.5-7B-Instruct | 27.7k words | 403 | 3.40 | 2026-07-03 |
| Hugo | Qwen2.5-7B-Instruct | ~739k words | 600 | 2.56 | 2026-07-06 |
| Dostoevsky | Qwen2.5-7B-Instruct | ~552k words | 600 | 2.448 | 2026-07-06 |
| Tolstoy | Qwen2.5-7B-Instruct | ~911k words | 600 | 2.448 | 2026-07-06 |

**Status:** all five retrained on `Qwen2.5-7B-Instruct` (loss down across the board
vs the 1.5B runs). 1.5B adapters preserved under `chat_model_1.5b/`.

**MPS note:** 7B LoRA SFT needs `--batch-size 1 --grad-accum 16`. At batch 2 the
prose authors (≥512-token passages) balloon the MPS allocator cache past RAM and
swap — step time explodes ~4 s → ~186 s. Dickinson (short poems) tolerated batch 2;
the novelists do not. At batch 1 it holds ~4.5 s/it, ~9 min/author.
