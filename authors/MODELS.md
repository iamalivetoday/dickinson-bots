# Trained models — live status

The adapter weights are gitignored (binaries), so this file is the record of what
each bot is currently running. `chat.py` / `salon.py` / `app.py` read the base from
each adapter's `BASE_MODEL.txt`. 1.5B adapters are backed up at
`authors/<author>/chat_model_1.5b/` (revert = copy that back over `chat_model/`).

| Author | Base model | Corpus | SFT ex. | Train loss | Updated |
|--------|-----------|-------:|--------:|-----------:|---------|
| Weil | Qwen2.5-1.5B-Instruct | ~517k words | 600 | 2.876 | 2026-06-26 |
| Dickinson | Qwen2.5-7B-Instruct | 27.7k words | 403 | 3.40 | 2026-07-03 |
| Hugo | Qwen2.5-1.5B-Instruct | ~739k words | 600 | 2.931 | 2026-06-29 |
| Dostoevsky | Qwen2.5-1.5B-Instruct | ~552k words | 600 | 3.012 | 2026-06-29 |
| Tolstoy | Qwen2.5-1.5B-Instruct | ~911k words | 600 | 2.66 | 2026-06-29 |

**In progress:** retraining all five on `Qwen2.5-7B-Instruct` (rows update as each
completes). Done: Dickinson (loss 3.40, ~3.8 min on M-series MPS, ~2.75 s/it).
Remaining: Weil, Hugo, Dostoevsky, Tolstoy.
