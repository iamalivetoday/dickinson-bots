# Fine-tuning writers' voices

Practice project: LoRA fine-tune a small base LLM on each author's prose so it
continues text in their voice. Modern redo of the original GPT-2 experiment.

## Layout
```
authors/<name>/data/raw/         source texts (see DATA_SOURCES.md)
authors/<name>/data/processed/   cleaned corpus  <- prepare_data.py
authors/<name>/model/            LoRA adapter    <- train.py
scripts/prepare_data.py          raw -> processed (cleaning)
scripts/train.py                 LoRA fine-tune (base model + adapter)
scripts/generate.py              sample text in the author's voice
```

## Setup (done once)
```
uv venv --python 3.12 .venv
uv pip install --python .venv torch transformers peft datasets accelerate
```
Runs on Apple MPS (M-series GPU). No CUDA / bitsandbytes needed.

## Workflow
```bash
.venv/bin/python scripts/prepare_data.py weil          # clean corpus
.venv/bin/python scripts/train.py weil                 # 3 epochs, Qwen2.5-0.5B base
.venv/bin/python scripts/generate.py weil "Attention is"
```

## Two flavors of model
1. **Continuation** (`train.py` → `generate.py`): a *base* model continued-pretrained
   on raw prose. You give it the start of a sentence; it finishes it in-voice.
   Not a chatbot — no notion of turns.
2. **Chat / multi-turn** (`build_chat_data.py` → `train_chat.py` → `chat.py`):
   an *Instruct* model supervised-fine-tuned on a conversational dataset, so you
   can actually talk back-and-forth with the author.

### Web UI (Gradio)
```bash
uv pip install --python .venv gradio     # once
.venv/bin/python scripts/app.py          # then open http://127.0.0.1:7860
```
Loads the base once + all adapters; **Chat** tab (pick an author, multi-turn) and
**Salon** tab (topic → the authors converse, variable length, streaming).
CLI equivalents: `scripts/chat.py <author>`, `scripts/salon.py "<topic>"`,
`scripts/compare.py "<question>"`.

### Chat workflow (Path B — proper conversational SFT)
```bash
# 1. Build a Q&A dataset by "reverse instruction": real author passages become the
#    assistant answers; a local model writes a fitting question for each.
.venv/bin/python scripts/build_chat_data.py weil --limit 600
# 2. SFT a chat model on it. Loss is masked on the prompt — only the author-voice
#    reply is learned. Starting from -Instruct keeps multi-turn ability.
caffeinate -is .venv/bin/python scripts/train_chat.py weil \
    --model Qwen/Qwen2.5-0.5B-Instruct --batch-size 2 --grad-accum 8
# 3. Talk to it (keeps conversation history across turns):
.venv/bin/python scripts/chat.py weil          # add --base to compare vs un-tuned
```
- `persona.py` holds the per-author system prompt used by BOTH train and chat.
- On MPS, wrap long runs in `caffeinate -is` (sleep stalls training) and note the
  `MPSCacheFlush` callback — MPS doesn't free its allocator cache between steps, so
  without it memory balloons into swap.

## Notes
- **Base model**: defaults to `Qwen/Qwen2.5-0.5B` (Apache-2.0, ungated, fast).
  For a stronger voice: `--model Qwen/Qwen2.5-1.5B` (or `-3B`). 48 GB RAM handles
  these easily with LoRA.
- **It's continued-pretraining on raw prose**, not instruction tuning — prompt it
  with the *start* of a sentence/passage, not a question.
- **Data quality > quantity for voice.** Corpus must be the author's own words;
  `prepare_data.py` strips editorial apparatus, footnotes, and other authors'
  sections (e.g. the translator/Miłosz material in the Weil source).
- Status: chat bots trained for **Weil** (~517k words), **Dickinson** (27.7k words,
  pure verse), **Hugo** (~741k words), **Dostoevsky** (~552k words, Crime & Punishment
  + Brothers Karamazov), and **Tolstoy** (~911k words, Anna Karenina + War and Peace).
  Le Guin not built — corpus only ~7.5k words (copyright); see `le_guin/data/README.md`.
- Gutenberg prose novels (Hugo/Dostoevsky/Tolstoy) are cleaned by
  `strip_gutenberg_novel` (skips translator prefaces + indented TOCs).

## Reference: first run (Weil, Qwen2.5-0.5B, 3 epochs)
loss 3.21 → 2.83, ~8.7 min on M5 Max. Samples picked up Weil's core vocabulary —
attention, affliction, the soul, the void, God.
```
"The soul is the only thing on earth which has anything to do with beauty…"
"Attention is the process of focusing… distraction from God…"
```
