#!/usr/bin/env python3
"""A literary salon: the fine-tuned author bots converse with each other.

All bots are LoRA adapters over the SAME base model, so we load the base once and
attach every adapter to it, switching the active voice each turn with
`set_adapter()` (cheap — the base weights are shared). Each author reacts, in
character, to what the previous speaker just said.

Usage:
    python scripts/salon.py "What makes a life worth living?"
    python scripts/salon.py "Is suffering necessary?" --authors weil dostoevsky tolstoy --rounds 2
"""
import argparse
import random
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from persona import system_for

ROOT = Path(__file__).resolve().parent.parent
NAME = {"weil": "Simone Weil", "dickinson": "Emily Dickinson", "hugo": "Victor Hugo",
        "dostoevsky": "Fyodor Dostoevsky", "tolstoy": "Leo Tolstoy"}

# Each "mode" carries a length instruction + matching token window, so replies
# actually range from a word to a paragraph (not just the ~100-word default).
MODES = {
    "quip":      ("Reply in a brief quip — a few words or a single line.",       1, 16),
    "short":     ("Reply in a sentence or two.",                                  8, 48),
    "paragraph": ("Reply at length, in a full paragraph.",                       40, 150),
}
# per-author temperament: how likely to pass / quip / speak briefly / expound.
# weights over (pass, quip, short, paragraph).
DISPOSITION = {
    "dickinson":  (0.20, 0.45, 0.25, 0.10),   # terse, gnomic — quips, often silent
    "weil":       (0.20, 0.10, 0.30, 0.40),   # measured, essayistic
    "dostoevsky": (0.10, 0.15, 0.30, 0.45),   # can't help expounding
    "hugo":       (0.10, 0.05, 0.25, 0.60),   # orator — long
    "tolstoy":    (0.15, 0.10, 0.30, 0.45),
}
DEFAULT_DISP = (0.15, 0.20, 0.35, 0.30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("topic")
    ap.add_argument("--authors", nargs="*")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=None, help="reproduce a given salon")
    args = ap.parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    authors = args.authors or sorted(
        p.parent.parent.name for p in ROOT.glob("authors/*/chat_model/BASE_MODEL.txt"))

    # one shared base model; attach every author's adapter to it
    base = (ROOT / "authors" / authors[0] / "chat_model" / "BASE_MODEL.txt").read_text().strip()
    tok = AutoTokenizer.from_pretrained(str(ROOT / "authors" / authors[0] / "chat_model"))
    dtype = torch.float32 if device == "mps" else torch.bfloat16
    base_model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype)
    model = None
    for a in authors:
        path = str(ROOT / "authors" / a / "chat_model")
        if model is None:
            model = PeftModel.from_pretrained(base_model, path, adapter_name=a)
        else:
            model.load_adapter(path, adapter_name=a)
    model = model.to(device).eval()

    def speak(author, user_msg, instruction, min_new, max_new):
        model.set_adapter(author)                          # switch the active voice
        msgs = [{"role": "system", "content": system_for(author) + " " + instruction},
                {"role": "user", "content": user_msg}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ins = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(**ins, max_new_tokens=max_new, min_new_tokens=min_new,
                                 do_sample=True, temperature=args.temp, top_p=0.9,
                                 repetition_penalty=1.2, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, ins.input_ids.shape[1]:], skip_special_tokens=True).strip()

    print(f"\n🕯  SALON — {args.topic}\n" + "=" * 70)
    transcript = [f"# The Salon\n\n**Topic: {args.topic}**\n"]
    prev = None
    modes = list(MODES)
    for r in range(args.rounds):
        for a in authors:
            # each author decides, by temperament, whether/how much to speak
            choice = random.choices(["pass"] + modes,
                                    weights=DISPOSITION.get(a, DEFAULT_DISP))[0]
            if choice == "pass":
                print(f"\n▸ {NAME[a]}:  …(holds their peace)")
                transcript.append(f"### {NAME[a]}\n\n*…(holds their peace)*\n")
                continue                                    # sit out; prev unchanged
            instruction, min_new, max_new = MODES[choice]
            if prev is None:
                user_msg = f"The question before us: {args.topic}\n\nGive your response."
            else:
                user_msg = (f"The question before us: {args.topic}\n\n"
                            f"{NAME[prev[0]]} just said:\n“{prev[1]}”\n\n"
                            f"Respond to them in your own voice.")
            reply = speak(a, user_msg, instruction, min_new, max_new)
            if not reply:                                   # model chose to say nothing
                print(f"\n▸ {NAME[a]}:  …(holds their peace)")
                transcript.append(f"### {NAME[a]}\n\n*…(holds their peace)*\n")
                continue
            print(f"\n▸ {NAME[a]} [{choice}]:\n{reply}")
            transcript.append(f"### {NAME[a]}\n\n{reply}\n")
            prev = (a, reply)

    slug = re.sub(r"[^a-z0-9]+", "_", args.topic.lower()).strip("_")[:40]
    out = ROOT / "authors" / f"salon_{slug}.md"
    out.write_text("\n".join(transcript), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
