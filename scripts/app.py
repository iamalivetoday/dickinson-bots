#!/usr/bin/env python3
"""Web UI (Gradio) for the fine-tuned writer bots.

Two tabs:
  - Chat:  pick an author and have a multi-turn conversation.
  - Salon: give a topic and watch the authors converse (streaming, variable length).

All bots share one base model; adapters are attached once and switched per turn.

Run:  .venv/bin/python scripts/app.py     then open the printed http://127.0.0.1 URL
"""
import random
from pathlib import Path

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from persona import system_for
from salon import DEFAULT_DISP, DISPOSITION, MODES, NAME

ROOT = Path(__file__).resolve().parent.parent
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
AUTHORS = sorted(p.parent.parent.name
                 for p in ROOT.glob("authors/*/chat_model/BASE_MODEL.txt"))

print(f"loading base + {len(AUTHORS)} adapters on {DEVICE} …")
_base = (ROOT / "authors" / AUTHORS[0] / "chat_model" / "BASE_MODEL.txt").read_text().strip()
TOK = AutoTokenizer.from_pretrained(str(ROOT / "authors" / AUTHORS[0] / "chat_model"))
_dtype = torch.float32 if DEVICE == "mps" else torch.bfloat16
_bm = AutoModelForCausalLM.from_pretrained(_base, dtype=_dtype)
MODEL = None
for a in AUTHORS:
    path = str(ROOT / "authors" / a / "chat_model")
    if MODEL is None:
        MODEL = PeftModel.from_pretrained(_bm, path, adapter_name=a)
    else:
        MODEL.load_adapter(path, adapter_name=a)      # attaches; don't reassign
MODEL = MODEL.to(DEVICE).eval()
print("ready.")


def gen(author, messages, temp, max_new, min_new=1):
    MODEL.set_adapter(author)
    text = TOK.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ins = TOK(text, return_tensors="pt", add_special_tokens=False).to(DEVICE)
    with torch.no_grad():
        out = MODEL.generate(**ins, max_new_tokens=int(max_new), min_new_tokens=int(min_new),
                             do_sample=True, temperature=temp, top_p=0.9,
                             repetition_penalty=1.2, pad_token_id=TOK.eos_token_id)
    return TOK.decode(out[0, ins.input_ids.shape[1]:], skip_special_tokens=True).strip()


def chat_fn(message, history, author, temp, max_new):
    msgs = [{"role": "system", "content": system_for(author)}]
    msgs += history                                   # gradio 'messages' format
    msgs.append({"role": "user", "content": message})
    return gen(author, msgs, temp, max_new)


def salon_fn(topic, rounds, temp):
    md, prev = "", None
    for _ in range(int(rounds)):
        for a in AUTHORS:
            choice = random.choices(["pass"] + list(MODES),
                                    weights=DISPOSITION.get(a, DEFAULT_DISP))[0]
            if choice == "pass":
                md += f"**{NAME[a]}:**  *…(holds their peace)*\n\n"; yield md; continue
            instr, mn, mx = MODES[choice]
            if prev is None:
                user = f"The question before us: {topic}\n\nGive your response."
            else:
                user = (f"The question before us: {topic}\n\n{NAME[prev[0]]} just said:\n"
                        f"“{prev[1]}”\n\nRespond to them in your own voice.")
            msgs = [{"role": "system", "content": system_for(a) + " " + instr},
                    {"role": "user", "content": user}]
            reply = gen(a, msgs, temp, mx, mn)
            if not reply:
                md += f"**{NAME[a]}:**  *…(holds their peace)*\n\n"; yield md; continue
            md += f"**{NAME[a]}**  *({choice})*:  {reply}\n\n"
            prev = (a, reply); yield md


with gr.Blocks(title="Fine-tuned Writers") as demo:
    gr.Markdown("# 🖋 Fine-tuned Writers\nChat with a writer, or convene a salon.")
    with gr.Tab("Chat"):
        author = gr.Dropdown(AUTHORS, value=AUTHORS[0], label="Author")
        temp = gr.Slider(0.1, 1.2, value=0.7, label="temperature")
        maxn = gr.Slider(16, 320, value=220, step=8, label="max new tokens")
        gr.ChatInterface(chat_fn, additional_inputs=[author, temp, maxn])
    with gr.Tab("Salon"):
        topic = gr.Textbox(label="Topic", value="What makes a life worth living?")
        with gr.Row():
            rounds = gr.Slider(1, 4, value=2, step=1, label="rounds")
            stemp = gr.Slider(0.1, 1.2, value=0.7, label="temperature")
        btn = gr.Button("Convene the salon", variant="primary")
        out = gr.Markdown()
        btn.click(salon_fn, [topic, rounds, stemp], out)

if __name__ == "__main__":
    demo.launch()
