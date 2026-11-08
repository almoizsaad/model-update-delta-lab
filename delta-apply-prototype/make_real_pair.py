#!/usr/bin/env python3
"""Create a real bf16 model pair (SmolLM2-135M and a 200-step fine-tune of it) for demo_real_pair.py.

Usage (Colab, GPU runtime): python3 make_real_pair.py [OUTDIR=/content/work]
Writes OUTDIR/b_v0/model.safetensors and OUTDIR/b_v1/model.safetensors (about 270 MB each).
The fine-tune follows the recipe of model-update-delta-lab (WikiText-2, 200 steps, batch 8, lr 2e-5, seed 1);
it is not guaranteed to be byte-identical to the file used there, so delta sizes may differ slightly.
"""
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "HuggingFaceTB/SmolLM2-135M"
out = Path(sys.argv[1] if len(sys.argv) > 1 else "/content/work")
out.mkdir(parents=True, exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODEL)
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
ids = tok("\n".join(t for t in ds["text"][:20000] if t.strip()), return_tensors="pt").input_ids[0]
n = len(ids) // 256
train = ids[:n * 256].view(n, 256)


def to_bf16(src, dst):
    AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).save_pretrained(dst, safe_serialization=True)


m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
m.save_pretrained(out / "v0_base_fp32", safe_serialization=True)
m = m.to("cuda")
m.train()
gen = torch.Generator().manual_seed(1)
opt = torch.optim.AdamW(m.parameters(), lr=2e-5, weight_decay=0.0)
for step in range(200):
    x = train[torch.randint(0, len(train), (8,), generator=gen)].to("cuda")
    loss = m(x, labels=x).loss
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    if step % 50 == 0:
        print(step, round(loss.item(), 3))
m.cpu().save_pretrained(out / "v1_fp32", safe_serialization=True)
to_bf16(out / "v0_base_fp32", out / "b_v0")
to_bf16(out / "v1_fp32", out / "b_v1")
for p in (out / "b_v0" / "model.safetensors", out / "b_v1" / "model.safetensors"):
    print(p, p.stat().st_size)
