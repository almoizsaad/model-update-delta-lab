# %% [markdown]
# # Model update delta lab (single script, runs top to bottom on Colab)
#
# Question: how many bytes must be sent to update a model on a device, for different
# kinds of update (full fine-tune, last layers only, LoRA, precision cast), and with
# which method?
#
# The numbers in REPORT.md come from the ORIGINAL interactive notebook run. This script
# is a cleaned, ordered version of the same cells. Re-running it should give very close
# but not bit-identical numbers (GPU nondeterminism; LoRA init is now seeded).
#
# Hypotheses registered before the runs (stated in the conversation before each stage):
#   H1 generic byte-level diff does not help after a dense fine-tune
#   H2 a separate LoRA adapter is much smaller than a diff of the merged model
#   H3 last-layers-only fine-tune: XOR+planes < 10% of raw
#   H4 bf16 halves the file, small quality loss
#   H5 XOR + byte planes beats generic diff
#   H6 byte planes help even without the old model
#   H7 chain v1->v2 behaves like v0->v1
#   H8 bf16 dense-fine-tune delta < 75% of raw bf16
#   H9 delta size grows with the number of fine-tuning steps
#   H10 (stage 6) the pattern H1/H5/H8 also holds for a second model (SmolLM2-360M, same family, 2.7x larger)
#   H11 (stage 5b) decoding a bf16 delta takes a few seconds on one CPU core

# %% setup
import os, sys, json, time, hashlib, subprocess, random, platform, shutil, struct
from pathlib import Path

def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)

sh("apt-get -qq install -y zstd xdelta3 > /dev/null 2>&1")
sh(f"{sys.executable} -m pip -q install transformers peft safetensors datasets zstandard")
sh(f"{sys.executable} -m pip -q uninstall -y torchao")   # Colab ships a torchao that peft rejects

import numpy as np
import torch, transformers
import zstandard as zstd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

W = Path("/content/work"); W.mkdir(exist_ok=True)
MODEL = "HuggingFaceTB/SmolLM2-135M"       # check the licence on the model card
SEED = 0
LONG, LEVEL = 30, 12                        # zstd CLI: window 2^30, level 12
dev = "cuda"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# %% helpers: files and measurement
def sha256(p, bs=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while (b := f.read(bs)):
            h.update(b)
    return h.hexdigest()

def size(p):
    return os.path.getsize(p)

def read_st(path):
    """Read a .safetensors file: (header dict, data region as uint8 array)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = np.fromfile(f, dtype=np.uint8)
    end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
    assert end == data.size, (end, data.size)
    return header, data

def tensor_stats(h_old, d_old, h_new, d_new):
    tot = changed = n_t = n_c = 0
    for k, v in h_new.items():
        if k == "__metadata__":
            continue
        s, e = v["data_offsets"]; so, eo = h_old[k]["data_offsets"]
        n_t += 1; tot += e - s
        if not np.array_equal(d_new[s:e], d_old[so:eo]):
            n_c += 1; changed += e - s
    return {"tensors": n_t, "tensors_changed": n_c, "bytes": tot, "bytes_in_changed_tensors": changed}

def planes(buf, itemsize):
    """Group bytes by position inside each element (plane-major layout)."""
    return np.ascontiguousarray(buf.reshape(-1, itemsize).T).reshape(-1)

def unplanes(p, itemsize):
    n = p.size // itemsize
    return np.ascontiguousarray(p.reshape(itemsize, n).T).reshape(-1)

cctx = zstd.ZstdCompressor(level=LEVEL, threads=-1)
dctx = zstd.ZstdDecompressor()

def zlen(a):
    return len(cctx.compress(a))

def stage2(old_path, new_path, itemsize=4):
    """Lossless delta variants: zstd of raw / planes / XOR / XOR+planes, with round trip check."""
    h_o, d_o = read_st(old_path); h_n, d_n = read_st(new_path)
    assert d_o.size == d_n.size and d_o.size % itemsize == 0
    out = {"data_bytes": int(d_n.size), "tensor_stats": tensor_stats(h_o, d_o, h_n, d_n)}
    x = np.bitwise_xor(d_o, d_n)
    px, pn = planes(x, itemsize), planes(d_n, itemsize)
    out["zstd_raw_new"] = zlen(d_n)
    out["zstd_planes_new"] = zlen(pn)
    out["zstd_xor"] = zlen(x)
    cx = cctx.compress(px)
    out["zstd_xor_planes"] = len(cx)
    n = d_n.size // itemsize
    out["per_plane_xor"] = [zlen(px[i * n:(i + 1) * n]) for i in range(itemsize)]
    out["per_plane_new"] = [zlen(pn[i * n:(i + 1) * n]) for i in range(itemsize)]
    rec = np.bitwise_xor(d_o, unplanes(np.frombuffer(dctx.decompress(cx), dtype=np.uint8), itemsize))
    out["roundtrip_ok"] = bool(np.array_equal(rec, d_n))
    return out

def run(cmd):
    t = time.time(); p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, round(time.time() - t, 2), p.stderr[-300:]

def zstd_full(new, out):
    return run(["zstd", f"-{LEVEL}", f"--long={LONG}", "-T0", "-f", "-q", "-o", str(out), str(new)])

def zstd_patch(old, new, out, recon):
    rc, t, e1 = run(["zstd", f"-{LEVEL}", f"--long={LONG}", "-T0", "-f", "-q", f"--patch-from={old}", "-o", str(out), str(new)])
    rc2, t2, e2 = run(["zstd", "-d", f"--long={LONG}", "-f", "-q", f"--patch-from={old}", "-o", str(recon), str(out)])
    return rc, t, t2, (rc == 0 and rc2 == 0 and sha256(recon) == sha256(new)), e1 + e2

def xdelta(old, new, out, recon):
    rc, t, e1 = run(["xdelta3", "-e", "-9", "-f", "-B", "1073741824", "-s", str(old), str(new), str(out)])
    rc2, t2, e2 = run(["xdelta3", "-d", "-f", "-B", "1073741824", "-s", str(old), str(out), str(recon)])
    return rc, t, t2, (rc == 0 and rc2 == 0 and sha256(recon) == sha256(new)), e1 + e2

def same_offset_fraction(old, new, cs):
    same = tot = 0
    with open(old, "rb") as a, open(new, "rb") as b:
        while True:
            x = a.read(cs); y = b.read(cs)
            if not y:
                break
            tot += 1; same += (x == y)
    return same / tot

def time_pipeline(old_p, new_p, itemsize=2):
    """Encode/decode time and size of the XOR + byte-planes + zstd pipeline."""
    _, d_o = read_st(old_p); _, d_n = read_st(new_p)
    t = time.time(); xx = np.bitwise_xor(d_o, d_n); px = planes(xx, itemsize)
    c = zstd.ZstdCompressor(level=LEVEL, threads=-1).compress(px); t_enc = time.time() - t
    t = time.time(); px2 = np.frombuffer(zstd.ZstdDecompressor().decompress(c), dtype=np.uint8)
    rec = np.bitwise_xor(d_o, unplanes(px2, itemsize)); t_dec = time.time() - t
    return {"bytes": len(c), "t_encode_s": round(t_enc, 1), "t_decode_s": round(t_dec, 1),
            "ok": bool(np.array_equal(rec, d_n))}

# %% helpers: data and training
tok = AutoTokenizer.from_pretrained(MODEL)
ds_tr = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
ds_va = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")

def blocks(ds, n_lines, L=256):
    text = "\n".join(t for t in ds["text"][:n_lines] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    n = len(ids) // L
    return ids[:n * L].view(n, L)

tr = blocks(ds_tr, 20000); va = blocks(ds_va, 2000)
print("train blocks", tuple(tr.shape), "val blocks", tuple(va.shape))

@torch.no_grad()
def val_loss(m, k=40):
    m.eval(); tot = 0
    for i in range(0, k, 8):
        x = va[i:i + 8].to(dev); tot += m(x, labels=x).loss.item() * len(x)
    return tot / k

def finetune(m, steps, lr, seed, bs=8):
    g = torch.Generator().manual_seed(seed)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    m.train()
    for s in range(steps):
        x = tr[torch.randint(0, len(tr), (bs,), generator=g)].to(dev)
        loss = m(x, labels=x).loss
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        if s % 50 == 0:
            print(s, round(loss.item(), 3))
    return m

def save_fp32(m, name):
    out = W / name; m.cpu().save_pretrained(out, safe_serialization=True); return out

def make_continue(src_dir, name, steps=200, lr=2e-5, seed=3):
    m = AutoModelForCausalLM.from_pretrained(src_dir, dtype=torch.float32).to(dev)
    finetune(m, steps, lr, seed)
    out = save_fp32(m, name); del m; torch.cuda.empty_cache(); return out

def make_head(name, last_k=2, steps=200, lr=2e-5, seed=2):
    m = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float32).to(dev)
    L = len(m.model.layers)
    keep = tuple(f"model.layers.{i}." for i in range(L - last_k, L)) + ("model.norm",)
    for n, p in m.named_parameters():
        p.requires_grad = n.startswith(keep)
    print(name, "trainable params:", sum(p.numel() for p in m.parameters() if p.requires_grad))
    finetune(m, steps, lr, seed)
    out = save_fp32(m, name); del m; torch.cuda.empty_cache(); return out

def make_lora(name, steps=200, lr=1e-4, seed=4):
    torch.manual_seed(seed)   # seeds the LoRA initialisation (not seeded in the original run)
    m = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float32).to(dev)
    cfg = LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0)
    pm = get_peft_model(m, cfg); pm.print_trainable_parameters()
    finetune(pm, steps, lr, seed)
    ad = W / (name + "_adapter"); pm.save_pretrained(ad)
    merged = pm.merge_and_unload()
    out = save_fp32(merged, name + "_merged"); del m, pm, merged; torch.cuda.empty_cache()
    return ad, out

def to_bf16(src, name):
    m = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16)
    out = W / name; m.save_pretrained(out, safe_serialization=True); del m; return out

def vloss(path, dtype=torch.float32):
    m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(dev)
    v = val_loss(m); del m; torch.cuda.empty_cache(); return round(v, 4)

def measure_pair(label, old_dir, new_dir, itemsize=4):
    old, new = old_dir / "model.safetensors", new_dir / "model.safetensors"
    t0 = time.time(); r = stage2(old, new, itemsize); r["t_stage2_s"] = round(time.time() - t0, 1)
    rc, t, t2, ok, e = zstd_patch(old, new, W / "tmp.patch", W / "tmp.recon")
    r["zstd_patch_cli"] = {"bytes": size(W / "tmp.patch"), "ok": ok, "t_create": t, "t_apply": t2}
    for f in ("tmp.patch", "tmp.recon"):
        if (W / f).exists():
            (W / f).unlink()
    raw = r["data_bytes"]; ts = r["tensor_stats"]
    print(f"{label:18s} tensors {ts['tensors_changed']}/{ts['tensors']}  "
          f"bytes in changed {100 * ts['bytes_in_changed_tensors'] / ts['bytes']:5.1f}%  "
          f"zstd {100 * r['zstd_raw_new'] / raw:5.1f}%  patch {100 * r['zstd_patch_cli']['bytes'] / raw:5.1f}%  "
          f"xor+planes {100 * r['zstd_xor_planes'] / raw:5.1f}%  ok={r['roundtrip_ok']}")
    return r

# %% stage 0: base model (fp32 copy of the bf16 original) and environment record
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
base_dir = W / "v0_base"
model.save_pretrained(base_dir, safe_serialization=True); tok.save_pretrained(base_dir)
del model
env = {"model": MODEL, "torch": torch.__version__, "transformers": transformers.__version__,
       "python": platform.python_version(), "seed": SEED,
       "zstd_cli": sh("zstd --version").stdout.strip(), "xdelta3": sh("xdelta3 -V 2>&1 | head -1").stdout.strip()}
(W / "env.json").write_text(json.dumps(env, indent=2)); print(env)
print("val loss v0:", vloss(base_dir))

# %% stage 1: variants (all fp32), then v0 -> v1 with generic tools
m = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float32).to(dev)
finetune(m, steps=200, lr=2e-5, seed=1)
v1_dir = save_fp32(m, "v1_fullft"); del m; torch.cuda.empty_cache()
v2_dir = make_continue(v1_dir, "v2_fullft")
head_dir = make_head("v1_head")
ad_dir, lora_dir = make_lora("v1_lora")

old, new = base_dir / "model.safetensors", v1_dir / "model.safetensors"
res1 = {"raw_new_bytes": size(new)}
rc, t, e = zstd_full(new, W / "v1.full.zst")
res1["zstd_full"] = {"bytes": size(W / "v1.full.zst"), "rc": rc, "t": t}
rc, t, t2, ok, e = zstd_patch(old, new, W / "v1.patch.zst", W / "v1.recon")
res1["zstd_patch"] = {"bytes": size(W / "v1.patch.zst"), "rc": rc, "t_create": t, "t_apply": t2, "roundtrip_ok": ok}
rc, t, t2, ok, e = xdelta(old, new, W / "v1.xd", W / "v1.recon2")
res1["xdelta3"] = {"bytes": size(W / "v1.xd") if (W / "v1.xd").exists() else None, "rc": rc,
                   "t_create": t, "t_apply": t2, "roundtrip_ok": ok}
res1["same_offset_unchanged"] = {f"{cs // 1024}KB": round(same_offset_fraction(old, new, cs), 4)
                                 for cs in (4096, 65536, 1 << 20)}
(W / "results_stage1.json").write_text(json.dumps(res1, indent=2))
print(json.dumps(res1, indent=2))

# %% stage 2/3: fp32 deltas for every variant
res3 = {"v0->v1 full": measure_pair("v0->v1 full", base_dir, v1_dir),
        "v1->v2 full": measure_pair("v1->v2 full", v1_dir, v2_dir),
        "v0->head(last2)": measure_pair("v0->head(last2)", base_dir, head_dir),
        "v0->lora merged": measure_pair("v0->lora merged", base_dir, lora_dir),
        "lora_adapter_bytes": size(ad_dir / "adapter_model.safetensors")}
for tag, dt in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
    m = AutoModelForCausalLM.from_pretrained(v1_dir, dtype=dt); out = W / f"v1_{tag}"
    m.save_pretrained(out, safe_serialization=True); del m
    _, d = read_st(out / "model.safetensors")
    res3[f"cast_{tag}"] = {"file_bytes": size(out / "model.safetensors"), "zstd_bytes": zlen(d),
                           "zstd_planes_bytes": zlen(planes(d, 2)), "val_loss": vloss(out, dt)}
res3["val_loss"] = {"v0": vloss(base_dir), "v1": vloss(v1_dir), "v2": vloss(v2_dir),
                    "head": vloss(head_dir), "lora_merged": vloss(lora_dir)}
(W / "results_stage3.json").write_text(json.dumps(res3, indent=2, default=str))
print(json.dumps(res3["val_loss"]))

# %% stage 4: the realistic case, bf16 files
b = {"v0": to_bf16(base_dir, "b_v0"), "v1": to_bf16(v1_dir, "b_v1"), "v2": to_bf16(v2_dir, "b_v2"),
     "head": to_bf16(head_dir, "b_head"), "lora": to_bf16(lora_dir, "b_lora_merged")}
res4 = {"bf16 v0->v1": measure_pair("bf16 v0->v1", b["v0"], b["v1"], 2),
        "bf16 v1->v2": measure_pair("bf16 v1->v2", b["v1"], b["v2"], 2),
        "bf16 head": measure_pair("bf16 head", b["v0"], b["head"], 2),
        "bf16 lora mrg": measure_pair("bf16 lora mrg", b["v0"], b["lora"], 2)}
ho, do = read_st(b["v0"] / "model.safetensors"); hn, dn = read_st(b["v1"] / "model.safetensors")
x = np.bitwise_xor(do, dn).view(np.uint16)
res4["diagnostics"] = {"identical_16bit_weights": round(float((x == 0).mean()), 4),
                       "identical_high_byte": round(float(((x >> 8) == 0).mean()), 4),
                       "unchanged_tensors": [k for k, v in hn.items() if k != "__metadata__" and
                                             np.array_equal(dn[v["data_offsets"][0]:v["data_offsets"][1]],
                                                            do[ho[k]["data_offsets"][0]:ho[k]["data_offsets"][1]])]}
res4["pipeline_time_v0_v1"] = time_pipeline(b["v0"] / "model.safetensors", b["v1"] / "model.safetensors")
(W / "results_stage4_bf16.json").write_text(json.dumps(res4, indent=2, default=str))
print(res4["diagnostics"], res4["pipeline_time_v0_v1"])

# %% stage 5: delta size versus amount of fine-tuning (bf16, from the base model)
sweep = {}
for n in (25, 100, 400, 1600):
    d32 = make_continue(base_dir, f"sw_{n}", steps=n)
    d16 = to_bf16(d32, f"sw_{n}_bf16")
    r = time_pipeline(b["v0"] / "model.safetensors", d16 / "model.safetensors")
    _, dn_ = read_st(d16 / "model.safetensors")
    xx = np.bitwise_xor(do, dn_).view(np.uint16)
    r["identical_weights"] = round(float((xx == 0).mean()), 4)
    r["pct_of_raw"] = round(100 * r["bytes"] / do.size, 1)
    r["val_loss_fp32"] = vloss(d32)
    sweep[n] = r; print(n, r)
    shutil.rmtree(d32); shutil.rmtree(d16)
(W / "results_sweep.json").write_text(json.dumps(sweep, indent=2))

# %% stage 5b: decode time on restricted CPU (runs on the CPU even in a GPU runtime)
# Decoding is pure numpy + zstd, so it never uses the GPU. We pin it to 1 core / 2 cores / all cores
# with `taskset`. Caveat: this is still a Colab server CPU, not a robot's ARM CPU; treat it as an upper
# bound on speed, and repeat on the real device before drawing conclusions about robots.
DECODE_WORKER = r"""
import sys, time, json, struct, numpy as np, zstandard as zstd
old, delta, it = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(old, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]; f.read(n); d_o = np.fromfile(f, dtype=np.uint8)
c = open(delta, "rb").read()
t0 = time.perf_counter()
px = np.frombuffer(zstd.ZstdDecompressor().decompress(c, max_output_size=d_o.size), dtype=np.uint8)
t1 = time.perf_counter()
x = np.ascontiguousarray(px.reshape(it, -1).T).reshape(-1)
out = np.bitwise_xor(d_o, x)
t2 = time.perf_counter()
print(json.dumps({"t_zstd_s": round(t1 - t0, 2), "t_unplane_xor_s": round(t2 - t1, 2), "t_total_s": round(t2 - t0, 2)}))
"""
(W / "decode_worker.py").write_text(DECODE_WORKER)

def save_delta(old_p, new_p, out, itemsize=2):
    _, d_o = read_st(old_p); _, d_n = read_st(new_p)
    Path(out).write_bytes(zstd.ZstdCompressor(level=LEVEL, threads=-1).compress(planes(np.bitwise_xor(d_o, d_n), itemsize)))
    return size(out)

def cpu_decode_test(label, old_p, new_p, reps=3):
    dp = W / f"{label}.delta.zst"; nbytes = save_delta(old_p, new_p, dp)
    cpu = sh("lscpu | grep -E 'Model name|^CPU\\(s\\)'").stdout.strip()
    res = {"delta_bytes": nbytes, "raw_bytes": size(new_p), "cpu": cpu, "runs": {}}
    ncpu = os.cpu_count()
    for cores in sorted({1, 2, ncpu}):
        pin = f"taskset -c 0-{cores - 1} " if shutil.which("taskset") else ""
        runs = []
        for _ in range(reps):
            r = sh(f"{pin}{sys.executable} -I {W / 'decode_worker.py'} {old_p} {dp} 2")
            runs.append(json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {"error": r.stderr[-200:]})
        res["runs"][f"{cores}_cores"] = runs
        print(label, cores, "cores:", runs)
    return res

cpu_res = {"135M_bf16_v0_to_v1": cpu_decode_test("m135", b["v0"] / "model.safetensors", b["v1"] / "model.safetensors")}
(W / "results_cpu_decode.json").write_text(json.dumps(cpu_res, indent=2))

# %% stage 6: second model (same family, 2.7x larger). Only the realistic bf16 case, 200 steps.
# Same tokenizer family, so the data pipeline is reused with the new tokenizer. To use a different
# architecture, change MODEL2 (check that the layer names contain "model.layers." and attention has q_proj/v_proj).
MODEL2 = "HuggingFaceTB/SmolLM2-360M"
tok = AutoTokenizer.from_pretrained(MODEL2)
tr = blocks(ds_tr, 20000); va = blocks(ds_va, 2000)
base_dir = W / "m2_v0"
m = AutoModelForCausalLM.from_pretrained(MODEL2, dtype=torch.float32); m.save_pretrained(base_dir, safe_serialization=True)
tok.save_pretrained(base_dir); del m
m2 = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float32).to(dev)
finetune(m2, steps=200, lr=2e-5, seed=1)
m2_v1 = save_fp32(m2, "m2_v1"); del m2; torch.cuda.empty_cache()
m2_head = make_head("m2_head")
m2_ad, m2_lora = make_lora("m2_lora")

res6 = {"model": MODEL2, "val_loss": {}, "pairs": {}}
for tag, d in (("v0", base_dir), ("v1", m2_v1), ("head", m2_head), ("lora_merged", m2_lora)):
    res6["val_loss"][tag] = vloss(d)
bb = {t: to_bf16(d, f"m2b_{t}") for t, d in (("v0", base_dir), ("v1", m2_v1), ("head", m2_head), ("lora", m2_lora))}
for lab, key in (("full ft", "v1"), ("last 2 layers", "head"), ("lora merged", "lora")):
    r = measure_pair(f"360M bf16 {lab}", bb["v0"], bb[key], 2)
    o_, n_ = bb["v0"] / "model.safetensors", bb[key] / "model.safetensors"
    rc, t, t2, ok, e = xdelta(o_, n_, W / "tmp.xd", W / "tmp.xd.recon")
    r["xdelta3"] = {"bytes": size(W / "tmp.xd") if (W / "tmp.xd").exists() else None, "ok": ok}
    for f in ("tmp.xd", "tmp.xd.recon"):
        if (W / f).exists():
            (W / f).unlink()
    res6["pairs"][lab] = r
res6["lora_adapter_bytes"] = size(m2_ad / "adapter_model.safetensors")
res6["pipeline_time_v0_v1"] = time_pipeline(bb["v0"] / "model.safetensors", bb["v1"] / "model.safetensors")
res6["cpu_decode"] = cpu_decode_test("m360", bb["v0"] / "model.safetensors", bb["v1"] / "model.safetensors")
(W / "results_model2.json").write_text(json.dumps(res6, indent=2, default=str))
print(json.dumps({k: res6[k] for k in ("val_loss", "lora_adapter_bytes", "pipeline_time_v0_v1")}, indent=2))

# %% export: only the small JSON files (not the models)
import zipfile
with zipfile.ZipFile("/content/results.zip", "w") as z:
    for f in sorted(W.glob("results_*.json")) + [W / "env.json"]:
        z.write(f, f.name)
try:
    from google.colab import files; files.download("/content/results.zip")
except Exception as e:
    print("download manually:", e)
