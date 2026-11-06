# Report: delta updates between model versions

All numbers come from one complete run of `delta_lab.py` on Colab (GPU runtime, 2 vCPU Intel Xeon @ 2.00 GHz); the JSON files are in `results/`. The rounded percentages of that run match those of my earlier interactive run. Percentages are relative to the size of the new file's tensor data (fp32: 538,060,032 bytes; bf16: 269,030,016 bytes for 135M; 723,642,240 bytes for 360M).

## Setup

- Models: SmolLM2-135M (main) and SmolLM2-360M (second model, same family and tokenizer, 2.7x larger). Both are published as bf16; the fp32 files are upcasts, which matters (see Corrections).
- Data: WikiText-2 (blocks of 256 tokens, 5,365 train / 533 validation blocks; validation loss on 40 blocks). AdamW, no weight decay, batch 8, learning rate 2e-5 (LoRA: 1e-4, r=8, alpha=16, q_proj and v_proj).
- Updates: full fine-tune (200 steps), a second full fine-tune on top (v1 to v2), last two layers only, LoRA (adapter and merged), and a step sweep (25 to 1600 steps).
- Methods: `zstd --patch-from` (level 12, `--long=30`), `xdelta3 -9`, and a simple lossless pipeline: XOR with the old file, regroup bytes by position inside each element ("byte planes"), then zstd level 12. Every lossless result was decoded and compared by hash or array equality (`roundtrip_ok`/`ok` are true everywhere).

## Hypotheses (written before each run) and outcomes

| ID | Hypothesis | Outcome |
|---|---|---|
| H1 | Generic byte-level diff does not help after a dense fine-tune | Supported. fp32 full fine-tune: zstd alone 498.6 MB (92.7%), `zstd --patch-from` 498.6 MB (92.7%), xdelta3 506.7 MB (94.2%). Unchanged bytes at the same offset: 0.0-0.01% for 4 KB, 64 KB and 1 MB chunks. Nuance: when whole tensors are unchanged, generic patching does help (last 2 layers, bf16: 4.0%) |
| H2 | A LoRA adapter is much smaller than the diff of the merged model | Supported. 135M: adapter 1.86 MB vs 35.1 MB (fp32) and 8.1 MB (bf16) merged delta, i.e. 19x and 4.4x. 360M (bf16): 3.29 MB vs 22.6 MB (6.9x) |
| H3 | Last-layers-only fine-tune: XOR+planes under 10% of raw | Supported. 135M: 3.4% (fp32), 1.4% (bf16). 360M (bf16): 1.4% |
| H4 | bf16 halves the file with small quality loss | Supported on this model. File 269.1 MB vs 538.1 MB. Validation loss 2.669 (bf16) vs 2.661 (fp32). fp16 gave 2.661 but compresses worse (zstd 92.2% of its size vs 76.7% for bf16) |
| H5 | XOR + byte planes beats generic diff | Supported. 135M full fine-tune: 63.7% (fp32), 25.1% (bf16) vs 92.7% and 76.1% for `--patch-from`. 360M (bf16): 23.6% vs 75.9% (patch) and 74.2% (xdelta3) |
| H6 | Byte planes help even without the old model | Supported but modest. zstd alone vs zstd on planes: fp32 498.6 to 456.7 MB (-8%), bf16 206.2 to 187.7 MB (-9%) |
| H7 | A chain v1 to v2 behaves like v0 to v1 | Supported. fp32 62.1% vs 63.7%; bf16 24.1% vs 25.1% |
| H8 | bf16 dense-fine-tune delta is under 75% of raw | Supported (25.1%) |
| H9 | Delta size grows with the number of fine-tuning steps | Supported, see sweep |
| H10 | The pattern also holds for a second model | Supported for a larger model of the same family (SmolLM2-360M). Not tested for a different architecture |
| H11 | Decoding a bf16 delta takes a few seconds on one CPU core | Supported on a server CPU, see below. Not tested on robot-class hardware |

## Where the savings come from

In the bf16 135M full fine-tune, 47.5% of the 16-bit weights are bit-identical after 200 steps, and the high byte (sign and most of the exponent) is identical for 98.8%. After XOR and regrouping, the high-byte plane compresses to 3.4 MB while the low-byte plane is still 64.1 MB. So the delta is almost entirely low-mantissa noise. This also explains why generic diffs fail: changed values are spread everywhere, and a shifted value changes bytes at every position.

## Step sweep (135M, bf16, from the base model)

| Steps | Delta, % of raw | Delta size | Identical 16-bit weights | Val loss (fp32 model) | Encode / decode |
|---|---|---|---|---|---|
| 25 | 14.8 | 39.9 MB | 74.9% | 2.8249 | 14.0 s / 1.6 s |
| 100 | 21.5 | 57.9 MB | 57.4% | 2.7078 | 16.4 s / 2.2 s |
| 400 | 28.5 | 76.8 MB | 38.0% | 2.6243 | 17.2 s / 2.6 s |
| 1600 | 36.2 | 97.5 MB | 19.0% | 2.5992 | 16.3 s / 2.0 s |

Roughly +3.5 percentage points per doubling of steps. At 1600 steps the delta (97.5 MB) is still under half of the bf16 file compressed alone (206.2 MB). Validation-loss gains diminish while the delta keeps growing. Base fp32 validation loss was 3.0235. Learning rate was not varied.

## Second model (SmolLM2-360M, bf16, 200 steps)

| Update | XOR+planes+zstd | `zstd --patch-from` | xdelta3 | Validation loss (base 2.6946) |
|---|---|---|---|---|
| Full fine-tune | 170.4 MB (23.6%) | 549.5 MB (75.9%) | 537.0 MB (74.2%) | 2.3627 |
| Last 2 layers | 10.3 MB (1.4%) | 30.0 MB (4.1%) | 29.4 MB (4.1%) | 2.5644 |
| LoRA merged | 22.6 MB (3.1%) | 61.5 MB (8.5%) | 69.7 MB (9.6%) | 2.4841 |

LoRA adapter: 3.29 MB. Encode 51 s, in-process decode 4.8 s. Numbers for the two bf16 conditions are in the same range as for 135M.

## Decode time on restricted CPU

Decoding (zstd decompress, undo byte planes, XOR with the old file's data) was run in a separate process pinned with `taskset`, 3 repetitions, file reading excluded. Machine: 2 vCPU Intel Xeon @ 2.00 GHz.

| Model | Delta | 1 core (3 runs) | 2 cores (3 runs) |
|---|---|---|---|
| 135M bf16 (269 MB) | 67.5 MB | 1.48 / 1.71 / 2.21 s | 1.42 / 1.43 / 1.44 s |
| 360M bf16 (724 MB) | 170.4 MB | 3.79 / 3.85 / 5.25 s | 3.76 / 3.81 / 4.64 s |

Two cores give no real speedup: the decoder is single-threaded. About 60-70% of the time is the regrouping and XOR in numpy, not zstd, so this part could likely be faster in compiled code. Memory use was not measured.

## Corrections made while working

- My first reading of the fp32 results ignored that the base model is a bf16 model upcast to fp32 (the low bytes of the original weights are zero). That makes fp32 numbers look different from a model trained in fp32. The bf16 files are the realistic case, and I report them first.
- The LoRA advantage (19x) holds only for fp32; in bf16 it is 4.4x (135M) and 6.9x (360M).
- I first expected generic diffs to be useless in every case; they are useful when entire tensors are unchanged (last-layers update: 4.0-4.9% vs 1.4-3.4% for XOR+planes).

## Limits

- Two models of one family (135M and 360M). Not tested: other architectures, billion-parameter models, or a different tokenizer.
- One seed per condition, one dataset (WikiText-2, which is not a realistic fine-tune), a 40-block validation set, and one fixed learning rate. 1600 steps is about 2.4 epochs of the training subset.
- Encoding and decoding times come from a shared Colab server CPU, not from an ARM or embedded device. Decode memory use is not measured.
- A delta needs the exact old file on the device and the same tensors in the same layout; it does not work if the architecture changes. I did not study chains longer than one step, rollback, or what happens if the device holds a corrupted old file (the hash of the result should be checked).
- I read ZipNN only after running these experiments (see Related work). The pipeline here is a simplified version of known techniques, so this is a measurement, not a new method.
- No claims here about security: a delta must still be signed and verified like any other package.

## Related work

ZipNN (Hershcovitch et al., arXiv 2411.05239) already describes lossless compression of model weights by separating the exponent bytes (exponent extraction, byte grouping), Huffman-only coding, XOR delta compression of checkpoints and fine-tuned models, periodic base checkpoints, multi-threaded decoding, and an open-source implementation. I read pages 1-10 of the paper (not the reference list). Overlap and comparison with this study:

- Standalone bf16: ZipNN reports Zstd 77.6% going to 68.8% with exponent extraction on Llama 3.1 BF16. Here zstd alone gives 76.7% and zstd on byte planes 69.8% (135M). This is consistent with their finding.
- Delta: they use XOR as well, show that deltas shrink as training converges (ResNet-18, Amber, Olmo checkpoints), and report for fine-tuned pairs on Hugging Face a delta of about 56% of size against 83.7% standalone. My bf16 results (25% after 200 steps, 14.8-36.2% across 25-1600 steps) are for a different model and training setup and are not a like-for-like comparison.
- Speed: their C implementation reports single-thread decompression of 1.65 GB/s (Llama 3.1 BF16, Apple M1 Max). My numpy decoder reached about 0.16-0.19 GB/s on one core of a Colab Xeon, i.e. several times slower on different hardware; I have not compared on the same machine.
- What this study adds, modestly: a comparison across update types (last layers, LoRA merged vs adapter) against generic diff baselines (`zstd --patch-from`, xdelta3), a step sweep with validation loss for LLM fine-tuning, the bf16-upcast pitfall for fp32 results, and decode timing pinned to one core. In the pages I read I did not see the LoRA adapter versus merged-delta comparison.
- Not covered by either: delta updates inside a package manager (versioning, signing the delta, fallback to a full download, rollback).

## Licences

Code: MIT. Models: SmolLM2-135M and SmolLM2-360M, Apache-2.0 according to the model cards (checked 6 November 2026; I checked the 135M card). Dataset: WikiText-2 via `Salesforce/wikitext`; its card lists `cc-by-sa-3.0` and `gfdl` in the metadata but names CC BY-SA 4.0 in its text, so I do not state a single version. This repository contains no model weights and no dataset files, only code and result numbers.

