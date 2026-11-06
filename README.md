# model-update-delta-lab

A small, measured study: **how many bytes must be sent to update a model on a device, depending on the kind of update and the method used?**

It uses a 135M-parameter model (SmolLM2-135M) and a few kinds of update: full fine-tune, last-layers-only fine-tune, LoRA (adapter and merged), and a precision cast. It compares generic diff tools (`zstd --patch-from`, `xdelta3`) with a simple lossless pipeline: XOR with the old file, byte-plane grouping, then zstd.

This is an independent measurement of known techniques, with some extra comparisons (update types, LoRA, step sweep, decode timing).

Context: model packages (for example conda packages holding weights) are re-downloaded in full on every new version. I wanted to know how much a delta could save and when it does not.

## Main results (details and caveats in `REPORT.md`)

- Generic byte-level diffs (`zstd --patch-from`, xdelta3) give almost nothing after a dense fine-tune (about 76% of the bf16 file, 93% of the fp32 file).
- XOR + byte planes + zstd: 25.1% of raw for bf16 (63.7% for fp32), one 200-step fine-tune of SmolLM2-135M. SmolLM2-360M: 23.6%.
- Fine-tuning only the last two layers: 1.4% (bf16). LoRA: 3.0-3.1% as a merged delta, but the adapter itself is 4.4-6.9x smaller than that.
- Delta size grows with the amount of fine-tuning: 14.8% / 21.5% / 28.5% / 36.2% of raw bf16 at 25 / 100 / 400 / 1600 steps.
- Decoding on one core of a Colab Xeon: about 1.7 s for 269 MB and 3.9 s for 724 MB (not measured on embedded hardware).

## Limits

Two small models of one family, one dataset (WikiText-2), one seed per condition, a 40-block validation set, a fixed learning rate, and timing only on a server CPU. The base models are bf16 upcast to fp32, which affects the fp32 numbers (see REPORT). This is not a new method: ZipNN (arXiv 2411.05239) already covers byte grouping and XOR delta compression for model weights. I read it after running the experiments; `REPORT.md` has a related-work section comparing the two.

## Run it

Open Colab with a GPU, paste `delta_lab.py` (cells are marked with `# %%`), run top to bottom (about 2 hours, 15-20 GB of disk). Results are written as JSON to `/content/work` and zipped at the end. A second full run reproduced the reported percentages; bit-identical results are not guaranteed on other GPUs or library versions.

## Files

- `delta_lab.py` - the whole experiment
- `REPORT.md` - hypotheses, results, corrections, limits
- `results/` - JSON files from the run reported in `REPORT.md` (`env.json` records versions)
