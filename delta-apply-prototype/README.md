# delta-apply-prototype

A small Python prototype of **verified delta updates for model files**: what a package manager would need around a delta before it may replace a model on a device. It is a follow-up to the measurements in the parent directory ([REPORT.md](../REPORT.md)), which measured how small such deltas can be. ZipNN (arXiv 2411.05239) already covers the compression side; this prototype is about what happens around it.

It is a prototype to specify and test behaviour, not a Pixi or rattler feature, and not production code.

## What it does

`mdelta.py` creates a delta (XOR with the old file, byte planes, zstd) inside a small container with a manifest, and applies it only if every check passes:

1. **Who published it** (trust policy): `disabled`, `warn` or `require`, with accepted publishers (Ed25519 public keys). Same modes as my earlier per-channel policy checker.
2. **Is it the version I asked for** (optional): `--expect-target-sha256`, for example a hash taken from a lock file.
3. **Is it for my old file**: the exact old file's sha256 and size.
4. **Is the delta intact**: payload sha256 and size.
5. **Decode with a size bound**: output is limited to the declared size, so a small payload cannot expand without limit.
6. **Is the result what the publisher meant**: sha256 of the reconstructed file against the signed target hash.
7. **Atomic write**: the output is written to a temporary file and renamed. The old file is never modified (that is the rollback), and the output path is untouched on any failure.

Integrity checks (3-6) are always on. Only the trust policy (1) has modes.

## Failure cases covered (tests/test_mdelta.py, 23 tests)

| Case | Result |
|---|---|
| Wrong or modified old file | `BASE_MISMATCH` |
| Flipped byte in the payload, truncated delta | `PAYLOAD_HASH_MISMATCH` |
| Not a delta file | `BAD_FORMAT` |
| Unsigned under `require` / under `warn` | `SIGNATURE_MISSING` / applies with a warning |
| Signed by a key that is not accepted | `PUBLISHER_NOT_ACCEPTED` |
| Wrong key claiming an accepted key id; manifest edited after signing | `SIGNATURE_INVALID` |
| Valid signature, but the payload decodes to a different file | `RESULT_MISMATCH` |
| Valid signature, payload that expands far beyond the declared size | `DECODE_ERROR` (bounded) |
| Inconsistent split/itemsize in a signed manifest | rejected, no crash |
| Files with different size or header layout | cannot be created (`LAYOUT_MISMATCH`): send the full file |
| Output equal to the old file | configuration error |
| Existing output file after a failed update | unchanged, no temp files left |
| Valid, signed, **older** delta (replay) | accepted unless the target hash is pinned; then `TARGET_PIN_MISMATCH` |

I checked that the tests detect their target: removing the result check, the base check, the payload check, the signature check or the pin check each makes at least one test fail.

## Result on a real pair

`make_real_pair.py` + `demo_real_pair.py` on SmolLM2-135M in bf16 and a 200-step fine-tune of it (269,060,552 bytes each), Colab. Raw output: `results/delta_demo.json`.

| Measure | Value |
|---|---|
| Signed delta file | 67,544,206 bytes (25.1% of the new file) |
| Create (reading, hashing, XOR, zstd, signing) | 21.4 s |
| Apply, total (policy `require`, target hash pinned) | 5.5 s, of which decoding 1.7 s; the rest (hashes, reading, writing, fsync) was not timed separately |
| Reconstructed file identical to the new file | yes |
| Failure scenarios on the real files (wrong base, flipped payload byte, truncated delta, unsigned, unknown publisher, manifest edited after signing, pinned to another version) | 7 of 7 rejected with the expected code, no output file left behind |

The pair was regenerated for this demo and is not guaranteed to be byte-identical to the one in the lab; the lab's payload-only figure was 67,529,204 bytes, so the container and header add very little.

## Limits

- **Freshness is not solved.** A valid signed older delta is accepted unless the caller pins the target hash from a trusted source. Real freshness needs signed version or expiry metadata (TUF-style); I have not implemented that.
- Ed25519 key pinning stands in for real attestations (Sigstore / CEP 27 / CEP 50). Keys here are plain files, with no rotation or revocation. A compromised accepted publisher is not addressed.
- Whole files are held in memory (about 4x one file); there is no streaming. Old and new files must have the same size and header layout.
- The decoder is single-threaded numpy; speed is not optimised (see the lab's decode timings).
- No Pixi, rattler or conda integration; no channel or repodata format; no fallback logic beyond returning a reject code (the caller must download the full file).
- Tested in unit tests on synthetic files and once on one real pair of model files (below): one pair, one Colab session, hardware not recorded. The replay case is covered only by the unit test.

## Use

```bash
pip install numpy zstandard cryptography        # Python 3.11+
python3 tests/test_mdelta.py                      # 23 tests

python3 mdelta.py keygen --key-id demo --out keys
python3 mdelta.py create --old v1.safetensors --new v2.safetensors --out v1_to_v2.mdelta \
    --sign-key keys/demo.private --key-id demo
cat > policy.toml <<'TOML'
mode = "require"
[[publisher]]
key_id = "demo"
public_key = "<contents of keys/demo.public>"
TOML
python3 mdelta.py apply --old v1.safetensors --delta v1_to_v2.mdelta --out v2.safetensors \
    --policy policy.toml --expect-target-sha256 <sha256 of v2 from a trusted source>
```

Exit codes: 0 applied, 1 rejected (a JSON line with the code on stderr), 2 usage or configuration error.

On a real pair of files (Colab with a GPU creates one): `python3 make_real_pair.py`, then `python3 demo_real_pair.py /content/work/b_v0/model.safetensors /content/work/b_v1/model.safetensors results.json`.


