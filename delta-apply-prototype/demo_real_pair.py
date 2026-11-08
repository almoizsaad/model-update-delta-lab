#!/usr/bin/env python3
"""Run mdelta on a real pair of model files and try the failure cases on them.

Usage: python3 demo_real_pair.py OLD.safetensors NEW.safetensors RESULTS.json [WORKDIR]
Needs about 4x the size of one model file in free RAM and 3x on disk.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mdelta  # noqa: E402
from mdelta import Policy, Rejected, apply_delta, sha256  # noqa: E402


def main(old_p, new_p, out_json, work="/tmp/mdelta_demo"):
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    old, new = Path(old_p).read_bytes(), Path(new_p).read_bytes()
    pub, other = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
    policy = Policy("require", {"demo": pub.public_key().public_bytes_raw()})
    res = {"old_bytes": len(old), "new_bytes": len(new), "scenarios": []}

    t = time.perf_counter()
    blob = mdelta.create_delta(old, new, 2, "demo", pub)
    res["t_create_s"] = round(time.perf_counter() - t, 2)
    res["delta_bytes"] = len(blob); res["delta_pct_of_new"] = round(100 * len(blob) / len(new), 2)
    delta = work / "good.mdelta"; delta.write_bytes(blob)
    manifest, _, sig, payload = mdelta.parse(blob)

    out = work / "out.bin"
    t = time.perf_counter()
    report = apply_delta(old_p, delta, out, policy, expect_target=sha256(new))
    res["t_apply_total_s"] = round(time.perf_counter() - t, 2)
    res["apply_report"] = report
    res["result_identical"] = out.read_bytes() == new
    out.unlink()

    def scenario(name, expected, delta_path, old_path=old_p, **kw):
        try:
            apply_delta(old_path, delta_path, out, policy, **kw); got = "APPLIED"
        except Rejected as e:
            got = e.code
        left = out.exists()
        if left:
            out.unlink()
        res["scenarios"].append({"name": name, "expected": expected, "observed": got,
                                 "output_left_behind": left, "pass": got == expected and not left})
        print(f"{name:28s} expected={expected:22s} observed={got:22s} {'ok' if got == expected and not left else 'FAIL'}")

    bad_old = bytearray(old); bad_old[-1] ^= 1
    (work / "bad_old.bin").write_bytes(bytes(bad_old)); del bad_old
    scenario("wrong_base", "BASE_MISMATCH", delta, old_path=work / "bad_old.bin")

    flipped = bytearray(blob); flipped[len(blob) // 2] ^= 0xFF
    (work / "flipped.mdelta").write_bytes(bytes(flipped)); del flipped
    scenario("payload_flipped_byte", "PAYLOAD_HASH_MISMATCH", work / "flipped.mdelta")

    (work / "truncated.mdelta").write_bytes(blob[:-4096])
    scenario("truncated_delta", "PAYLOAD_HASH_MISMATCH", work / "truncated.mdelta")

    (work / "unsigned.mdelta").write_bytes(mdelta.pack(manifest, {}, payload))
    scenario("unsigned_under_require", "SIGNATURE_MISSING", work / "unsigned.mdelta")

    (work / "other.mdelta").write_bytes(mdelta.pack(manifest, mdelta.sign_block(manifest, "intruder", other), payload))
    scenario("unknown_publisher", "PUBLISHER_NOT_ACCEPTED", work / "other.mdelta")

    forged = dict(manifest, target_sha256=sha256(old))        # attacker edits the manifest, keeps the old signature
    (work / "forged.mdelta").write_bytes(mdelta.pack(forged, sig, payload))
    scenario("manifest_edited_after_signing", "SIGNATURE_INVALID", work / "forged.mdelta")

    scenario("pinned_to_other_version", "TARGET_PIN_MISMATCH", delta, expect_target=sha256(old))

    res["all_pass"] = all(s["pass"] for s in res["scenarios"]) and res["result_identical"]
    Path(out_json).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("delta_bytes", "delta_pct_of_new", "t_create_s", "t_apply_total_s", "result_identical", "all_pass")}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(*sys.argv[1:5])
