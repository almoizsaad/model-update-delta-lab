"""Failure-case tests for mdelta. Run: python3 tests/test_mdelta.py  (needs numpy, zstandard, cryptography)."""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import zstandard as zstd
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mdelta  # noqa: E402
from mdelta import Policy, Rejected, apply_delta, create_delta, sha256  # noqa: E402


def make_file(n, seed, change_from=None, frac=0.5):
    """A safetensors-like file: 8-byte length + padded JSON header + n bf16-like values."""
    rng = np.random.default_rng(seed)
    header = json.dumps({"w": {"dtype": "BF16", "shape": [n], "data_offsets": [0, 2 * n]}}).encode()
    header += b" " * ((-len(header)) % 8)
    prefix = struct.pack("<Q", len(header)) + header
    if change_from is None:
        hi, lo = rng.choice(np.arange(0x3C, 0x40, dtype=np.uint8), n), rng.integers(0, 256, n, dtype=np.uint8)
    else:
        arr = np.frombuffer(change_from[len(prefix):], dtype=np.uint8)
        lo, hi = arr[0::2].copy(), arr[1::2].copy()
        mask = rng.random(n) < frac
        lo[mask] = rng.integers(0, 256, int(mask.sum()), dtype=np.uint8)
    data = np.empty(2 * n, dtype=np.uint8); data[0::2], data[1::2] = lo, hi
    return prefix + data.tobytes()


class Base(unittest.TestCase):
    N = 1 << 15

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.d = Path(self.tmp.name)
        self.old = make_file(self.N, 1)
        self.new1 = make_file(self.N, 2, self.old, 0.5)
        self.new2 = make_file(self.N, 3, self.old, 0.3)
        self.pub_key, self.other_key = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
        self.policy = Policy("require", {"pub": self.pub_key.public_key().public_bytes_raw()})
        self.old_p = self.d / "old.bin"; self.old_p.write_bytes(self.old)
        self.out = self.d / "out.bin"

    def tearDown(self):
        self.tmp.cleanup()

    def delta(self, new=None, key="pub", key_obj=None, name="d.mdelta"):
        blob = create_delta(self.old, new or self.new1, 2, key, key_obj or (self.pub_key if key else None))
        p = self.d / name; p.write_bytes(blob); return p

    def reject(self, delta_p, code, **kw):
        with self.assertRaises(Rejected) as cm:
            apply_delta(self.old_p, delta_p, self.out, **kw)
        self.assertEqual(cm.exception.code, code)
        self.assertFalse(self.out.exists(), "output must not exist after a rejected update")


class TestHappyPath(Base):
    def test_signed_roundtrip_and_smaller_than_full(self):
        p = self.delta()
        r = apply_delta(self.old_p, p, self.out, self.policy)
        self.assertEqual(self.out.read_bytes(), self.new1)
        self.assertEqual(r["signed_by"], "pub"); self.assertEqual(r["warnings"], [])
        self.assertLess(p.stat().st_size, len(self.new1) * 0.8)

    def test_unsigned_with_trust_disabled(self):
        apply_delta(self.old_p, self.delta(key=None), self.out)
        self.assertEqual(self.out.read_bytes(), self.new1)

    def test_deterministic(self):
        a = create_delta(self.old, self.new1, 2, "pub", self.pub_key)
        b = create_delta(self.old, self.new1, 2, "pub", self.pub_key)
        self.assertEqual(a, b)


class TestWrongInputs(Base):
    def test_wrong_base(self):
        p = self.delta()
        bad = bytearray(self.old); bad[-1] ^= 1; self.old_p.write_bytes(bytes(bad))
        self.reject(p, "BASE_MISMATCH", policy=self.policy)

    def test_wrong_size_base(self):
        p = self.delta(); self.old_p.write_bytes(self.old[:-8])
        self.reject(p, "BASE_MISMATCH", policy=self.policy)

    def test_payload_flipped_byte(self):
        b = bytearray(self.delta().read_bytes()); b[-5] ^= 0xFF
        p = self.d / "bad.mdelta"; p.write_bytes(bytes(b))
        self.reject(p, "PAYLOAD_HASH_MISMATCH", policy=self.policy)

    def test_truncated_delta(self):
        p = self.d / "t.mdelta"; p.write_bytes(self.delta().read_bytes()[:-100])
        self.reject(p, "PAYLOAD_HASH_MISMATCH", policy=self.policy)

    def test_garbage_file(self):
        p = self.d / "g.mdelta"; p.write_bytes(b"not a delta at all")
        self.reject(p, "BAD_FORMAT", policy=self.policy)

    def test_different_layout_cannot_be_created(self):
        with self.assertRaises(Rejected) as cm:
            create_delta(self.old, make_file(self.N * 2, 5))
        self.assertEqual(cm.exception.code, "LAYOUT_MISMATCH")

    def test_output_may_not_be_the_old_file(self):
        with self.assertRaises(mdelta.ConfigError):
            apply_delta(self.old_p, self.delta(), self.old_p, self.policy)


class TestTrust(Base):
    def test_unsigned_under_require(self):
        self.reject(self.delta(key=None), "SIGNATURE_MISSING", policy=self.policy)

    def test_unsigned_under_warn_applies_with_warning(self):
        r = apply_delta(self.old_p, self.delta(key=None), self.out, Policy("warn"))
        self.assertEqual(r["warnings"], ["SIGNATURE_MISSING"]); self.assertEqual(self.out.read_bytes(), self.new1)

    def test_unknown_publisher(self):
        self.reject(self.delta(key="someone-else", key_obj=self.other_key), "PUBLISHER_NOT_ACCEPTED", policy=self.policy)

    def test_wrong_key_claiming_accepted_id(self):
        self.reject(self.delta(key="pub", key_obj=self.other_key), "SIGNATURE_INVALID", policy=self.policy)

    def test_manifest_tampered_after_signing(self):
        blob = bytearray(self.delta().read_bytes())
        m_off = len(mdelta.MAGIC) + 8
        mlen = struct.unpack_from("<Q", blob, len(mdelta.MAGIC))[0]
        manifest = bytes(blob[m_off:m_off + mlen]).replace(sha256(self.new1).encode(), sha256(self.new2).encode())
        blob[m_off:m_off + mlen] = manifest
        p = self.d / "m.mdelta"; p.write_bytes(bytes(blob))
        self.reject(p, "SIGNATURE_INVALID", policy=self.policy)

    def test_signed_but_wrong_policy_mode_values(self):
        with self.assertRaises(mdelta.ConfigError):
            Policy("require", {})            # require without a publisher is a configuration error
        with self.assertRaises(mdelta.ConfigError):
            Policy("strict")


class TestPublisherErrors(Base):
    """A publisher whose signature is valid can still ship a bad delta; hash checks catch it."""

    def forged(self, manifest, payload):
        return self.sign_and_write(manifest, payload)

    def sign_and_write(self, manifest, payload):
        p = self.d / "f.mdelta"
        p.write_bytes(mdelta.pack(manifest, mdelta.sign_block(manifest, "pub", self.pub_key), payload)); return p

    def test_signed_manifest_whose_payload_decodes_to_something_else(self):
        manifest, payload = mdelta.build_parts(self.old, self.new1)
        manifest["target_sha256"] = sha256(self.new2)            # claims new2, payload encodes new1
        self.reject(self.sign_and_write(manifest, payload), "RESULT_MISMATCH", policy=self.policy)

    def test_decompression_bomb_is_bounded(self):
        manifest, _ = mdelta.build_parts(self.old, self.new1)
        payload = zstd.ZstdCompressor().compress(b"\0" * (len(self.old) * 50))
        manifest["payload_size"], manifest["payload_sha256"] = len(payload), sha256(payload)
        self.reject(self.sign_and_write(manifest, payload), "DECODE_ERROR", policy=self.policy)

    def test_inconsistent_split_is_rejected_not_crashing(self):
        manifest, payload = mdelta.build_parts(self.old, self.new1)
        manifest["itemsize"] = 8                                   # data length is not a multiple of 8 after a bad split
        manifest["split"] = manifest["split"] + 1
        p = self.sign_and_write(manifest, payload)
        with self.assertRaises(Rejected):
            apply_delta(self.old_p, p, self.out, self.policy)
        self.assertFalse(self.out.exists())


class TestFreshness(Base):
    """Documents a known limit: a valid, signed, OLDER delta is accepted unless the caller pins the target hash."""

    def test_replay_of_older_valid_delta(self):
        older, newer = self.delta(self.new1, name="older.mdelta"), self.delta(self.new2, name="newer.mdelta")
        apply_delta(self.old_p, older, self.out, self.policy)            # accepted: nothing says it is outdated
        self.out.unlink()
        self.reject(older, "TARGET_PIN_MISMATCH", policy=self.policy, expect_target=sha256(self.new2))
        apply_delta(self.old_p, newer, self.out, self.policy, expect_target=sha256(self.new2))
        self.assertEqual(self.out.read_bytes(), self.new2)


class TestAtomicity(Base):
    def test_existing_output_untouched_on_failure(self):
        self.out.write_bytes(b"previous good file")
        b = bytearray(self.delta().read_bytes()); b[-5] ^= 0xFF
        p = self.d / "bad.mdelta"; p.write_bytes(bytes(b))
        with self.assertRaises(Rejected):
            apply_delta(self.old_p, p, self.out, self.policy)
        self.assertEqual(self.out.read_bytes(), b"previous good file")
        self.assertEqual([x.name for x in self.d.iterdir() if ".tmp" in x.name], [])

    def test_no_temp_files_left_after_success(self):
        apply_delta(self.old_p, self.delta(), self.out, self.policy)
        self.assertEqual([x.name for x in self.d.iterdir() if ".tmp" in x.name], [])


class TestCli(Base):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(ROOT / "mdelta.py"), *map(str, args)], capture_output=True, text=True)

    def test_end_to_end_and_exit_codes(self):
        new_p = self.d / "new.bin"; new_p.write_bytes(self.new1)
        r = self.run_cli("keygen", "--key-id", "demo", "--out", self.d / "keys"); self.assertEqual(r.returncode, 0, r.stderr)
        pub = (self.d / "keys" / "demo.public").read_text().strip()
        (self.d / "policy.toml").write_text(f'mode = "require"\n[[publisher]]\nkey_id = "demo"\npublic_key = "{pub}"\n')
        r = self.run_cli("create", "--old", self.old_p, "--new", new_p, "--out", self.d / "d.mdelta",
                         "--sign-key", self.d / "keys" / "demo.private", "--key-id", "demo"); self.assertEqual(r.returncode, 0, r.stderr)
        r = self.run_cli("apply", "--old", self.old_p, "--delta", self.d / "d.mdelta", "--out", self.out,
                         "--policy", self.d / "policy.toml", "--expect-target-sha256", sha256(self.new1))
        self.assertEqual(r.returncode, 0, r.stderr); self.assertEqual(self.out.read_bytes(), self.new1)
        self.out.unlink()
        r = self.run_cli("apply", "--old", self.old_p, "--delta", self.d / "d.mdelta", "--out", self.out,
                         "--policy", self.d / "policy.toml", "--expect-target-sha256", sha256(self.new2))
        self.assertEqual(r.returncode, 1); self.assertIn("TARGET_PIN_MISMATCH", r.stderr)
        r = self.run_cli("apply", "--old", self.old_p, "--delta", self.d / "d.mdelta", "--out", self.out, "--policy", self.d / "missing.toml")
        self.assertEqual(r.returncode, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
