#!/usr/bin/env python3
"""mdelta: prototype of verified delta updates for model files.

A delta turns an exact old file into an exact new file. Applying one is only safe
if (1) it is bound to the old file, (2) the delta itself is intact, (3) the result
is the file the publisher meant, and (4) the publisher is one you accept. This
prototype checks all four, in that order of cost, and never touches the output
path unless every check passed.

Integrity checks (hashes) are always on. Only the trust policy (signature and
accepted publishers) has modes: disabled / warn / require.

Prototype limits: whole files are held in memory; old and new must have the same
size; Ed25519 key pinning stands in for real attestations (Sigstore / CEP 27);
freshness is NOT solved (see --expect-target-sha256 and the README).
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import tempfile
import time
import tomllib
from pathlib import Path

import numpy as np
import zstandard as zstd
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

MAGIC = b"MDELTA1\n"
CODEC = "xor-planes-zstd"
LEVEL = 12
MAX_META = 1 << 20          # manifest / signature block size cap
MODES = ("disabled", "warn", "require")
REQUIRED = {"format": int, "codec": str, "itemsize": int, "split": int, "base_size": int, "base_sha256": str,
            "target_size": int, "target_sha256": str, "payload_size": int, "payload_sha256": str}


class Rejected(Exception):
    """The update must not be applied. `code` is stable and machine readable."""

    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code, self.detail = code, detail


class ConfigError(Exception):
    pass


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def _hex64(s):
    return isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdef" for c in s)


# ---------------------------------------------------------------- byte layout
def split_point(buf):
    """Size of the safetensors header block (8 + header length), or 0 if the file does not look like one."""
    if len(buf) >= 9 and bytes(buf[8:9]) == b"{":
        n = struct.unpack("<Q", bytes(buf[:8]))[0]
        if 8 + n <= len(buf):
            return 8 + n
    return 0


def planes(a, itemsize):
    return np.ascontiguousarray(a.reshape(-1, itemsize).T).reshape(-1)


def unplanes(p, itemsize):
    n = p.size // itemsize
    return np.ascontiguousarray(p.reshape(itemsize, n).T).reshape(-1)


# ------------------------------------------------------------------- container
def canonical(manifest):
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()


def pack(manifest, sigblock, payload):
    m, s = canonical(manifest), json.dumps(sigblock, sort_keys=True).encode()
    return MAGIC + struct.pack("<Q", len(m)) + m + struct.pack("<Q", len(s)) + s + payload


def sign_block(manifest, key_id, private_key):
    sig = private_key.sign(MAGIC + canonical(manifest))
    return {"alg": "ed25519", "key_id": key_id, "sig": sig.hex()}


def parse(blob):
    """-> (manifest dict, manifest bytes, signature block dict, payload bytes). Raises Rejected(BAD_FORMAT)."""
    try:
        if not blob.startswith(MAGIC):
            raise ValueError("bad magic")
        pos = len(MAGIC)
        (mlen,) = struct.unpack_from("<Q", blob, pos); pos += 8
        if mlen > MAX_META:
            raise ValueError("manifest too large")
        mbytes = bytes(blob[pos:pos + mlen]); pos += mlen
        if len(mbytes) != mlen:
            raise ValueError("truncated manifest")
        (slen,) = struct.unpack_from("<Q", blob, pos); pos += 8
        if slen > MAX_META:
            raise ValueError("signature block too large")
        sbytes = bytes(blob[pos:pos + slen]); pos += slen
        if len(sbytes) != slen:
            raise ValueError("truncated signature block")
        manifest, sigblock = json.loads(mbytes), json.loads(sbytes)
        for k, t in REQUIRED.items():
            if not isinstance(manifest.get(k), t) or isinstance(manifest.get(k), bool):
                raise ValueError(f"manifest field {k!r} missing or wrong type")
        if manifest["format"] != 1 or manifest["codec"] != CODEC:
            raise ValueError("unsupported format or codec")
        if manifest["itemsize"] not in (1, 2, 4, 8) or manifest["split"] < 0:
            raise ValueError("bad itemsize or split")
        if manifest["base_size"] != manifest["target_size"]:
            raise ValueError("format 1 needs equal sizes")
        if not (_hex64(manifest["base_sha256"]) and _hex64(manifest["target_sha256"]) and _hex64(manifest["payload_sha256"])):
            raise ValueError("bad hash field")
        if not isinstance(sigblock, dict):
            raise ValueError("bad signature block")
        return manifest, mbytes, sigblock, bytes(blob[pos:])
    except (ValueError, struct.error, json.JSONDecodeError, UnicodeDecodeError) as e:
        raise Rejected("BAD_FORMAT", str(e))


# ---------------------------------------------------------------------- create
def build_parts(old, new, itemsize=2):
    """-> (manifest, payload). `old` and `new` are bytes-like and must have the same size and header layout."""
    old, new = np.frombuffer(old, dtype=np.uint8), np.frombuffer(new, dtype=np.uint8)
    if old.size != new.size:
        raise Rejected("LAYOUT_MISMATCH", "files have different sizes; send the full file instead")
    split = split_point(new)
    if split_point(old) != split:
        raise Rejected("LAYOUT_MISMATCH", "different header layout; send the full file instead")
    if (new.size - split) % itemsize:
        itemsize = 1
    x = np.bitwise_xor(old, new)
    stream = x[:split].tobytes() + planes(x[split:], itemsize).tobytes()
    payload = zstd.ZstdCompressor(level=LEVEL, threads=0).compress(stream)
    manifest = {"format": 1, "codec": CODEC, "itemsize": itemsize, "split": split,
                "base_size": int(old.size), "base_sha256": sha256(old.tobytes()),
                "target_size": int(new.size), "target_sha256": sha256(new.tobytes()),
                "payload_size": len(payload), "payload_sha256": sha256(payload)}
    return manifest, payload


def create_delta(old, new, itemsize=2, key_id=None, private_key=None):
    manifest, payload = build_parts(old, new, itemsize)
    sig = sign_block(manifest, key_id, private_key) if private_key is not None else {}
    return pack(manifest, sig, payload)


# ----------------------------------------------------------------------- trust
class Policy:
    def __init__(self, mode="disabled", publishers=None):
        if mode not in MODES:
            raise ConfigError(f"mode must be one of {MODES}")
        self.mode, self.publishers = mode, dict(publishers or {})
        if mode == "require" and not self.publishers:
            raise ConfigError("mode 'require' needs at least one publisher")


def load_policy(path):
    try:
        data = tomllib.loads(Path(path).read_text())
        pubs = {p["key_id"]: bytes.fromhex(p["public_key"]) for p in data.get("publisher", [])}
        return Policy(data.get("mode", "disabled"), pubs)
    except (OSError, tomllib.TOMLDecodeError, KeyError, ValueError, TypeError) as e:
        raise ConfigError(f"cannot read policy {path}: {e}")


def check_trust(manifest, sigblock, policy):
    """Return a list of warning codes, or raise Rejected under mode 'require'."""
    if policy.mode == "disabled":
        return []
    finding = None
    if not sigblock:
        finding = ("SIGNATURE_MISSING", "delta is not signed")
    elif sigblock.get("key_id") not in policy.publishers:
        finding = ("PUBLISHER_NOT_ACCEPTED", f"key_id {sigblock.get('key_id')!r} is not an accepted publisher")
    else:
        try:
            Ed25519PublicKey.from_public_bytes(policy.publishers[sigblock["key_id"]]).verify(
                bytes.fromhex(sigblock["sig"]), MAGIC + canonical(manifest))
        except (InvalidSignature, ValueError, KeyError, TypeError):
            finding = ("SIGNATURE_INVALID", "signature does not match the manifest")
    if finding is None:
        return []
    if policy.mode == "require":
        raise Rejected(*finding)
    return [finding[0]]


# ----------------------------------------------------------------------- apply
def apply_delta(old_path, delta_path, out_path, policy=None, expect_target=None):
    """Verify everything, reconstruct, verify the result, then atomically write `out_path`.
    On any failure `out_path` is left exactly as it was. Returns a report dict."""
    policy = policy or Policy()
    old_path, delta_path, out_path = Path(old_path), Path(delta_path), Path(out_path)
    if out_path.resolve() == old_path.resolve():
        raise ConfigError("output must not be the old file (keep it for rollback)")
    manifest, mbytes, sigblock, payload = parse(delta_path.read_bytes())
    warnings = check_trust(manifest, sigblock, policy)                       # 1. who published it
    if expect_target is not None and manifest["target_sha256"] != expect_target:   # 2. is it the version I asked for
        raise Rejected("TARGET_PIN_MISMATCH", "delta produces a different file than the pinned hash")
    old = np.fromfile(old_path, dtype=np.uint8)
    if old.size != manifest["base_size"] or sha256(old.tobytes()) != manifest["base_sha256"]:   # 3. right old file
        raise Rejected("BASE_MISMATCH", "the old file is not the one this delta was made for")
    if len(payload) != manifest["payload_size"] or sha256(payload) != manifest["payload_sha256"]:   # 4. delta intact
        raise Rejected("PAYLOAD_HASH_MISMATCH", "delta payload is corrupted or truncated")
    t0 = time.perf_counter()
    try:                                                                       # 5. decode, bounded by the declared size
        stream = zstd.ZstdDecompressor().decompress(payload, max_output_size=manifest["target_size"])
    except zstd.ZstdError as e:
        raise Rejected("DECODE_ERROR", str(e))
    if len(stream) != manifest["target_size"]:
        raise Rejected("DECODE_ERROR", "decoded size differs from the declared size")
    s, split, it = np.frombuffer(stream, dtype=np.uint8), manifest["split"], manifest["itemsize"]
    if split > s.size or (s.size - split) % it:
        raise Rejected("DECODE_ERROR", "inconsistent split or itemsize")
    x = np.concatenate([s[:split], unplanes(s[split:], it)])
    new = np.bitwise_xor(old, x).tobytes()
    t_decode = time.perf_counter() - t0
    if sha256(new) != manifest["target_sha256"]:                               # 6. the file the publisher meant
        raise Rejected("RESULT_MISMATCH", "reconstructed file does not match the signed target hash")
    out_path.parent.mkdir(parents=True, exist_ok=True)                         # 7. atomic write
    fd, tmp = tempfile.mkstemp(dir=out_path.parent, prefix=out_path.name + ".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(new); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, out_path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return {"ok": True, "warnings": warnings, "target_sha256": manifest["target_sha256"], "bytes_out": len(new),
            "delta_bytes": delta_path.stat().st_size, "t_decode_s": round(t_decode, 3),
            "signed_by": sigblock.get("key_id") if sigblock else None}


# ------------------------------------------------------------------------- CLI
def _load_private(path):
    return Ed25519PrivateKey.from_private_bytes(bytes.fromhex(Path(path).read_text().strip()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("keygen"); k.add_argument("--key-id", required=True); k.add_argument("--out", required=True)
    c = sub.add_parser("create")
    c.add_argument("--old", required=True); c.add_argument("--new", required=True); c.add_argument("--out", required=True)
    c.add_argument("--itemsize", type=int, default=2); c.add_argument("--sign-key"); c.add_argument("--key-id")
    a = sub.add_parser("apply")
    a.add_argument("--old", required=True); a.add_argument("--delta", required=True); a.add_argument("--out", required=True)
    a.add_argument("--policy"); a.add_argument("--expect-target-sha256"); a.add_argument("--report")
    i = sub.add_parser("inspect"); i.add_argument("--delta", required=True)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "keygen":
            out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
            priv = Ed25519PrivateKey.generate()
            (out / f"{args.key_id}.private").write_text(priv.private_bytes(
                serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption()).hex() + "\n")
            os.chmod(out / f"{args.key_id}.private", 0o600)
            pub = priv.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
            (out / f"{args.key_id}.public").write_text(pub + "\n")
            print(f"key_id={args.key_id}\npublic_key={pub}")
        elif args.cmd == "create":
            if bool(args.sign_key) != bool(args.key_id):
                raise ConfigError("--sign-key and --key-id go together")
            blob = create_delta(Path(args.old).read_bytes(), Path(args.new).read_bytes(), args.itemsize,
                                args.key_id, _load_private(args.sign_key) if args.sign_key else None)
            Path(args.out).write_bytes(blob)
            print(json.dumps({"delta_bytes": len(blob), "signed": bool(args.sign_key)}))
        elif args.cmd == "apply":
            policy = load_policy(args.policy) if args.policy else Policy()
            report = apply_delta(args.old, args.delta, args.out, policy, args.expect_target_sha256)
            print(json.dumps(report, indent=2))
            if args.report:
                Path(args.report).write_text(json.dumps(report, indent=2))
        elif args.cmd == "inspect":
            m, _, s, p = parse(Path(args.delta).read_bytes())
            print(json.dumps({"manifest": m, "signature": {k: s.get(k) for k in ("alg", "key_id")} if s else None}, indent=2))
        return 0
    except Rejected as e:
        print(json.dumps({"ok": False, "code": e.code, "detail": e.detail}), file=sys.stderr)
        if getattr(args, "report", None):
            Path(args.report).write_text(json.dumps({"ok": False, "code": e.code, "detail": e.detail}, indent=2))
        return 1
    except (ConfigError, OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
