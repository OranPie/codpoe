from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ProviderSecretService:
    path: Path

    def save(self, user_key: str, payload: dict[str, Any]) -> None:
        key = self._validate_user_key(user_key)
        blob = self._encrypt(key, payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(blob, ensure_ascii=True), encoding="utf-8")

    def load(self, user_key: str) -> dict[str, Any]:
        key = self._validate_user_key(user_key)
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))
        raw = self.path.read_text(encoding="utf-8")
        blob = json.loads(raw)
        return self._decrypt(key, blob)

    @staticmethod
    def _validate_user_key(value: str) -> str:
        key = (value or "").strip()
        if len(key) < 4:
            raise ValueError("user_key must be at least 4 chars")
        return key

    @staticmethod
    def _encrypt(user_key: str, payload: dict[str, Any]) -> dict[str, str | int]:
        salt = os.urandom(16)
        nonce = os.urandom(16)
        key = hashlib.scrypt(user_key.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32)
        plain = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        stream = ProviderSecretService._keystream(key, nonce, len(plain))
        cipher = bytes(a ^ b for a, b in zip(plain, stream))
        mac = hmac.new(key, salt + nonce + cipher, hashlib.sha256).digest()
        return {
            "v": 1,
            "kdf": "scrypt",
            "salt": base64.b64encode(salt).decode("ascii"),
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ciphertext": base64.b64encode(cipher).decode("ascii"),
            "mac": base64.b64encode(mac).decode("ascii"),
        }

    @staticmethod
    def _decrypt(user_key: str, blob: dict[str, Any]) -> dict[str, Any]:
        salt = base64.b64decode(str(blob["salt"]))
        nonce = base64.b64decode(str(blob["nonce"]))
        cipher = base64.b64decode(str(blob["ciphertext"]))
        mac = base64.b64decode(str(blob["mac"]))
        key = hashlib.scrypt(user_key.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32)
        check = hmac.new(key, salt + nonce + cipher, hashlib.sha256).digest()
        if not hmac.compare_digest(mac, check):
            raise ValueError("invalid user_key or corrupted secret file")
        stream = ProviderSecretService._keystream(key, nonce, len(cipher))
        plain = bytes(a ^ b for a, b in zip(cipher, stream))
        return json.loads(plain.decode("utf-8"))

    @staticmethod
    def _keystream(key: bytes, nonce: bytes, size: int) -> bytes:
        out = bytearray()
        counter = 0
        while len(out) < size:
            block = hashlib.sha256(key + nonce + counter.to_bytes(8, "big")).digest()
            out.extend(block)
            counter += 1
        return bytes(out[:size])
