"""Photo storage behind a tiny interface (local disk now, S3-compatible later)."""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Protocol

ALLOWED_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
MIN_PHOTOS, MAX_PHOTOS = 2, 4


class UploadError(ValueError):
    pass


def sniff_image_type(data: bytes) -> str | None:
    """Content type from magic bytes; the client-declared type is never trusted."""
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def validate_image(data: bytes, max_bytes: int) -> str:
    if not data:
        raise UploadError("קובץ ריק")
    if len(data) > max_bytes:
        raise UploadError(f"התמונה גדולה מדי (עד {max_bytes // (1024 * 1024)}MB)")
    ctype = sniff_image_type(data)
    if ctype not in ALLOWED_TYPES:
        raise UploadError("אפשר להעלות רק JPG, PNG או WEBP")
    return ctype


class Storage(Protocol):
    def save(self, data: bytes, content_type: str, prefix: str) -> str: ...
    def load(self, key: str) -> bytes: ...
    def delete(self, key: str) -> None: ...


class LocalStorage:
    """Files under `root`, with random names. TODO(s3): add an S3Storage with the same API."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root.resolve() not in p.parents:
            raise UploadError("bad storage key")
        return p

    def save(self, data: bytes, content_type: str, prefix: str) -> str:
        key = f"{prefix}/{secrets.token_urlsafe(16)}{ALLOWED_TYPES[content_type]}"
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return key

    def load(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def delete(self, key: str) -> None:
        try:
            self._path(key).unlink()
        except FileNotFoundError:
            pass
