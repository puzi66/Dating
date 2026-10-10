"""Photo storage behind a tiny interface: local disk, or the database (for hosts with no persistent disk)."""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from .config import Settings
    from .db import Database

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
    # `session`: the caller's open DB session. DbStorage writes through it so the blob commits
    # (or rolls back) together with the Photo row; disk storage ignores it.
    def save(self, data: bytes, content_type: str, prefix: str, session: Session | None = None) -> str: ...
    def load(self, key: str) -> bytes: ...
    def delete(self, key: str, session: Session | None = None) -> None: ...


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

    def save(self, data: bytes, content_type: str, prefix: str, session: Session | None = None) -> str:
        key = f"{prefix}/{secrets.token_urlsafe(16)}{ALLOWED_TYPES[content_type]}"
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return key

    def load(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def delete(self, key: str, session: Session | None = None) -> None:
        try:
            self._path(key).unlink()
        except FileNotFoundError:
            pass


class DbStorage:
    """Photo bytes in the `stored_blobs` table. Fine for a pilot (~200 photos); use S3 beyond that."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def save(self, data: bytes, content_type: str, prefix: str, session: Session | None = None) -> str:
        from .models import StoredBlob
        key = f"{prefix}/{secrets.token_urlsafe(16)}{ALLOWED_TYPES[content_type]}"
        if session is not None:
            session.add(StoredBlob(key=key, content_type=content_type, data=data))
            return key
        with self.db.SessionLocal() as s:
            s.add(StoredBlob(key=key, content_type=content_type, data=data))
            s.commit()
        return key

    def load(self, key: str) -> bytes:
        from .models import StoredBlob
        with self.db.SessionLocal() as s:
            blob = s.get(StoredBlob, key)
            if blob is None:
                raise FileNotFoundError(key)
            return blob.data

    def delete(self, key: str, session: Session | None = None) -> None:
        from .models import StoredBlob
        if session is not None:
            blob = session.get(StoredBlob, key)
            if blob is not None:
                session.delete(blob)
            return
        with self.db.SessionLocal() as s:
            blob = s.get(StoredBlob, key)
            if blob is not None:
                s.delete(blob)
                s.commit()


def make_storage(settings: Settings, db: Database) -> Storage:
    if settings.storage_backend == "db":
        return DbStorage(db)
    if settings.storage_backend == "local":
        return LocalStorage(settings.upload_dir)
    raise ValueError(f"unknown STORAGE={settings.storage_backend!r} (expected 'local' or 'db')")
