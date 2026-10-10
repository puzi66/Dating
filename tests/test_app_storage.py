"""Photo storage backends: the database backend must behave like the disk one."""

import pytest

from app.db import Database
from app.storage import DbStorage, LocalStorage


@pytest.fixture(params=["local", "db"])
def storage(request, tmp_path):
    if request.param == "local":
        return LocalStorage(tmp_path / "up")
    db = Database(f"sqlite:///{tmp_path / 's.db'}")
    db.create_all()
    return DbStorage(db)


def test_roundtrip_and_delete(storage):
    key = storage.save(b"\x89PNG\r\n\x1a\nabc", "image/png", prefix="p1")
    assert key.startswith("p1/") and key.endswith(".png")
    assert storage.load(key) == b"\x89PNG\r\n\x1a\nabc"
    storage.delete(key)
    with pytest.raises(FileNotFoundError):
        storage.load(key)
    storage.delete(key)  # idempotent


def test_keys_are_unique(storage):
    assert storage.save(b"x", "image/png", "p1") != storage.save(b"x", "image/png", "p1")
