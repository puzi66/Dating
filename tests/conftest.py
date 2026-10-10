"""Fixtures for the pilot-app tests (the engine tests don't use any of these)."""

from __future__ import annotations

import os
import re
import tempfile

# Must be set before `app.config` is imported anywhere.
_TMP = tempfile.mkdtemp(prefix="shalosh-test-")
os.environ.setdefault("DATA_DIR", _TMP)
os.environ["FAKE_LLM"] = "1"
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ.setdefault("SECRET_KEY", "test-secret-key")
os.environ["ADMIN_USER"] = "admin"
os.environ["ADMIN_PASSWORD"] = "pw"

import pytest  # noqa: E402

ADMIN = ("admin", "pw")


@pytest.fixture
def fake_llm():
    from app.llm import FakeLLM
    return FakeLLM()


@pytest.fixture
def app_env(tmp_path, fake_llm):
    from dataclasses import replace

    from app.config import settings
    from app.db import Database
    from app.main import create_app
    from app.storage import LocalStorage

    s = replace(settings, database_url=f"sqlite:///{tmp_path / 'test.db'}", upload_dir=tmp_path / "uploads",
                max_upload_bytes=200_000, interview_max_user_turns=5, interview_min_user_turns=2)
    db = Database(s.database_url)
    app = create_app(settings=s, db=db, llm=fake_llm, storage=LocalStorage(s.upload_dir))
    return app, db


@pytest.fixture
def app(app_env):
    return app_env[0]


@pytest.fixture
def db(app_env):
    with app_env[1].SessionLocal() as s:
        yield s


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)


def new_client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)


def csrf_of(html: str) -> str:
    m = re.search(r'name="csrf-token" content="([^"]+)"', html) or re.search(
        r'name="csrf_token" value="([^"]+)"', html)
    assert m, "no csrf token in page"
    return m.group(1)


PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64


@pytest.fixture
def make_participant(db, app):
    """Create a participant directly in the DB. complete=True gives a paid, fully profiled user."""
    from app.choices import CITY_COORDS
    from app.models import Participant, Photo, utcnow
    from app.profile import neutral_visual_vector, values_vector
    from app.prompts import VALUE_LABELS_HE
    from app.security import new_token

    counter = {"n": 0}

    def _make(first: str | None = None, gender: str = "female", seeking=("male",), age: int = 30,
              city: str = "Tel Aviv-Yafo", complete: bool = True, paid: bool = True, paused: bool = False,
              values: dict | None = None, **kw) -> Participant:
        counter["n"] += 1
        n = counter["n"]
        p = Participant(token=new_token(), first_name=first or f"P{n}", last_name="", phone=f"050-000{n:04d}",
                        paused=paused)
        if complete:
            now = utcnow()
            _, lat, lon = CITY_COORDS[city]
            scores = values or {name: (n * 7 + i * 3) % 11 for i, name in enumerate(VALUE_LABELS_HE)}
            p.consented_at = p.facts_done_at = p.interview_done_at = p.visual_done_at = now
            p.interview_ended = True
            p.gender, p.seeking, p.age, p.age_min, p.age_max = gender, list(seeking), age, 20, 45
            p.city, p.lat, p.lon, p.max_distance_km = city, lat, lon, 100
            p.intent, p.has_kids, p.wants_kids, p.accepts_partner_with_kids = "serious", False, "yes", True
            p.religiosity, p.religiosity_tolerance, p.smoking, p.max_partner_smoking = 0, 1, 0, 1
            p.summary = "summary"
            p.values_scores, p.values_vec = scores, values_vector(scores)
            p.looking_for = ["a", "b", "c"]
            p.red_flags = ["SECRET-RED-FLAG"]
            p.visual_taste_vec = p.appearance_vec = neutral_visual_vector()
            if paid:
                p.paid_at = now
        for k, v in kw.items():
            setattr(p, k, v)
        db.add(p)
        db.flush()
        if complete:
            storage = app.state.storage
            for i in range(2):
                db.add(Photo(participant_id=p.id, storage_key=storage.save(PNG, "image/png", f"p{p.id}"),
                             content_type="image/png", ordinal=i))
        db.commit()
        return p

    return _make


def login(client, p) -> str:
    """Log in through the secret link; returns the CSRF token of the session."""
    r = client.get(f"/p/{p.token}", follow_redirects=True)
    assert r.status_code == 200, r.text
    return csrf_of(r.text)
