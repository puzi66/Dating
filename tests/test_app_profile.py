"""Interview streaming + profile extraction parsing, using the fake Claude client."""

from __future__ import annotations

import asyncio
import json
import math

import pytest

from app.interview import api_history, extract_and_store
from app.llm import MarkerFilter
from app.models import utcnow
from app.profile import ProfileParseError, parse_profile, values_vector
from app.prompts import END_MARKER, VALUE_KEYS, extraction_schema
from matching.scoring import VALUE_DIMENSION_NAMES
from tests.conftest import csrf_of, login

def good_profile(**over) -> dict:
    d = {
        "summary": "נועה, 29, אוהבת טבע ואנשים.",
        "values": {k: 5 for k in VALUE_KEYS} | {"family": 9, "humor": 10, "career_ambition": 2},
        "lifestyle": {"daily_rhythm": "בוקר", "social_energy": "באמצע", "weekend": "טיולים",
                      "health": "ריצה", "work": "הייטק", "home_and_pets": "כלב"},
        "looking_for": ["הומור", "יציבות", "סקרנות"],
        "red_flags": [],
    }
    d.update(over)
    return d


def test_parse_valid_profile_and_vector_mapping():
    p = parse_profile(good_profile())
    assert p.values["family"] == 9 and p.values["humor"] == 10 and p.values["career ambition"] == 2
    assert len(p.values) == 32 and p.looking_for == ["הומור", "יציבות", "סקרנות"]
    vec = values_vector(p.values)
    assert len(vec) == 32 and math.isclose(math.fsum(x * x for x in vec), 1.0)
    idx = list(VALUE_DIMENSION_NAMES)
    assert vec[idx.index("humor")] > vec[idx.index("family")] > 0 > vec[idx.index("career ambition")]


def test_parse_accepts_json_string_with_fences_and_clamps():
    raw = "```json\n" + json.dumps(good_profile(values={"family": 14, "humor": -3, "honesty": "7.6"})) + "\n```"
    p = parse_profile(raw)
    assert p.values["family"] == 10 and p.values["humor"] == 0 and p.values["honesty"] == 8
    assert p.values["kindness"] == 5            # missing dims default to neutral


def test_parse_normalises_lists():
    p = parse_profile(good_profile(looking_for=["a", "", "b", "c", "d"], red_flags="לא בטוח/ה שמוכן/ה"))
    assert p.looking_for == ["a", "b", "c"] and p.red_flags == ["לא בטוח/ה שמוכן/ה"]


@pytest.mark.parametrize("bad", [
    "not json", "[]", json.dumps({"values": {}}),
    json.dumps(good_profile(summary="")), json.dumps(good_profile(looking_for=[])),
    json.dumps(good_profile(values={"nonsense": 3})), json.dumps(good_profile(values={"family": "high"})),
])
def test_parse_rejects_bad_output(bad):
    with pytest.raises(ProfileParseError):
        parse_profile(bad)


def test_all_equal_scores_give_zero_vector():
    assert values_vector({n: 7 for n in VALUE_DIMENSION_NAMES}) == [0.0] * 32


def test_extraction_schema_is_strict():
    s = extraction_schema()
    assert s["additionalProperties"] is False
    assert set(s["properties"]["values"]["required"]) == set(VALUE_KEYS) and len(VALUE_KEYS) == 32


def test_marker_filter_handles_split_marker():
    f = MarkerFilter()
    chunks = ["תודה רבה", " [[E", "N", "D]", "]"]
    out = "".join(f.feed(c) for c in chunks) + f.flush()
    assert out == "תודה רבה " and f.found
    g = MarkerFilter()
    assert g.feed("a [[x") + g.flush() == "a [[x" and not g.found


def test_extract_and_store_with_fake_client(db, make_participant, fake_llm):
    p = make_participant(complete=False, consented_at=utcnow())
    fake_llm.extract_response = good_profile(red_flags=["נשמע/ת עדיין בקשר"])
    assert asyncio.run(extract_and_store(db, p, fake_llm))
    assert p.summary.startswith("נועה") and p.red_flags == ["נשמע/ת עדיין בקשר"]
    assert len(p.values_vec) == 32 and len(p.visual_taste_vec) == 16 and p.profile_error == ""


def test_extract_failure_is_recorded_not_fatal(db, make_participant, fake_llm):
    p = make_participant(complete=False)
    fake_llm.extract_response = "garbage"
    assert not asyncio.run(extract_and_store(db, p, fake_llm))
    assert "not JSON" in p.profile_error
    assert sum(1 for c in fake_llm.calls if c[0] == "extract") == 2      # retried once


def _sse(text: str) -> list[dict]:
    return [json.loads(line[5:]) for line in text.split("\n\n") if line.startswith("data:")]


def test_interview_streaming_turn_cap_and_extraction(client, db, make_participant, fake_llm):
    from app.models import Participant
    p = make_participant(complete=False, consented_at=utcnow(), facts_done_at=utcnow(), gender="female",
                         age=29, city="Haifa", intent="serious", wants_kids="yes", religiosity=0, smoking=0)
    tok = login(client, p)
    h = {"X-CSRF-Token": tok}
    # missing CSRF header
    assert client.post("/p/interview/message", json={"text": ""}).status_code == 403
    # opening message from עדי
    ev = _sse(client.post("/p/interview/message", json={"text": ""}, headers=h).text)
    assert ev[-1]["done"] and not ev[-1]["ended"]
    assert "עדי" in "".join(e.get("delta", "") for e in ev)
    for i in range(5):      # test settings cap the interview at 5 user turns
        ev = _sse(client.post("/p/interview/message", json={"text": f"תשובה {i}"}, headers=h).text)
    assert ev[-1]["ended"] is True
    full = "".join(e.get("delta", "") for e in ev)
    assert END_MARKER not in full and "תודה" in full
    # the cap: interview closed for further messages
    assert client.post("/p/interview/message", json={"text": "עוד"}, headers=h).status_code == 409
    db.expire_all()
    p = db.get(Participant, p.id)
    assert [m.role for m in p.messages][:3] == ["assistant", "user", "assistant"] and p.interview_ended
    assert sum(1 for m in p.messages if m.role == "user") == 5
    # history passed to Claude starts with a user turn and alternates
    hist = api_history(p)
    assert hist[0]["role"] == "user" and all(a["role"] != b["role"] for a, b in zip(hist, hist[1:]))
    # finish -> extraction via fake client -> visual step
    r = client.post("/p/interview/finish", data={"csrf_token": tok}, follow_redirects=False)
    assert r.headers["location"] == "/p/visual"
    db.expire_all()
    p = db.get(Participant, p.id)
    assert p.interview_done_at and p.summary and len(p.values_vec) == 32
    page = client.get("/p/visual")
    r = client.post("/p/visual", data={"csrf_token": csrf_of(page.text), "looks_matter": "3", "clear_type": "2",
                                       "grows_over_time": "5"}, follow_redirects=False)
    assert r.headers["location"] == "/p/home"
    db.expire_all()
    assert db.get(Participant, p.id).profile_complete


def test_interview_finish_needs_minimum_turns(client, make_participant):
    p = make_participant(complete=False, consented_at=utcnow(), facts_done_at=utcnow())
    tok = login(client, p)
    r = client.post("/p/interview/finish", data={"csrf_token": tok}, follow_redirects=False)
    assert r.headers["location"] == "/p/interview"
