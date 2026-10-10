"""Token links, sessions, CSRF, admin auth, upload limits and data isolation."""

from __future__ import annotations

from tests.conftest import ADMIN, JPEG, PNG, csrf_of, login, new_client


def test_token_link_logs_in_and_strips_token_from_url(client, make_participant):
    p = make_participant(complete=False)
    r = client.get(f"/p/{p.token}", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/p/"
    assert "shalosh_session" in r.cookies or "shalosh_session" in client.cookies
    r = client.get("/p/", follow_redirects=False)
    assert r.headers["location"] == "/p/welcome"
    r = client.get("/p/welcome")
    assert r.status_code == 200 and p.first_name in r.text
    assert r.headers["referrer-policy"] == "no-referrer"


def test_bad_or_missing_token_is_rejected(client, make_participant):
    p = make_participant(complete=False)
    assert client.get("/p/" + "x" * 43).status_code == 404
    assert client.get("/p/" + p.token[:-1] + ("A" if p.token[-1] != "A" else "B")).status_code == 404
    # no session cookie at all
    r = client.get("/p/welcome", headers={"accept": "text/html"})
    assert r.status_code == 401 and "הקישור האישי" in r.text


def test_rotating_token_invalidates_old_session(client, db, make_participant):
    p = make_participant(complete=False)
    login(client, p)
    assert client.get("/p/welcome").status_code == 200
    from app.security import new_token
    p.token = new_token()
    db.commit()
    assert client.get("/p/welcome").status_code == 401


def test_csrf_required_on_post(client, make_participant):
    p = make_participant(complete=False)
    tok = login(client, p)
    assert client.post("/p/welcome", data={"consent": "1"}).status_code == 403
    assert client.post("/p/welcome", data={"consent": "1", "csrf_token": "wrong"}).status_code == 403
    r = client.post("/p/welcome", data={"consent": "1", "csrf_token": tok}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/p/facts"
    # a different browser can't reuse the token
    other = new_client(client.app)
    login(other, make_participant(complete=False))
    assert other.post("/p/welcome", data={"consent": "1", "csrf_token": tok}).status_code == 403


def test_admin_requires_basic_auth(client):
    assert client.get("/admin").status_code == 401
    assert client.get("/admin", auth=("admin", "nope")).status_code == 401
    assert client.get("/admin", auth=ADMIN).status_code == 200


def test_admin_create_participant_generates_long_token(client, db):
    r = client.get("/admin/participants/new", auth=ADMIN)
    tok = csrf_of(r.text)
    assert client.post("/admin/participants/new", data={"first_name": "נועה", "phone": "050-1234567"},
                       auth=ADMIN).status_code == 403
    r = client.post("/admin/participants/new", auth=ADMIN,
                    data={"first_name": "נועה", "phone": "050-1234567", "csrf_token": tok})
    assert r.status_code == 200
    from app.models import Participant
    p = db.query(Participant).filter_by(first_name="נועה").one()
    assert len(p.token) >= 40 and p.token in r.text


def _facts(tok: str) -> dict:
    return {"csrf_token": tok, "gender": "female", "seeking": "male", "age": "30", "age_min": "28",
            "age_max": "38", "city": "Haifa", "max_distance_km": "50", "intent": "serious", "has_kids": "0",
            "wants_kids": "yes", "accepts_partner_with_kids": "1", "religiosity": "0", "smoking": "0",
            "max_partner_smoking": "1"}


def test_facts_form_upload_limits(client, db, make_participant):
    from app.models import utcnow
    p = make_participant(complete=False, consented_at=utcnow())
    tok = login(client, p)
    # wrong type (claims image/png but is text)
    r = client.post("/p/facts", data=_facts(tok),
                    files=[("photos", ("a.png", b"hello world", "image/png")), ("photos", ("b.png", PNG, "image/png"))])
    assert r.status_code == 400 and "JPG" in r.text
    # too big (limit is 200KB in tests)
    big = PNG + b"\x00" * 300_000
    r = client.post("/p/facts", data=_facts(tok),
                    files=[("photos", ("a.png", big, "image/png")), ("photos", ("b.png", PNG, "image/png"))])
    assert r.status_code == 400 and "גדולה" in r.text
    # too few
    r = client.post("/p/facts", data=_facts(tok), files=[("photos", ("a.png", PNG, "image/png"))])
    assert r.status_code == 400
    # too many
    r = client.post("/p/facts", data=_facts(tok), files=[("photos", (f"{i}.png", PNG, "image/png")) for i in range(5)])
    assert r.status_code == 400
    # ok: 2 photos, content type sniffed from bytes (declared type is ignored)
    r = client.post("/p/facts", data=_facts(tok), follow_redirects=False,
                    files=[("photos", ("a.jpg", JPEG, "application/octet-stream")), ("photos", ("b.png", PNG, "image/png"))])
    assert r.status_code == 303 and r.headers["location"] == "/p/interview"
    db.expire_all()
    db.refresh(p)
    assert sorted(ph.content_type for ph in p.photos) == ["image/jpeg", "image/png"]
    assert p.city == "Haifa" and p.lat and p.seeking == ["male"]


def test_participant_cannot_see_other_participants_data(client, db, make_participant):
    from app import services as S
    a = make_participant("Alice", gender="female", seeking=("male",))
    b = make_participant("Bob", gender="male", seeking=("female",))
    c = make_participant("Carol", gender="female", seeking=("male",))
    login(client, a)
    b_photo, c_photo = b.photos[0].id, c.photos[0].id
    # before any drop: nobody else's photos
    assert client.get(f"/media/{b_photo}").status_code == 404
    assert client.get(f"/media/{a.photos[0].id}").status_code == 200
    # publish a drop with A+B and B+C
    drop = S.generate_draft_drop(db)
    pairs = {(pr.user_low_id, pr.user_high_id) for pr in drop.proposals}
    assert pairs == {(a.id, b.id), (b.id, c.id)}
    for pr in drop.proposals:
        pr.why_text = "why"
    db.commit()
    S.publish_drop(db, drop)
    assert client.get(f"/media/{b_photo}").status_code == 200        # in A's own drop
    assert client.get(f"/media/{c_photo}").status_code == 404        # C is in B's drop, not A's
    page = client.get("/p/drop").text
    assert "Bob" in page and "Carol" not in page
    assert "SECRET-RED-FLAG" not in page and "summary" not in page
    assert b.phone not in page
    # proposal ids of other people's pairs can't be answered
    other = next(pr for pr in drop.proposals if a.id not in (pr.user_low_id, pr.user_high_id))
    tok = csrf_of(page)
    r = client.post("/p/drop/answer", data={"csrf_token": tok, "proposal_id": other.id, "answer": "yes"})
    assert r.status_code == 400
    # a debrief link for someone else's match is a 404
    assert client.get("/p/debrief/999").status_code == 404
