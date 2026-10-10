"""Drop generation exclusions, matchmaker edits (cap of 3), publishing, answers and matches."""

from __future__ import annotations

import random
from collections import Counter
from datetime import timedelta

import pytest

from app import services as S
from app.models import MatchRow, ProposalRow, utcnow
from tests.conftest import ADMIN, csrf_of, login


def _pool(make_participant, women=4, men=4):
    ws = [make_participant(f"W{i}", gender="female", seeking=("male",)) for i in range(women)]
    ms = [make_participant(f"M{i}", gender="male", seeking=("female",)) for i in range(men)]
    return ws, ms


def _publish(db, drop):
    for pr in drop.proposals:
        pr.why_text = pr.why_text or "שניכם אוהבים טבע"
    db.commit()
    return S.publish_drop(db, drop)


def _users(drop):
    return {u for pr in drop.proposals for u in (pr.user_low_id, pr.user_high_id)}


def test_draft_drop_uses_engine_and_respects_cap(db, make_participant):
    _pool(make_participant, 5, 5)
    drop = S.generate_draft_drop(db)
    assert drop.status == "draft" and drop.proposals
    load = S.drop_load(drop)
    assert max(load.values()) <= 3
    for pr in drop.proposals:
        assert pr.user_low_id < pr.user_high_id
        assert 0 < pr.mutual_score <= 1 and len(pr.features["low_high"]) == 3
    assert drop.stats["pool"] == 10


def test_generation_excludes_paused_unpaid_incomplete_and_in_match(db, make_participant):
    ws, ms = _pool(make_participant, 3, 3)
    paused = make_participant("Paused", gender="female", seeking=("male",), paused=True)
    unpaid = make_participant("Unpaid", gender="female", seeking=("male",), paid=False)
    incomplete = make_participant("Inc", complete=False)
    # an active match between W0 and M0
    drop0 = S.generate_draft_drop(db)
    pr0 = next(pr for pr in drop0.proposals if {pr.user_low_id, pr.user_high_id} == {ws[0].id, ms[0].id})
    _publish(db, drop0)
    S.answer_proposal(db, ws[0], pr0.id, "yes")
    _, m = S.answer_proposal(db, ms[0], pr0.id, "yes")
    assert m is not None

    drop = S.generate_draft_drop(db)
    users = _users(drop)
    for excluded in (paused, unpaid, incomplete, ws[0], ms[0]):
        assert excluded.id not in users
    assert ws[0].id in drop.stats["excluded_in_match"] and ms[0].id in drop.stats["excluded_in_match"]
    assert paused.id in drop.stats["excluded_paused"]
    # pairs already shown in drop0 never come back
    shown = {(pr.user_low_id, pr.user_high_id) for pr in drop0.proposals}
    assert not shown & {(pr.user_low_id, pr.user_high_id) for pr in drop.proposals}


def test_closing_match_frees_both_users(db, make_participant):
    ws, ms = _pool(make_participant, 2, 2)
    drop0 = S.generate_draft_drop(db)
    pr0 = drop0.proposals[0]
    a, b = pr0.user_low_id, pr0.user_high_id
    _publish(db, drop0)
    pa, pb = db.get(type(ws[0]), a), db.get(type(ws[0]), b)
    S.answer_proposal(db, pa, pr0.id, "yes")
    _, m = S.answer_proposal(db, pb, pr0.id, "yes")
    assert S.participant_status(pa, db=db) == "in_match"
    S.close_match(db, m)
    assert S.participant_status(pa, db=db) == "active"
    assert S.active_match_for(db, a) is None


def test_draft_regeneration_replaces_previous_draft(db, make_participant):
    _pool(make_participant, 3, 3)
    d1 = S.generate_draft_drop(db)
    d2 = S.generate_draft_drop(db)
    assert db.get(type(d1), d1.id) is None or d1.id == d2.id
    assert db.query(type(d2)).filter_by(status="draft").count() == 1


def test_swap_keeps_cap_of_three(db, make_participant):
    # 2 women, 4 men (straight): each woman gets the cap of 3, so both are full.
    w = make_participant("W", gender="female", seeking=("male",))
    w2 = make_participant("W2", gender="female", seeking=("male",))
    for i in range(4):
        make_participant(f"M{i}", gender="male", seeking=("female",))
    drop = S.generate_draft_drop(db)
    load = S.drop_load(drop)
    assert load[w.id] == 3 and load[w2.id] == 3
    with_w = {pr.other(w.id) for pr in drop.proposals if w.id in (pr.user_low_id, pr.user_high_id)}
    # a pair (W2, man) where the man is not paired with W: swapping his partner to W must fail on the cap
    pr = next(pr for pr in drop.proposals
              if w2.id in (pr.user_low_id, pr.user_high_id) and pr.other(w2.id) not in with_w)
    man = pr.other(w2.id)
    cands = {c["id"]: c for c in S.swap_candidates(db, drop, pr, man)}
    assert w.id in cands and not cands[w.id]["available"]
    with pytest.raises(S.CapError):
        S.swap_pair(db, drop, pr.id, man, w.id)
    db.refresh(drop)
    assert S.drop_load(drop)[w.id] == 3 and max(S.drop_load(drop).values()) <= 3
    # after removing one of W's pairs she has room, and the same swap succeeds
    S.remove_pair(db, drop, next(p.id for p in drop.proposals if w.id in (p.user_low_id, p.user_high_id)))
    S.swap_pair(db, drop, pr.id, man, w.id)
    db.refresh(drop)
    assert S.drop_load(drop)[w.id] == 3 and max(S.drop_load(drop).values()) <= 3


def test_swap_to_available_candidate(db, make_participant):
    _pool(make_participant, 4, 4)
    drop = S.generate_draft_drop(db)
    removed = drop.proposals[-1]
    freed = {removed.user_low_id, removed.user_high_id}
    S.remove_pair(db, drop, removed.id)
    db.refresh(drop)
    for pr in list(drop.proposals):
        for keep in (pr.user_low_id, pr.user_high_id):
            cands = [c for c in S.swap_candidates(db, drop, pr, keep) if c["available"] and c["id"] in freed]
            if not cands:
                continue
            old_id = pr.id
            new = S.swap_pair(db, drop, old_id, keep, cands[0]["id"])
            db.refresh(drop)
            assert new.source == "swap" and {keep, cands[0]["id"]} == {new.user_low_id, new.user_high_id}
            assert max(S.drop_load(drop).values()) <= 3
            assert old_id not in {p.id for p in drop.proposals}
            return
    pytest.fail("expected a swap into the freed capacity")


def test_random_matchmaker_edits_never_exceed_cap(db, make_participant):
    _pool(make_participant, 6, 6)
    drop = S.generate_draft_drop(db)
    rng = random.Random(3)
    for _ in range(25):
        db.refresh(drop)
        if not drop.proposals:
            break
        pr = rng.choice(drop.proposals)
        if rng.random() < 0.25:
            S.remove_pair(db, drop, pr.id)
            continue
        keep = rng.choice([pr.user_low_id, pr.user_high_id])
        cands = S.swap_candidates(db, drop, pr, keep)
        if not cands:
            continue
        c = rng.choice(cands)
        try:
            S.swap_pair(db, drop, pr.id, keep, c["id"])
            assert c["available"]
        except S.CapError:
            assert not c["available"]
        db.refresh(drop)
        assert max(S.drop_load(drop).values(), default=0) <= 3
        keys = Counter((p.user_low_id, p.user_high_id) for p in drop.proposals)
        assert max(keys.values(), default=0) == 1


def test_swap_rejects_infeasible_candidate(db, make_participant):
    w = make_participant("W", gender="female", seeking=("male",))
    m = make_participant("M", gender="male", seeking=("female",))
    w2 = make_participant("W2", gender="female", seeking=("female",))   # doesn't want men
    drop = S.generate_draft_drop(db)
    pr = drop.proposals[0]
    with pytest.raises(S.DropError):
        S.swap_pair(db, drop, pr.id, m.id, w2.id)


def test_publish_requires_texts_and_sets_48h_deadline(db, make_participant):
    _pool(make_participant, 2, 2)
    drop = S.generate_draft_drop(db)
    with pytest.raises(S.DropError):
        S.publish_drop(db, drop)
    now = utcnow()
    for pr in drop.proposals:
        pr.why_text = "x"
    db.commit()
    S.publish_drop(db, drop, now)
    assert drop.status == "published" and drop.deadline_at == now + timedelta(hours=48)
    with pytest.raises(S.DropError):
        S.remove_pair(db, drop, drop.proposals[0].id)   # published drops are frozen


def test_why_texts_never_include_red_flags(db, make_participant, fake_llm):
    import asyncio
    _pool(make_participant, 1, 1)
    drop = S.generate_draft_drop(db)
    seen = []

    async def spy(brief: str) -> str:
        seen.append(brief)
        return "טקסט"
    fake_llm.why_text = spy
    n, errors = asyncio.run(S.generate_why_texts(db, drop, fake_llm))
    assert n == 1 and not errors and drop.proposals[0].why_text == "טקסט"
    assert "SECRET-RED-FLAG" not in seen[0] and "summary" not in seen[0]


def test_answers_are_final_and_deadline_enforced(db, make_participant):
    ws, ms = _pool(make_participant, 1, 1)
    drop = S.generate_draft_drop(db)
    pr = drop.proposals[0]
    w, m = ws[0], ms[0]
    with pytest.raises(S.AnswerError):          # not published yet
        S.answer_proposal(db, w, pr.id, "yes")
    now = utcnow()
    _publish(db, drop)
    S.answer_proposal(db, w, pr.id, "no", now)
    with pytest.raises(S.AnswerError):          # final
        S.answer_proposal(db, w, pr.id, "yes", now)
    with pytest.raises(S.AnswerError):          # bad value
        S.answer_proposal(db, m, pr.id, "maybe", now)
    with pytest.raises(S.AnswerError):          # past the deadline
        S.answer_proposal(db, m, pr.id, "yes", drop.deadline_at + timedelta(seconds=1))
    pr2, match = S.answer_proposal(db, m, pr.id, "yes", now)
    assert match is None and pr2.response_low != pr2.response_high


def test_mutual_yes_creates_exactly_one_match(db, make_participant):
    ws, ms = _pool(make_participant, 2, 2)
    drop = S.generate_draft_drop(db)
    _publish(db, drop)
    w0 = ws[0]
    mine = [pr for pr in drop.proposals if w0.id in (pr.user_low_id, pr.user_high_id)]
    assert len(mine) == 2
    first, second = mine
    # first mutual -> match
    S.answer_proposal(db, w0, first.id, "yes")
    other1 = db.get(type(w0), first.other(w0.id))
    _, m1 = S.answer_proposal(db, other1, first.id, "yes")
    assert m1 is not None and {m1.user_low_id, m1.user_high_id} == {w0.id, other1.id}
    # second mutual for the same user -> superseded, no second match (exclusivity)
    S.answer_proposal(db, w0, second.id, "yes")
    other2 = db.get(type(w0), second.other(w0.id))
    pr2, m2 = S.answer_proposal(db, other2, second.id, "yes")
    assert m2 is None and pr2.superseded and pr2.is_mutual
    assert db.query(MatchRow).count() == 1


def test_participant_flow_drop_to_match_via_http(client, app, db, make_participant):
    w = make_participant("Wendy", gender="female", seeking=("male",))
    m = make_participant("Moshe", gender="male", seeking=("female",))
    drop = S.generate_draft_drop(db)
    _publish(db, drop)
    pr = drop.proposals[0]
    tok = login(client, w)
    page = client.get("/p/drop")
    assert "Moshe" in page.text and "שניכם" in page.text
    r = client.post("/p/drop/answer", data={"csrf_token": tok, "proposal_id": pr.id, "answer": "yes"},
                    follow_redirects=False)
    assert r.headers["location"] == "/p/drop"
    r = client.post("/p/drop/answer", data={"csrf_token": tok, "proposal_id": pr.id, "answer": "no"})
    assert r.status_code == 400 and "סופיות" in r.text
    from tests.conftest import new_client
    c2 = new_client(app)
    tok2 = login(c2, m)
    r = c2.post("/p/drop/answer", data={"csrf_token": tok2, "proposal_id": pr.id, "answer": "yes"},
                follow_redirects=True)
    assert "זה הדדי" in r.text and "Wendy" in r.text
    # admin sees both phone numbers highlighted
    r = client.get("/admin/responses", auth=ADMIN)
    assert w.phone in r.text and m.phone in r.text and "התאמה הדדית" in r.text


def test_admin_drop_review_page_shows_breakdown(client, db, make_participant):
    _pool(make_participant, 2, 2)
    r = client.get("/admin/drops", auth=ADMIN)
    r = client.post("/admin/drops/generate", data={"csrf_token": csrf_of(r.text)}, auth=ADMIN)
    assert r.status_code == 200 and "S(" in r.text and "ערכים" in r.text
    pr = db.query(ProposalRow).first()
    r = client.get(f"/admin/drops/{pr.drop_id}?swap={pr.id}&keep={pr.user_low_id}", auth=ADMIN)
    assert r.status_code == 200 and "החלפה" in r.text
