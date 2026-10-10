"""Debrief -> matching.feedback weight updates, and the metrics page."""

from __future__ import annotations

import pytest

from app import services as S
from app.models import Debrief, LearningStateRow
from matching.feedback import LearningState, update_from_debrief
from matching.models import DateDebrief, PreferenceWeights
from matching.scoring import Features
from tests.conftest import ADMIN, csrf_of, login


def _matched_pair(db, make_participant):
    w = make_participant("W", gender="female", seeking=("male",))
    m = make_participant("M", gender="male", seeking=("female",))
    drop = S.generate_draft_drop(db)
    for pr in drop.proposals:
        pr.why_text = "x"
    db.commit()
    S.publish_drop(db, drop)
    pr = drop.proposals[0]
    S.answer_proposal(db, w, pr.id, "yes")
    _, match = S.answer_proposal(db, m, pr.id, "yes")
    assert match is not None
    return w, m, match


def test_debrief_updates_weights_exactly_like_engine(db, make_participant):
    w, m, match = _matched_pair(db, make_participant)
    pr = match.proposal
    key = "low_high" if w.id == pr.user_low_id else "high_low"
    f = Features(*pr.features[key])

    # expected: the engine's own update on a fresh state
    exp_state = LearningState()
    exp_w = PreferenceWeights(w.id)
    update_from_debrief(exp_state, exp_w, DateDebrief(match.id, w.id, m.id, 5, True), f)

    d = S.submit_debrief(db, w, match.id, 5, "yes", "  הפתיע לטובה  ", "yes")
    db.refresh(w)
    assert (w.w_values, w.w_visual, w.w_lifestyle) == pytest.approx(exp_w.as_tuple())
    assert w.w_updates == 1 and w.w_bias == pytest.approx(exp_w.bias)
    row = db.get(LearningStateRow, 1)
    assert row.global_weights == pytest.approx(list(exp_state.global_weights.as_tuple()))
    assert row.feature_mean == pytest.approx(exp_state.feature_mean)
    assert d.weights_before == pytest.approx([1 / 3] * 3) and d.weights_after == pytest.approx(list(exp_w.as_tuple()))
    assert d.surprise_text == "הפתיע לטובה"
    assert match.date_happened_at is not None            # a debrief implies the date happened


def test_debrief_direction_and_unsure(db, make_participant):
    w, m, match = _matched_pair(db, make_participant)
    S.submit_debrief(db, m, match.id, 3, "unsure", "", "partly")
    db.refresh(m)
    assert m.w_updates == 1
    assert sum((m.w_values, m.w_visual, m.w_lifestyle)) == pytest.approx(1.0)
    # second debrief by the same person is rejected; the other side still can
    with pytest.raises(S.DebriefError):
        S.submit_debrief(db, m, match.id, 4, "yes", "", "yes")
    S.submit_debrief(db, w, match.id, 1, "no", "", "no")
    assert db.query(Debrief).count() == 2
    assert db.get(LearningStateRow, 1).global_updates == 2


def test_debrief_validation_and_ownership(db, make_participant):
    w, m, match = _matched_pair(db, make_participant)
    outsider = make_participant("X", gender="female", seeking=("male",))
    with pytest.raises(S.DebriefError):
        S.submit_debrief(db, outsider, match.id, 4, "yes", "", "yes")
    with pytest.raises(S.DebriefError):
        S.submit_debrief(db, w, match.id, 6, "yes", "", "yes")
    with pytest.raises(S.DebriefError):
        S.submit_debrief(db, w, match.id, 4, "maybe", "", "yes")


def test_learned_weights_feed_the_next_drop(db, make_participant):
    """After a debrief the rater's effective weights move away from the prior."""
    w, m, match = _matched_pair(db, make_participant)
    before = S.build_pool(db).eff[w.id]
    S.submit_debrief(db, w, match.id, 5, "yes", "", "yes")
    S.close_match(db, match)
    after = S.build_pool(db).eff[w.id]
    assert before != pytest.approx(after)


def test_debrief_via_link_and_metrics(client, db, make_participant):
    w, m, match = _matched_pair(db, make_participant)
    r = client.get(f"/p/{w.token}/debrief/{match.id}", follow_redirects=True)
    assert "איך היה עם M" in r.text
    tok = csrf_of(r.text)
    r = client.post(f"/p/debrief/{match.id}", data={"csrf_token": tok, "rating": "4", "meet_again": "yes",
                                                    "surprise_text": "שיחה קלה", "looked_like_photos": "yes"})
    assert r.status_code == 200 and "תודה" in r.text
    login(client, m)
    r = client.post(f"/p/debrief/{match.id}", data={"csrf_token": csrf_of(client.get('/p/home').text),
                                                    "rating": "5", "meet_again": "yes", "looked_like_photos": "partly"})
    assert r.status_code == 200
    # survey for H2
    client.post("/p/survey", data={"csrf_token": csrf_of(client.get('/p/home').text), "score": "4"})
    match.replied_within_48h = True
    db.commit()
    met = S.compute_metrics(db)
    assert met["H3_mutual_accept_pct"] == 100.0
    assert met["H4_replied_48h_pct"] == 100.0
    assert met["H5_date_within_7d_pct"] == 100.0
    assert met["second_date_pct"] == 100.0
    assert met["H2_three_is_enough_avg"] == 4
    assert met["H6_paid_of_invited_pct"] == 100.0
    r = client.get("/admin/metrics", auth=ADMIN)
    assert r.status_code == 200 and "H1" in r.text
    csv = client.get("/admin/export/pairs.csv", auth=ADMIN)
    assert csv.status_code == 200 and "mutual_score" in csv.text and "True" in csv.text
    assert "H3_mutual_accept_pct" in client.get("/admin/export/metrics.csv", auth=ADMIN).text
