"""Business logic: engine bridge, Drop workflow, answers/matches, debrief feedback, metrics.

Everything that touches the matching engine goes through here; routes stay thin.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import delete, func, or_, select
from sqlalchemy.orm import Session

from matching.drop import allocate_drop
from matching.feedback import (LearningState, debrief_label, effective_weights, sgd_step,
                               update_from_debrief, USER_LR, GLOBAL_LR)
from matching.filters import PoolState, candidate_graph, mutually_compatible
from matching.models import (CAPACITY_PER_USER, REPLY_WINDOW, DateDebrief, Gender, Intent, KidsWish,
                             PreferenceWeights, Profile, Religiosity, Smoking, pair_key)
from matching.scoring import Features, explain_pair, score_pair

from .llm import LLMClient, LLMError
from .models import (Debrief, DropRow, LearningStateRow, MatchRow, Participant, ParticipantStatus,
                     ProposalRow, SurveyResponse, utcnow)
from .prompts import VALUE_LABELS_HE, signal_label_he

log = logging.getLogger("app.services")
IL = ZoneInfo("Asia/Jerusalem")
DROP_WEEKDAY = 6          # Sunday (Mon=0)
DROP_HOUR = 20
DATE_WINDOW = timedelta(days=7)


class DropError(ValueError):
    pass


class CapError(DropError):
    pass


class AnswerError(ValueError):
    pass


class DebriefError(ValueError):
    pass


# =========================================================================== time helpers

def to_il(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt.replace(tzinfo=ZoneInfo("UTC")).astimezone(IL)


def next_drop_at(now: datetime) -> datetime:
    """Next Sunday 20:00 Israel time, as naive UTC."""
    local = to_il(now)
    assert local is not None
    days = (DROP_WEEKDAY - local.weekday()) % 7
    cand = (local + timedelta(days=days)).replace(hour=DROP_HOUR, minute=0, second=0, microsecond=0)
    if cand <= local:
        cand += timedelta(days=7)
    return cand.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)


def week_start(now: datetime) -> date:
    """The Sunday (Israel time) that starts the current pilot week."""
    local = to_il(now)
    assert local is not None
    return (local - timedelta(days=(local.weekday() - DROP_WEEKDAY) % 7)).date()


# =========================================================================== status

def active_match_user_ids(db: Session) -> set[int]:
    rows = db.execute(select(MatchRow.user_low_id, MatchRow.user_high_id).where(MatchRow.is_active)).all()
    return {u for r in rows for u in r}


def participant_status(p: Participant, in_match: set[int] | None = None, db: Session | None = None) -> str:
    if in_match is None:
        in_match = active_match_user_ids(db) if db is not None else set()
    if p.paused:
        return ParticipantStatus.PAUSED
    if p.id in in_match:
        return ParticipantStatus.IN_MATCH
    if p.profile_complete and p.paid_at:
        return ParticipantStatus.ACTIVE
    if p.profile_complete:
        return ParticipantStatus.PROFILE_DONE
    if p.paid_at:
        return ParticipantStatus.PAID
    if p.consented_at:
        return ParticipantStatus.CONSENTED
    return ParticipantStatus.INVITED


def active_match_for(db: Session, pid: int) -> MatchRow | None:
    return db.scalar(select(MatchRow).where(
        MatchRow.is_active, or_(MatchRow.user_low_id == pid, MatchRow.user_high_id == pid)))


# =========================================================================== engine bridge

def to_engine_profile(p: Participant) -> Profile | None:
    if not p.profile_complete or p.age is None or p.lat is None:
        return None
    from .choices import accepted_intents_for
    return Profile(
        user_id=p.id, gender=Gender(p.gender), age=p.age, city=p.city or "", lat=p.lat, lon=p.lon,
        intent=Intent(p.intent), has_kids=bool(p.has_kids), wants_kids=KidsWish(p.wants_kids),
        religiosity=Religiosity(p.religiosity), smoking=Smoking(p.smoking),
        seeking=frozenset(Gender(g) for g in (p.seeking or [])),
        age_min=p.age_min, age_max=p.age_max, max_distance_km=float(p.max_distance_km or 50),
        accepted_intents=frozenset(Intent(i) for i in accepted_intents_for(p.intent)),
        accepts_partner_with_kids=bool(p.accepts_partner_with_kids),
        religiosity_tolerance=int(p.religiosity_tolerance),
        max_partner_smoking=Smoking(p.max_partner_smoking),
        values_vec=tuple(p.values_vec or [0.0] * 32),
        visual_taste_vec=tuple(p.visual_taste_vec or [0.0] * 16),
        appearance_vec=tuple(p.appearance_vec or [0.0] * 16),
    )


def user_weights(p: Participant) -> PreferenceWeights:
    return PreferenceWeights(p.id, p.w_values, p.w_visual, p.w_lifestyle, p.w_bias, p.w_updates)


def store_weights(p: Participant, w: PreferenceWeights) -> None:
    p.w_values, p.w_visual, p.w_lifestyle = w.as_tuple()
    p.w_bias, p.w_updates = w.bias, w.n_updates


def load_learning(db: Session) -> tuple[LearningState, LearningStateRow]:
    row = db.get(LearningStateRow, 1)
    if row is None:
        row = LearningStateRow(id=1)
        db.add(row)
        db.flush()
    gw = PreferenceWeights(0, *row.global_weights, bias=row.global_bias, n_updates=row.global_updates)
    return LearningState(global_weights=gw, feature_mean=list(row.feature_mean)), row


def save_learning(state: LearningState, row: LearningStateRow) -> None:
    row.global_weights = list(state.global_weights.as_tuple())
    row.global_bias = state.global_weights.bias
    row.global_updates = state.global_weights.n_updates
    row.feature_mean = list(state.feature_mean)
    row.updated_at = utcnow()


def shown_pairs(db: Session) -> set[tuple[int, int]]:
    rows = db.execute(
        select(ProposalRow.user_low_id, ProposalRow.user_high_id)
        .join(DropRow, ProposalRow.drop_id == DropRow.id)
        .where(DropRow.status != "draft")).all()
    return {(a, b) for a, b in rows}


@dataclass
class Pool:
    participants: dict[int, Participant]
    profiles: dict[int, Profile]
    state: PoolState
    eff: dict[int, tuple[float, float, float]]

    def score(self, a: int, b: int):  # noqa: ANN201
        lo, hi = pair_key(a, b)
        return score_pair(self.profiles[lo], self.profiles[hi], self.eff[lo], self.eff[hi])


def build_pool(db: Session) -> Pool:
    """Engine inputs for this week: paid + complete profiles; paused users, users in an
    active exclusive match and previously shown pairs are excluded via PoolState."""
    parts = db.scalars(select(Participant).where(Participant.paid_at.is_not(None))).all()
    participants = {p.id: p for p in parts if p.profile_complete}
    profiles = {pid: prof for pid, p in participants.items() if (prof := to_engine_profile(p))}
    state = PoolState(
        shown_pairs=shown_pairs(db),
        in_exclusive=active_match_user_ids(db),
        paused={pid for pid, p in participants.items() if p.paused},
    )
    learning, _ = load_learning(db)
    eff = {pid: effective_weights(user_weights(participants[pid]), learning.global_weights)
           for pid in profiles}
    return Pool(participants, profiles, state, eff)


def _proposal_from_score(drop_id: int, pool: Pool, a: int, b: int, source: str) -> ProposalRow:
    lo, hi = pair_key(a, b)
    ps = pool.score(lo, hi)
    expl = explain_pair(pool.profiles[lo], pool.profiles[hi], ps)
    return ProposalRow(
        drop_id=drop_id, user_low_id=lo, user_high_id=hi, mutual_score=ps.mutual,
        score_low_to_high=ps.s_ab, score_high_to_low=ps.s_ba,
        features={"low_high": list(ps.f_ab.as_tuple()), "high_low": list(ps.f_ba.as_tuple())},
        explanation=_pilot_explanation(expl.to_dict()), source=source,
    )


def _pilot_explanation(expl: dict) -> dict:
    """Pilot visual vectors are neutral (identical for everyone), so the engine's
    'mutual visual chemistry' signal is meaningless here and is dropped. Labels get Hebrew."""
    sigs = [s | {"label_he": signal_label_he(s)} for s in expl.get("top_signals", []) if s["kind"] != "visual"]
    return expl | {"top_signals": sigs}


# =========================================================================== Drop workflow

def current_draft(db: Session) -> DropRow | None:
    return db.scalar(select(DropRow).where(DropRow.status == "draft").order_by(DropRow.id.desc()))


def generate_draft_drop(db: Session, now: datetime | None = None) -> DropRow:
    now = now or utcnow()
    db.execute(delete(ProposalRow).where(ProposalRow.drop_id.in_(
        select(DropRow.id).where(DropRow.status == "draft"))))
    db.execute(delete(DropRow).where(DropRow.status == "draft"))
    pool = build_pool(db)
    graph = candidate_graph(pool.profiles, pool.state)
    scores = {}
    for u, vs in graph.items():
        for v in vs:
            if u < v:
                scores[(u, v)] = pool.score(u, v).mutual
    result = allocate_drop(graph, scores)
    drop = DropRow(week_start=week_start(next_drop_at(now)), status="draft", stats={
        "pool": len(pool.profiles), "eligible": len(graph),
        "with_candidates": sum(1 for v in graph.values() if v),
        "uncovered": sorted(result.uncovered), "pairs": len(result.pairs),
        "excluded_paused": sorted(pool.state.paused & set(pool.profiles)),
        "excluded_in_match": sorted(pool.state.in_exclusive & set(pool.profiles)),
    })
    db.add(drop)
    db.flush()
    for p in result.pairs:
        db.add(_proposal_from_score(drop.id, pool, p.a, p.b, "engine"))
    db.commit()
    db.refresh(drop)
    return drop


def drop_load(drop: DropRow) -> dict[int, int]:
    load: dict[int, int] = defaultdict(int)
    for pr in drop.proposals:
        load[pr.user_low_id] += 1
        load[pr.user_high_id] += 1
    return load


def _require_draft(drop: DropRow) -> None:
    if drop.status != "draft":
        raise DropError("אפשר לערוך רק דרופ בטיוטה")


def remove_pair(db: Session, drop: DropRow, proposal_id: int) -> None:
    _require_draft(drop)
    pr = db.get(ProposalRow, proposal_id)
    if pr is None or pr.drop_id != drop.id:
        raise DropError("הזוג לא נמצא בדרופ")
    db.delete(pr)
    db.commit()
    db.refresh(drop)


def swap_candidates(db: Session, drop: DropRow, proposal: ProposalRow, keep_user: int) -> list[dict]:
    """Ranked alternatives for `keep_user` in place of their current partner in `proposal`.

    Only engine-feasible partners (hard filters both ways, not paused / in a match / shown
    before) who are not already paired with keep_user in this drop. `available` is False when
    the candidate already has CAPACITY_PER_USER proposals in this drop."""
    if keep_user not in (proposal.user_low_id, proposal.user_high_id):
        raise DropError("המשתמש לא חלק מהזוג")
    pool = build_pool(db)
    if keep_user not in pool.profiles or not pool.state.is_available(keep_user):
        return []
    load = drop_load(drop)
    already = {pr.other(keep_user) for pr in drop.proposals
               if keep_user in (pr.user_low_id, pr.user_high_id)}
    me = pool.profiles[keep_user]
    out = []
    for cid, prof in pool.profiles.items():
        if cid == keep_user or cid in already or not pool.state.is_available(cid):
            continue
        if pool.state.pair_excluded(keep_user, cid) or not mutually_compatible(me, prof):
            continue
        ps = pool.score(keep_user, cid)
        out.append({"id": cid, "name": pool.participants[cid].display_name, "mutual": ps.mutual,
                    "load": load.get(cid, 0), "available": load.get(cid, 0) < CAPACITY_PER_USER})
    out.sort(key=lambda r: (not r["available"], -r["mutual"]))
    return out


def swap_pair(db: Session, drop: DropRow, proposal_id: int, keep_user: int, new_partner: int) -> ProposalRow:
    """Replace keep_user's partner in a draft pair. Keeps the per-user cap of 3."""
    _require_draft(drop)
    pr = db.get(ProposalRow, proposal_id)
    if pr is None or pr.drop_id != drop.id:
        raise DropError("הזוג לא נמצא בדרופ")
    cands = {c["id"]: c for c in swap_candidates(db, drop, pr, keep_user)}
    c = cands.get(new_partner)
    if c is None:
        raise DropError("המועמד/ת לא עומד/ת בפילטרים או כבר הוצג/ה")
    if not c["available"]:
        raise CapError(f"ל{c['name']} כבר יש {CAPACITY_PER_USER} הצעות בדרופ הזה")
    pool = build_pool(db)
    db.delete(pr)
    db.flush()
    new = _proposal_from_score(drop.id, pool, keep_user, new_partner, "swap")
    db.add(new)
    db.flush()
    db.refresh(drop)
    if max(drop_load(drop).values(), default=0) > CAPACITY_PER_USER:   # defence in depth
        db.rollback()
        raise CapError("חריגה מתקרת 3 הצעות")
    db.commit()
    return new


def pair_brief(a: Participant, b: Participant, explanation: dict) -> str:
    """Only information that may be shared with both sides: first name, age, city, the
    'looking for' lines, top values and the engine's shared signals. Never red flags,
    summaries or transcript content."""
    def side(p: Participant) -> str:
        top = sorted((p.values_scores or {}).items(), key=lambda kv: -kv[1])[:5]
        vals = ", ".join(VALUE_LABELS_HE.get(k, k) for k, _ in top)
        looking = "; ".join(p.looking_for or [])
        weekend = (p.lifestyle or {}).get("weekend", "")
        return (f"- {p.first_name}, {p.age}, {p.city}. ערכים בולטים: {vals}. "
                f"מחפש/ת: {looking}. סוף שבוע טוב: {weekend}")
    shared = ", ".join(signal_label_he(s) for s in explanation.get("top_signals", []) if s["kind"] != "visual")
    return f"{side(a)}\n{side(b)}\nמשותף לפי המנוע: {shared or 'לא צוין'}"


async def generate_why_texts(db: Session, drop: DropRow, llm: LLMClient, only_missing: bool = True,
                             concurrency: int = 4) -> tuple[int, list[str]]:
    sem = asyncio.Semaphore(concurrency)
    errors: list[str] = []
    todo = [pr for pr in drop.proposals if not (only_missing and pr.why_text.strip())]

    async def one(pr: ProposalRow) -> None:
        brief = pair_brief(pr.user_low, pr.user_high, pr.explanation or {})
        async with sem:
            try:
                pr.why_text = (await llm.why_text(brief)).strip()
            except LLMError as e:
                errors.append(f"{pr.user_low.first_name}+{pr.user_high.first_name}: {e}")

    await asyncio.gather(*(one(pr) for pr in todo))
    db.commit()
    return len(todo) - len(errors), errors


def publish_drop(db: Session, drop: DropRow, now: datetime | None = None) -> DropRow:
    now = now or utcnow()
    _require_draft(drop)
    if not drop.proposals:
        raise DropError("אין זוגות בדרופ")
    if any(not pr.why_text.strip() for pr in drop.proposals):
        raise DropError("לכל זוג צריך טקסט 'למה שניכם' לפני פרסום")
    if max(drop_load(drop).values()) > CAPACITY_PER_USER:
        raise CapError("חריגה מתקרת 3 הצעות")
    blocked = active_match_user_ids(db) | {p.id for p in db.scalars(
        select(Participant).where(Participant.paused))}
    bad = [pr for pr in drop.proposals if {pr.user_low_id, pr.user_high_id} & blocked]
    if bad:
        raise DropError("יש בדרופ משתתפים שנכנסו להתאמה או הושהו מאז היצירה. צרו טיוטה מחדש או הסירו אותם")
    for old in db.scalars(select(DropRow).where(DropRow.status == "published")):
        old.status = "closed"
    drop.status = "published"
    drop.published_at = now
    drop.deadline_at = now + REPLY_WINDOW
    db.commit()
    return drop


# =========================================================================== participant side

def participant_proposals(db: Session, pid: int) -> tuple[DropRow | None, list[ProposalRow]]:
    """The participant's cards in the latest published drop (only their own)."""
    drop = db.scalar(select(DropRow).where(DropRow.status == "published").order_by(DropRow.id.desc()))
    if drop is None:
        return None, []
    prs = [pr for pr in drop.proposals if pid in (pr.user_low_id, pr.user_high_id)]
    return drop, prs


def answer_proposal(db: Session, p: Participant, proposal_id: int, answer: str,
                    now: datetime | None = None) -> tuple[ProposalRow, MatchRow | None]:
    now = now or utcnow()
    if answer not in ("yes", "no"):
        raise AnswerError("תשובה לא חוקית")
    pr = db.get(ProposalRow, proposal_id)
    if pr is None or p.id not in (pr.user_low_id, pr.user_high_id):
        raise AnswerError("ההצעה לא נמצאה")
    drop = pr.drop
    if drop.status != "published" or drop.deadline_at is None:
        raise AnswerError("ההצעה לא פתוחה לתשובות")
    if now >= drop.deadline_at:
        raise AnswerError("הזמן לענות על ההצעה הזו עבר")
    if pr.response_of(p.id) != "pending":
        raise AnswerError("כבר ענית על ההצעה הזו, והתשובות סופיות")
    if pr.user_low_id == p.id:
        pr.response_low, pr.responded_low_at = answer, now
    else:
        pr.response_high, pr.responded_high_at = answer, now
    match = _maybe_create_match(db, pr, now)
    db.commit()
    return pr, match


def _maybe_create_match(db: Session, pr: ProposalRow, now: datetime) -> MatchRow | None:
    """Mutual yes -> exclusive match, unless one side is already in one (first mutual wins)."""
    if not pr.is_mutual:
        return None
    busy = active_match_user_ids(db)
    if pr.user_low_id in busy or pr.user_high_id in busy:
        pr.superseded = True
        return None
    m = MatchRow(proposal_id=pr.id, user_low_id=pr.user_low_id, user_high_id=pr.user_high_id, created_at=now)
    db.add(m)
    db.flush()
    return m


def can_view_photos_of(db: Session, viewer_id: int, owner_id: int) -> bool:
    """A participant may see their own photos, the people in their own cards of the current
    published drop, and their active match. Nobody else."""
    if viewer_id == owner_id:
        return True
    _, prs = participant_proposals(db, viewer_id)
    if any(owner_id in (pr.user_low_id, pr.user_high_id) for pr in prs):
        return True
    m = active_match_for(db, viewer_id)
    return m is not None and m.other(viewer_id) == owner_id


# =========================================================================== matches + debriefs

def close_match(db: Session, m: MatchRow, reason: str = "closed_by_team", now: datetime | None = None) -> None:
    m.is_active = False
    m.closed_at = now or utcnow()
    m.close_reason = reason
    db.commit()


def mark_date(db: Session, m: MatchRow, now: datetime | None = None) -> None:
    if m.date_happened_at is None:
        m.date_happened_at = now or utcnow()
    db.commit()


def apply_debrief_feedback(db: Session, d: Debrief, m: MatchRow) -> None:
    """Feed one debrief into matching.feedback (per-user weights + population prior).

    yes/no go through `update_from_debrief` unchanged. "unsure" uses the same SGD step with
    a label halfway between yes and no for that rating."""
    rater = db.get(Participant, d.rater_id)
    pr = m.proposal
    key = "low_high" if d.rater_id == pr.user_low_id else "high_low"
    f = Features(*pr.features[key])
    state, row = load_learning(db)
    w = user_weights(rater)
    d.weights_before = list(w.as_tuple())
    if d.meet_again in ("yes", "no"):
        update_from_debrief(state, w, DateDebrief(m.id, d.rater_id, d.ratee_id, d.rating,
                                                  d.meet_again == "yes"), f)
    else:
        y = 0.5 * (debrief_label(d.rating, True) + debrief_label(d.rating, False))
        center = list(state.feature_mean)
        if w.n_updates == 0:
            w.set_from(state.global_weights.as_tuple())
            w.bias = state.global_weights.bias
        sgd_step(w, f, y, center, USER_LR)
        sgd_step(state.global_weights, f, y, center, GLOBAL_LR)
        state.observe_features(f)
    d.weights_after = list(w.as_tuple())
    store_weights(rater, w)
    save_learning(state, row)


def submit_debrief(db: Session, p: Participant, match_id: int, rating: int, meet_again: str,
                   surprise_text: str, looked_like_photos: str, now: datetime | None = None) -> Debrief:
    m = db.get(MatchRow, match_id)
    if m is None or p.id not in (m.user_low_id, m.user_high_id):
        raise DebriefError("ההתאמה לא נמצאה")
    if not 1 <= rating <= 5 or meet_again not in ("yes", "no", "unsure") \
            or looked_like_photos not in ("yes", "partly", "no"):
        raise DebriefError("חסרות תשובות")
    if db.scalar(select(Debrief).where(Debrief.match_id == m.id, Debrief.rater_id == p.id)):
        raise DebriefError("כבר מילאת את התחקיר הזה. תודה!")
    d = Debrief(match_id=m.id, rater_id=p.id, ratee_id=m.other(p.id), rating=rating,
                meet_again=meet_again, surprise_text=surprise_text.strip()[:2000],
                looked_like_photos=looked_like_photos, created_at=now or utcnow())
    db.add(d)
    if m.date_happened_at is None:
        m.date_happened_at = now or utcnow()
    apply_debrief_feedback(db, d, m)
    db.commit()
    return d


def record_survey(db: Session, p: Participant, score: int, now: datetime | None = None) -> None:
    if not 1 <= score <= 5:
        raise ValueError("score")
    ws = week_start(now or utcnow())
    if db.scalar(select(SurveyResponse).where(SurveyResponse.participant_id == p.id,
                                              SurveyResponse.week_start == ws)):
        return
    db.add(SurveyResponse(participant_id=p.id, week_start=ws, score=score))
    db.commit()


def survey_done_this_week(db: Session, p: Participant, now: datetime | None = None) -> bool:
    ws = week_start(now or utcnow())
    return db.scalar(select(SurveyResponse.id).where(SurveyResponse.participant_id == p.id,
                                                     SurveyResponse.week_start == ws)) is not None


# =========================================================================== metrics

def _pct(n: int, d: int) -> float | None:
    return round(100.0 * n / d, 1) if d else None


def compute_metrics(db: Session) -> dict:
    parts = db.scalars(select(Participant)).all()
    invited = len(parts)
    consented = sum(1 for p in parts if p.consented_at)
    interviewed = sum(1 for p in parts if p.interview_done_at)
    paid = sum(1 for p in parts if p.paid_at)

    surveys = db.scalars(select(SurveyResponse)).all()
    h2 = round(sum(s.score for s in surveys) / len(surveys), 2) if surveys else None

    props = db.scalars(select(ProposalRow).join(DropRow).where(DropRow.status != "draft")).all()
    mutual = sum(1 for pr in props if pr.is_mutual)
    both_answered = sum(1 for pr in props if pr.response_low != "pending" and pr.response_high != "pending")
    yes_answers = sum((pr.response_low == "yes") + (pr.response_high == "yes") for pr in props)
    answers = sum((pr.response_low != "pending") + (pr.response_high != "pending") for pr in props)

    matches = db.scalars(select(MatchRow)).all()
    replied_marked = [m for m in matches if m.replied_within_48h is not None]
    replied = sum(1 for m in replied_marked if m.replied_within_48h)
    dated7 = sum(1 for m in matches if m.date_happened_at and m.date_happened_at - m.created_at <= DATE_WINDOW)
    dated = [m for m in matches if m.date_happened_at]
    second = 0
    for m in dated:
        ds = {d.rater_id: d.meet_again for d in m.debriefs}
        if ds.get(m.user_low_id) == "yes" and ds.get(m.user_high_id) == "yes":
            second += 1

    per_drop = []
    for drop in db.scalars(select(DropRow).where(DropRow.status != "draft").order_by(DropRow.id)):
        prs = drop.proposals
        per_drop.append({
            "id": drop.id, "week_start": drop.week_start.isoformat(), "status": drop.status,
            "pairs": len(prs), "mutual": sum(1 for pr in prs if pr.is_mutual),
            "users": len({u for pr in prs for u in (pr.user_low_id, pr.user_high_id)}),
        })

    return {
        "counts": {"invited": invited, "consented": consented, "interviewed": interviewed, "paid": paid,
                   "proposals": len(props), "matches": len(matches), "dates": len(dated),
                   "debriefs": sum(len(m.debriefs) for m in matches), "survey_answers": len(surveys)},
        "H1_interview_completion_pct": _pct(interviewed, consented),
        "H2_three_is_enough_avg": h2,
        "H3_mutual_accept_pct": _pct(mutual, len(props)),
        "H3_mutual_of_fully_answered_pct": _pct(mutual, both_answered),
        "yes_rate_pct": _pct(yes_answers, answers),
        "H4_replied_48h_pct": _pct(replied, len(replied_marked)),
        "H5_date_within_7d_pct": _pct(dated7, len(matches)),
        "H6_paid_of_invited_pct": _pct(paid, invited),
        "second_date_pct": _pct(second, len(dated)),
        "per_drop": per_drop,
    }


def count(db: Session, model) -> int:  # noqa: ANN001
    return db.scalar(select(func.count()).select_from(model)) or 0
