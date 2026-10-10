"""Participant side (Hebrew, RTL, mobile-first). Auth = secret link -> signed session cookie."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse, Response, StreamingResponse
from sqlalchemy.orm import Session

from . import services as S
from .choices import (CITY_COORDS, DISTANCES, GENDERS, INTENTS, LOOKED_LIKE, MEET_AGAIN,
                      PARTNER_SMOKING, RELIGIOSITY, SEEKING, SMOKING, VISUAL_STATEMENTS, WANTS_KIDS)
from .interview import add_message, api_history, extract_and_store, user_turns
from .llm import LLMError, MarkerFilter
from .models import MatchRow, Participant, Photo, utcnow
from .profile import neutral_visual_vector
from .security import current_participant, find_by_token, login_participant, verify_csrf
from .storage import MAX_PHOTOS, MIN_PHOTOS, UploadError, validate_image
from .web import get_db, render

log = logging.getLogger("app.participant")
router = APIRouter()

FEE_NIS = 49


def me(request: Request, db: Session = Depends(get_db)) -> Participant:
    return current_participant(request, db)


def next_step(p: Participant) -> str:
    if not p.consented_at:
        return "/p/welcome"
    if not p.facts_done_at:
        return "/p/facts"
    if not p.interview_done_at:
        return "/p/interview"
    if not p.visual_done_at:
        return "/p/visual"
    return "/p/home"


def go(url: str) -> RedirectResponse:
    return RedirectResponse(url, status_code=303)


# --------------------------------------------------------------------------- entry

@router.get("/p/")
def p_root(p: Participant = Depends(me)):
    return go(next_step(p))


@router.get("/p/welcome")
def welcome(request: Request, p: Participant = Depends(me)):
    return render(request, "welcome.html", p=p, fee=FEE_NIS)


@router.post("/p/welcome", dependencies=[Depends(verify_csrf)])
async def welcome_post(request: Request, p: Participant = Depends(me), db: Session = Depends(get_db)):
    form = await request.form()
    if form.get("consent") != "1":
        return render(request, "welcome.html", p=p, fee=FEE_NIS, error="צריך לאשר את התנאים כדי להמשיך",
                      status_code=400)
    if not p.consented_at:
        p.consented_at = utcnow()
        db.commit()
    return go(next_step(p))


# --------------------------------------------------------------------------- facts

FACT_CHOICES = dict(genders=GENDERS, seeking=SEEKING, intents=INTENTS, wants_kids=WANTS_KIDS,
                    religiosity=RELIGIOSITY, smoking=SMOKING, partner_smoking=PARTNER_SMOKING,
                    distances=DISTANCES, cities=CITY_COORDS, min_photos=MIN_PHOTOS, max_photos=MAX_PHOTOS)


def _facts_page(request: Request, p: Participant, errors: list[str] | None = None, status: int = 200):
    return render(request, "facts.html", p=p, errors=errors or [], status_code=status, **FACT_CHOICES)


@router.get("/p/facts")
def facts(request: Request, p: Participant = Depends(me)):
    if not p.consented_at:
        return go("/p/welcome")
    return _facts_page(request, p)


def _int(form, name: str, lo: int, hi: int, errors: list[str], label: str) -> int | None:  # noqa: ANN001
    try:
        v = int(str(form.get(name, "")).strip())
    except ValueError:
        errors.append(f"{label}: חסר")
        return None
    if not lo <= v <= hi:
        errors.append(f"{label}: ערך לא תקין")
        return None
    return v


@router.post("/p/facts", dependencies=[Depends(verify_csrf)])
async def facts_post(request: Request, p: Participant = Depends(me), db: Session = Depends(get_db)):
    settings = request.app.state.settings
    storage = request.app.state.storage
    form = await request.form(max_files=MAX_PHOTOS + 2)
    errors: list[str] = []
    allowed = lambda key, choices: str(form.get(key, "")) in {str(k) for k, _ in choices}  # noqa: E731

    gender = str(form.get("gender", ""))
    if not allowed("gender", GENDERS):
        errors.append("מגדר: חסר")
    seeking = [g for g in form.getlist("seeking") if g in dict(SEEKING)]
    if not seeking:
        errors.append("את מי מחפשים: חסר")
    age = _int(form, "age", 18, 99, errors, "גיל")
    age_min = _int(form, "age_min", 18, 99, errors, "גיל מינימלי")
    age_max = _int(form, "age_max", 18, 99, errors, "גיל מקסימלי")
    if age_min and age_max and age_min > age_max:
        errors.append("טווח גילאים: המינימום גדול מהמקסימום")
    city = str(form.get("city", ""))
    if city not in CITY_COORDS:
        errors.append("עיר: חסר")
    max_dist = _int(form, "max_distance_km", 1, 1000, errors, "מרחק")
    if not allowed("intent", INTENTS):
        errors.append("כוונה: חסר")
    has_kids = form.get("has_kids")
    if has_kids not in ("0", "1"):
        errors.append("יש ילדים: חסר")
    if not allowed("wants_kids", WANTS_KIDS):
        errors.append("רוצה ילדים: חסר")
    accepts_kids = form.get("accepts_partner_with_kids")
    if accepts_kids not in ("0", "1"):
        errors.append("בן/בת זוג עם ילדים: חסר")
    religiosity = _int(form, "religiosity", 0, 3, errors, "דתיות")
    smoking = _int(form, "smoking", 0, 2, errors, "עישון")
    partner_smoking = _int(form, "max_partner_smoking", 0, 2, errors, "עישון אצל בן/בת הזוג")

    uploads = [f for f in form.getlist("photos") if getattr(f, "filename", "")]
    new_photos: list[tuple[bytes, str]] = []
    for f in uploads:
        data = await f.read(settings.max_upload_bytes + 1)
        try:
            new_photos.append((data, validate_image(data, settings.max_upload_bytes)))
        except UploadError as e:
            errors.append(f"{f.filename}: {e}")
    remove_ids = {int(x) for x in form.getlist("remove_photo") if str(x).isdigit()}
    kept = [ph for ph in p.photos if ph.id not in remove_ids]
    total = len(kept) + len(new_photos)
    if total < MIN_PHOTOS or total > MAX_PHOTOS:
        errors.append(f"צריך {MIN_PHOTOS}–{MAX_PHOTOS} תמונות (כרגע {total})")

    if errors:
        return _facts_page(request, p, errors, 400)

    p.gender, p.seeking, p.age, p.age_min, p.age_max = gender, seeking, age, age_min, age_max
    p.city = city
    _, p.lat, p.lon = CITY_COORDS[city]
    p.max_distance_km = float(max_dist)
    p.intent = str(form["intent"])
    p.has_kids = has_kids == "1"
    p.wants_kids = str(form["wants_kids"])
    p.accepts_partner_with_kids = accepts_kids == "1"
    p.religiosity, p.smoking, p.max_partner_smoking = religiosity, smoking, partner_smoking
    for ph in p.photos:
        if ph.id in remove_ids:
            storage.delete(ph.storage_key, session=db)
            db.delete(ph)
    for i, (data, ctype) in enumerate(new_photos):
        key = storage.save(data, ctype, prefix=f"p{p.id}", session=db)
        db.add(Photo(participant_id=p.id, storage_key=key, content_type=ctype, ordinal=len(kept) + i))
    p.facts_done_at = p.facts_done_at or utcnow()
    db.commit()
    return go(next_step(p))


# --------------------------------------------------------------------------- interview

@router.get("/p/interview")
def interview(request: Request, p: Participant = Depends(me)):
    if not p.facts_done_at:
        return go(next_step(p))
    if p.interview_done_at:
        return go("/p/visual")
    s = request.app.state.settings
    return render(request, "interview.html", p=p, messages=p.messages, turns=user_turns(p),
                  max_turns=s.interview_max_user_turns, min_turns=s.interview_min_user_turns)


def _sse(obj: dict) -> bytes:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode()


@router.post("/p/interview/message", dependencies=[Depends(verify_csrf)])
async def interview_message(request: Request, p: Participant = Depends(me), db: Session = Depends(get_db)):
    settings = request.app.state.settings
    llm = request.app.state.llm
    if p.interview_done_at or p.interview_ended or not p.facts_done_at:
        raise HTTPException(409, "interview closed")
    body = await request.json()
    text = str(body.get("text", "")).strip()[:4000]
    turns = user_turns(p)
    last_role = p.messages[-1].role if p.messages else None
    if text:
        if last_role == "user":
            raise HTTPException(409, "waiting for עדי")
        if turns >= settings.interview_max_user_turns:
            raise HTTPException(409, "turn limit reached")
        add_message(db, p, "user", text)
        turns += 1
    elif last_role == "assistant":
        raise HTTPException(400, "empty message")
    # else: opening turn (no messages yet) or retry after a failed reply
    history = api_history(p)
    wrap_up = turns >= settings.interview_max_user_turns
    pid = p.id
    session_factory = request.app.state.db.SessionLocal

    async def gen():
        flt = MarkerFilter()
        parts: list[str] = []
        try:
            async for chunk in llm.stream_interview(history, wrap_up):
                out = flt.feed(chunk)
                if out:
                    parts.append(out)
                    yield _sse({"delta": out})
            tail = flt.flush()
            if tail:
                parts.append(tail)
                yield _sse({"delta": tail})
        except LLMError as e:
            yield _sse({"error": f"{e}. אפשר לנסות שוב."})
            return
        reply = "".join(parts).strip()
        ended = flt.found or wrap_up
        with session_factory() as s2:
            p2 = s2.get(Participant, pid)
            add_message(s2, p2, "assistant", reply)
            if ended:
                p2.interview_ended = True
                s2.commit()
        yield _sse({"done": True, "ended": ended, "turns": turns})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/p/interview/finish", dependencies=[Depends(verify_csrf)])
async def interview_finish(request: Request, p: Participant = Depends(me), db: Session = Depends(get_db)):
    s = request.app.state.settings
    if p.interview_done_at:
        return go(next_step(p))
    if not (p.interview_ended or user_turns(p) >= s.interview_min_user_turns):
        return go("/p/interview")
    await extract_and_store(db, p, request.app.state.llm)
    p.interview_done_at = utcnow()
    p.interview_ended = True
    db.commit()
    return go(next_step(p))


# --------------------------------------------------------------------------- visual taste

@router.get("/p/visual")
def visual(request: Request, p: Participant = Depends(me)):
    if not p.interview_done_at:
        return go(next_step(p))
    return render(request, "visual.html", p=p, statements=VISUAL_STATEMENTS)


@router.post("/p/visual", dependencies=[Depends(verify_csrf)])
async def visual_post(request: Request, p: Participant = Depends(me), db: Session = Depends(get_db)):
    form = await request.form()
    answers = {}
    for key, _ in VISUAL_STATEMENTS:
        v = str(form.get(key, ""))
        if v not in {"1", "2", "3", "4", "5"}:
            return render(request, "visual.html", p=p, statements=VISUAL_STATEMENTS,
                          error="דרג/י את כל שלושת המשפטים", status_code=400)
        answers[key] = int(v)
    p.visual_answers = answers
    # TODO(avatar-calibration): the real product calibrates taste from 6 rounds of synthetic
    # avatars and embeds the participant's photos. The pilot stores neutral vectors so every
    # pair gets the same visual feature; the answers above are kept for later analysis.
    p.visual_taste_vec = neutral_visual_vector()
    p.appearance_vec = neutral_visual_vector()
    p.visual_done_at = p.visual_done_at or utcnow()
    db.commit()
    return go("/p/home")


# --------------------------------------------------------------------------- waiting / home

@router.get("/p/home")
def home(request: Request, p: Participant = Depends(me), db: Session = Depends(get_db)):
    if not p.profile_complete:
        return go(next_step(p))
    now = utcnow()
    drop, cards = S.participant_proposals(db, p.id)
    open_cards = [c for c in cards if drop and drop.deadline_at and now < drop.deadline_at
                  and c.response_of(p.id) == "pending"]
    match = S.active_match_for(db, p.id)
    partner = db.get(Participant, match.other(p.id)) if match else None
    return render(request, "home.html", p=p, next_drop=S.next_drop_at(now), drop=drop, cards=cards,
                  open_cards=open_cards, match=match, partner=partner,
                  status=S.participant_status(p, db=db), survey_done=S.survey_done_this_week(db, p, now))


@router.post("/p/survey", dependencies=[Depends(verify_csrf)])
async def survey(request: Request, p: Participant = Depends(me), db: Session = Depends(get_db)):
    form = await request.form()
    try:
        S.record_survey(db, p, int(str(form.get("score", "0"))))
    except ValueError:
        pass
    return go("/p/home")


# --------------------------------------------------------------------------- the Drop

def _card(db: Session, viewer: Participant, pr) -> dict:  # noqa: ANN001
    other = db.get(Participant, pr.other(viewer.id))
    return {"proposal": pr, "id": pr.id, "first_name": other.first_name, "age": other.age,
            "city": CITY_COORDS.get(other.city, (other.city,))[0], "photos": [ph.id for ph in other.photos],
            "why": pr.why_text, "my_answer": pr.response_of(viewer.id)}


@router.get("/p/drop")
def drop_view(request: Request, p: Participant = Depends(me), db: Session = Depends(get_db)):
    drop, prs = S.participant_proposals(db, p.id)
    if drop is None or not prs:
        return go("/p/home")
    now = utcnow()
    return render(request, "drop.html", p=p, drop=drop, cards=[_card(db, p, pr) for pr in prs],
                  open=drop.deadline_at is not None and now < drop.deadline_at,
                  error=request.query_params.get("e", ""))


@router.post("/p/drop/answer", dependencies=[Depends(verify_csrf)])
async def drop_answer(request: Request, p: Participant = Depends(me), db: Session = Depends(get_db)):
    form = await request.form()
    try:
        pr, match = S.answer_proposal(db, p, int(str(form.get("proposal_id", "0"))), str(form.get("answer", "")))
    except (S.AnswerError, ValueError) as e:
        return render(request, "message.html", title="אי אפשר לשמור את התשובה", body=str(e),
                      link="/p/drop", link_text="חזרה ל-Drop", status_code=400)
    if match is not None:
        return go("/p/match")
    return go("/p/drop")


@router.get("/p/match")
def match_view(request: Request, p: Participant = Depends(me), db: Session = Depends(get_db)):
    m = S.active_match_for(db, p.id)
    if m is None:
        return go("/p/home")
    partner = db.get(Participant, m.other(p.id))
    return render(request, "match.html", p=p, match=m, partner=partner,
                  photo=partner.photos[0].id if partner.photos else None)


# --------------------------------------------------------------------------- debrief

def _my_match(db: Session, p: Participant, match_id: int) -> MatchRow:
    m = db.get(MatchRow, match_id)
    if m is None or p.id not in (m.user_low_id, m.user_high_id):
        raise HTTPException(404)
    return m


@router.get("/p/debrief/{match_id}")
def debrief(request: Request, match_id: int, p: Participant = Depends(me), db: Session = Depends(get_db)):
    m = _my_match(db, p, match_id)
    partner = db.get(Participant, m.other(p.id))
    done = any(d.rater_id == p.id for d in m.debriefs)
    return render(request, "debrief.html", p=p, match=m, partner=partner, done=done,
                  meet_again=MEET_AGAIN, looked_like=LOOKED_LIKE)


@router.post("/p/debrief/{match_id}", dependencies=[Depends(verify_csrf)])
async def debrief_post(request: Request, match_id: int, p: Participant = Depends(me),
                       db: Session = Depends(get_db)):
    m = _my_match(db, p, match_id)
    form = await request.form()
    try:
        S.submit_debrief(db, p, m.id, int(str(form.get("rating", "0")) or 0), str(form.get("meet_again", "")),
                         str(form.get("surprise_text", "")), str(form.get("looked_like_photos", "")))
    except (S.DebriefError, ValueError) as e:
        partner = db.get(Participant, m.other(p.id))
        return render(request, "debrief.html", p=p, match=m, partner=partner, done=False, error=str(e),
                      meet_again=MEET_AGAIN, looked_like=LOOKED_LIKE, status_code=400)
    return render(request, "message.html", title="תודה!", body="עדי קיבלה את התשובות. הן נשארות אצלנו "
                  "ולא מועברות לצד השני, ומלמדות אותנו מה באמת עובד לך.", link="/p/home", link_text="למסך הבית")


# --------------------------------------------------------------------------- photos

@router.get("/media/{photo_id}")
def media(request: Request, photo_id: int, p: Participant = Depends(me), db: Session = Depends(get_db)):
    ph = db.get(Photo, photo_id)
    if ph is None or not S.can_view_photos_of(db, p.id, ph.participant_id):
        raise HTTPException(404)
    return Response(request.app.state.storage.load(ph.storage_key), media_type=ph.content_type,
                    headers={"Cache-Control": "private, max-age=600"})


# --------------------------------------------------------------------------- secret links (last: catch-all)

@router.get("/p/{token}/debrief/{match_id}")
def token_debrief(request: Request, token: str, match_id: int, db: Session = Depends(get_db)):
    p = find_by_token(db, token)
    if p is None:
        raise HTTPException(404)
    login_participant(request, p)
    return go(f"/p/debrief/{match_id}")


@router.get("/p/{token}")
def token_login(request: Request, token: str, db: Session = Depends(get_db)):
    p = find_by_token(db, token)
    if p is None:
        raise HTTPException(404)
    login_participant(request, p)
    return go("/p/")   # drop the token from the address bar right away
