"""Matchmaker / team console under /admin (HTTP basic auth from ADMIN_USER / ADMIN_PASSWORD)."""

from __future__ import annotations

import csv
import io
import re

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from matching.models import CAPACITY_PER_USER
from matching.scoring import VALUE_DIMENSION_NAMES

from . import services as S
from .choices import CITY_COORDS, LABELS, VISUAL_STATEMENTS
from .interview import extract_and_store
from .models import STATUS_LABELS_HE, DropRow, MatchRow, Participant, Photo, ProposalRow
from .profile import values_labels, values_vector
from .prompts import LIFESTYLE_FIELDS, VALUE_LABELS_HE
from .security import check_admin, new_token, verify_csrf
from .web import get_db, render

router = APIRouter(prefix="/admin", dependencies=[Depends(check_admin)])


def go(url: str) -> RedirectResponse:
    return RedirectResponse(url, status_code=303)


def invite_url(request: Request, p: Participant) -> str:
    return f"{request.app.state.settings.base_url}/p/{p.token}"


def wa_phone(phone: str) -> str:
    digits = re.sub(r"\D", "", phone or "")
    if digits.startswith("0"):
        digits = "972" + digits[1:]
    return digits


def _participant_or_404(db: Session, pid: int) -> Participant:
    p = db.get(Participant, pid)
    if p is None:
        raise HTTPException(404)
    return p


# --------------------------------------------------------------------------- participants

@router.get("")
@router.get("/")
def participants(request: Request, db: Session = Depends(get_db)):
    parts = db.scalars(select(Participant).order_by(Participant.id)).all()
    in_match = S.active_match_user_ids(db)
    rows = [{"p": p, "status": S.participant_status(p, in_match), "invite": invite_url(request, p),
             "wa": wa_phone(p.phone)} for p in parts]
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return render(request, "admin_participants.html", rows=rows, counts=counts, labels=STATUS_LABELS_HE,
                  llm_name=request.app.state.llm.name)


@router.get("/participants/new")
def new_participant(request: Request):
    return render(request, "admin_new.html")


@router.post("/participants/new", dependencies=[Depends(verify_csrf)])
async def new_participant_post(request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    first = str(form.get("first_name", "")).strip()[:80]
    last = str(form.get("last_name", "")).strip()[:80]
    phone = str(form.get("phone", "")).strip()[:32]
    if not first or not re.fullmatch(r"[0-9+\-\s()]{7,32}", phone):
        return render(request, "admin_new.html", error="צריך שם פרטי וטלפון תקין", status_code=400)
    p = Participant(first_name=first, last_name=last, phone=phone, token=new_token())
    db.add(p)
    db.commit()
    return go(f"/admin/participants/{p.id}?created=1")


@router.get("/participants/{pid}")
def participant_detail(request: Request, pid: int, db: Session = Depends(get_db)):
    p = _participant_or_404(db, pid)
    matches = db.scalars(select(MatchRow).where(
        (MatchRow.user_low_id == pid) | (MatchRow.user_high_id == pid)).order_by(MatchRow.id.desc())).all()
    return render(request, "admin_participant.html", p=p, status=S.participant_status(p, db=db),
                  labels=STATUS_LABELS_HE, invite=invite_url(request, p), wa=wa_phone(p.phone),
                  values=values_labels(p.values_scores or {}), value_names=VALUE_DIMENSION_NAMES,
                  value_he=VALUE_LABELS_HE, lifestyle_fields=LIFESTYLE_FIELDS, fact_labels=LABELS,
                  cities=CITY_COORDS, visual_statements=VISUAL_STATEMENTS, matches=matches,
                  created=request.query_params.get("created"), msg=request.query_params.get("msg", ""))


@router.post("/participants/{pid}/action", dependencies=[Depends(verify_csrf)])
async def participant_action(request: Request, pid: int, db: Session = Depends(get_db)):
    p = _participant_or_404(db, pid)
    form = await request.form()
    action = form.get("action")
    if action == "mark_paid":
        p.paid_at = p.paid_at or S.utcnow()
    elif action == "unmark_paid":
        p.paid_at = None
    elif action == "pause":
        p.paused = True
    elif action == "unpause":
        p.paused = False
    elif action == "rotate_token":
        p.token = new_token()
    elif action == "reopen_interview":
        p.interview_done_at = None
        p.interview_ended = False
    else:
        raise HTTPException(400, "unknown action")
    db.commit()
    back = str(form.get("back", ""))
    return go(back if back.startswith("/admin") else f"/admin/participants/{pid}")


@router.post("/participants/{pid}/extract", dependencies=[Depends(verify_csrf)])
async def participant_extract(request: Request, pid: int, db: Session = Depends(get_db)):
    p = _participant_or_404(db, pid)
    ok = await extract_and_store(db, p, request.app.state.llm)
    return go(f"/admin/participants/{pid}?msg={'הפרופיל חולץ מחדש' if ok else 'החילוץ נכשל'}")


@router.post("/participants/{pid}/profile", dependencies=[Depends(verify_csrf)])
async def participant_profile_save(request: Request, pid: int, db: Session = Depends(get_db)):
    p = _participant_or_404(db, pid)
    form = await request.form()
    p.first_name = str(form.get("first_name", p.first_name)).strip()[:80] or p.first_name
    p.last_name = str(form.get("last_name", p.last_name)).strip()[:80]
    p.phone = str(form.get("phone", p.phone)).strip()[:32] or p.phone
    p.notes = str(form.get("notes", "")).strip()[:4000]
    p.summary = str(form.get("summary", "")).strip()[:4000]
    scores = {}
    for i, name in enumerate(VALUE_DIMENSION_NAMES):
        try:
            scores[name] = max(0, min(10, int(str(form.get(f"v{i}", "5")))))
        except ValueError:
            scores[name] = 5
    p.values_scores = scores
    p.values_vec = values_vector(scores)
    p.lifestyle = {k: str(form.get(f"life_{k}", "")).strip()[:300] for k in LIFESTYLE_FIELDS}
    p.looking_for = [s for s in (str(form.get(f"look{i}", "")).strip()[:300] for i in range(3)) if s]
    p.red_flags = [s.strip() for s in str(form.get("red_flags", "")).splitlines() if s.strip()]
    db.commit()
    return go(f"/admin/participants/{pid}?msg=נשמר")


@router.get("/media/{photo_id}")
def admin_media(request: Request, photo_id: int, db: Session = Depends(get_db)):
    ph = db.get(Photo, photo_id)
    if ph is None:
        raise HTTPException(404)
    return Response(request.app.state.storage.load(ph.storage_key), media_type=ph.content_type,
                    headers={"Cache-Control": "private, max-age=600"})


# --------------------------------------------------------------------------- Drop workflow

@router.get("/drops")
def drops(request: Request, db: Session = Depends(get_db)):
    draft = S.current_draft(db)
    if draft:
        return go(f"/admin/drops/{draft.id}")
    all_drops = db.scalars(select(DropRow).order_by(DropRow.id.desc())).all()
    return render(request, "admin_drops.html", drops=all_drops, error=request.query_params.get("e", ""))


@router.post("/drops/generate", dependencies=[Depends(verify_csrf)])
def drops_generate(request: Request, db: Session = Depends(get_db)):
    drop = S.generate_draft_drop(db)
    return go(f"/admin/drops/{drop.id}")


def _drop_or_404(db: Session, drop_id: int) -> DropRow:
    d = db.get(DropRow, drop_id)
    if d is None:
        raise HTTPException(404)
    return d


@router.get("/drops/{drop_id}")
def drop_review(request: Request, drop_id: int, db: Session = Depends(get_db)):
    drop = _drop_or_404(db, drop_id)
    load = S.drop_load(drop)
    names = {p.id: p for p in db.scalars(select(Participant))}
    swap = None
    if "swap" in request.query_params and drop.status == "draft":
        pr = db.get(ProposalRow, int(request.query_params["swap"]))
        keep = int(request.query_params.get("keep", "0"))
        if pr and pr.drop_id == drop.id and keep in (pr.user_low_id, pr.user_high_id):
            swap = {"proposal": pr, "keep": names[keep], "replace": names[pr.other(keep)],
                    "candidates": S.swap_candidates(db, drop, pr, keep)}
    return render(request, "admin_drop.html", drop=drop, load=load, cap=CAPACITY_PER_USER, names=names,
                  swap=swap, error=request.query_params.get("e", ""), msg=request.query_params.get("msg", ""),
                  llm_name=request.app.state.llm.name)


def _drop_redirect(drop_id: int, error: str = "", msg: str = "") -> RedirectResponse:
    from urllib.parse import urlencode
    q = urlencode({k: v for k, v in (("e", error), ("msg", msg)) if v})
    return go(f"/admin/drops/{drop_id}" + (f"?{q}" if q else ""))


@router.post("/drops/{drop_id}/remove", dependencies=[Depends(verify_csrf)])
async def drop_remove(request: Request, drop_id: int, db: Session = Depends(get_db)):
    drop = _drop_or_404(db, drop_id)
    form = await request.form()
    try:
        S.remove_pair(db, drop, int(str(form.get("proposal_id", "0"))))
    except S.DropError as e:
        return _drop_redirect(drop_id, error=str(e))
    return _drop_redirect(drop_id, msg="הזוג הוסר")


@router.post("/drops/{drop_id}/swap", dependencies=[Depends(verify_csrf)])
async def drop_swap(request: Request, drop_id: int, db: Session = Depends(get_db)):
    drop = _drop_or_404(db, drop_id)
    form = await request.form()
    try:
        S.swap_pair(db, drop, int(str(form["proposal_id"])), int(str(form["keep"])), int(str(form["new_partner"])))
    except (S.DropError, KeyError, ValueError) as e:
        return _drop_redirect(drop_id, error=str(e))
    return _drop_redirect(drop_id, msg="ההחלפה בוצעה")


@router.post("/drops/{drop_id}/why", dependencies=[Depends(verify_csrf)])
async def drop_why(request: Request, drop_id: int, db: Session = Depends(get_db)):
    drop = _drop_or_404(db, drop_id)
    form = await request.form()
    if drop.status != "draft":
        return _drop_redirect(drop_id, error="אפשר לייצר טקסטים רק לטיוטה")
    n, errors = await S.generate_why_texts(db, drop, request.app.state.llm,
                                           only_missing=form.get("all") != "1")
    if errors:
        return _drop_redirect(drop_id, error="; ".join(errors)[:500])
    return _drop_redirect(drop_id, msg=f"נוצרו {n} טקסטים")


@router.post("/drops/{drop_id}/text", dependencies=[Depends(verify_csrf)])
async def drop_text(request: Request, drop_id: int, db: Session = Depends(get_db)):
    drop = _drop_or_404(db, drop_id)
    form = await request.form()
    pr = db.get(ProposalRow, int(str(form.get("proposal_id", "0"))))
    if pr is None or pr.drop_id != drop.id or drop.status != "draft":
        return _drop_redirect(drop_id, error="אי אפשר לערוך")
    pr.why_text = str(form.get("why_text", "")).strip()[:1500]
    db.commit()
    return _drop_redirect(drop_id, msg="הטקסט נשמר")


@router.post("/drops/{drop_id}/publish", dependencies=[Depends(verify_csrf)])
def drop_publish(request: Request, drop_id: int, db: Session = Depends(get_db)):
    drop = _drop_or_404(db, drop_id)
    try:
        S.publish_drop(db, drop)
    except S.DropError as e:
        return _drop_redirect(drop_id, error=str(e))
    return go("/admin/responses")


# --------------------------------------------------------------------------- responses

@router.get("/responses")
def responses(request: Request, db: Session = Depends(get_db)):
    drops_ = db.scalars(select(DropRow).where(DropRow.status != "draft").order_by(DropRow.id.desc())).all()
    sel = request.query_params.get("drop")
    drop = next((d for d in drops_ if str(d.id) == sel), drops_[0] if drops_ else None)
    matches = db.scalars(select(MatchRow).order_by(MatchRow.is_active.desc(), MatchRow.id.desc())).all()
    names = {p.id: p for p in db.scalars(select(Participant))}
    return render(request, "admin_responses.html", drops=drops_, drop=drop, matches=matches, names=names,
                  base_url=request.app.state.settings.base_url, wa=wa_phone, now=S.utcnow())


@router.post("/matches/{match_id}/action", dependencies=[Depends(verify_csrf)])
async def match_action(request: Request, match_id: int, db: Session = Depends(get_db)):
    m = db.get(MatchRow, match_id)
    if m is None:
        raise HTTPException(404)
    form = await request.form()
    action = form.get("action")
    if action == "group_opened":
        m.group_opened_at = m.group_opened_at or S.utcnow()
        db.commit()
    elif action == "replied_yes":
        m.replied_within_48h = True
        db.commit()
    elif action == "replied_no":
        m.replied_within_48h = False
        db.commit()
    elif action == "date":
        S.mark_date(db, m)
    elif action == "close":
        S.close_match(db, m, str(form.get("reason", "closed_by_team"))[:32] or "closed_by_team")
    else:
        raise HTTPException(400, "unknown action")
    return go("/admin/responses#matches")


# --------------------------------------------------------------------------- metrics + CSV

@router.get("/metrics")
def metrics(request: Request, db: Session = Depends(get_db)):
    return render(request, "admin_metrics.html", m=S.compute_metrics(db))


def _csv(rows: list[list], filename: str) -> Response:
    buf = io.StringIO()
    csv.writer(buf).writerows(rows)
    return Response("﻿" + buf.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.get("/export/metrics.csv")
def export_metrics(db: Session = Depends(get_db)):
    m = S.compute_metrics(db)
    rows = [["metric", "value"]]
    rows += [[k, v] for k, v in m.items() if k not in ("counts", "per_drop")]
    rows += [[f"count_{k}", v] for k, v in m["counts"].items()]
    return _csv(rows, "metrics.csv")


@router.get("/export/pairs.csv")
def export_pairs(db: Session = Depends(get_db)):
    rows = [["drop_id", "week_start", "drop_status", "proposal_id", "user_low_id", "user_low_name",
             "user_high_id", "user_high_name", "mutual_score", "source", "response_low", "response_high",
             "mutual", "superseded", "match_id", "replied_within_48h", "date_happened_at",
             "debrief_low_rating", "debrief_low_meet_again", "debrief_high_rating", "debrief_high_meet_again"]]
    matches = {m.proposal_id: m for m in db.scalars(select(MatchRow))}
    for pr in db.scalars(select(ProposalRow).join(DropRow).where(DropRow.status != "draft")
                         .order_by(ProposalRow.id)):
        m = matches.get(pr.id)
        ds = {d.rater_id: d for d in (m.debriefs if m else [])}
        dl, dh = ds.get(pr.user_low_id), ds.get(pr.user_high_id)
        rows.append([pr.drop_id, pr.drop.week_start, pr.drop.status, pr.id, pr.user_low_id,
                     pr.user_low.display_name, pr.user_high_id, pr.user_high.display_name,
                     round(pr.mutual_score, 4), pr.source, pr.response_low, pr.response_high, pr.is_mutual,
                     pr.superseded, m.id if m else "", "" if not m or m.replied_within_48h is None
                     else m.replied_within_48h, m.date_happened_at if m else "",
                     dl.rating if dl else "", dl.meet_again if dl else "",
                     dh.rating if dh else "", dh.meet_again if dh else ""])
    return _csv(rows, "pairs.csv")
