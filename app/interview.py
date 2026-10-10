"""Interview orchestration: history building, turn limits, profile extraction."""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from .llm import LLMClient, LLMError
from .models import InterviewMessage, Participant, utcnow
from .profile import (ProfileParseError, facts_text, neutral_visual_vector, parse_profile,
                      transcript_text, values_vector)
from .prompts import OPENING_USER_TURN

log = logging.getLogger("app.interview")


def user_turns(p: Participant) -> int:
    return sum(1 for m in p.messages if m.role == "user")


def api_history(p: Participant) -> list[dict]:
    """Stored transcript starts with עדי's opening, so a synthetic first user turn is
    prepended (the Messages API needs the conversation to start with a user turn).
    Consecutive same-role messages (e.g. a retried turn) are merged."""
    out: list[dict] = [{"role": "user", "content": OPENING_USER_TURN}]
    for m in p.messages:
        if out and out[-1]["role"] == m.role:
            out[-1]["content"] += "\n\n" + m.content
        else:
            out.append({"role": m.role, "content": m.content})
    return out


def apply_extracted(p: Participant, raw: dict | str) -> None:
    prof = parse_profile(raw)
    p.summary = prof.summary
    p.values_scores = prof.values
    p.values_vec = values_vector(prof.values)
    p.lifestyle = prof.lifestyle
    p.looking_for = prof.looking_for
    p.red_flags = prof.red_flags
    p.profile_error = ""
    if not p.visual_taste_vec:
        p.visual_taste_vec = neutral_visual_vector()
        p.appearance_vec = neutral_visual_vector()


async def extract_and_store(db: Session, p: Participant, llm: LLMClient, attempts: int = 2) -> bool:
    """Run the structured extraction and persist it. Returns False (and records the error
    for the admin) if Claude fails twice; the participant can continue either way."""
    transcript = transcript_text(p.messages)
    err = ""
    for _ in range(attempts):
        try:
            raw = await llm.extract_profile(transcript, facts_text(p))
            apply_extracted(p, raw)
            db.commit()
            return True
        except (LLMError, ProfileParseError) as e:
            err = str(e)
            log.warning("profile extraction failed for participant %s: %s", p.id, e)
    p.profile_error = f"{utcnow():%Y-%m-%d %H:%M} {err}"
    db.commit()
    return False


def add_message(db: Session, p: Participant, role: str, content: str) -> InterviewMessage:
    m = InterviewMessage(participant_id=p.id, role=role, content=content)
    db.add(m)
    db.commit()
    db.refresh(p)
    return m
