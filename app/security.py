"""Participant token links, signed-session auth, CSRF and admin basic auth."""

from __future__ import annotations

import base64
import binascii
import hmac
import secrets

from fastapi import HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Participant

TOKEN_BYTES = 32  # -> 43 url-safe chars, ~256 bits


def new_token() -> str:
    return secrets.token_urlsafe(TOKEN_BYTES)


# --------------------------------------------------------------------------- participant

def login_participant(request: Request, p: Participant) -> None:
    """Bind the signed session cookie to the participant (and to the current token,
    so regenerating a participant's link logs out old sessions)."""
    request.session.clear()
    request.session["pid"] = p.id
    request.session["tok"] = p.token[:12]
    ensure_csrf(request)


def find_by_token(db: Session, token: str) -> Participant | None:
    if not token or len(token) < 20 or len(token) > 64:
        return None
    p = db.scalar(select(Participant).where(Participant.token == token))
    if p is None or not hmac.compare_digest(p.token, token):
        return None
    return p


def current_participant(request: Request, db: Session) -> Participant:
    pid = request.session.get("pid")
    if not isinstance(pid, int):
        raise HTTPException(status_code=401, detail="not logged in")
    p = db.get(Participant, pid)
    if p is None or not hmac.compare_digest(p.token[:12], str(request.session.get("tok", ""))):
        request.session.clear()
        raise HTTPException(status_code=401, detail="session expired")
    return p


# --------------------------------------------------------------------------- CSRF

def ensure_csrf(request: Request) -> str:
    tok = request.session.get("csrf")
    if not tok:
        tok = secrets.token_urlsafe(24)
        request.session["csrf"] = tok
    return tok


async def verify_csrf(request: Request) -> None:
    """Synchronizer-token check for every state-changing request.

    Accepts the token from the `X-CSRF-Token` header (fetch) or the
    `csrf_token` form field (HTML forms)."""
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return
    expected = request.session.get("csrf")
    sent = request.headers.get("x-csrf-token")
    if not sent:
        ctype = request.headers.get("content-type", "")
        if "form" in ctype:
            form = await request.form()
            sent = form.get("csrf_token")  # type: ignore[assignment]
    if not expected or not sent or not hmac.compare_digest(str(expected), str(sent)):
        raise HTTPException(status_code=403, detail="CSRF token missing or invalid")


# --------------------------------------------------------------------------- admin

def check_admin(request: Request) -> str:
    settings = request.app.state.settings
    if not settings.admin_user or not settings.admin_password:
        raise HTTPException(status_code=503, detail="admin disabled: set ADMIN_USER and ADMIN_PASSWORD")
    auth = request.headers.get("authorization", "")
    user = pw = ""
    if auth.lower().startswith("basic "):
        try:
            user, _, pw = base64.b64decode(auth[6:]).decode("utf-8").partition(":")
        except (binascii.Error, UnicodeDecodeError):
            pass
    ok_user = hmac.compare_digest(user.encode(), settings.admin_user.encode())
    ok_pw = hmac.compare_digest(pw.encode(), settings.admin_password.encode())
    if not (ok_user and ok_pw):
        raise HTTPException(status_code=401, detail="admin login required",
                            headers={"WWW-Authenticate": 'Basic realm="shalosh-admin", charset="UTF-8"'})
    return user
