"""Shared web helpers: templates, DB dependency, formatting filters."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from fastapi import Request
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from .security import ensure_csrf
from .services import to_il

HERE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(HERE / "templates"))

HE_DAYS = ["שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת", "ראשון"]


def fmt_il(dt: datetime | None, with_day: bool = True) -> str:
    local = to_il(dt)
    if local is None:
        return ""
    s = local.strftime("%d.%m %H:%M")
    return f"{HE_DAYS[local.weekday()]} {s}" if with_day else s


def pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.0f}%"


templates.env.filters["il"] = fmt_il
templates.env.filters["pct"] = pct
templates.env.globals["iso_utc"] = lambda dt: dt.isoformat() + "Z" if dt else ""


def get_db(request: Request) -> Iterator[Session]:
    yield from request.app.state.db.session()


def render(request: Request, name: str, status_code: int = 200, **ctx) -> object:
    ctx.setdefault("csrf_token", ensure_csrf(request))
    ctx.setdefault("settings", request.app.state.settings)
    return templates.TemplateResponse(request, name, ctx, status_code=status_code)
