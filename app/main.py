"""FastAPI app factory.   uvicorn app.main:app"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from .config import Settings, settings as default_settings
from .db import Database
from .llm import LLMClient, make_llm
from .storage import LocalStorage, Storage
from .web import HERE, render

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

MAX_BODY = 40 * 1024 * 1024   # whole request (4 photos x 8MB + form)


def create_app(settings: Settings | None = None, db: Database | None = None,
               llm: LLMClient | None = None, storage: Storage | None = None) -> FastAPI:
    settings = settings or default_settings
    app = FastAPI(title="שלוש – פיילוט", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.settings = settings
    app.state.db = db or Database(settings.database_url)
    app.state.db.create_all()
    app.state.llm = llm or make_llm(settings)
    app.state.storage = storage or LocalStorage(settings.upload_dir)

    from .routes_admin import router as admin_router
    from .routes_participant import router as participant_router
    app.include_router(admin_router)
    app.include_router(participant_router)
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

    @app.middleware("http")
    async def security_headers(request: Request, call_next):  # noqa: ANN001, ANN202
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > MAX_BODY:
            return JSONResponse({"detail": "request too large"}, status_code=413)
        resp = await call_next(request)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")   # secret links must never leak
        resp.headers.setdefault("Cache-Control", "no-store")
        return resp

    # Signed cookie session (itsdangerous). Added last so it wraps everything.
    app.add_middleware(SessionMiddleware, secret_key=settings.secret_key, session_cookie="shalosh_session",
                       max_age=60 * 60 * 24 * 60, same_site="lax", https_only=settings.cookie_secure)

    @app.exception_handler(HTTPException)
    async def friendly_errors(request: Request, exc: HTTPException):  # noqa: ANN202
        path = request.url.path
        wants_html = "text/html" in request.headers.get("accept", "")
        if path.startswith("/admin") or not wants_html:
            return await http_exception_handler(request, exc)
        if exc.status_code == 401:
            return render(request, "message.html", status_code=401, title="צריך להיכנס מהקישור האישי",
                          body="פתחו שוב את הקישור האישי ששלחנו לכם בוואטסאפ. אם הוא לא עובד, כתבו לנו ונשלח חדש.")
        if exc.status_code == 404:
            return render(request, "message.html", status_code=404, title="לא מצאנו את הדף",
                          body="ייתכן שהקישור לא מלא או שפג תוקפו. כתבו לצוות ונשלח קישור חדש.")
        return render(request, "message.html", status_code=exc.status_code, title="משהו השתבש",
                      body=str(exc.detail))

    @app.get("/")
    def index(request: Request):  # noqa: ANN202
        if request.session.get("pid"):
            return RedirectResponse("/p/", status_code=303)
        return render(request, "message.html", title="שלוש",
                      body="שלוש הצעות בשבוע. שיחה אחת. אדם אחד. הכניסה לפיילוט היא בקישור אישי בלבד.")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True, "llm": app.state.llm.name}

    return app


app = create_app()
