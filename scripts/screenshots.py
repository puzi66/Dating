"""End-to-end click-through of the main flows with Playwright, saving mobile screenshots.

    python scripts/screenshots.py                # -> docs/screenshots/*.png

Runs a fresh app (temp data dir, seeded, fake Claude) under uvicorn and drives it with
Chromium at 390x844. Uses the preinstalled browsers in PLAYWRIGHT_BROWSERS_PATH
(default /opt/pw-browsers); it never runs `playwright install`.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "screenshots"
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "/opt/pw-browsers")
MOBILE = {"viewport": {"width": 390, "height": 844}, "device_scale_factor": 2, "is_mobile": True,
          "has_touch": True, "locale": "he-IL"}
ADMIN = ("team", "pilot-pass")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def png(color: tuple[int, int, int]) -> bytes:
    sys.path.insert(0, str(ROOT))
    from app.seed import placeholder_png
    return placeholder_png(240, 300, color, (241, 210, 182), 1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = Path(tempfile.mkdtemp(prefix="shalosh-e2e-"))
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    env = os.environ | {"DATA_DIR": str(data), "FAKE_LLM": "1", "ANTHROPIC_API_KEY": "", "ADMIN_USER": ADMIN[0],
                        "ADMIN_PASSWORD": ADMIN[1], "SECRET_KEY": "e2e-secret", "BASE_URL": base}
    subprocess.run([sys.executable, "-m", "app.seed", "--reset"], cwd=ROOT, env=env, check=True,
                   stdout=subprocess.DEVNULL)
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port)],
                              cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(base + "/healthz", timeout=1)
                break
            except OSError:
                time.sleep(0.1)
        run(base, env, data)
    finally:
        server.terminate()
        server.wait(10)


def run(base: str, env: dict, data: Path) -> None:
    from playwright.sync_api import expect, sync_playwright

    os.environ.update({k: env[k] for k in ("DATA_DIR", "FAKE_LLM", "SECRET_KEY")})
    sys.path.insert(0, str(ROOT))
    from sqlalchemy import select

    from app.config import load_settings
    from app.db import Database
    from app.models import Participant, ProposalRow
    db = Database(load_settings().database_url)

    photo_a, photo_b = data / "a.png", data / "b.png"
    photo_a.write_bytes(png((63, 90, 91)))
    photo_b.write_bytes(png((91, 74, 99)))

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        admin = browser.new_context(**MOBILE, http_credentials={"username": ADMIN[0], "password": ADMIN[1]})
        ap = admin.new_page()

        # ---- admin creates a participant -> personal link
        ap.goto(base + "/admin/participants/new")
        ap.fill("input[name=first_name]", "רוני")
        ap.fill("input[name=last_name]", "אלמוג")
        ap.fill("input[name=phone]", "054-1234567")
        ap.click("button[type=submit]")
        link = ap.locator("[data-copy]").first.get_attribute("data-copy")
        assert link and "/p/" in link

        # ---- new participant: welcome -> facts -> interview -> visual -> waiting
        newbie = browser.new_context(**MOBILE)
        p = newbie.new_page()
        p.goto(link)
        expect(p.locator("h1")).to_contain_text("שלוש הצעות")
        p.wait_for_timeout(300)
        p.screenshot(path=str(OUT / "01-welcome.png"), full_page=True)
        p.check("input[name=consent]")
        p.click("form.form button[type=submit]")
        expect(p.locator("h2")).to_contain_text("עובדות")
        for name, value in [("gender", "female"), ("intent", "serious"), ("has_kids", "0"), ("wants_kids", "yes"),
                            ("accepts_partner_with_kids", "1"), ("religiosity", "0"), ("smoking", "0"),
                            ("max_partner_smoking", "1")]:
            p.check(f"input[name={name}][value='{value}']", force=True)
        p.check("input[name=seeking][value=male]", force=True)
        p.fill("input[name=age]", "29")
        p.fill("input[name=age_min]", "27")
        p.fill("input[name=age_max]", "36")
        p.select_option("select[name=city]", "Tel Aviv-Yafo")
        p.set_input_files("input[name=photos]", [str(photo_a), str(photo_b)])
        p.screenshot(path=str(OUT / "02-facts.png"), full_page=True)
        p.click("form.form button[type=submit]")

        expect(p.locator(".chat-head")).to_contain_text("עדי")
        expect(p.locator(".msg.assistant").first).to_contain_text("עדי", timeout=15000)
        answers = ["ים בבוקר, ואז ארוחת שישי אצל ההורים בחולון 🙂",
                   "אני מורה לביולוגיה, ומה שממלא אותי זה הרגע שתלמיד פתאום מבין משהו.",
                   "שאני לא מוותרת על אנשים.",
                   "חוסר כנות. גם בדברים קטנים.",
                   "היה לי קשר של שלוש שנים. למדתי שאני צריכה מישהו שמדבר על מה שהוא מרגיש.",
                   "מישהו יציב, מצחיק, שרוצה משפחה בעוד כמה שנים."]
        for i, a in enumerate(answers):
            p.fill("#msg-input", a)
            p.click("#send-btn")
            expect(p.locator(".msg.assistant")).to_have_count(i + 2, timeout=15000)
            expect(p.locator("#send-btn")).to_be_enabled(timeout=15000)
            if i == 2:
                p.screenshot(path=str(OUT / "03-interview.png"))
        expect(p.locator("#finish-box")).to_be_visible(timeout=15000)
        p.screenshot(path=str(OUT / "03b-interview-end.png"))
        p.click("#finish-box button")
        expect(p.locator("h2")).to_contain_text("משיכה", timeout=20000)
        for key in ("looks_matter", "clear_type", "grows_over_time"):
            p.check(f"input[name={key}][value='3']", force=True)
        p.click("form.form button[type=submit]")
        expect(p.locator(".countdown")).to_be_visible()
        p.screenshot(path=str(OUT / "04-waiting.png"), full_page=True)

        # ---- matchmaker: generate draft, why-texts, review, swap UI, publish
        ap.goto(base + "/admin/drops")
        ap.click("text=יצירת טיוטת Drop")
        expect(ap.locator(".pair").first).to_be_visible()
        ap.click("button:has-text('יצירת טקסטים')")
        expect(ap.locator(".ok")).to_contain_text("נוצרו")
        ap.goto(ap.url.split("?")[0])
        ap.screenshot(path=str(OUT / "05-admin-drop-review.png"), full_page=False)
        ap.locator(".pair").first.locator("a.sbtn").first.click()
        expect(ap.locator("#swap")).to_be_visible()
        ap.locator("#swap").screenshot(path=str(OUT / "05b-admin-swap.png"))
        ap.goto(ap.url.split("?")[0])
        ap.once("dialog", lambda d: d.accept())
        ap.click("button:has-text('פרסום')")
        expect(ap.locator("h2").first).to_contain_text("תשובות")

        # ---- two seeded participants who were paired: both say yes -> match
        with db.SessionLocal() as s:
            pr = s.scalars(select(ProposalRow).order_by(ProposalRow.id)).first()
            a = s.get(Participant, pr.user_low_id)
            b = s.get(Participant, pr.user_high_id)
            a_link, b_link, pr_id = f"{base}/p/{a.token}", f"{base}/p/{b.token}", pr.id

        ca = browser.new_context(**MOBILE)
        pa = ca.new_page()
        pa.goto(a_link)
        expect(pa.locator("text=הצעות מחכות לתשובה")).to_be_visible()
        pa.click("text=הצעות מחכות לתשובה")
        expect(pa.locator(".pcard").first).to_be_visible()
        pa.wait_for_timeout(300)
        pa.screenshot(path=str(OUT / "06-drop.png"), full_page=True)
        pa.locator(f"form:has(input[name=proposal_id][value='{pr_id}']) button.btn").click()
        expect(pa.locator(".answered").first).to_be_visible()

        cb = browser.new_context(**MOBILE)
        pb = cb.new_page()
        pb.goto(b_link)
        pb.goto(base + "/p/drop")
        pb.locator(f"form:has(input[name=proposal_id][value='{pr_id}']) button.btn").click()
        expect(pb.locator("h2")).to_contain_text("זה הדדי")
        pb.wait_for_timeout(300)
        pb.screenshot(path=str(OUT / "07-match.png"), full_page=True)

        # ---- team: responses dashboard, open group, date happened; debrief via link
        ap.goto(base + "/admin/responses")
        expect(ap.locator(".pill:has-text('התאמה הדדית')").first).to_be_visible()
        ap.click("button:has-text('פתחנו קבוצה')")
        ap.click("button:has-text('הדייט התקיים')")
        ap.goto(base + "/admin/responses")
        ap.screenshot(path=str(OUT / "08-admin-responses.png"), full_page=True)
        debrief_link = re.search(r"(http://\S+/debrief/\d+)", ap.locator("details").first.text_content()).group(1)
        pd = browser.new_context(**MOBILE).new_page()
        pd.goto(debrief_link)
        expect(pd.locator("h2")).to_contain_text("איך היה")
        pd.check("input[name=rating][value='4']", force=True)
        pd.check("input[name=meet_again][value='yes']", force=True)
        pd.fill("textarea[name=surprise_text]", "הרבה יותר מצחיק ממה שציפיתי")
        pd.check("input[name=looked_like_photos][value='yes']", force=True)
        pd.screenshot(path=str(OUT / "09-debrief.png"), full_page=True)
        pd.click("form.form button[type=submit]")
        expect(pd.locator("h2")).to_contain_text("תודה")

        ap.goto(base + "/admin/metrics")
        ap.screenshot(path=str(OUT / "10-admin-metrics.png"), full_page=True)
        ap.goto(base + "/admin")
        ap.screenshot(path=str(OUT / "11-admin-participants.png"), full_page=False)
        browser.close()
    print("screenshots in", OUT)


if __name__ == "__main__":
    main()
