"""Seed the local DB with fake participants that have complete profiles.

    python -m app.seed            # adds 10 complete + 2 in-progress participants
    python -m app.seed --reset    # wipes the DB (and local uploads) first

All names, phones and photos are fake (photos are generated placeholder PNGs).
"""

from __future__ import annotations

import argparse
import math
import random
import struct
import zlib

from sqlalchemy import select

from .choices import CITY_COORDS
from .config import settings
from .db import Base, Database
from .models import InterviewMessage, Participant, Photo, utcnow
from .profile import neutral_visual_vector, values_vector
from .prompts import LIFESTYLE_FIELDS, VALUE_LABELS_HE
from .security import new_token
from .storage import Storage, make_storage

PEOPLE = [
    # first, last, gender, age, city, religiosity, intent, values emphasis
    ("נועה", "לוי", "female", 29, "Tel Aviv-Yafo", 0, "serious", ["humor", "travel", "creativity", "emotional openness"]),
    ("מאיה", "כהן", "female", 31, "Ramat Gan", 0, "serious", ["family", "kindness", "stability", "cooking & food"]),
    ("שירה", "מזרחי", "female", 27, "Tel Aviv-Yafo", 1, "marriage", ["family", "tradition", "loyalty", "community"]),
    ("יעל", "פרידמן", "female", 34, "Herzliya", 0, "serious", ["career ambition", "health & fitness", "independence", "travel"]),
    ("תמר", "אברהם", "female", 30, "Holon", 1, "serious", ["music", "humor", "playfulness", "spontaneity"]),
    ("איתי", "רוזן", "male", 32, "Tel Aviv-Yafo", 0, "serious", ["humor", "travel", "intellectual curiosity", "creativity"]),
    ("עומר", "שפירא", "male", 30, "Petah Tikva", 1, "marriage", ["family", "tradition", "stability", "generosity"]),
    ("דניאל", "ביטון", "male", 33, "Ramat Gan", 0, "serious", ["kindness", "family", "cooking & food", "emotional openness"]),
    ("יונתן", "גולן", "male", 29, "Rishon LeZion", 0, "serious", ["health & fitness", "nature & outdoors", "adventure", "optimism"]),
    ("אלון", "דהן", "male", 35, "Kfar Saba", 0, "serious", ["career ambition", "making an impact", "reading", "independence"]),
]
IN_PROGRESS = [("רוני", "ששון"), ("גל", "אוחיון")]

PALETTES = [((63, 90, 91), (232, 149, 111)), ((47, 72, 88), (201, 162, 107)), ((91, 74, 99), (233, 195, 160)),
            ((90, 58, 34), (241, 210, 182)), ((74, 85, 104), (217, 169, 130)), ((94, 90, 58), (227, 181, 143))]


def placeholder_png(w: int, h: int, bg: tuple, fg: tuple, seed: int) -> bytes:
    """A simple portrait-ish placeholder (gradient + head/shoulders silhouette), pure stdlib."""
    rows = []
    cx, cy, r = w / 2, h * 0.42, w * 0.19
    for y in range(h):
        row = bytearray([0])
        t = y / h
        for x in range(w):
            col = [int(bg[i] * (1 - 0.35 * t)) for i in range(3)]
            in_head = (x - cx) ** 2 + ((y - cy) * 0.85) ** 2 < r * r
            in_body = y > h * 0.66 and abs(x - cx) < w * 0.36 * math.sqrt(max(0.0, (y - h * 0.62) / (h * 0.38)))
            if in_head or in_body:
                col = list(fg) if in_head else [int(c * 0.8) for c in fg]
            row.extend(col)
        rows.append(bytes(row))
    raw = b"".join(rows)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def seed(db: Database, storage: Storage, rng: random.Random) -> list[Participant]:
    created = []
    now = utcnow()
    with db.SessionLocal() as s:
        for i, (first, last, gender, age, city, rel, intent, emph) in enumerate(PEOPLE):
            scores = {name: rng.randint(3, 6) for name in VALUE_LABELS_HE}
            for name in emph:
                scores[name] = rng.randint(8, 10)
            label, lat, lon = CITY_COORDS[city]
            p = Participant(
                token=new_token(), first_name=first, last_name=last, phone=f"050-55{i:02d}{rng.randint(100, 999)}",
                consented_at=now, paid_at=now, facts_done_at=now, interview_done_at=now, visual_done_at=now,
                interview_ended=True,
                gender=gender, seeking=["male"] if gender == "female" else ["female"], age=age,
                age_min=max(18, age - 5), age_max=age + 6, city=city, lat=lat, lon=lon, max_distance_km=50,
                intent=intent, has_kids=False, wants_kids="yes" if intent == "marriage" else rng.choice(["yes", "open"]),
                accepts_partner_with_kids=True, religiosity=rel, religiosity_tolerance=1, smoking=0,
                max_partner_smoking=1,
                summary=f"{first} ({age}, {label}) מחפש/ת קשר רציני. בולטים אצל{'ה' if gender == 'female' else 'ו'}: "
                        + ", ".join(VALUE_LABELS_HE[e] for e in emph) + ". (פרופיל לדוגמה)",
                values_scores=scores, values_vec=values_vector(scores),
                lifestyle={k: "לדוגמה" for k in LIFESTYLE_FIELDS} | {"weekend": rng.choice(
                    ["טיול בצפון וקפה טוב", "ים בבוקר וחברים בערב", "ארוחת שישי משפחתית", "הופעה וחומוס"])},
                looking_for=[f"מישהו/י עם {VALUE_LABELS_HE[emph[0]]}", "שיחות שלא נגמרות", "שותפות אמיתית ביומיום"],
                red_flags=["לדוגמה: הזכיר/ה שעדיין בקשר עם האקס"] if i == 3 else [],
                visual_answers={"looks_matter": 3, "clear_type": 2, "grows_over_time": 4},
                visual_taste_vec=neutral_visual_vector(), appearance_vec=neutral_visual_vector(),
            )
            s.add(p)
            s.flush()
            s.add_all([
                InterviewMessage(participant_id=p.id, role="assistant",
                                 content="היי, אני עדי. איך נראה סוף שבוע טוב בשבילך?"),
                InterviewMessage(participant_id=p.id, role="user", content=p.lifestyle["weekend"]),
                InterviewMessage(participant_id=p.id, role="assistant", content="נשמע מעולה. ומה הכי חשוב לך בקשר?"),
                InterviewMessage(participant_id=p.id, role="user",
                                 content=" ו".join(VALUE_LABELS_HE[e] for e in emph[:2])),
            ])
            bg, fg = PALETTES[i % len(PALETTES)]
            for k in range(2):
                data = placeholder_png(240, 300, bg if k == 0 else fg, fg if k == 0 else bg, i * 10 + k)
                key = storage.save(data, "image/png", prefix=f"p{p.id}", session=s)
                s.add(Photo(participant_id=p.id, storage_key=key, content_type="image/png", ordinal=k))
            created.append(p)
        for first, last in IN_PROGRESS:
            p = Participant(token=new_token(), first_name=first, last_name=last,
                            phone=f"052-{rng.randint(1000000, 9999999)}")
            if first == "רוני":
                p.consented_at = now
            s.add(p)
            created.append(p)
        s.commit()
        return list(s.scalars(select(Participant).order_by(Participant.id)))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reset", action="store_true", help="drop all tables and local uploads first")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    db = Database(settings.database_url)
    if args.reset:
        Base.metadata.drop_all(db.engine)
        import shutil
        shutil.rmtree(settings.upload_dir, ignore_errors=True)
    db.create_all()
    parts = seed(db, make_storage(settings, db), random.Random(args.seed))
    print(f"Seeded into {settings.database_url}")
    for p in parts:
        print(f"  #{p.id:<3} {p.display_name:<16} {settings.base_url}/p/{p.token}")


if __name__ == "__main__":
    main()
