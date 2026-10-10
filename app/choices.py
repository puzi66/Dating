"""Form choices with Hebrew labels, mapped onto the engine's enums.

The city list (names + lat/lon) is reused from `matching.simulate.CITIES`.
"""

from __future__ import annotations

from matching.models import Intent
from matching.simulate import CITIES

CITY_HE = {
    "Tel Aviv-Yafo": "תל אביב-יפו", "Jerusalem": "ירושלים", "Haifa": "חיפה",
    "Rishon LeZion": "ראשון לציון", "Petah Tikva": "פתח תקווה", "Ashdod": "אשדוד",
    "Netanya": "נתניה", "Beer Sheva": "באר שבע", "Holon": "חולון", "Ramat Gan": "רמת גן",
    "Rehovot": "רחובות", "Herzliya": "הרצליה", "Kfar Saba": "כפר סבא", "Ra'anana": "רעננה",
    "Modi'in": "מודיעין", "Tiberias": "טבריה", "Eilat": "אילת",
}
# key -> (Hebrew label, lat, lon)
CITY_COORDS: dict[str, tuple[str, float, float]] = {
    name: (CITY_HE.get(name, name), lat, lon) for name, lat, lon, _w, _mix in CITIES
}

GENDERS = [("female", "אישה"), ("male", "גבר"), ("nonbinary", "א-בינארי/ת")]
SEEKING = [("female", "נשים"), ("male", "גברים"), ("nonbinary", "א-בינאריים/ות")]
INTENTS = [("serious", "קשר רציני"), ("marriage", "קשר לחתונה ומשפחה"), ("casual", "משהו קליל, נראה לאן זה הולך")]
WANTS_KIDS = [("yes", "כן, רוצה ילדים"), ("open", "פתוח/ה לזה"), ("no", "לא רוצה ילדים")]
RELIGIOSITY = [(0, "חילוני/ת"), (1, "מסורתי/ת"), (2, "דתי/ה"), (3, "חרדי/ת")]
SMOKING = [(0, "לא מעשן/ת"), (1, "לפעמים"), (2, "מעשן/ת")]
PARTNER_SMOKING = [(0, "רק לא מעשנים"), (1, "לפעמים זה בסדר"), (2, "לא משנה לי")]
DISTANCES = [(10, "עד 10 ק\"מ"), (25, "עד 25 ק\"מ"), (50, "עד 50 ק\"מ"), (100, "עד 100 ק\"מ"), (500, "כל הארץ")]

LABELS = {
    "gender": dict(GENDERS), "intent": dict(INTENTS), "wants_kids": dict(WANTS_KIDS),
    "religiosity": dict(RELIGIOSITY), "smoking": dict(SMOKING),
}


def accepted_intents_for(intent: str) -> list[str]:
    """The form asks only for one's own intent; the accepted set is derived from it
    (the same rule the simulation uses): casual daters accept casual/serious,
    everyone else accepts serious/marriage."""
    if intent == Intent.CASUAL.value:
        return [Intent.CASUAL.value, Intent.SERIOUS.value]
    return [Intent.SERIOUS.value, Intent.MARRIAGE.value]


# Visual-taste step (simplified for the pilot): 3 statements rated 1-5.
VISUAL_STATEMENTS = [
    ("looks_matter", "המראה החיצוני משחק תפקיד גדול בהחלטה שלי אם לצאת לדייט"),
    ("clear_type", "יש לי \"טייפ\" ברור של מי שמושך אותי"),
    ("grows_over_time", "משיכה אצלי נבנית עם הזמן, לא תמיד ברגע הראשון"),
]

MEET_AGAIN = [("yes", "כן"), ("unsure", "לא בטוח/ה"), ("no", "לא")]
LOOKED_LIKE = [("yes", "כן"), ("partly", "חלקית"), ("no", "לא")]
