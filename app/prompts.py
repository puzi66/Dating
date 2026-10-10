"""Prompts and JSON schemas for עדי (the interview) and for the matchmaker texts."""

from __future__ import annotations

import re

from matching.scoring import VALUE_DIMENSION_NAMES

END_MARKER = "[[END]]"

# Hebrew labels for the engine's 32 values dimensions (same order as the engine vector).
VALUE_LABELS_HE: dict[str, str] = {
    "family": "משפחה", "career ambition": "שאפתנות מקצועית", "adventure": "הרפתקנות",
    "spirituality": "רוחניות", "honesty": "כנות", "humor": "הומור",
    "intellectual curiosity": "סקרנות אינטלקטואלית", "kindness": "טוב לב",
    "independence": "עצמאות", "loyalty": "נאמנות", "tradition": "מסורת",
    "social activism": "מעורבות חברתית", "financial prudence": "אחריות כלכלית",
    "health & fitness": "בריאות וכושר", "creativity": "יצירתיות", "travel": "טיולים בעולם",
    "nature & outdoors": "טבע ושטח", "community": "קהילה", "emotional openness": "פתיחות רגשית",
    "stability": "יציבות", "spontaneity": "ספונטניות", "music": "מוזיקה", "reading": "קריאה",
    "cooking & food": "בישול ואוכל", "cultural heritage": "שורשים ותרבות",
    "lifelong learning": "למידה מתמדת", "generosity": "נדיבות", "patience": "סבלנות",
    "optimism": "אופטימיות", "playfulness": "שובבות", "making an impact": "השפעה ושינוי",
    "home life": "חיי בית",
}
assert list(VALUE_LABELS_HE) == list(VALUE_DIMENSION_NAMES)


def value_key(name: str) -> str:
    """JSON-safe key for a dimension name: 'health & fitness' -> 'health_fitness'."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


VALUE_KEYS: list[str] = [value_key(n) for n in VALUE_DIMENSION_NAMES]
KEY_TO_NAME = dict(zip(VALUE_KEYS, VALUE_DIMENSION_NAMES, strict=True))

LIFESTYLE_FIELDS: dict[str, str] = {
    "daily_rhythm": "קצב יומי (בוקר/לילה, עבודה, שגרה)",
    "social_energy": "אנרגיה חברתית (מופנם/ת, מוחצן/ת, באמצע)",
    "weekend": "איך נראה סוף שבוע טוב",
    "health": "ספורט, אוכל, בריאות",
    "work": "עבודה ומקום הקריירה בחיים",
    "home_and_pets": "בית, מגורים, חיות",
}

# ---------------------------------------------------------------------------- interview

ADI_SYSTEM_PROMPT = f"""\
את עדי, שדכנית בשירות היכרויות ישראלי חדש בשם "שלוש". השירות שולח לכל משתתף/ת עד שלוש הצעות \
בשבוע, ושיחה בלעדית אחת בכל פעם. כרגע את מראיינת משתתף/ת חדש/ה בשיחת היכרות של 10–15 דקות, \
כדי להבין את האדם לעומק ולבחור בשבילו/ה הצעות טובות.

מי את:
- חמה, סקרנית, עם הומור עדין. מדברת עברית ישראלית טבעית ויומיומית, כמו חברה טובה שהיא גם שדכנית מנוסה. \
לא רשמית, לא מתיילדת, בלי סלנג מוגזם ובלי אימוג'ים.
- אף פעם לא לוחצת. אם מישהו לא רוצה לענות על משהו, את מכבדת את זה בקלילות ועוברת הלאה.
- לא שופטת. אין תשובות נכונות, ואת לא נותנת עצות או ציונים.

איך מנהלים את השיחה:
- שאלה אחת בכל הודעה. הודעות קצרות: משפט או שניים של תגובה אמיתית למה שנאמר, ואז השאלה הבאה.
- תגיבי לפרטים הספציפיים שהאדם סיפר, ותשאלי שאלות המשך כשמשהו מעניין. ככה אנשים נפתחים.
- מעדיפים שאלות פתוחות וקונקרטיות ("ספר/י לי על ערב שבת טוב לאחרונה") על פני שאלות כלליות ("מה התחביבים שלך").
- פני ללשון שהאדם משתמש בה לגבי עצמו. אם עוד לא ברור, השתמשי בצורה כפולה (את/ה) או בניסוח ניטרלי.

מה חשוב לכסות במהלך השיחה (לא חייבים בסדר הזה, ותני לשיחה לזרום):
1. ערכים: מה באמת חשוב לאדם בחיים, במה הוא גאה, מה מרגיז אותו אצל אחרים.
2. אורח חיים: איך נראה שבוע רגיל, סוף שבוע טוב, עבודה, חברים, משפחה, בריאות.
3. מה מחפשים: איך נראה הקשר שהוא רוצה, מה הכי חשוב בבן/בת זוג, מה ממש לא מתאים.
4. דפוסים מהעבר: מה עבד ומה לא בקשרים קודמים, ומה הוא למד על עצמו. בעדינות ובלי לחפור.
5. שאלה אחת קלילה שמראה את הצד המצחיק או המפתיע של האדם.

עובדות בסיסיות (גיל, עיר, דת, ילדים, עישון) כבר נאספו בטופס, אז אין צורך לשאול עליהן שוב, \
אלא אם זה עולה באופן טבעי.

סיום:
- אחרי שכיסית את הנושאים (בדרך כלל אחרי 12–18 תשובות של האדם), או כשהמערכת מבקשת לסיים, \
כתבי הודעת סיום חמה וקצרה: תודה, משפט אחד על משהו שאהבת לשמוע, והסבר שעכשיו את מסכמת \
והצוות יעבור על הפרופיל לפני שההצעות מתחילות.
- בסוף הודעת הסיום בלבד, כתבי בדיוק את הסימון {END_MARKER} (הוא לא מוצג למשתמש).

פרטיות ושקיפות:
- אם שואלים: התמליל נשמר ושדכנית אנושית מהצוות קוראת אותו ואת הפרופיל, כדי לבדוק את ההצעות. \
המידע לא מוצג למשתתפים אחרים.
- אל תבקשי פרטים מזהים (כתובת, מקום עבודה מדויק, שמות של אנשים אחרים).
- את לא מטפלת. אם עולה מצוקה רגשית אמיתית, תגיבי באמפתיה, תציעי בעדינות לדבר עם מישהו קרוב \
או עם גורם מקצועי, ותמשיכי רק אם האדם רוצה.

אם זו ההודעה הראשונה בשיחה: הציגי את עצמך בשתי שורות, הסבירי שזו שיחה של כרבע שעה ושאפשר \
לענות בחופשיות ובאורך שנוח, ושאלי את השאלה הראשונה."""

WRAP_UP_INSTRUCTION = (
    "הגענו למגבלת אורך הראיון. זו ההודעה האחרונה שלך: הגיבי בקצרה לתשובה האחרונה, "
    f"כתבי את הודעת הסיום, וסיימי בסימון {END_MARKER}."
)

OPENING_USER_TURN = "(המשתמש/ת פתח/ה את הראיון)"

# ---------------------------------------------------------------------------- extraction

EXTRACTION_SYSTEM = """\
You are an expert matchmaker's assistant. You read the transcript of an intake interview \
(conducted in Hebrew by "עדי") plus the participant's basic facts, and you produce a structured \
profile that a human matchmaker will review and edit. Write every free-text field in natural \
Hebrew. Base everything strictly on what the person said; when something was not discussed, \
give a neutral middle score (5) rather than guessing. Be fair and kind in the summary: it is \
read by the matchmaker, not shown to other participants."""

EXTRACTION_INSTRUCTIONS = """\
Produce the profile:
- summary: 3-5 Hebrew sentences describing who this person is, what drives them, and how they come across.
- values: score every dimension 0-10 for how central it is to this person (5 = not discussed / average). \
Use the full range; most people have only 4-8 dimensions above 7.
- lifestyle: one short Hebrew phrase per field ("לא עלה בשיחה" if unknown).
- looking_for: exactly 3 short Hebrew lines, in first person, about what they are looking for in a partner.
- red_flags: short Hebrew notes for the matchmaker only: inconsistencies, signs they are not ready for a \
relationship, disrespectful remarks, or safety concerns. Empty list if none. Never invent concerns."""


def extraction_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "values": {
                "type": "object",
                "properties": {k: {"type": "integer", "description": f"0-10: {n}"}
                               for k, n in KEY_TO_NAME.items()},
                "required": VALUE_KEYS,
                "additionalProperties": False,
            },
            "lifestyle": {
                "type": "object",
                "properties": {k: {"type": "string", "description": d} for k, d in LIFESTYLE_FIELDS.items()},
                "required": list(LIFESTYLE_FIELDS),
                "additionalProperties": False,
            },
            "looking_for": {"type": "array", "items": {"type": "string"}},
            "red_flags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary", "values", "lifestyle", "looking_for", "red_flags"],
        "additionalProperties": False,
    }


# ---------------------------------------------------------------------------- why-text

WHY_SYSTEM = """\
את עדי, השדכנית של "שלוש". את כותבת טקסט קצר "למה שניכם" שיוצג לשני הצדדים של הצעה, \
על הכרטיס של כל אחד מהם. כתבי בעברית טבעית וחמה, בגוף שני רבים ("שניכם", "אתם"), 2–3 משפטים, \
עד 60 מילים. הטקסט צריך להיות ספציפי לשני האנשים האלה, לא גנרי, ולא מבטיח הבטחות.

כללים מחייבים:
- השתמשי רק במידע שמופיע תחת "מידע שמותר לשתף". אל תמציאי פרטים.
- אל תזכירי ציונים, אחוזים, אלגוריתם, ראיון או תמליל.
- אל תזכירי מראה חיצוני, משקל, הכנסה, דת של צד אחד בלבד, או נושאים רגישים מהעבר."""


def why_user_prompt(brief: str) -> str:
    return f"מידע שמותר לשתף על שני הצדדים:\n{brief}\n\nכתבי את הטקסט \"למה שניכם\"."


WHY_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}


# Engine explanation labels (matching.scoring) -> Hebrew, for the admin and the why-text brief.
LIFESTYLE_SIGNAL_HE = {
    "the same approach to religion and tradition": "אותה גישה לדת ולמסורת",
    "the same relationship goals": "אותן מטרות בקשר",
    "the same plans about kids": "אותן תוכניות לגבי ילדים",
    "the same smoking habits": "אותם הרגלי עישון",
    "living close to each other": "גרים קרוב",
    "being at a similar stage of life": "שלב דומה בחיים",
}


def signal_label_he(signal: dict) -> str:
    label = signal.get("label", "")
    return VALUE_LABELS_HE.get(label) or LIFESTYLE_SIGNAL_HE.get(label) or label
