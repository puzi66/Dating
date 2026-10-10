"""Turning interview output into engine inputs.

* parse_profile: validates Claude's structured output (defensively: the schema is enforced
  by the API, but the same parser also handles matchmaker edits and older/fake outputs).
* values_vector: maps 0-10 scores onto the engine's 32-dim values vector.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

from matching.models import VALUES_DIM, VISUAL_DIM
from matching.scoring import VALUE_DIMENSION_NAMES

from .choices import LABELS
from .prompts import KEY_TO_NAME, LIFESTYLE_FIELDS, VALUE_KEYS, VALUE_LABELS_HE, value_key


class ProfileParseError(ValueError):
    pass


@dataclass
class ExtractedProfile:
    summary: str
    values: dict[str, int]                 # engine dimension NAME -> 0..10
    lifestyle: dict[str, str]
    looking_for: list[str]
    red_flags: list[str] = field(default_factory=list)


def _clean_str(x: object, limit: int = 2000) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()[:limit]


def parse_profile(raw: str | dict) -> ExtractedProfile:
    if isinstance(raw, str):
        text = raw.strip()
        fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
        if fence:
            text = fence.group(1)
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as e:
            raise ProfileParseError(f"not JSON: {e}") from e
    if not isinstance(raw, dict):
        raise ProfileParseError("profile must be a JSON object")

    summary = _clean_str(raw.get("summary"))
    if not summary:
        raise ProfileParseError("missing summary")

    vals_in = raw.get("values")
    if not isinstance(vals_in, dict):
        raise ProfileParseError("missing values object")
    values: dict[str, int] = {}
    for name in VALUE_DIMENSION_NAMES:
        # accept snake keys ("health_fitness") or the raw dimension name
        v = vals_in.get(value_key(name), vals_in.get(name))
        if v is None:
            v = 5
        try:
            v = float(v)
        except (TypeError, ValueError) as e:
            raise ProfileParseError(f"value for {name!r} is not a number") from e
        if math.isnan(v):
            v = 5
        values[name] = int(round(min(10.0, max(0.0, v))))
    if not any(k in vals_in for k in (*VALUE_KEYS, *VALUE_DIMENSION_NAMES)):
        raise ProfileParseError("values object has none of the expected dimensions")

    life_in = raw.get("lifestyle") if isinstance(raw.get("lifestyle"), dict) else {}
    lifestyle = {k: _clean_str(life_in.get(k), 300) for k in LIFESTYLE_FIELDS}

    lf = raw.get("looking_for")
    if isinstance(lf, str):
        lf = [s for s in re.split(r"[\n•]+", lf) if s.strip()]
    looking = [_clean_str(s, 300) for s in (lf or []) if _clean_str(s)]
    if not looking:
        raise ProfileParseError("missing looking_for")
    looking = looking[:3]

    rf = raw.get("red_flags") or []
    if isinstance(rf, str):
        rf = [rf]
    red_flags = [_clean_str(s, 500) for s in rf if _clean_str(s)]
    return ExtractedProfile(summary, values, lifestyle, looking, red_flags)


def values_vector(scores: dict[str, int | float]) -> list[float]:
    """0-10 scores -> L2-normalised 32-vector in engine order.

    Scores are centred on the person's own mean first: people tend to rate everything
    as important, and what distinguishes them is which values stand out *relative to
    their own baseline*. Cosine similarity of the centred vectors then compares
    priorities. All-equal scores give the zero vector (engine treats cosine as 0 -> 0.5).
    """
    raw = [float(scores.get(n, 5)) for n in VALUE_DIMENSION_NAMES]
    mean = sum(raw) / len(raw)
    centred = [x - mean for x in raw]
    n = math.sqrt(sum(x * x for x in centred))
    if n == 0:
        return [0.0] * VALUES_DIM
    return [x / n for x in centred]


def neutral_visual_vector() -> list[float]:
    """TODO(avatar-calibration): replace with the real 6-round avatar calibration.
    For the pilot every participant gets the same unit vector for both taste and
    appearance, so visual similarity is identical (=1.0) for every pair and the
    visual component does not change the ranking."""
    return [1 / math.sqrt(VISUAL_DIM)] * VISUAL_DIM


def values_labels(scores: dict) -> list[tuple[str, str, int]]:
    """(engine name, Hebrew label, score) sorted by score desc."""
    rows = [(n, VALUE_LABELS_HE[n], int(scores.get(n, 5))) for n in VALUE_DIMENSION_NAMES]
    return sorted(rows, key=lambda r: -r[2])


def facts_text(p) -> str:  # noqa: ANN001
    """Basic facts as plain Hebrew lines, for the extraction prompt."""
    parts = [
        f"שם פרטי: {p.first_name}",
        f"גיל: {p.age}",
        f"עיר: {p.city}",
        f"מגדר: {LABELS['gender'].get(p.gender, p.gender)}",
        f"כוונה: {LABELS['intent'].get(p.intent, p.intent)}",
        f"דתיות: {LABELS['religiosity'].get(p.religiosity, p.religiosity)}",
        f"ילדים: {'יש' if p.has_kids else 'אין'}; רוצה ילדים: {LABELS['wants_kids'].get(p.wants_kids, p.wants_kids)}",
        f"עישון: {LABELS['smoking'].get(p.smoking, p.smoking)}",
    ]
    return "\n".join(parts)


def transcript_text(messages) -> str:  # noqa: ANN001
    who = {"user": "משתתף/ת", "assistant": "עדי"}
    return "\n\n".join(f"{who.get(m.role, m.role)}: {m.content}" for m in messages)


__all__ = ["ExtractedProfile", "ProfileParseError", "parse_profile", "values_vector",
           "neutral_visual_vector", "values_labels", "facts_text", "transcript_text", "KEY_TO_NAME"]
