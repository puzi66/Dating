"""Directional and mutual compatibility scores, plus "why the two of you".

Features for the directed pair a -> b (all in [0, 1]):

* values    -- cosine(a.values_vec, b.values_vec), mapped from [-1, 1] to [0, 1]
* visual    -- cosine(a.visual_taste_vec, b.appearance_vec), same mapping
               (directional: how much b's look matches what a is drawn to)
* lifestyle -- soft compatibility of the hard facts that survived filtering
               (religiosity gap, intent, kids, smoking, distance, age gap)

S(a -> b) = w_a . features(a -> b), with a's learned weights (sum to 1).
Mutual   = sqrt(S(a -> b) * S(b -> a))   (geometric mean)

The geometric mean is what punishes one-sided attraction: (0.9, 0.1) gives
0.30 while the arithmetic mean would give 0.50, the same as a balanced
(0.5, 0.5) pair.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from .filters import distance_km
from .models import KidsWish, PreferenceWeights, Profile, VALUES_DIM

FEATURE_NAMES = ("values", "visual", "lifestyle")

# Human-readable names of the 32 values dimensions produced by the AI
# interview. Placeholder taxonomy until the real interview model exists.
VALUE_DIMENSION_NAMES: tuple[str, ...] = (
    "family", "career ambition", "adventure", "spirituality",
    "honesty", "humor", "intellectual curiosity", "kindness",
    "independence", "loyalty", "tradition", "social activism",
    "financial prudence", "health & fitness", "creativity", "travel",
    "nature & outdoors", "community", "emotional openness", "stability",
    "spontaneity", "music", "reading", "cooking & food",
    "cultural heritage", "lifelong learning", "generosity", "patience",
    "optimism", "playfulness", "making an impact", "home life",
)
assert len(VALUE_DIMENSION_NAMES) == VALUES_DIM


def dot(u: Sequence[float], v: Sequence[float]) -> float:
    return math.fsum(x * y for x, y in zip(u, v, strict=True))


def norm(u: Sequence[float]) -> float:
    return math.sqrt(dot(u, u))


def cosine(u: Sequence[float], v: Sequence[float]) -> float:
    nu, nv = norm(u), norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return max(-1.0, min(1.0, dot(u, v) / (nu * nv)))


def _unit(c: float) -> float:
    return (c + 1.0) / 2.0


@dataclass(frozen=True, slots=True)
class Features:
    values: float
    visual: float
    lifestyle: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.values, self.visual, self.lifestyle)


def lifestyle_components(a: Profile, b: Profile) -> dict[str, float]:
    """Soft lifestyle similarity per facet, each in [0, 1]. Symmetric."""
    if a.wants_kids == b.wants_kids:
        kids = 1.0
    elif KidsWish.OPEN in (a.wants_kids, b.wants_kids):
        kids = 0.6
    else:
        kids = 0.0
    return {
        "religiosity": 1.0 - abs(int(a.religiosity) - int(b.religiosity)) / 3.0,
        "intent": 1.0 if a.intent == b.intent else 0.5,
        "kids": kids,
        "smoking": 1.0 - abs(int(a.smoking) - int(b.smoking)) / 2.0,
        "distance": max(0.0, 1.0 - distance_km(a, b) / 100.0),
        "age": max(0.0, 1.0 - abs(a.age - b.age) / 15.0),
    }


def lifestyle_compat(a: Profile, b: Profile) -> float:
    comps = lifestyle_components(a, b)
    return sum(comps.values()) / len(comps)


def pair_features(a: Profile, b: Profile) -> Features:
    """Directed features a -> b."""
    return Features(
        values=_unit(cosine(a.values_vec, b.values_vec)),
        visual=_unit(cosine(a.visual_taste_vec, b.appearance_vec)),
        lifestyle=lifestyle_compat(a, b),
    )


def normalized(w: Sequence[float]) -> tuple[float, ...]:
    s = sum(max(0.0, x) for x in w)
    if s <= 0:
        return tuple(1.0 / len(w) for _ in w)
    return tuple(max(0.0, x) / s for x in w)


def score_from_features(f: Features, weights: Sequence[float]) -> float:
    w = normalized(weights)
    return dot(w, f.as_tuple())


def directional_score(a: Profile, b: Profile, weights: PreferenceWeights | Sequence[float]) -> float:
    """S(a -> b) using a's learned weights."""
    w = weights.as_tuple() if isinstance(weights, PreferenceWeights) else weights
    return score_from_features(pair_features(a, b), w)


def mutual_score(s_ab: float, s_ba: float) -> float:
    """Geometric mean: penalises asymmetric (one-sided) attraction."""
    return math.sqrt(max(0.0, s_ab) * max(0.0, s_ba))


# ---------------------------------------------------------------------------
# Explanation ("why the two of you")
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Signal:
    kind: str        # "value" | "lifestyle" | "visual"
    label: str
    strength: float  # in [0, 1], used for ranking


@dataclass(slots=True)
class Explanation:
    mutual_score: float
    score_a_to_b: float
    score_b_to_a: float
    top_signals: list[Signal] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mutual_score": round(self.mutual_score, 4),
            "score_a_to_b": round(self.score_a_to_b, 4),
            "score_b_to_a": round(self.score_b_to_a, 4),
            "top_signals": [
                {"kind": s.kind, "label": s.label, "strength": round(s.strength, 3)}
                for s in self.top_signals
            ],
        }

    def summary(self) -> str:
        """English fallback; the UI renders its own localized copy from signals."""
        labels = [s.label for s in self.top_signals]
        return "You both share: " + ", ".join(labels) if labels else ""


_LIFESTYLE_LABELS = {
    "religiosity": "the same approach to religion and tradition",
    "intent": "the same relationship goals",
    "kids": "the same plans about kids",
    "smoking": "the same smoking habits",
    "distance": "living close to each other",
    "age": "being at a similar stage of life",
}


def explain(a: Profile, b: Profile, s_ab: float, s_ba: float,
            f_ab: Features, f_ba: Features, top_k: int = 4) -> Explanation:
    signals: list[Signal] = []

    # Shared values: dimensions where both vectors are strongly positive.
    na, nb = norm(a.values_vec) or 1.0, norm(b.values_vec) or 1.0
    for i, name in enumerate(VALUE_DIMENSION_NAMES):
        x, y = a.values_vec[i] / na, b.values_vec[i] / nb
        if x > 0 and y > 0:
            signals.append(Signal("value", name, min(1.0, 4.0 * math.sqrt(x * y))))

    for facet, sim in lifestyle_components(a, b).items():
        if sim >= 0.9:
            signals.append(Signal("lifestyle", _LIFESTYLE_LABELS[facet], 0.6 * sim))

    # Visual only counts as a *shared* signal when it is mutual.
    mutual_visual = min(f_ab.visual, f_ba.visual)
    if mutual_visual >= 0.7:
        signals.append(Signal("visual", "mutual visual chemistry", mutual_visual))

    signals.sort(key=lambda s: (-s.strength, s.kind, s.label))
    # keep at most 2 value signals so the text is not one-dimensional
    picked: list[Signal] = []
    n_values = 0
    for s in signals:
        if s.kind == "value":
            if n_values >= 2:
                continue
            n_values += 1
        picked.append(s)
        if len(picked) == top_k:
            break
    return Explanation(mutual_score(s_ab, s_ba), s_ab, s_ba, picked)


@dataclass(frozen=True, slots=True)
class PairScore:
    a: int
    b: int
    s_ab: float
    s_ba: float
    mutual: float
    f_ab: Features
    f_ba: Features


def score_pair(a: Profile, b: Profile, wa: Sequence[float], wb: Sequence[float]) -> PairScore:
    f_ab, f_ba = pair_features(a, b), pair_features(b, a)
    s_ab, s_ba = score_from_features(f_ab, wa), score_from_features(f_ba, wb)
    return PairScore(a.user_id, b.user_id, s_ab, s_ba, mutual_score(s_ab, s_ba), f_ab, f_ba)


def explain_pair(a: Profile, b: Profile, ps: PairScore) -> Explanation:
    return explain(a, b, ps.s_ab, ps.s_ba, ps.f_ab, ps.f_ba)
