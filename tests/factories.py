"""Small helpers to build profiles for tests."""

from __future__ import annotations

import math
import random

from matching.models import (VALUES_DIM, VISUAL_DIM, Gender, Intent, KidsWish, Profile,
                             Religiosity, Smoking)

TEL_AVIV = (32.0853, 34.7818)
JERUSALEM = (31.7683, 35.2137)
HAIFA = (32.7940, 34.9896)


def unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return tuple(x / n for x in v)


def basis(dim: int, i: int, scale: float = 1.0):
    return tuple(scale if j == i else 0.0 for j in range(dim))


def make_profile(uid: int, **kw) -> Profile:
    """A permissive default profile: woman, 30, Tel Aviv, seeks men."""
    lat, lon = kw.pop("loc", TEL_AVIV)
    defaults = dict(
        user_id=uid, gender=Gender.FEMALE, age=30, city="Tel Aviv-Yafo", lat=lat, lon=lon,
        intent=Intent.SERIOUS, has_kids=False, wants_kids=KidsWish.YES,
        religiosity=Religiosity.SECULAR, smoking=Smoking.NO,
        seeking=frozenset({Gender.MALE}), age_min=25, age_max=40, max_distance_km=50,
        accepted_intents=frozenset({Intent.SERIOUS, Intent.MARRIAGE}),
        accepts_partner_with_kids=True, religiosity_tolerance=1,
        max_partner_smoking=Smoking.SOMETIMES,
        values_vec=unit([1.0] * VALUES_DIM), visual_taste_vec=unit([1.0] * VISUAL_DIM),
        appearance_vec=unit([1.0] * VISUAL_DIM),
    )
    defaults.update(kw)
    return Profile(**defaults)


def make_man(uid: int, **kw) -> Profile:
    kw.setdefault("gender", Gender.MALE)
    kw.setdefault("seeking", frozenset({Gender.FEMALE}))
    return make_profile(uid, **kw)


def random_population(n: int, seed: int = 0):
    """Population from the simulator generator (realistic, mixed orientations)."""
    from matching.simulate import generate_population
    return generate_population(n, random.Random(seed)).profiles
