"""Hard filters (dealbreakers) and pool-level exclusions.

A pair (a, b) is feasible only if a accepts b AND b accepts a, the pair was
never shown before, neither blocked the other, and both users are available
(active, not paused, not inside an exclusive conversation).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from .models import KidsWish, Profile, pair_key

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(h)))


def distance_km(a: Profile, b: Profile) -> float:
    return haversine_km(a.lat, a.lon, b.lat, b.lon)


def one_way_rejections(a: Profile, b: Profile) -> list[str]:
    """Reasons why `a` would reject `b` (empty list means a accepts b)."""
    reasons: list[str] = []
    if b.gender not in a.seeking:
        reasons.append("gender")
    if not a.age_min <= b.age <= a.age_max:
        reasons.append("age")
    if distance_km(a, b) > a.max_distance_km:
        reasons.append("distance")
    if b.intent not in a.accepted_intents:
        reasons.append("intent")
    if b.has_kids and not a.accepts_partner_with_kids:
        reasons.append("has_kids")
    if {a.wants_kids, b.wants_kids} == {KidsWish.YES, KidsWish.NO}:
        reasons.append("wants_kids")
    if abs(int(a.religiosity) - int(b.religiosity)) > a.religiosity_tolerance:
        reasons.append("religiosity")
    if b.smoking > a.max_partner_smoking:
        reasons.append("smoking")
    return reasons


def accepts(a: Profile, b: Profile) -> bool:
    return not one_way_rejections(a, b)


def mutually_compatible(a: Profile, b: Profile) -> bool:
    """Hard filters in BOTH directions. Symmetric by construction."""
    return a.user_id != b.user_id and accepts(a, b) and accepts(b, a)


@dataclass
class PoolState:
    """Everything outside the two profiles that can exclude a pair."""

    shown_pairs: set[tuple[int, int]] = field(default_factory=set)
    blocked_pairs: set[tuple[int, int]] = field(default_factory=set)
    in_exclusive: set[int] = field(default_factory=set)
    paused: set[int] = field(default_factory=set)

    def is_available(self, user_id: int) -> bool:
        return user_id not in self.in_exclusive and user_id not in self.paused

    def pair_excluded(self, a: int, b: int) -> bool:
        k = pair_key(a, b)
        return k in self.shown_pairs or k in self.blocked_pairs

    def mark_shown(self, pairs: Iterable[tuple[int, int]]) -> None:
        for a, b in pairs:
            self.shown_pairs.add(pair_key(a, b))

    def block(self, blocker: int, blocked: int) -> None:
        self.blocked_pairs.add(pair_key(blocker, blocked))


def eligible_users(profiles: Mapping[int, Profile], state: PoolState) -> list[int]:
    return sorted(uid for uid in profiles if state.is_available(uid))


def candidate_graph(profiles: Mapping[int, Profile], state: PoolState) -> dict[int, set[int]]:
    """Undirected feasibility graph over available users.

    O(n^2) pairwise check; fine for a prototype. At scale this is replaced by
    a pgvector ANN prefilter + SQL hard filters (see README).
    """
    ids = eligible_users(profiles, state)
    graph: dict[int, set[int]] = {uid: set() for uid in ids}
    for i, a_id in enumerate(ids):
        a = profiles[a_id]
        for b_id in ids[i + 1:]:
            b = profiles[b_id]
            # cheap symmetric pre-checks before the full filter
            if b.gender not in a.seeking or a.gender not in b.seeking:
                continue
            if state.pair_excluded(a_id, b_id):
                continue
            if mutually_compatible(a, b):
                graph[a_id].add(b_id)
                graph[b_id].add(a_id)
    return graph
