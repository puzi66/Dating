"""Weekly Drop allocation: capacity-constrained b-matching with b = 3.

Why not Gale-Shapley
--------------------
Stable marriage assumes a bipartite market (two sides, each ranking the
other). Our compatibility graph is NOT bipartite: same-sex and bisexual users
mean an edge can connect any two users, so the problem is the *stable
roommates* setting, where a stable solution may not exist at all. Stability is
also the wrong objective here: we do not want a single partner per user, we
want up to three good, mutually-plausible proposals per user per week, and we
care about coverage and fairness at least as much as about the top pairs.

So we model a Drop as a b-matching on a general graph:
    vertices = available users, edges = mutually feasible pairs,
    edge weight = mutual score (geometric mean of the two directions),
    capacity b(v) = 3 for every user.
A proposal is a pair shown to BOTH users, so every chosen edge consumes one
unit of capacity at each endpoint. Because capacity is per user, a popular
user can appear in at most 3 proposals; the demand that would otherwise pile
onto them is forced to spread across the rest of the pool. This is the
structural reason the Drop is fairer than "everyone sees their own top 3".

Algorithm
---------
Pass 1 (scarcity-first coverage). Users are served in increasing order of
    *residual* feasible degree (candidates that still have free capacity),
    recomputed lazily with a heap. Each uncovered user gets their best
    still-available partner. If a user's candidates are all full, a repair
    step swaps out one of the candidate's pairs whose other endpoint would
    still remain covered. Users with a tiny feasible set go first, before
    popular users get filled up by people who have many alternatives.
Pass 2 / Pass 3 (greedy fill). Edges are scanned in decreasing mutual score;
    pass k adds an edge if both endpoints have fewer than k proposals. Filling
    level-by-level keeps the load even (no one gets a 3rd proposal while a
    neighbour with the same options still has 1).

Approximation guarantee
-----------------------
The plain greedy rule ("sort edges by weight, take an edge if both endpoints
still have capacity") is a classic 1/2-approximation for maximum-weight
b-matching: every edge of the optimum that greedy skipped was blocked by a
saturated endpoint, and each greedy edge can be charged for at most two such
optimal edges (one per endpoint), each of weight no larger than its own. The
final pass here is exactly that greedy over the residual capacities, so the
output is a *maximal* b-matching. Passes 1-2 intentionally trade a little
total weight for coverage and evenness, so the strict 1/2 bound applies to
`greedy_b_matching` (exposed below and reported by the simulation for
comparison). In the simulation there is no measurable price: our total
mutual score is 3-8% HIGHER than plain greedy, because level-by-level filling
leaves fewer users stranded with unused capacity and so places more edges
(see README). That is an empirical observation, not a guarantee.

TODO(exact-bmatching): replace the heuristic with an exact max-weight
b-matching. For the general (non-bipartite) graph this needs blossom-based
b-matching; a practical route is min-cost flow on the bipartite double cover
(each user split into a "proposer" and "receiver" copy, capacities 3, costs
= -mutual score) followed by rounding, or an ILP with a coverage constraint
(sum_e x_e >= 1 for every non-isolated vertex) solved with OR-Tools/HiGHS.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field

from .models import CAPACITY_PER_USER, pair_key


@dataclass(frozen=True, slots=True)
class ProposedPair:
    a: int          # low user id
    b: int          # high user id
    mutual: float
    pass_no: int    # 1 = scarcity/coverage pass, 2/3 = greedy fill


@dataclass
class DropResult:
    pairs: list[ProposedPair]
    load: dict[int, int]
    uncovered: set[int] = field(default_factory=set)   # had candidates but got nothing

    def proposals_for(self, user_id: int) -> list[int]:
        out = []
        for p in self.pairs:
            if p.a == user_id:
                out.append(p.b)
            elif p.b == user_id:
                out.append(p.a)
        return out

    @property
    def total_weight(self) -> float:
        return sum(p.mutual for p in self.pairs)


def _sorted_edges(candidates: Mapping[int, set[int]],
                  scores: Mapping[tuple[int, int], float],
                  min_score: float) -> list[tuple[float, int, int]]:
    edges = []
    for u, vs in candidates.items():
        for v in vs:
            if u < v:
                s = scores[(u, v)]
                if s >= min_score:
                    edges.append((s, u, v))
    edges.sort(key=lambda e: (-e[0], e[1], e[2]))
    return edges


def allocate_drop(candidates: Mapping[int, set[int]],
                  scores: Mapping[tuple[int, int], float],
                  capacity: int = CAPACITY_PER_USER,
                  min_score: float = 0.0) -> DropResult:
    """Scarcity-first coverage pass, then level-by-level greedy fill.

    candidates: undirected feasibility graph (already hard-filtered, already
                excluding shown/blocked pairs and unavailable users).
    scores:     mutual score per canonical pair (low, high).
    """
    # Drop edges below the quality floor up front.
    adj: dict[int, set[int]] = {
        u: {v for v in vs if scores[pair_key(u, v)] >= min_score}
        for u, vs in candidates.items()
    }
    load: dict[int, int] = defaultdict(int)
    chosen: dict[tuple[int, int], int] = {}          # pair -> pass number
    partners: dict[int, set[int]] = defaultdict(set)

    def add(u: int, v: int, pass_no: int) -> None:
        chosen[pair_key(u, v)] = pass_no
        load[u] += 1
        load[v] += 1
        partners[u].add(v)
        partners[v].add(u)

    def remove(u: int, v: int) -> None:
        del chosen[pair_key(u, v)]
        load[u] -= 1
        load[v] -= 1
        partners[u].discard(v)
        partners[v].discard(u)

    def residual(u: int) -> list[int]:
        return [v for v in adj[u] if load[v] < capacity and pair_key(u, v) not in chosen]

    # ---- Pass 1: scarcity-first coverage ---------------------------------
    heap = [(len(vs), u) for u, vs in adj.items() if vs]
    heapq.heapify(heap)
    while heap:
        deg, u = heapq.heappop(heap)
        if load[u] > 0:
            continue
        options = residual(u)
        if len(options) != deg:          # stale key: reinsert with fresh degree
            heapq.heappush(heap, (len(options), u))
            continue
        if options:
            v = max(options, key=lambda x: (scores[pair_key(u, x)], -load[x], -x))
            add(u, v, 1)
            continue
        # Repair: every candidate is full. Steal a slot from candidate v by
        # dropping one of v's pairs (v, w) where w keeps >= 1 proposal.
        best: tuple[float, int, int] | None = None
        for v in adj[u]:
            if pair_key(u, v) in chosen:
                continue
            for w in partners[v]:
                if load[w] >= 2:
                    delta = scores[pair_key(u, v)] - scores[pair_key(v, w)]
                    if best is None or delta > best[0]:
                        best = (delta, v, w)
        if best is not None:
            _, v, w = best
            remove(v, w)
            add(u, v, 1)

    # ---- Passes 2 and 3: greedy fill, one capacity level at a time -------
    edges = _sorted_edges(adj, scores, min_score)
    for level in range(2, capacity + 1):
        for s, u, v in edges:
            if (u, v) in chosen:
                continue
            if load[u] < level and load[v] < level:
                add(u, v, level)

    pairs = [ProposedPair(a, b, scores[(a, b)], p) for (a, b), p in chosen.items()]
    pairs.sort(key=lambda p: (p.a, p.b))
    uncovered = {u for u, vs in adj.items() if vs and load[u] == 0}
    return DropResult(pairs, {u: load[u] for u in candidates}, uncovered)


def greedy_b_matching(candidates: Mapping[int, set[int]],
                      scores: Mapping[tuple[int, int], float],
                      capacity: int = CAPACITY_PER_USER,
                      min_score: float = 0.0) -> DropResult:
    """Plain greedy max-weight b-matching (the 1/2-approximation baseline)."""
    load: dict[int, int] = defaultdict(int)
    pairs: list[ProposedPair] = []
    for s, u, v in _sorted_edges(candidates, scores, min_score):
        if load[u] < capacity and load[v] < capacity:
            load[u] += 1
            load[v] += 1
            pairs.append(ProposedPair(u, v, s, 0))
    uncovered = {u for u, vs in candidates.items() if vs and load[u] == 0}
    return DropResult(pairs, {u: load[u] for u in candidates}, uncovered)


def naive_top_k(candidates: Mapping[int, set[int]],
                one_sided: Mapping[tuple[int, int], float],
                k: int = CAPACITY_PER_USER) -> dict[int, int]:
    """Baseline: each user sees their own top-k by one-sided score S(a->b),
    with no capacity limit on the receiving side.

    one_sided is keyed by DIRECTED pairs (a, b). Returns how many times each
    user is *received* (appears in someone else's top-k).
    """
    received: dict[int, int] = {u: 0 for u in candidates}
    for a, vs in candidates.items():
        top = sorted(vs, key=lambda b: (-one_sided[(a, b)], b))[:k]
        for b in top:
            received[b] += 1
    return received
