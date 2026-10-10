"""Synthetic end-to-end simulation of the weekly Drop and the feedback loop.

    python -m matching.simulate --users 500 --weeks 4 --seed 1

A synthetic Israeli population is generated with a HIDDEN ground truth:
every user has true preference weights over (values, visual, lifestyle) that
the engine does not see (values matter most for most people, visual matters
most for the first click). Accept/decline is driven by a visual-heavy first
impression; date outcomes are driven by the hidden true compatibility. The
engine starts from uniform weights and learns from debriefs only.

The same seed is also run with learning disabled (control), so the
second-date trend can be separated from pool depletion effects.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field

from .drop import allocate_drop, greedy_b_matching, naive_top_k
from .feedback import LearningState, effective_weights, update_from_debrief
from .filters import PoolState, candidate_graph
from .models import (VALUES_DIM, VISUAL_DIM, DateDebrief, Gender, Intent, KidsWish,
                     PreferenceWeights, Profile, Religiosity, Smoking, pair_key)
from .scoring import Features, pair_features, score_pair, score_from_features

# name, lat, lon, population weight, religiosity mix (secular, traditional, religious, orthodox)
CITIES: list[tuple[str, float, float, float, tuple[float, float, float, float]]] = [
    ("Tel Aviv-Yafo", 32.0853, 34.7818, 14, (0.72, 0.18, 0.08, 0.02)),
    ("Jerusalem", 31.7683, 35.2137, 12, (0.25, 0.25, 0.30, 0.20)),
    ("Haifa", 32.7940, 34.9896, 9, (0.60, 0.25, 0.12, 0.03)),
    ("Rishon LeZion", 31.9730, 34.7925, 7, (0.50, 0.35, 0.12, 0.03)),
    ("Petah Tikva", 32.0840, 34.8878, 7, (0.45, 0.35, 0.15, 0.05)),
    ("Ashdod", 31.8014, 34.6435, 6, (0.35, 0.35, 0.20, 0.10)),
    ("Netanya", 32.3215, 34.8532, 6, (0.45, 0.35, 0.15, 0.05)),
    ("Beer Sheva", 31.2518, 34.7913, 6, (0.45, 0.35, 0.15, 0.05)),
    ("Holon", 32.0158, 34.7874, 5, (0.50, 0.35, 0.12, 0.03)),
    ("Ramat Gan", 32.0684, 34.8248, 5, (0.60, 0.25, 0.12, 0.03)),
    ("Rehovot", 31.8928, 34.8113, 4, (0.50, 0.30, 0.15, 0.05)),
    ("Herzliya", 32.1624, 34.8447, 4, (0.65, 0.25, 0.08, 0.02)),
    ("Kfar Saba", 32.1750, 34.9069, 4, (0.60, 0.28, 0.10, 0.02)),
    ("Ra'anana", 32.1848, 34.8713, 3, (0.55, 0.25, 0.15, 0.05)),
    ("Modi'in", 31.8980, 35.0104, 4, (0.45, 0.30, 0.20, 0.05)),
    ("Tiberias", 32.7922, 35.5312, 1, (0.35, 0.35, 0.20, 0.10)),
    ("Eilat", 29.5577, 34.9519, 1, (0.60, 0.30, 0.08, 0.02)),
]

N_VALUE_ARCHETYPES = 8
TRUE_WEIGHT_ALPHA = (6.0, 1.5, 2.5)      # hidden truth: values >> lifestyle > visual


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

def _unit_vec(v: list[float]) -> tuple[float, ...]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return tuple(x / n for x in v)


def _gauss_vec(rng: random.Random, dim: int, sd: float = 1.0) -> list[float]:
    return [rng.gauss(0.0, sd) for _ in range(dim)]


def _dirichlet(rng: random.Random, alpha: tuple[float, ...]) -> tuple[float, ...]:
    g = [rng.gammavariate(a, 1.0) for a in alpha]
    s = sum(g)
    return tuple(x / s for x in g)


@dataclass
class Population:
    """Synthetic users plus the hidden ground truth the engine never sees."""

    rng: random.Random
    profiles: dict[int, Profile] = field(default_factory=dict)
    true_weights: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    orientation: dict[int, str] = field(default_factory=dict)
    archetypes: list[tuple[float, ...]] = field(default_factory=list)
    beauty: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        self.archetypes = [_unit_vec(_gauss_vec(self.rng, VALUES_DIM)) for _ in range(N_VALUE_ARCHETYPES)]
        self.beauty = _unit_vec(_gauss_vec(self.rng, VISUAL_DIM))   # shared beauty standard

    def add_users(self, n: int) -> list[int]:
        start = max(self.profiles, default=0) + 1
        new_ids = list(range(start, start + n))
        for uid in new_ids:
            self._make_user(uid)
        return new_ids

    def _make_user(self, uid: int) -> None:
        rng = self.rng
        name, lat, lon, _, rel_mix = rng.choices(CITIES, weights=[c[3] for c in CITIES])[0]
        lat += rng.gauss(0, 0.025)
        lon += rng.gauss(0, 0.025)

        r = rng.random()
        gender = Gender.MALE if r < 0.49 else Gender.FEMALE if r < 0.98 else Gender.NONBINARY
        o = rng.random()
        if gender is Gender.NONBINARY:
            orient = "queer"
            seeking = frozenset(rng.sample(list(Gender), rng.randint(1, 3)))
        elif o < 0.88:
            orient = "straight"
            seeking = frozenset({Gender.FEMALE if gender is Gender.MALE else Gender.MALE})
        elif o < 0.95:
            orient = "gay" if gender is Gender.MALE else "lesbian"
            seeking = frozenset({gender})
        else:
            orient = "bi"
            seeking = frozenset({Gender.MALE, Gender.FEMALE} | ({Gender.NONBINARY} if rng.random() < 0.5 else set()))
        self.orientation[uid] = orient

        age = int(round(rng.triangular(21, 46, 29)))
        age_min = max(18, age - rng.randint(4, 9) - (2 if gender is Gender.MALE else 0))
        age_max = max(age + 1, age + rng.randint(4, 10) - (2 if gender is Gender.MALE else 0))

        religiosity = Religiosity(rng.choices(range(4), weights=rel_mix)[0])
        rel_tol = 0 if religiosity is Religiosity.ORTHODOX and rng.random() < 0.6 else rng.choice((1, 1, 1, 2))

        intent = rng.choices([Intent.CASUAL, Intent.SERIOUS, Intent.MARRIAGE],
                             weights=[0.15, 0.60, 0.25] if religiosity < 2 else [0.03, 0.47, 0.50])[0]
        if intent is Intent.CASUAL:
            accepted = {Intent.CASUAL, Intent.SERIOUS}
        elif intent is Intent.SERIOUS:
            accepted = {Intent.SERIOUS, Intent.MARRIAGE} | ({Intent.CASUAL} if rng.random() < 0.3 else set())
        else:
            accepted = {Intent.SERIOUS, Intent.MARRIAGE}

        has_kids = rng.random() < (0.30 if age >= 35 else 0.07)
        wants_kids = rng.choices([KidsWish.YES, KidsWish.NO, KidsWish.OPEN],
                                 weights=[0.55, 0.12, 0.33] if religiosity < 2 else [0.85, 0.02, 0.13])[0]
        smoking = Smoking(rng.choices(range(3), weights=[0.75, 0.15, 0.10])[0])
        max_partner_smoking = (Smoking.YES if smoking is not Smoking.NO
                               else rng.choice((Smoking.NO, Smoking.SOMETIMES, Smoking.SOMETIMES)))

        arch = self.archetypes[rng.randrange(N_VALUE_ARCHETYPES)]
        values = _unit_vec([a + rng.gauss(0, 0.2) for a in arch])

        attractiveness = rng.uniform(-0.4, 1.4)          # drives popularity
        appearance = _unit_vec([attractiveness * b + rng.gauss(0, 0.25) for b in self.beauty])
        taste = _unit_vec([0.7 * b + rng.gauss(0, 0.25) for b in self.beauty])

        self.profiles[uid] = Profile(
            user_id=uid, gender=gender, age=age, city=name, lat=lat, lon=lon,
            intent=intent, has_kids=has_kids, wants_kids=wants_kids,
            religiosity=religiosity, smoking=smoking, seeking=seeking,
            age_min=age_min, age_max=age_max,
            max_distance_km=rng.choice((20, 30, 50, 80, 120)),
            accepted_intents=frozenset(accepted),
            accepts_partner_with_kids=rng.random() < 0.75,
            religiosity_tolerance=rel_tol, max_partner_smoking=max_partner_smoking,
            values_vec=values, visual_taste_vec=taste, appearance_vec=appearance,
        )
        self.true_weights[uid] = _dirichlet(rng, TRUE_WEIGHT_ALPHA)


def generate_population(n: int, rng: random.Random) -> Population:
    pop = Population(rng)
    pop.add_users(n)
    return pop


# ---------------------------------------------------------------------------
# Hidden behaviour model
# ---------------------------------------------------------------------------

def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))


ACCEPT_THRESHOLD = 0.60
DATE_THRESHOLD = 0.635
P_REPLY_48H = 0.85
P_DATE_GIVEN_REPLY = 0.9
P_COUPLE_GIVEN_SECOND = 0.5
P_WEEKLY_PAUSE = 0.03


def first_impression(f: Features) -> float:
    """What drives the click in the Drop: mostly looks."""
    return 0.6 * f.visual + 0.25 * f.values + 0.15 * f.lifestyle


def true_compat(pop: Population, a: int, f_ab: Features) -> float:
    return score_from_features(f_ab, pop.true_weights[a])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def gini(xs: list[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    total = sum(xs)
    if n == 0 or total == 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return (2 * cum) / (n * total) - (n + 1) / n


@dataclass
class WeekStats:
    week: int
    eligible: int = 0
    excluded_exclusive: int = 0
    paused: int = 0
    with_candidates: int = 0
    covered: int = 0
    proposals: int = 0
    max_load: int = 0
    mutual_accepts: int = 0
    matches: int = 0
    dates: int = 0
    second_dates: int = 0
    couples: int = 0
    gini_bmatch: float = 0.0
    gini_greedy: float = 0.0
    gini_naive: float = 0.0
    max_received_naive: int = 0
    coverage_greedy: float = 0.0
    weight_ratio_vs_greedy: float = 0.0
    mean_true_compat: float = 0.0
    rank_corr: float = 0.0
    total_users: int = 0
    new_users: int = 0
    global_w: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def coverage(self) -> float:
        return self.covered / self.eligible if self.eligible else 0.0

    @property
    def coverage_feasible(self) -> float:
        return self.covered / self.with_candidates if self.with_candidates else 0.0

    @property
    def mean_proposals(self) -> float:
        return 2 * self.proposals / self.eligible if self.eligible else 0.0

    @property
    def mutual_accept_rate(self) -> float:
        return self.mutual_accepts / self.proposals if self.proposals else 0.0

    @property
    def second_date_rate(self) -> float:
        return self.second_dates / self.dates if self.dates else 0.0


@dataclass
class SimResult:
    weeks: list[WeekStats] = field(default_factory=list)
    final_mean_user_w: tuple[float, float, float] = (0.0, 0.0, 0.0)
    true_mean_w: tuple[float, float, float] = (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=v.__getitem__)
        r = [0.0] * len(v)
        for rank, i in enumerate(order):
            r[i] = float(rank)
        return r
    if len(xs) < 3:
        return 0.0
    return statistics.correlation(ranks(xs), ranks(ys))


def run_simulation(n_users: int, n_weeks: int, seed: int, learn: bool = True,
                   new_users_per_week: int | None = None) -> SimResult:
    """Run n_weeks of Drops. learn=False keeps uniform weights (control arm).

    new_users_per_week defaults to 10% of n_users (sign-ups refresh the pool;
    without them a 500-user pool is exhausted by the never-repeat rule).
    """
    if new_users_per_week is None:
        new_users_per_week = n_users // 10
    pop = generate_population(n_users, random.Random(seed))
    rng = random.Random(seed * 7919 + 17)   # behaviour RNG (same for both arms)
    profiles = pop.profiles
    state = PoolState()
    weights = {uid: PreferenceWeights(uid) for uid in profiles}
    learning = LearningState()
    couples: set[int] = set()
    carryover: set[int] = set()            # still in an active exclusive conversation

    # a few pre-existing blocks to exercise the path
    ids = list(profiles)
    for _ in range(n_users // 50):
        a, b = rng.sample(ids, 2)
        state.block(a, b)

    result = SimResult()
    for week in range(1, n_weeks + 1):
        ws = WeekStats(week)
        if week > 1 and new_users_per_week:
            for uid in pop.add_users(new_users_per_week):
                weights[uid] = PreferenceWeights(uid)
                ids.append(uid)
            ws.new_users = new_users_per_week
        ws.total_users = len(profiles)
        paused_now = {u for u in ids if u not in couples and rng.random() < P_WEEKLY_PAUSE}
        state.paused = couples | paused_now
        state.in_exclusive = set(carryover)
        ws.paused = len(state.paused)
        ws.excluded_exclusive = len(state.in_exclusive - state.paused)

        graph = candidate_graph(profiles, state)
        ws.eligible = len(graph)
        ws.with_candidates = sum(1 for v in graph.values() if v)

        eff = {u: effective_weights(weights[u], learning.global_weights) if learn
               else (1 / 3, 1 / 3, 1 / 3) for u in graph}
        scores: dict[tuple[int, int], float] = {}
        one_sided: dict[tuple[int, int], float] = {}
        feats: dict[tuple[int, int], Features] = {}
        for u, vs in graph.items():
            for v in vs:
                if u < v:
                    ps = score_pair(profiles[u], profiles[v], eff[u], eff[v])
                    scores[(u, v)] = ps.mutual
                    one_sided[(u, v)], one_sided[(v, u)] = ps.s_ab, ps.s_ba
                    feats[(u, v)], feats[(v, u)] = ps.f_ab, ps.f_ba

        drop = allocate_drop(graph, scores)
        greedy = greedy_b_matching(graph, scores)
        naive = naive_top_k(graph, one_sided)

        ws.proposals = len(drop.pairs)
        ws.covered = sum(1 for u in graph if drop.load[u] > 0)
        ws.max_load = max(drop.load.values(), default=0)
        ws.gini_bmatch = gini([drop.load[u] for u in graph])
        ws.gini_greedy = gini([greedy.load[u] for u in graph])
        ws.gini_naive = gini([naive[u] for u in graph])
        ws.max_received_naive = max(naive.values(), default=0)
        ws.coverage_greedy = (sum(1 for u in graph if greedy.load[u] > 0) / ws.eligible) if ws.eligible else 0.0
        ws.weight_ratio_vs_greedy = drop.total_weight / greedy.total_weight if greedy.total_weight else 0.0
        engine_s, hidden_s = [], []
        for (u, v), s in scores.items():
            engine_s.append(s)
            hidden_s.append(math.sqrt(true_compat(pop, u, feats[(u, v)]) * true_compat(pop, v, feats[(v, u)])))
        ws.rank_corr = spearman(engine_s, hidden_s)
        if drop.pairs:
            ws.mean_true_compat = statistics.fmean(
                math.sqrt(true_compat(pop, p.a, feats[(p.a, p.b)]) * true_compat(pop, p.b, feats[(p.b, p.a)]))
                for p in drop.pairs)

        state.mark_shown((p.a, p.b) for p in drop.pairs)

        # --- responses: both sides decide independently -------------------
        mutual: list[tuple[int, int]] = []
        for p in drop.pairs:
            acc = []
            for x, y in ((p.a, p.b), (p.b, p.a)):
                pr = _sigmoid(12 * (first_impression(feats[(x, y)]) - ACCEPT_THRESHOLD))
                acc.append(rng.random() < pr)
            if all(acc):
                mutual.append((p.a, p.b))
            elif rng.random() < 0.02:
                state.block(p.a, p.b)
        ws.mutual_accepts = len(mutual)

        # --- exclusivity: first mutual accept wins, the rest expire -------
        rng.shuffle(mutual)
        in_conv: set[int] = set()
        matches = []
        for a, b in mutual:
            if a in in_conv or b in in_conv:
                continue
            in_conv.update((a, b))
            matches.append((a, b))
        ws.matches = len(matches)

        # --- conversations, dates, debriefs, feedback ----------------------
        carryover = set()
        for a, b in matches:
            if rng.random() > P_REPLY_48H or rng.random() > P_DATE_GIVEN_REPLY:
                continue                                  # no reply / no date: back to pool
            ws.dates += 1
            chem = rng.gauss(0, 0.03)                     # unobservable shared chemistry
            wants = {}
            for x, y in ((a, b), (b, a)):
                t = true_compat(pop, x, feats[(x, y)]) + chem
                want = rng.random() < _sigmoid(25 * (t - DATE_THRESHOLD))
                rating = max(1, min(5, round(1 + 4 * _sigmoid(15 * (t - DATE_THRESHOLD)) + rng.gauss(0, 0.5))))
                wants[x] = want
                if learn:
                    update_from_debrief(learning, weights[x],
                                        DateDebrief(0, x, y, rating, want), feats[(x, y)])
            if wants[a] and wants[b]:
                ws.second_dates += 1
                if rng.random() < P_COUPLE_GIVEN_SECOND:
                    couples.update((a, b))
                    ws.couples += 1
                else:
                    carryover.update((a, b))               # still exclusive next Drop
        ws.global_w = learning.global_weights.as_tuple()
        result.weeks.append(ws)

    learned = [weights[u].as_tuple() for u in profiles if weights[u].n_updates > 0]
    if learned:
        result.final_mean_user_w = tuple(statistics.fmean(w[i] for w in learned) for i in range(3))  # type: ignore[assignment]
    result.true_mean_w = tuple(statistics.fmean(w[i] for w in pop.true_weights.values()) for i in range(3))  # type: ignore[assignment]
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt_w(w: tuple[float, float, float]) -> str:
    return "v={:.2f} vis={:.2f} life={:.2f}".format(*w)


def format_report(res: SimResult, control: SimResult | None, n_users: int, seed: int) -> str:
    out: list[str] = []
    out.append(f"=== Matching engine simulation: {n_users} users, {len(res.weeks)} weeks, seed {seed} ===")
    out.append("")
    out.append("Per-week Drop (b-matching, b=3, scarcity-first + greedy fill)")
    hdr = (f"{'week':>4} {'users':>5} {'elig':>5} {'excl':>5} {'cover':>7} {'cov|feas':>8} {'props':>5} "
           f"{'mean/u':>6} {'max':>3} {'mutAcc':>7} {'match':>5} {'dates':>5} {'2nd%':>6}")
    out.append(hdr)
    out.append("-" * len(hdr))
    for w in res.weeks:
        out.append(f"{w.week:>4} {w.total_users:>5} {w.eligible:>5} {w.excluded_exclusive:>5} {w.coverage:>7.1%} "
                   f"{w.coverage_feasible:>8.1%} {w.proposals:>5} {w.mean_proposals:>6.2f} {w.max_load:>3} "
                   f"{w.mutual_accept_rate:>7.1%} {w.matches:>5} {w.dates:>5} {w.second_date_rate:>6.1%}")
    out.append("  elig = available users (not paused, not in an exclusive conversation)")
    out.append("  excl = users skipped because their exclusive conversation is still active")
    out.append("  cover = % of eligible users with >=1 proposal; cov|feas = same, among users with >=1 feasible candidate")
    out.append("  users = registered users incl. weekly sign-ups; props = proposed pairs (each shown to both users)")
    out.append("")
    out.append("Fairness: Gini of proposals received (lower = more equal)")
    hdr2 = f"{'week':>4} {'b-match':>8} {'greedy':>8} {'naive':>8} {'naiveMax':>8} {'covGreedy':>9} {'W/Wgreedy':>9}"
    out.append(hdr2)
    out.append("-" * len(hdr2))
    for w in res.weeks:
        out.append(f"{w.week:>4} {w.gini_bmatch:>8.3f} {w.gini_greedy:>8.3f} {w.gini_naive:>8.3f} "
                   f"{w.max_received_naive:>8} {w.coverage_greedy:>9.1%} {w.weight_ratio_vs_greedy:>9.3f}")
    out.append("  naive = each user gets own top-3 by one-sided S(a->b), no capacity on the receiving side")
    out.append("  naiveMax = most proposals any single user would receive under naive (b-match max is 3)")
    out.append("  W/Wgreedy = total mutual score of our Drop / plain greedy b-matching (price of coverage)")
    out.append("")
    out.append("Learning: second-date rate (date -> both want a 2nd date), learning vs. no-learning control")
    out.append(f"{'week':>4} {'2nd%':>7} {'ctl 2nd%':>8} {'lift':>7} {'rho':>6} {'ctl rho':>7} {'trueQ':>6} {'ctl trueQ':>9}  global weights after week")
    for i, w in enumerate(res.weeks):
        c = control.weeks[i] if control else None
        out.append(f"{w.week:>4} {w.second_date_rate:>7.1%} "
                   + (f"{c.second_date_rate:>8.1%} " if c else f"{'-':>8} ")
                   + (f"{100 * (w.second_date_rate - c.second_date_rate):>+6.1f}pp " if c else f"{'-':>7} ")
                   + f"{w.rank_corr:>6.3f} "
                   + (f"{c.rank_corr:>7.3f} " if c else f"{'-':>7} ")
                   + f"{w.mean_true_compat:>6.3f} "
                   + (f"{c.mean_true_compat:>9.3f}" if c else f"{'-':>9}")
                   + f"  {_fmt_w(w.global_w)}")
    out.append("  rho   = Spearman correlation between engine mutual score and HIDDEN true compatibility,")
    out.append("          over all feasible pairs that week (how well the engine ranks)")
    out.append("  trueQ = mean HIDDEN true mutual compatibility of the pairs actually proposed")
    out.append("  lift  = learning minus control in the same week. Both arms decline in absolute terms because")
    out.append("          the never-repeat rule uses up the best pairs first and happy couples leave the pool;")
    out.append("          the control isolates that depletion effect from the effect of learning.")
    tot_d = sum(w.dates for w in res.weeks)
    tot_s = sum(w.second_dates for w in res.weeks)
    line = f"overall {tot_s}/{tot_d} = {tot_s / tot_d:.1%}" if tot_d else "overall n/a"
    if control:
        cd = sum(w.dates for w in control.weeks)
        cs = sum(w.second_dates for w in control.weeks)
        line += f"   (no-learn {cs}/{cd} = {cs / cd:.1%})" if cd else ""
    out.append(line)
    out.append(f"couples formed: {sum(w.couples for w in res.weeks)}"
               + (f" (no-learn {sum(w.couples for w in control.weeks)})" if control else ""))
    out.append("")
    out.append(f"Learned mean user weights : {_fmt_w(res.final_mean_user_w)}")
    out.append(f"Hidden true mean weights  : {_fmt_w(res.true_mean_w)}")
    out.append("Initial weights           : v=0.33 vis=0.33 life=0.33")
    out.append("")
    out.append("NOTE: synthetic population and synthetic behaviour model; numbers validate the")
    out.append("mechanics (capacity, coverage, fairness, learning signal), not real-world outcomes.")
    return "\n".join(out)


def format_multi_seed(pairs: list[tuple[SimResult, SimResult]], seeds: list[int]) -> str:
    """Per-week second-date rate pooled over several seeds (less noise)."""
    n_weeks = len(pairs[0][0].weeks)
    out = [f"Second-date rate pooled over seeds {seeds[0]}..{seeds[-1]} ({len(seeds)} runs per arm)",
           f"{'week':>4} {'learning':>14} {'control':>14} {'lift':>8} {'rho':>6} {'ctl rho':>7}"]
    for i in range(n_weeks):
        ld = sum(r.weeks[i].dates for r, _ in pairs)
        ls = sum(r.weeks[i].second_dates for r, _ in pairs)
        cd = sum(c.weeks[i].dates for _, c in pairs)
        cs = sum(c.weeks[i].second_dates for _, c in pairs)
        rho = statistics.fmean(r.weeks[i].rank_corr for r, _ in pairs)
        crho = statistics.fmean(c.weeks[i].rank_corr for _, c in pairs)
        lr_, cr_ = (ls / ld if ld else 0.0), (cs / cd if cd else 0.0)
        out.append(f"{i + 1:>4} {f'{ls}/{ld}':>8} {lr_:>5.1%} {f'{cs}/{cd}':>8} {cr_:>5.1%} "
                   f"{100 * (lr_ - cr_):>+6.1f}pp {rho:>6.3f} {crho:>7.3f}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Simulate weekly Drops on a synthetic population.")
    ap.add_argument("--users", type=int, default=500)
    ap.add_argument("--weeks", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--new-users-per-week", type=int, default=None,
                    help="weekly sign-ups (default: 10%% of --users)")
    ap.add_argument("--no-control", action="store_true", help="skip the no-learning control run")
    ap.add_argument("--seeds", type=int, default=1,
                    help="also pool the learning/control comparison over this many seeds")
    args = ap.parse_args(argv)
    res = run_simulation(args.users, args.weeks, args.seed, learn=True,
                         new_users_per_week=args.new_users_per_week)
    control = None if args.no_control else run_simulation(
        args.users, args.weeks, args.seed, learn=False, new_users_per_week=args.new_users_per_week)
    print(format_report(res, control, args.users, args.seed))
    if args.seeds > 1 and not args.no_control:
        seeds = list(range(args.seed, args.seed + args.seeds))
        runs = [(res, control)] + [
            (run_simulation(args.users, args.weeks, s, True, args.new_users_per_week),
             run_simulation(args.users, args.weeks, s, False, args.new_users_per_week))
            for s in seeds[1:]]
        print()
        print(format_multi_seed(runs, seeds))  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
