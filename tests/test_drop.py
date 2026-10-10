import itertools
import random

import pytest

from matching.drop import allocate_drop, greedy_b_matching, naive_top_k
from matching.filters import PoolState, candidate_graph
from matching.models import pair_key
from matching.scoring import score_pair
from matching.simulate import gini
from tests.factories import random_population

UNIFORM = (1 / 3, 1 / 3, 1 / 3)


def _scored_graph(profiles, state):
    g = candidate_graph(profiles, state)
    scores, one_sided = {}, {}
    for u, vs in g.items():
        for v in vs:
            if u < v:
                ps = score_pair(profiles[u], profiles[v], UNIFORM, UNIFORM)
                scores[(u, v)] = ps.mutual
                one_sided[(u, v)], one_sided[(v, u)] = ps.s_ab, ps.s_ba
    return g, scores, one_sided


@pytest.fixture(scope="module")
def population():
    return random_population(300, seed=11)


def _assert_valid(result, graph, cap=3):
    seen = set()
    for p in result.pairs:
        assert p.a < p.b
        assert p.b in graph[p.a], "proposal must be a feasible edge"
        assert (p.a, p.b) not in seen, "no duplicate pair"
        seen.add((p.a, p.b))
    counts = {u: 0 for u in graph}
    for p in result.pairs:
        counts[p.a] += 1
        counts[p.b] += 1
    assert counts == {u: result.load[u] for u in graph}
    assert max(counts.values(), default=0) <= cap


def test_no_user_exceeds_three_proposals(population):
    g, scores, _ = _scored_graph(population, PoolState())
    result = allocate_drop(g, scores)
    _assert_valid(result, g)
    assert max(result.load.values()) == 3


def test_hub_with_many_admirers_is_capped():
    # star: user 0 is everyone's best option
    graph = {0: set(range(1, 11)), **{i: {0} for i in range(1, 11)}}
    scores = {(0, i): 0.9 for i in range(1, 11)}
    result = allocate_drop(graph, scores)
    assert result.load[0] == 3
    assert len(result.pairs) == 3


def test_already_shown_pairs_never_repeat(population):
    state = PoolState()
    all_pairs = set()
    for _week in range(6):
        g, scores, _ = _scored_graph(population, state)
        result = allocate_drop(g, scores)
        _assert_valid(result, g)
        week_pairs = {(p.a, p.b) for p in result.pairs}
        assert not (week_pairs & all_pairs), "a pair was shown twice"
        assert not (week_pairs & state.shown_pairs)
        all_pairs |= week_pairs
        state.mark_shown(week_pairs)
    assert len(all_pairs) > 0


def test_users_in_active_exclusive_conversation_get_nothing(population):
    g0, scores0, _ = _scored_graph(population, PoolState())
    first = allocate_drop(g0, scores0)
    # the first 20 pairs "matched" and are now in exclusive conversations
    exclusive = {u for p in first.pairs[:20] for u in (p.a, p.b)}
    state = PoolState(in_exclusive=exclusive)
    g, scores, _ = _scored_graph(population, state)
    result = allocate_drop(g, scores)
    for p in result.pairs:
        assert p.a not in exclusive and p.b not in exclusive
    assert all(result.load.get(u, 0) == 0 for u in exclusive)


def test_scarcity_pass_covers_user_that_greedy_starves():
    # Hub H=0 is the top choice of A,B,C (score 0.9) and the ONLY option for L=9.
    graph = {0: {1, 2, 3, 9}, 1: {0, 4}, 2: {0, 5}, 3: {0, 6},
             4: {1}, 5: {2}, 6: {3}, 9: {0}}
    scores = {(0, 1): 0.9, (0, 2): 0.9, (0, 3): 0.9, (0, 9): 0.4,
              (1, 4): 0.5, (2, 5): 0.5, (3, 6): 0.5}
    greedy = greedy_b_matching(graph, scores)
    assert greedy.load[9] == 0, "precondition: plain greedy leaves L uncovered"
    ours = allocate_drop(graph, scores)
    _assert_valid(ours, graph)
    assert ours.load[9] == 1
    assert not ours.uncovered


def test_repair_step_frees_a_slot_on_a_full_candidate():
    # Hubs 8 and 22 are filled by their leaves; hub 0 is filled by 12, 15 and 3,
    # and 3 later also gets 20. When 30 (options: 0 and 8, both full) is served,
    # the repair step drops (0, 3) -- 3 keeps 20 -- and gives 0's slot to 30.
    edges = [(0, 3), (0, 12), (0, 15), (0, 30), (3, 20), (20, 22), (22, 23), (22, 24),
             (22, 25), (8, 30), (5, 8), (6, 8), (7, 8)]
    graph: dict[int, set[int]] = {}
    for u, v in edges:
        graph.setdefault(u, set()).add(v)
        graph.setdefault(v, set()).add(u)
    scores = {pair_key(u, v): 0.5 for u, v in edges}
    scores[(0, 3)] = 0.9
    ours = allocate_drop(graph, scores)
    _assert_valid(ours, graph)
    pairs = {(p.a, p.b) for p in ours.pairs}
    assert (0, 30) in pairs and (3, 20) in pairs
    assert all(ours.load[u] >= 1 for u in graph)
    assert greedy_b_matching(graph, scores).load[30] == 0


def test_scarcity_pass_gives_full_coverage_on_population(population):
    g, scores, _ = _scored_graph(population, PoolState())
    ours = allocate_drop(g, scores)
    greedy = greedy_b_matching(g, scores)
    feasible = [u for u, vs in g.items() if vs]
    covered = sum(1 for u in feasible if ours.load[u] > 0)
    covered_greedy = sum(1 for u in feasible if greedy.load[u] > 0)
    assert covered == len(feasible)
    assert covered >= covered_greedy


def test_bmatching_is_fairer_than_naive_top3(population):
    g, scores, one_sided = _scored_graph(population, PoolState())
    ours = allocate_drop(g, scores)
    naive = naive_top_k(g, one_sided)
    assert max(naive.values()) > 3
    assert gini([ours.load[u] for u in g]) < gini([naive[u] for u in g])


def _brute_force_opt(graph, scores, cap):
    edges = sorted(scores)
    best = 0.0
    for r in range(len(edges) + 1):
        for subset in itertools.combinations(edges, r):
            load = {}
            ok = True
            for u, v in subset:
                load[u] = load.get(u, 0) + 1
                load[v] = load.get(v, 0) + 1
                if load[u] > cap or load[v] > cap:
                    ok = False
                    break
            if ok:
                best = max(best, sum(scores[e] for e in subset))
    return best


@pytest.mark.parametrize("seed", range(8))
def test_greedy_is_half_approximation_on_small_graphs(seed):
    rng = random.Random(seed)
    n, cap = 7, 2
    graph = {u: set() for u in range(n)}
    scores = {}
    for u, v in itertools.combinations(range(n), 2):
        if rng.random() < 0.5 and len(scores) < 13:
            graph[u].add(v)
            graph[v].add(u)
            scores[pair_key(u, v)] = rng.random()
    opt = _brute_force_opt(graph, scores, cap)
    greedy = greedy_b_matching(graph, scores, capacity=cap)
    ours = allocate_drop(graph, scores, capacity=cap)
    assert greedy.total_weight >= 0.5 * opt - 1e-9
    _assert_valid(ours, graph, cap)
    assert ours.total_weight <= opt + 1e-9


def test_min_score_floor_is_respected():
    graph = {1: {2, 3}, 2: {1}, 3: {1}}
    scores = {(1, 2): 0.2, (1, 3): 0.6}
    ours = allocate_drop(graph, scores, min_score=0.5)
    assert {(p.a, p.b) for p in ours.pairs} == {(1, 3)}
