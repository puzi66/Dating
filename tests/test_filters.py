import itertools

import pytest

from matching.filters import (PoolState, accepts, candidate_graph, haversine_km,
                              mutually_compatible, one_way_rejections)
from matching.models import Gender, Intent, KidsWish, Religiosity, Smoking, pair_key
from tests.factories import HAIFA, JERUSALEM, TEL_AVIV, make_man, make_profile, random_population


def test_haversine_tel_aviv_jerusalem():
    d = haversine_km(*TEL_AVIV, *JERUSALEM)
    assert 50 < d < 58


def test_filter_symmetry_on_random_population():
    profiles = random_population(160, seed=3)
    one_sided = 0
    for a, b in itertools.combinations(profiles.values(), 2):
        ab, ba = mutually_compatible(a, b), mutually_compatible(b, a)
        assert ab == ba
        if accepts(a, b) != accepts(b, a):
            one_sided += 1
            assert not ab, "a one-way acceptance must never yield a feasible pair"
    assert one_sided > 0, "population should contain one-way acceptances for the test to be meaningful"


@pytest.mark.parametrize("field,b_kwargs,a_kwargs", [
    ("gender", dict(gender=Gender.FEMALE, seeking=frozenset({Gender.FEMALE})), {}),
    ("age", dict(age=45), {}),
    ("distance", dict(loc=HAIFA), dict(max_distance_km=30)),
    ("intent", dict(intent=Intent.CASUAL, accepted_intents=frozenset({Intent.CASUAL, Intent.SERIOUS})), {}),
    ("has_kids", dict(has_kids=True), dict(accepts_partner_with_kids=False)),
    ("wants_kids", dict(wants_kids=KidsWish.NO), dict(wants_kids=KidsWish.YES)),
    ("religiosity", dict(religiosity=Religiosity.ORTHODOX, religiosity_tolerance=3), {}),
    ("smoking", dict(smoking=Smoking.YES, max_partner_smoking=Smoking.YES), dict(max_partner_smoking=Smoking.NO)),
])
def test_each_dealbreaker_blocks_pair_in_both_orders(field, b_kwargs, a_kwargs):
    # a: straight woman; b: a man that violates exactly one of a's dealbreakers
    a = make_profile(1, **a_kwargs)
    b_kwargs = dict(b_kwargs)
    if "gender" not in b_kwargs:
        b = make_man(2, **b_kwargs)
    else:
        b = make_profile(2, **b_kwargs)
    assert field in one_way_rejections(a, b) + one_way_rejections(b, a)
    assert not mutually_compatible(a, b)
    assert not mutually_compatible(b, a)


def test_same_sex_pair_is_supported():
    a = make_profile(1, seeking=frozenset({Gender.FEMALE}))
    b = make_profile(2, seeking=frozenset({Gender.FEMALE, Gender.MALE}))
    assert mutually_compatible(a, b)


def test_one_way_only_is_rejected():
    # b (Jerusalem) would travel 60 km, a (Tel Aviv) only 30 km
    a = make_profile(1, max_distance_km=30)
    b = make_man(2, loc=JERUSALEM, max_distance_km=60)
    assert accepts(b, a) and not accepts(a, b)
    assert not mutually_compatible(a, b)


def _small_pool():
    women = {i: make_profile(i) for i in range(1, 5)}
    men = {i: make_man(i) for i in range(5, 9)}
    return {**women, **men}


def test_candidate_graph_excludes_shown_and_blocked_pairs():
    profiles = _small_pool()
    state = PoolState()
    state.mark_shown([(1, 5)])
    state.block(6, 2)
    g = candidate_graph(profiles, state)
    assert 5 not in g[1] and 1 not in g[5]
    assert 6 not in g[2] and 2 not in g[6]
    assert 6 in g[1]
    assert pair_key(5, 1) in state.shown_pairs


def test_candidate_graph_excludes_exclusive_and_paused_users():
    profiles = _small_pool()
    state = PoolState(in_exclusive={1, 5}, paused={2})
    g = candidate_graph(profiles, state)
    for uid in (1, 2, 5):
        assert uid not in g
        assert all(uid not in vs for vs in g.values())
    assert g[3] == {6, 7, 8}
