import math

from matching.models import VALUES_DIM, VISUAL_DIM, PreferenceWeights
from matching.scoring import (VALUE_DIMENSION_NAMES, directional_score, explain_pair, mutual_score,
                              pair_features, score_pair)
from tests.factories import basis, make_man, make_profile, unit

UNIFORM = (1 / 3, 1 / 3, 1 / 3)


def test_geometric_mean_penalizes_asymmetry_numbers():
    balanced = mutual_score(0.5, 0.5)
    lopsided = mutual_score(0.9, 0.1)
    assert math.isclose(balanced, 0.5)
    assert math.isclose(lopsided, 0.3)
    # same arithmetic mean, much lower mutual score
    assert lopsided < balanced
    assert mutual_score(0.8, 0.0) == 0.0


def test_geometric_mean_penalizes_one_sided_visual_attraction():
    look_x = basis(VISUAL_DIM, 0)
    look_y = basis(VISUAL_DIM, 1)
    # a is drawn to look_x and b has it; b is drawn to look_y but a does not have it
    a = make_profile(1, visual_taste_vec=look_x, appearance_vec=tuple(-x for x in look_y))
    b = make_man(2, visual_taste_vec=look_y, appearance_vec=look_x)
    one_sided = score_pair(a, b, UNIFORM, UNIFORM)
    assert one_sided.s_ab > one_sided.s_ba

    # symmetric pair whose directional scores have the same arithmetic mean
    c = make_profile(3, visual_taste_vec=look_x, appearance_vec=look_x)
    d = make_man(4, visual_taste_vec=look_x, appearance_vec=look_x)
    sym = score_pair(c, d, UNIFORM, UNIFORM)
    target = (one_sided.s_ab + one_sided.s_ba) / 2
    assert one_sided.mutual < target  # GM < AM whenever the sides differ
    assert math.isclose(sym.mutual, (sym.s_ab + sym.s_ba) / 2)


def test_directional_score_uses_raters_weights():
    a = make_profile(1, visual_taste_vec=basis(VISUAL_DIM, 0))
    b = make_man(2, appearance_vec=basis(VISUAL_DIM, 0), values_vec=unit([-1.0] * VALUES_DIM))
    looks_first = directional_score(a, b, (0.1, 0.8, 0.1))
    values_first = directional_score(a, b, PreferenceWeights(1, 0.8, 0.1, 0.1))
    assert looks_first > values_first


def test_features_are_in_unit_interval():
    a = make_profile(1, values_vec=basis(VALUES_DIM, 0))
    b = make_man(2, values_vec=basis(VALUES_DIM, 0, -1.0))
    f = pair_features(a, b)
    for x in f.as_tuple():
        assert 0.0 <= x <= 1.0
    assert f.values == 0.0


def test_explanation_lists_shared_values():
    shared = [0.0] * VALUES_DIM
    shared[0] = 1.0   # family
    shared[5] = 0.8   # humor
    a = make_profile(1, values_vec=unit(shared))
    b = make_man(2, values_vec=unit(shared))
    ps = score_pair(a, b, UNIFORM, UNIFORM)
    exp = explain_pair(a, b, ps)
    labels = [s.label for s in exp.top_signals]
    assert VALUE_DIMENSION_NAMES[0] in labels
    assert VALUE_DIMENSION_NAMES[5] in labels
    assert 1 <= len(exp.top_signals) <= 4
    d = exp.to_dict()
    assert d["top_signals"] and math.isclose(d["mutual_score"], round(ps.mutual, 4))
