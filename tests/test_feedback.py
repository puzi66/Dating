import math
import random

import pytest

from matching.feedback import (W_MAX, W_MIN, LearningState, effective_weights, project_weights,
                               update_from_debrief)
from matching.models import DateDebrief, PreferenceWeights
from matching.scoring import Features


def _debrief(rater=1, rating=5, wants=True):
    return DateDebrief(match_id=1, rater_id=rater, ratee_id=2, rating=rating, wants_second_date=wants)


def test_good_date_on_values_raises_values_weight():
    w = PreferenceWeights(1)
    before = w.as_tuple()
    x = Features(values=0.9, visual=0.2, lifestyle=0.5)   # values-heavy, looks-light pair
    update_from_debrief(LearningState(), w, _debrief(rating=5, wants=True), x)
    assert w.w_values > before[0]
    assert w.w_visual < before[1]
    assert math.isclose(sum(w.as_tuple()), 1.0)
    assert w.n_updates == 1


def test_bad_date_on_looks_lowers_visual_weight():
    w = PreferenceWeights(1)
    before = w.as_tuple()
    x = Features(values=0.3, visual=0.95, lifestyle=0.5)  # chosen for looks, values mismatch
    update_from_debrief(LearningState(), w, _debrief(rating=1, wants=False), x)
    assert w.w_visual < before[1]
    assert w.w_values > before[0]


def test_global_prior_moves_too():
    state = LearningState()
    w = PreferenceWeights(1)
    g_before = state.global_weights.w_values
    update_from_debrief(state, w, _debrief(), Features(0.9, 0.2, 0.5))
    assert state.global_weights.w_values > g_before


def test_rater_must_own_the_weights():
    with pytest.raises(ValueError):
        update_from_debrief(LearningState(), PreferenceWeights(7), _debrief(rater=1), Features(0.5, 0.5, 0.5))


def test_weights_stay_clamped_and_normalized():
    rng = random.Random(0)
    state, w = LearningState(), PreferenceWeights(1)
    for _ in range(500):
        x = Features(rng.random(), rng.random(), rng.random())
        update_from_debrief(state, w, _debrief(rating=rng.randint(1, 5), wants=rng.random() < 0.5), x)
        t = w.as_tuple()
        assert math.isclose(sum(t), 1.0, abs_tol=1e-9)
        assert all(W_MIN - 1e-9 <= v <= W_MAX + 1e-9 for v in t)


def test_learns_dominant_feature_from_outcomes():
    # Hidden truth: only values similarity predicts a second date.
    rng = random.Random(42)
    state, w = LearningState(), PreferenceWeights(1)
    for _ in range(400):
        x = Features(rng.uniform(0.3, 0.9), rng.uniform(0.3, 0.9), rng.uniform(0.3, 0.9))
        wants = rng.random() < 1 / (1 + math.exp(-12 * (x.values - 0.6)))
        update_from_debrief(state, w, _debrief(rating=4 if wants else 2, wants=wants), x)
    assert w.w_values == max(w.as_tuple())
    assert w.w_values > 0.5


def test_project_weights():
    p = project_weights((5.0, 0.0, 0.0))
    assert math.isclose(sum(p), 1.0)
    assert all(W_MIN - 1e-9 <= v <= W_MAX + 1e-9 for v in p)


def test_effective_weights_shrink_to_prior_for_new_users():
    prior = PreferenceWeights(0, 0.6, 0.1, 0.3)
    new_user = PreferenceWeights(1, 0.1, 0.8, 0.1, n_updates=0)
    assert all(math.isclose(a, b) for a, b in zip(effective_weights(new_user, prior), prior.as_tuple()))
    veteran = PreferenceWeights(1, 0.1, 0.8, 0.1, n_updates=400)
    eff = effective_weights(veteran, prior)
    assert eff[1] > 0.75
