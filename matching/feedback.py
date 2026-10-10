"""Online learning of per-user preference weights from date debriefs.

Model (per rater a, about the date with b):
    x = features(a -> b) - mu            (mu: running population mean)
    p = sigmoid(bias_a + SCALE * w_a . x)
    y = soft label from the debrief (wants a second date, nudged by rating)

One SGD step on the logistic loss: w += lr * (y - p) * x, bias likewise.
Then the weights are projected back onto the feasible set used by the scorer:
each weight clamped to [W_MIN, W_MAX] and the vector normalized to sum to 1.

Intuition: a date that went well while values-similarity was above average
pushes w_values up; a date that went badly while visual attraction was high
pushes w_visual down.

Because one user generates only a handful of debriefs, the scorer uses
`effective_weights`, which shrinks a user's weights towards a population
prior that is updated by every debrief (a cheap hierarchical model that also
handles cold start).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from .models import DateDebrief, PreferenceWeights
from .scoring import Features

W_MIN = 0.05
W_MAX = 0.85
SCALE = 6.0          # features differ by ~0.1-0.3, scale so the logit moves
USER_LR = 0.6
GLOBAL_LR = 0.08
PRIOR_STRENGTH = 4.0  # pseudo-debriefs that a user's own weights must outweigh
MEAN_EMA = 0.02


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def project_weights(w: Sequence[float], lo: float = W_MIN, hi: float = W_MAX) -> tuple[float, ...]:
    """Clamp each weight into [lo, hi] and renormalize to sum 1.

    Exact "water-filling": coordinates that hit a bound are fixed there, the
    remaining mass is shared by the free coordinates in proportion to their
    current values, and this repeats until nothing violates a bound (at most
    len(w) rounds). Requires len(w) * lo <= 1 <= len(w) * hi.
    """
    n = len(w)
    v = [max(1e-9, x) for x in w]
    fixed: dict[int, float] = {}
    for _ in range(n + 1):
        free = [i for i in range(n) if i not in fixed]
        mass = 1.0 - sum(fixed.values())
        s = sum(v[i] for i in free)
        out = [fixed.get(i, 0.0) for i in range(n)]
        for i in free:
            out[i] = mass * v[i] / s
        # Fix upper-bound violations first: capping them hands mass to the
        # others, which may resolve lower-bound violations by itself.
        violated = {i: hi for i in free if out[i] > hi}
        if not violated:
            violated = {i: lo for i in free if out[i] < lo}
        if not violated:
            return tuple(out)
        fixed.update(violated)
    return tuple(out)


def debrief_label(rating: int, wants_second_date: bool) -> float:
    """Soft target in [0, 1]: mostly the y/n answer, nudged by the 1-5 rating."""
    return 0.8 * (1.0 if wants_second_date else 0.0) + 0.2 * (rating - 1) / 4.0


def sgd_step(weights: PreferenceWeights, x: Features, y: float,
             center: Sequence[float], lr: float) -> PreferenceWeights:
    """Single logistic-regression SGD step, in place. Returns the same object."""
    xc = [xi - ci for xi, ci in zip(x.as_tuple(), center)]
    w = weights.as_tuple()
    z = weights.bias + SCALE * math.fsum(wi * xi for wi, xi in zip(w, xc))
    g = y - sigmoid(z)
    new_w = [wi + lr * g * xi for wi, xi in zip(w, xc)]
    weights.set_from(project_weights(new_w))  # type: ignore[arg-type]
    weights.bias = max(-3.0, min(3.0, weights.bias + lr * 0.5 * g))
    weights.n_updates += 1
    return weights


@dataclass
class LearningState:
    """Population-level prior plus running feature mean (for centring)."""

    global_weights: PreferenceWeights = field(default_factory=lambda: PreferenceWeights(user_id=0))
    feature_mean: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])

    def observe_features(self, x: Features) -> None:
        for i, xi in enumerate(x.as_tuple()):
            self.feature_mean[i] += MEAN_EMA * (xi - self.feature_mean[i])


def update_from_debrief(state: LearningState, user_weights: PreferenceWeights,
                        debrief: DateDebrief, features_rater_to_ratee: Features,
                        user_lr: float = USER_LR, global_lr: float = GLOBAL_LR) -> PreferenceWeights:
    """Apply one debrief to the rater's weights and to the population prior."""
    if user_weights.user_id != debrief.rater_id:
        raise ValueError("debrief rater does not own these weights")
    y = debrief_label(debrief.rating, debrief.wants_second_date)
    center = list(state.feature_mean)
    if user_weights.n_updates == 0:
        # cold start: a user's first personal update starts from the prior
        user_weights.set_from(state.global_weights.as_tuple())
        user_weights.bias = state.global_weights.bias
    sgd_step(user_weights, features_rater_to_ratee, y, center, user_lr)
    sgd_step(state.global_weights, features_rater_to_ratee, y, center, global_lr)
    state.observe_features(features_rater_to_ratee)
    return user_weights


def effective_weights(user: PreferenceWeights, prior: PreferenceWeights,
                      prior_strength: float = PRIOR_STRENGTH) -> tuple[float, float, float]:
    """Shrink a user's own weights towards the population prior.

    With n own updates the user's weights get weight n / (n + prior_strength).
    A brand-new user therefore scores with the population weights.
    """
    n = user.n_updates
    lam = n / (n + prior_strength)
    mixed = [lam * u + (1 - lam) * g for u, g in zip(user.as_tuple(), prior.as_tuple())]
    return project_weights(mixed)  # type: ignore[return-value]
