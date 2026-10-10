"""Dataclasses mirroring schema.sql.

These are plain in-memory records; a real service would load them from
Postgres. Vectors are tuples of floats (L2-normalised by convention).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum, IntEnum

VALUES_DIM = 32
VISUAL_DIM = 16
CAPACITY_PER_USER = 3
REPLY_WINDOW = timedelta(hours=48)
DECISION_WINDOW = timedelta(days=7)

Vector = tuple[float, ...]


class UserStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    BANNED = "banned"
    DELETED = "deleted"


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    NONBINARY = "nonbinary"


class Intent(str, Enum):
    CASUAL = "casual"
    SERIOUS = "serious"
    MARRIAGE = "marriage"


class KidsWish(str, Enum):
    YES = "yes"
    NO = "no"
    OPEN = "open"


class Smoking(IntEnum):
    NO = 0
    SOMETIMES = 1
    YES = 2


class Religiosity(IntEnum):
    SECULAR = 0
    TRADITIONAL = 1
    RELIGIOUS = 2
    ORTHODOX = 3


class DropStatus(str, Enum):
    BUILDING = "building"
    PUBLISHED = "published"
    CLOSED = "closed"


class ProposalStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    EXPIRED = "expired"


class MatchStatus(str, Enum):
    ACTIVE = "active"
    ENDED_COUPLE = "ended_couple"
    ENDED_NO_REPLY = "ended_no_reply"
    ENDED_DECLINED = "ended_declined"
    ENDED_EXPIRED = "ended_expired"


def pair_key(a: int, b: int) -> tuple[int, int]:
    """Canonical (low, high) representation of an unordered pair."""
    if a == b:
        raise ValueError("a pair needs two different users")
    return (a, b) if a < b else (b, a)


@dataclass(slots=True)
class User:
    id: int
    display_name: str
    status: UserStatus = UserStatus.ACTIVE
    created_at: datetime | None = None


@dataclass(slots=True)
class Profile:
    user_id: int
    # (a) hard facts
    gender: Gender
    age: int
    city: str
    lat: float
    lon: float
    intent: Intent
    has_kids: bool
    wants_kids: KidsWish
    religiosity: Religiosity
    smoking: Smoking
    # (a) dealbreakers
    seeking: frozenset[Gender]
    age_min: int
    age_max: int
    max_distance_km: float
    accepted_intents: frozenset[Intent]
    accepts_partner_with_kids: bool = True
    religiosity_tolerance: int = 1
    max_partner_smoking: Smoking = Smoking.SOMETIMES
    # (b) values from the AI interview, (c) visual taste / appearance
    values_vec: Vector = field(default_factory=lambda: (0.0,) * VALUES_DIM)
    visual_taste_vec: Vector = field(default_factory=lambda: (0.0,) * VISUAL_DIM)
    appearance_vec: Vector = field(default_factory=lambda: (0.0,) * VISUAL_DIM)


@dataclass(slots=True)
class PreferenceWeights:
    """Learned weights over the three score components (sum to 1)."""

    user_id: int
    w_values: float = 1 / 3
    w_visual: float = 1 / 3
    w_lifestyle: float = 1 / 3
    bias: float = 0.0
    n_updates: int = 0

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.w_values, self.w_visual, self.w_lifestyle)

    def set_from(self, w: tuple[float, float, float]) -> None:
        self.w_values, self.w_visual, self.w_lifestyle = w


@dataclass(slots=True)
class Drop:
    id: int
    week_start: date
    status: DropStatus = DropStatus.BUILDING
    algorithm_version: str = "bmatch-scarcity-v1"
    capacity_per_user: int = CAPACITY_PER_USER


@dataclass(slots=True)
class Proposal:
    id: int
    drop_id: int
    user_low_id: int
    user_high_id: int
    mutual_score: float
    score_low_to_high: float
    score_high_to_low: float
    explanation: dict = field(default_factory=dict)
    response_low: ProposalStatus = ProposalStatus.PENDING
    response_high: ProposalStatus = ProposalStatus.PENDING

    @property
    def status(self) -> ProposalStatus:
        responses = (self.response_low, self.response_high)
        if ProposalStatus.DECLINED in responses:
            return ProposalStatus.DECLINED
        if ProposalStatus.EXPIRED in responses:
            return ProposalStatus.EXPIRED
        if all(r is ProposalStatus.ACCEPTED for r in responses):
            return ProposalStatus.ACCEPTED
        return ProposalStatus.PENDING

    def respond(self, user_id: int, response: ProposalStatus) -> None:
        if user_id == self.user_low_id:
            self.response_low = response
        elif user_id == self.user_high_id:
            self.response_high = response
        else:
            raise ValueError(f"user {user_id} is not part of proposal {self.id}")


@dataclass(slots=True)
class Match:
    """An exclusive conversation created when both sides accept."""

    id: int
    proposal_id: int
    user_low_id: int
    user_high_id: int
    started_at: datetime
    status: MatchStatus = MatchStatus.ACTIVE
    first_reply_at: datetime | None = None
    ended_at: datetime | None = None

    @property
    def reply_deadline(self) -> datetime:
        return self.started_at + REPLY_WINDOW

    @property
    def decision_deadline(self) -> datetime:
        return self.started_at + DECISION_WINDOW

    @property
    def participants(self) -> tuple[int, int]:
        return (self.user_low_id, self.user_high_id)


@dataclass(slots=True)
class DateDebrief:
    match_id: int
    rater_id: int
    ratee_id: int
    rating: int
    wants_second_date: bool

    def __post_init__(self) -> None:
        if not 1 <= self.rating <= 5:
            raise ValueError("rating must be in 1..5")


@dataclass(slots=True, frozen=True)
class ShownPair:
    user_low_id: int
    user_high_id: int
    first_drop_id: int


@dataclass(slots=True, frozen=True)
class Block:
    blocker_id: int
    blocked_id: int
