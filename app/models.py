"""ORM tables for the pilot. Vectors are stored as JSON lists (no pgvector needed at 48 users).

Times are naive UTC everywhere (`utcnow()`), converted to Asia/Jerusalem only for display.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from sqlalchemy import (JSON, Boolean, Date, DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text,
                        UniqueConstraint)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class ParticipantStatus:
    INVITED = "invited"
    CONSENTED = "consented"
    PAID = "paid"
    PROFILE_DONE = "profile_done"
    ACTIVE = "active"
    IN_MATCH = "in_match"
    PAUSED = "paused"


STATUS_LABELS_HE = {
    "invited": "הוזמן/ה",
    "consented": "אישר/ה תנאים",
    "paid": "שילם/ה",
    "profile_done": "פרופיל מוכן",
    "active": "פעיל/ה",
    "in_match": "בהתאמה",
    "paused": "מושהה",
}


class Participant(Base):
    __tablename__ = "participants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    token: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    first_name: Mapped[str] = mapped_column(String(80))
    last_name: Mapped[str] = mapped_column(String(80), default="")
    phone: Mapped[str] = mapped_column(String(32))
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    consented_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    paid_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    paused: Mapped[bool] = mapped_column(Boolean, default=False)
    facts_done_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    interview_done_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    interview_ended: Mapped[bool] = mapped_column(Boolean, default=False)   # עדי sent her closing message
    profile_error: Mapped[str] = mapped_column(Text, default="")             # last extraction failure
    visual_done_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # (a) hard facts + dealbreakers (engine Profile fields)
    gender: Mapped[str | None] = mapped_column(String(16), nullable=True)
    seeking: Mapped[list] = mapped_column(JSON, default=list)
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    age_min: Mapped[int | None] = mapped_column(Integer, nullable=True)
    age_max: Mapped[int | None] = mapped_column(Integer, nullable=True)
    city: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_distance_km: Mapped[float | None] = mapped_column(Float, nullable=True)
    intent: Mapped[str | None] = mapped_column(String(16), nullable=True)
    has_kids: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    wants_kids: Mapped[str | None] = mapped_column(String(8), nullable=True)
    accepts_partner_with_kids: Mapped[bool] = mapped_column(Boolean, default=True)
    religiosity: Mapped[int | None] = mapped_column(Integer, nullable=True)
    religiosity_tolerance: Mapped[int] = mapped_column(Integer, default=1)
    smoking: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_partner_smoking: Mapped[int] = mapped_column(Integer, default=1)

    # (b) profile extracted from the interview (editable by the matchmaker)
    summary: Mapped[str] = mapped_column(Text, default="")
    values_scores: Mapped[dict] = mapped_column(JSON, default=dict)   # dim name -> 0..10
    values_vec: Mapped[list] = mapped_column(JSON, default=list)      # 32 floats, L2-normalised
    lifestyle: Mapped[dict] = mapped_column(JSON, default=dict)
    looking_for: Mapped[list] = mapped_column(JSON, default=list)
    red_flags: Mapped[list] = mapped_column(JSON, default=list)

    # (c) visual taste: simplified in the pilot (neutral vectors)
    visual_answers: Mapped[dict] = mapped_column(JSON, default=dict)
    visual_taste_vec: Mapped[list] = mapped_column(JSON, default=list)
    appearance_vec: Mapped[list] = mapped_column(JSON, default=list)

    # learned preference weights (feedback.py)
    w_values: Mapped[float] = mapped_column(Float, default=1 / 3)
    w_visual: Mapped[float] = mapped_column(Float, default=1 / 3)
    w_lifestyle: Mapped[float] = mapped_column(Float, default=1 / 3)
    w_bias: Mapped[float] = mapped_column(Float, default=0.0)
    w_updates: Mapped[int] = mapped_column(Integer, default=0)

    photos: Mapped[list[Photo]] = relationship(back_populates="participant", order_by="Photo.ordinal",
                                               cascade="all, delete-orphan")
    messages: Mapped[list[InterviewMessage]] = relationship(
        back_populates="participant", order_by="InterviewMessage.id", cascade="all, delete-orphan")

    @property
    def display_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()

    @property
    def profile_complete(self) -> bool:
        return bool(self.facts_done_at and self.interview_done_at and self.visual_done_at)


class Photo(Base):
    __tablename__ = "photos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    participant_id: Mapped[int] = mapped_column(ForeignKey("participants.id", ondelete="CASCADE"), index=True)
    storage_key: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(64))
    ordinal: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    participant: Mapped[Participant] = relationship(back_populates="photos")


class StoredBlob(Base):
    """Photo bytes kept in the database (STORAGE=db), for hosts without a persistent disk."""
    __tablename__ = "stored_blobs"

    key: Mapped[str] = mapped_column(String(255), primary_key=True)
    content_type: Mapped[str] = mapped_column(String(64))
    data: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class InterviewMessage(Base):
    __tablename__ = "interview_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    participant_id: Mapped[int] = mapped_column(ForeignKey("participants.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(16))   # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    participant: Mapped[Participant] = relationship(back_populates="messages")


class DropRow(Base):
    __tablename__ = "drops"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    week_start: Mapped[date] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(16), default="draft")   # draft | published | closed
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    published_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    deadline_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    algorithm_version: Mapped[str] = mapped_column(String(32), default="bmatch-scarcity-v1")
    stats: Mapped[dict] = mapped_column(JSON, default=dict)

    proposals: Mapped[list[ProposalRow]] = relationship(back_populates="drop", cascade="all, delete-orphan",
                                                        order_by="ProposalRow.id")


class ProposalRow(Base):
    """One row per pair (user_low_id < user_high_id), shown to both sides."""

    __tablename__ = "proposals"
    __table_args__ = (UniqueConstraint("drop_id", "user_low_id", "user_high_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    drop_id: Mapped[int] = mapped_column(ForeignKey("drops.id", ondelete="CASCADE"), index=True)
    user_low_id: Mapped[int] = mapped_column(ForeignKey("participants.id"), index=True)
    user_high_id: Mapped[int] = mapped_column(ForeignKey("participants.id"), index=True)
    mutual_score: Mapped[float] = mapped_column(Float)
    score_low_to_high: Mapped[float] = mapped_column(Float)
    score_high_to_low: Mapped[float] = mapped_column(Float)
    features: Mapped[dict] = mapped_column(JSON, default=dict)      # {"low_high": [v,vis,life], "high_low": [...]}
    explanation: Mapped[dict] = mapped_column(JSON, default=dict)
    source: Mapped[str] = mapped_column(String(16), default="engine")  # engine | swap
    why_text: Mapped[str] = mapped_column(Text, default="")
    response_low: Mapped[str] = mapped_column(String(8), default="pending")   # pending | yes | no
    response_high: Mapped[str] = mapped_column(String(8), default="pending")
    responded_low_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    responded_high_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # set when both said yes but one side was already in another exclusive match
    superseded: Mapped[bool] = mapped_column(Boolean, default=False)

    drop: Mapped[DropRow] = relationship(back_populates="proposals")
    user_low: Mapped[Participant] = relationship(foreign_keys=[user_low_id])
    user_high: Mapped[Participant] = relationship(foreign_keys=[user_high_id])

    def other(self, pid: int) -> int:
        if pid == self.user_low_id:
            return self.user_high_id
        if pid == self.user_high_id:
            return self.user_low_id
        raise ValueError("participant not in proposal")

    def response_of(self, pid: int) -> str:
        return self.response_low if pid == self.user_low_id else self.response_high

    @property
    def is_mutual(self) -> bool:
        return self.response_low == "yes" and self.response_high == "yes"


class MatchRow(Base):
    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    proposal_id: Mapped[int] = mapped_column(ForeignKey("proposals.id"), unique=True)
    user_low_id: Mapped[int] = mapped_column(ForeignKey("participants.id"), index=True)
    user_high_id: Mapped[int] = mapped_column(ForeignKey("participants.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    group_opened_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    replied_within_48h: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    date_happened_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    close_reason: Mapped[str] = mapped_column(String(32), default="")

    proposal: Mapped[ProposalRow] = relationship()
    user_low: Mapped[Participant] = relationship(foreign_keys=[user_low_id])
    user_high: Mapped[Participant] = relationship(foreign_keys=[user_high_id])
    debriefs: Mapped[list[Debrief]] = relationship(back_populates="match", cascade="all, delete-orphan")

    def other(self, pid: int) -> int:
        return self.user_high_id if pid == self.user_low_id else self.user_low_id


class Debrief(Base):
    __tablename__ = "debriefs"
    __table_args__ = (UniqueConstraint("match_id", "rater_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id", ondelete="CASCADE"), index=True)
    rater_id: Mapped[int] = mapped_column(ForeignKey("participants.id"))
    ratee_id: Mapped[int] = mapped_column(ForeignKey("participants.id"))
    rating: Mapped[int] = mapped_column(Integer)
    meet_again: Mapped[str] = mapped_column(String(8))          # yes | no | unsure
    surprise_text: Mapped[str] = mapped_column(Text, default="")
    looked_like_photos: Mapped[str] = mapped_column(String(8))  # yes | partly | no
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    weights_before: Mapped[list] = mapped_column(JSON, default=list)
    weights_after: Mapped[list] = mapped_column(JSON, default=list)

    match: Mapped[MatchRow] = relationship(back_populates="debriefs")


class SurveyResponse(Base):
    """Weekly 1-question survey on the waiting screen (H2: "3 is enough")."""

    __tablename__ = "survey_responses"
    __table_args__ = (UniqueConstraint("participant_id", "week_start"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    participant_id: Mapped[int] = mapped_column(ForeignKey("participants.id", ondelete="CASCADE"))
    week_start: Mapped[date] = mapped_column(Date)
    score: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class LearningStateRow(Base):
    """Singleton: population prior + running feature mean (feedback.LearningState)."""

    __tablename__ = "learning_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    global_weights: Mapped[list] = mapped_column(JSON, default=lambda: [1 / 3, 1 / 3, 1 / 3])
    global_bias: Mapped[float] = mapped_column(Float, default=0.0)
    global_updates: Mapped[int] = mapped_column(Integer, default=0)
    feature_mean: Mapped[list] = mapped_column(JSON, default=lambda: [0.5, 0.5, 0.5])
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
