-- Matching engine schema (PostgreSQL 16+ with pgvector >= 0.5 for HNSW).
--
-- Conventions
--   * Pairs are always stored canonically as (user_low_id, user_high_id) with
--     user_low_id < user_high_id, so "A shown to B" and "B shown to A" are the
--     same row and uniqueness is trivial to enforce.
--   * Vectors are L2-normalised by the application before insert, so cosine
--     distance (<=>) and inner product are interchangeable.

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- Enumerations
-- ---------------------------------------------------------------------------
CREATE TYPE user_status      AS ENUM ('active', 'paused', 'banned', 'deleted');
CREATE TYPE gender           AS ENUM ('male', 'female', 'nonbinary');
CREATE TYPE relationship_intent AS ENUM ('casual', 'serious', 'marriage');
CREATE TYPE kids_wish        AS ENUM ('yes', 'no', 'open');
CREATE TYPE smoking_level    AS ENUM ('no', 'sometimes', 'yes');   -- ordered
CREATE TYPE drop_status      AS ENUM ('building', 'published', 'closed');
CREATE TYPE proposal_status  AS ENUM ('pending', 'accepted', 'declined', 'expired');
CREATE TYPE match_status     AS ENUM ('active', 'ended_couple', 'ended_no_reply',
                                      'ended_declined', 'ended_expired');

-- ---------------------------------------------------------------------------
-- Users and profiles
-- ---------------------------------------------------------------------------
CREATE TABLE users (
    id              BIGSERIAL PRIMARY KEY,
    email           TEXT        NOT NULL UNIQUE,
    display_name    TEXT        NOT NULL,
    status          user_status NOT NULL DEFAULT 'active',
    paused_until    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_active_at  TIMESTAMPTZ
);
CREATE INDEX users_active_idx ON users (id) WHERE status = 'active';

CREATE TABLE profiles (
    user_id                    BIGINT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,

    -- (a) hard facts
    birth_date                 DATE NOT NULL,
    gender                     gender NOT NULL,
    city                       TEXT NOT NULL,
    lat                        DOUBLE PRECISION NOT NULL CHECK (lat BETWEEN -90 AND 90),
    lon                        DOUBLE PRECISION NOT NULL CHECK (lon BETWEEN -180 AND 180),
    intent                     relationship_intent NOT NULL,
    has_kids                   BOOLEAN NOT NULL,
    wants_kids                 kids_wish NOT NULL,
    religiosity                SMALLINT NOT NULL CHECK (religiosity BETWEEN 0 AND 3),
                               -- 0 secular, 1 traditional, 2 religious, 3 orthodox
    smoking                    smoking_level NOT NULL,

    -- (a) dealbreakers: what this user accepts in a partner
    seeking_genders            gender[] NOT NULL CHECK (cardinality(seeking_genders) > 0),
    age_min                    SMALLINT NOT NULL CHECK (age_min >= 18),
    age_max                    SMALLINT NOT NULL CHECK (age_max >= age_min),
    max_distance_km            SMALLINT NOT NULL CHECK (max_distance_km > 0),
    accepted_intents           relationship_intent[] NOT NULL,
    accepts_partner_with_kids  BOOLEAN NOT NULL DEFAULT TRUE,
    religiosity_tolerance      SMALLINT NOT NULL DEFAULT 1 CHECK (religiosity_tolerance BETWEEN 0 AND 3),
    max_partner_smoking        smoking_level NOT NULL DEFAULT 'sometimes',

    -- (b) values vector from the AI interview
    values_vec                 vector(32),
    interview_completed_at     TIMESTAMPTZ,

    -- (c) visual taste (6 calibration rounds) and how the user looks to others,
    --     both in the same 16-d avatar embedding space
    visual_taste_vec           vector(16),
    appearance_vec             vector(16),
    calibration_rounds_done    SMALLINT NOT NULL DEFAULT 0 CHECK (calibration_rounds_done BETWEEN 0 AND 6),

    updated_at                 TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ANN indexes. values_vec is the main retrieval key (symmetric similarity).
CREATE INDEX profiles_values_hnsw ON profiles
    USING hnsw (values_vec vector_cosine_ops) WITH (m = 16, ef_construction = 64);
-- "Who looks like what I like": query appearance_vec with my visual_taste_vec.
CREATE INDEX profiles_appearance_hnsw ON profiles
    USING hnsw (appearance_vec vector_cosine_ops) WITH (m = 16, ef_construction = 64);
-- Reverse direction: "who would like how I look": query visual_taste_vec with my appearance_vec.
CREATE INDEX profiles_visual_taste_hnsw ON profiles
    USING hnsw (visual_taste_vec vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- Hard-filter prefilters (equality first, then range).
CREATE INDEX profiles_gender_birth_idx ON profiles (gender, birth_date);
CREATE INDEX profiles_seeking_gin      ON profiles USING gin (seeking_genders);
CREATE INDEX profiles_latlon_idx       ON profiles (lat, lon);
CREATE INDEX profiles_religiosity_idx  ON profiles (religiosity, intent);

-- ---------------------------------------------------------------------------
-- Learned per-user preference weights (updated online from debriefs)
-- ---------------------------------------------------------------------------
CREATE TABLE user_preference_weights (
    user_id      BIGINT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    w_values     REAL NOT NULL DEFAULT 0.3334,
    w_visual     REAL NOT NULL DEFAULT 0.3333,
    w_lifestyle  REAL NOT NULL DEFAULT 0.3333,
    bias         REAL NOT NULL DEFAULT 0,
    n_updates    INTEGER NOT NULL DEFAULT 0,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (w_values BETWEEN 0 AND 1 AND w_visual BETWEEN 0 AND 1 AND w_lifestyle BETWEEN 0 AND 1),
    CHECK (abs(w_values + w_visual + w_lifestyle - 1) < 0.001)
);

-- Population-level prior that new users' weights shrink towards (cold start).
CREATE TABLE global_learning_state (
    id                 SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    w_values           REAL NOT NULL,
    w_visual           REAL NOT NULL,
    w_lifestyle        REAL NOT NULL,
    bias               REAL NOT NULL DEFAULT 0,
    feature_mean       REAL[] NOT NULL,          -- running mean used to centre features
    n_updates          BIGINT NOT NULL DEFAULT 0,
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Weekly drops and proposals
-- ---------------------------------------------------------------------------
CREATE TABLE drops (
    id                 BIGSERIAL PRIMARY KEY,
    week_start         DATE NOT NULL UNIQUE,
    status             drop_status NOT NULL DEFAULT 'building',
    algorithm_version  TEXT NOT NULL,
    capacity_per_user  SMALLINT NOT NULL DEFAULT 3,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    published_at       TIMESTAMPTZ
);

-- One row per proposed PAIR. Both users see each other in the same drop and
-- each answers independently; the aggregate status is 'accepted' only when
-- both sides accepted.
CREATE TABLE proposals (
    id               BIGSERIAL PRIMARY KEY,
    drop_id          BIGINT NOT NULL REFERENCES drops(id) ON DELETE CASCADE,
    user_low_id      BIGINT NOT NULL REFERENCES users(id),
    user_high_id     BIGINT NOT NULL REFERENCES users(id),
    mutual_score     REAL NOT NULL,
    score_low_to_high REAL NOT NULL,
    score_high_to_low REAL NOT NULL,
    explanation      JSONB NOT NULL DEFAULT '{}'::jsonb,   -- "why the two of you"
    response_low     proposal_status NOT NULL DEFAULT 'pending',
    response_high    proposal_status NOT NULL DEFAULT 'pending',
    status           proposal_status NOT NULL DEFAULT 'pending',
    expires_at       TIMESTAMPTZ NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (user_low_id < user_high_id),
    CHECK (status <> 'accepted' OR (response_low = 'accepted' AND response_high = 'accepted')),
    UNIQUE (drop_id, user_low_id, user_high_id)
);
CREATE INDEX proposals_drop_low_idx  ON proposals (drop_id, user_low_id);
CREATE INDEX proposals_drop_high_idx ON proposals (drop_id, user_high_id);
CREATE INDEX proposals_pending_idx   ON proposals (expires_at) WHERE status = 'pending';

-- Hard guarantee: no user appears in more than capacity_per_user proposals per drop.
CREATE OR REPLACE FUNCTION enforce_proposal_capacity() RETURNS trigger AS $$
DECLARE
    cap SMALLINT;
    n   INTEGER;
    uid BIGINT;
BEGIN
    SELECT capacity_per_user INTO cap FROM drops WHERE id = NEW.drop_id;
    FOREACH uid IN ARRAY ARRAY[NEW.user_low_id, NEW.user_high_id] LOOP
        PERFORM pg_advisory_xact_lock(NEW.drop_id::int, uid::int);
        SELECT count(*) INTO n FROM proposals
         WHERE drop_id = NEW.drop_id AND (user_low_id = uid OR user_high_id = uid);
        IF n >= cap THEN
            RAISE EXCEPTION 'user % already has % proposals in drop %', uid, n, NEW.drop_id;
        END IF;
    END LOOP;
    RETURN NEW;
END $$ LANGUAGE plpgsql;

CREATE TRIGGER proposals_capacity BEFORE INSERT ON proposals
    FOR EACH ROW EXECUTE FUNCTION enforce_proposal_capacity();

-- ---------------------------------------------------------------------------
-- History of pairs already shown to each other (never shown again)
-- ---------------------------------------------------------------------------
CREATE TABLE shown_pairs (
    user_low_id     BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    user_high_id    BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    first_drop_id   BIGINT NOT NULL REFERENCES drops(id),
    shown_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_low_id, user_high_id),
    CHECK (user_low_id < user_high_id)
);
CREATE INDEX shown_pairs_high_idx ON shown_pairs (user_high_id);

CREATE TABLE blocks (
    blocker_id  BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    blocked_id  BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (blocker_id, blocked_id),
    CHECK (blocker_id <> blocked_id)
);
CREATE INDEX blocks_blocked_idx ON blocks (blocked_id);

-- ---------------------------------------------------------------------------
-- Matches = exclusive conversations
-- ---------------------------------------------------------------------------
CREATE TABLE matches (
    id                 BIGSERIAL PRIMARY KEY,
    proposal_id        BIGINT NOT NULL UNIQUE REFERENCES proposals(id),
    user_low_id        BIGINT NOT NULL REFERENCES users(id),
    user_high_id       BIGINT NOT NULL REFERENCES users(id),
    status             match_status NOT NULL DEFAULT 'active',
    started_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    reply_deadline     TIMESTAMPTZ NOT NULL,      -- started_at + 48h
    decision_deadline  TIMESTAMPTZ NOT NULL,      -- started_at + 7 days
    first_reply_at     TIMESTAMPTZ,
    ended_at           TIMESTAMPTZ,
    CHECK (user_low_id < user_high_id),
    CHECK (reply_deadline <= decision_deadline)
);
CREATE INDEX matches_active_deadlines_idx ON matches (reply_deadline, decision_deadline)
    WHERE status = 'active';

-- One row per participant so that "at most ONE active exclusive conversation
-- per user" is a plain partial unique index.
CREATE TABLE match_participants (
    match_id   BIGINT NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    user_id    BIGINT NOT NULL REFERENCES users(id),
    is_active  BOOLEAN NOT NULL DEFAULT TRUE,
    PRIMARY KEY (match_id, user_id)
);
CREATE UNIQUE INDEX match_participants_one_active
    ON match_participants (user_id) WHERE is_active;

CREATE TABLE messages (
    id         BIGSERIAL PRIMARY KEY,
    match_id   BIGINT NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    sender_id  BIGINT NOT NULL REFERENCES users(id),
    body       TEXT NOT NULL,
    sent_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX messages_match_idx ON messages (match_id, sent_at);

-- ---------------------------------------------------------------------------
-- Post-date debriefs (feed the online weight update)
-- ---------------------------------------------------------------------------
CREATE TABLE date_debriefs (
    id                 BIGSERIAL PRIMARY KEY,
    match_id           BIGINT NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    rater_id           BIGINT NOT NULL REFERENCES users(id),
    ratee_id           BIGINT NOT NULL REFERENCES users(id),
    rating             SMALLINT NOT NULL CHECK (rating BETWEEN 1 AND 5),
    wants_second_date  BOOLEAN NOT NULL,
    -- snapshot of the directional feature vector at proposal time, so the
    -- weight update is reproducible even after profiles change
    features           REAL[] NOT NULL,
    notes              TEXT,
    applied_to_weights BOOLEAN NOT NULL DEFAULT FALSE,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (match_id, rater_id),
    CHECK (rater_id <> ratee_id)
);
CREATE INDEX date_debriefs_unapplied_idx ON date_debriefs (created_at) WHERE NOT applied_to_weights;
CREATE INDEX date_debriefs_rater_idx ON date_debriefs (rater_id);
