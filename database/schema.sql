-- GymVision AI initial schema baseline.
-- This file represents only the approved tables, columns, keys, and constraints
-- whose SQL definitions are explicit in the approved database design.
-- No extension setup, defaults, additional indexes, or inferred nullability are included.

CREATE TYPE member_status AS ENUM (
    'active',
    'suspended',
    'banned'
);

CREATE TYPE enrollment_status AS ENUM (
    'pending',
    'enrolled',
    'failed'
);

CREATE TYPE membership_status AS ENUM (
    'active',
    'expired',
    'cancelled',
    'frozen'
);

CREATE TYPE consent_status AS ENUM (
    'granted',
    'revoked'
);

CREATE TYPE device_status AS ENUM (
    'online',
    'offline',
    'disabled'
);

CREATE TYPE access_decision AS ENUM (
    'allowed',
    'denied'
);

CREATE TYPE access_reason AS ENUM (
    'granted',
    'unknown_face',
    'membership_expired',
    'no_active_membership',
    'member_suspended',
    'no_consent',
    'liveness_failed',
    'low_quality_image'
);

CREATE TYPE admin_role AS ENUM (
    'owner',
    'receptionist',
    'viewer'
);

CREATE TABLE members (
    member_id UUID PRIMARY KEY,
    full_name VARCHAR(150) NOT NULL,
    phone VARCHAR(20) NOT NULL,
    email VARCHAR(255),
    date_of_birth DATE,
    member_status member_status,
    enrollment_status enrollment_status,
    deleted_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ
);

CREATE TABLE membership_plans (
    plan_id UUID PRIMARY KEY,
    name VARCHAR(100) UNIQUE,
    duration_days INT,
    price NUMERIC(10,2),
    is_active BOOLEAN,
    created_at TIMESTAMPTZ
);

CREATE TABLE admin_users (
    admin_id UUID PRIMARY KEY,
    username VARCHAR(50) UNIQUE,
    password_hash VARCHAR(255),
    role admin_role,
    is_active BOOLEAN,
    last_login_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ
);

CREATE TABLE devices (
    device_id UUID PRIMARY KEY,
    device_name VARCHAR(100),
    device_location VARCHAR(150),
    device_status device_status,
    api_key_hash VARCHAR(255),
    last_seen_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ
);

CREATE TABLE memberships (
    membership_id UUID PRIMARY KEY,
    member_id UUID NOT NULL REFERENCES members (member_id),
    plan_id UUID NOT NULL REFERENCES membership_plans (plan_id),
    start_date DATE,
    end_date DATE,
    membership_status membership_status,
    created_at TIMESTAMPTZ
);

CREATE TABLE face_embeddings (
    embedding_id UUID PRIMARY KEY,
    member_id UUID NOT NULL REFERENCES members (member_id),
    embedding VECTOR(512),
    model_version VARCHAR(50),
    quality_score REAL,
    is_active BOOLEAN,
    created_at TIMESTAMPTZ
);

CREATE TABLE biometric_consents (
    consent_id UUID PRIMARY KEY,
    member_id UUID NOT NULL REFERENCES members (member_id),
    consent_status consent_status,
    consent_version VARCHAR(20),
    consent_date TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ
);

CREATE TABLE access_logs (
    log_id UUID PRIMARY KEY,
    member_id UUID REFERENCES members (member_id),
    device_id UUID NOT NULL REFERENCES devices (device_id),
    reviewed_by UUID REFERENCES admin_users (admin_id),
    access_decision access_decision,
    reason access_reason,
    similarity_score REAL,
    liveness_score REAL,
    accessed_at TIMESTAMPTZ
);

CREATE TABLE audit_logs (
    audit_id UUID PRIMARY KEY,
    admin_id UUID REFERENCES admin_users (admin_id),
    action VARCHAR(100),
    entity_type VARCHAR(50),
    entity_id UUID,
    details JSONB,
    created_at TIMESTAMPTZ
);
