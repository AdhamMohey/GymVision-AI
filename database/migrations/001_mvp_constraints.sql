-- Pre-approved MVP additions only. Run after database/migrations/000_enable_vector.sql
-- and database/schema.sql on a fresh PostgreSQL 17 database.

-- The ERD requires cosine similarity over VECTOR(512).
CREATE INDEX IF NOT EXISTS face_embeddings_hnsw_cosine_idx
    ON face_embeddings USING hnsw (embedding vector_cosine_ops);

-- Contact uniqueness applies only to records that have not been soft-deleted.
CREATE UNIQUE INDEX IF NOT EXISTS members_phone_non_deleted_uidx
    ON members (phone) WHERE deleted_at IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS members_email_non_deleted_uidx
    ON members (email) WHERE deleted_at IS NULL AND email IS NOT NULL;

-- Obvious required status/decision columns needed for reliable business decisions.
ALTER TABLE members
    ALTER COLUMN member_status SET NOT NULL,
    ALTER COLUMN enrollment_status SET NOT NULL;

ALTER TABLE memberships
    ALTER COLUMN membership_status SET NOT NULL;

ALTER TABLE face_embeddings
    ALTER COLUMN embedding SET NOT NULL,
    ALTER COLUMN model_version SET NOT NULL,
    ALTER COLUMN is_active SET NOT NULL;

ALTER TABLE biometric_consents
    ALTER COLUMN consent_status SET NOT NULL,
    ALTER COLUMN consent_version SET NOT NULL,
    ALTER COLUMN consent_date SET NOT NULL;

ALTER TABLE devices
    ALTER COLUMN device_status SET NOT NULL,
    ALTER COLUMN api_key_hash SET NOT NULL;

ALTER TABLE access_logs
    ALTER COLUMN access_decision SET NOT NULL,
    ALTER COLUMN reason SET NOT NULL,
    ALTER COLUMN accessed_at SET NOT NULL;

ALTER TABLE admin_users
    ALTER COLUMN username SET NOT NULL,
    ALTER COLUMN password_hash SET NOT NULL,
    ALTER COLUMN role SET NOT NULL,
    ALTER COLUMN is_active SET NOT NULL;

ALTER TABLE audit_logs
    ALTER COLUMN action SET NOT NULL,
    ALTER COLUMN entity_type SET NOT NULL,
    ALTER COLUMN created_at SET NOT NULL;

-- Prevent overlapping inclusive membership date ranges for the same member.
CREATE OR REPLACE FUNCTION prevent_membership_overlap()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.start_date IS NULL OR NEW.end_date IS NULL THEN
        RETURN NEW;
    END IF;
    IF NEW.end_date < NEW.start_date THEN
        RAISE EXCEPTION 'membership end date precedes start date'
            USING ERRCODE = '22007';
    END IF;
    -- Serialize overlap checks for this member to avoid concurrent insert races.
    PERFORM pg_advisory_xact_lock(hashtextextended(NEW.member_id::text, 0));
    IF EXISTS (
        SELECT 1
        FROM memberships existing
        WHERE existing.member_id = NEW.member_id
          AND existing.membership_id <> NEW.membership_id
          AND daterange(existing.start_date, existing.end_date, '[]')
              && daterange(NEW.start_date, NEW.end_date, '[]')
    ) THEN
        RAISE EXCEPTION 'membership dates overlap for member %', NEW.member_id
            USING ERRCODE = '23P01';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER memberships_prevent_overlap
    BEFORE INSERT OR UPDATE OF member_id, start_date, end_date ON memberships
    FOR EACH ROW EXECUTE FUNCTION prevent_membership_overlap();

-- Audit history is append-only.
CREATE OR REPLACE FUNCTION prevent_audit_log_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'audit_logs rows are append-only'
        USING ERRCODE = '55000';
END;
$$;

CREATE TRIGGER audit_logs_append_only
    BEFORE UPDATE OR DELETE ON audit_logs
    FOR EACH ROW EXECUTE FUNCTION prevent_audit_log_mutation();
