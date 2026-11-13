# GymVision AI Database Design

## 1. Purpose and Scope

This document records the approved PostgreSQL database baseline for GymVision AI. It describes the approved tables, columns, relationships, database rules, and vector-search direction. It is a specification only: it does not create or migrate database objects.

This document distinguishes:

- **APPROVED DESIGN** — decisions explicitly present in the approved ERD or project requirements.
- **IMPLEMENTATION NOTES — REVIEW REQUIRED** — implementation details that remain open and must not be treated as approved schema decisions.

No table, column, relationship, or data type may be added, removed, renamed, or changed without the project owner's explicit approval.

## 2. Database Technology

### APPROVED DESIGN

- Database: PostgreSQL 17.
- Vector extension: pgvector.
- Architecture: local-first; member data and biometric processing remain local.
- Expected scale: approximately 5,000 gym members.
- Face embeddings: `VECTOR(512)`.
- Face matching: cosine similarity.
- Vector index family: HNSW.

### IMPLEMENTATION NOTES — REVIEW REQUIRED

Extension installation and activation, deployment details, and environment-specific PostgreSQL configuration have not been specified here. Review them before implementation. Do not infer a hosted service or a different database technology.

## 3. Approved ERD Reference

The authoritative source is the original, unchanged **GymVision AI — PostgreSQL ERD v1.0** image:

[`docs/diagrams/gymvision_erd_v1.png`](diagrams/gymvision_erd_v1.png)

The image is the single source of truth for the approved visual schema, keys, relationships, and cardinalities. This document supplements the image in text; it does not replace or alter it. If a discrepancy is found, stop and ask the project owner to resolve it before implementation.

The reference image was verified at the path above when this document was prepared.

## 4. Tables and Column Specifications

The nine tables below are the approved baseline. Nullability is stated only where explicitly established by the approved design or the ERD. Columns listed as UNCONFIRMED below have no established NULL/NOT NULL status; do not infer one. Primary keys (PKs) and foreign keys (FKs) are identified explicitly.

### `members`

- `member_id` — UUID, PK
- `full_name` — VARCHAR(150), NOT NULL
- `phone` — VARCHAR(20), NOT NULL, UNIQUE among non-deleted members
- `email` — VARCHAR(255), nullable, UNIQUE among non-deleted members
- `date_of_birth` — DATE, nullable
- `member_status` — ENUM(`active`, `suspended`, `banned`)
- `enrollment_status` — ENUM(`pending`, `enrolled`, `failed`)
- `deleted_at` — TIMESTAMPTZ, nullable
- `created_at` — TIMESTAMPTZ
- `updated_at` — TIMESTAMPTZ

### `membership_plans`

- `plan_id` — UUID, PK
- `name` — VARCHAR(100), UNIQUE
- `duration_days` — INT
- `price` — NUMERIC(10,2)
- `is_active` — BOOLEAN
- `created_at` — TIMESTAMPTZ

### `memberships`

- `membership_id` — UUID, PK
- `member_id` — UUID, NOT NULL, FK to `members.member_id` (required relationship in ERD)
- `plan_id` — UUID, NOT NULL, FK to `membership_plans.plan_id` (required relationship in ERD)
- `start_date` — DATE
- `end_date` — DATE
- `membership_status` — ENUM(`active`, `expired`, `cancelled`, `frozen`)
- `created_at` — TIMESTAMPTZ

### `face_embeddings`

- `embedding_id` — UUID, PK
- `member_id` — UUID, NOT NULL, FK to `members.member_id` (required relationship in ERD)
- `embedding` — VECTOR(512)
- `model_version` — VARCHAR(50)
- `quality_score` — REAL, nullable
- `is_active` — BOOLEAN
- `created_at` — TIMESTAMPTZ

### `biometric_consents`

- `consent_id` — UUID, PK
- `member_id` — UUID, NOT NULL, FK to `members.member_id` (required relationship in ERD)
- `consent_status` — ENUM(`granted`, `revoked`)
- `consent_version` — VARCHAR(20)
- `consent_date` — TIMESTAMPTZ
- `revoked_at` — TIMESTAMPTZ, nullable

### `devices`

- `device_id` — UUID, PK
- `device_name` — VARCHAR(100)
- `device_location` — VARCHAR(150)
- `device_status` — ENUM(`online`, `offline`, `disabled`)
- `api_key_hash` — VARCHAR(255)
- `last_seen_at` — TIMESTAMPTZ, nullable
- `created_at` — TIMESTAMPTZ

### `access_logs`

- `log_id` — UUID, PK
- `member_id` — UUID, nullable FK to `members.member_id`; NULL is used for unknown faces
- `device_id` — UUID, NOT NULL, FK to `devices.device_id` (required relationship in ERD)
- `reviewed_by` — UUID, nullable FK to `admin_users.admin_id`
- `access_decision` — ENUM(`allowed`, `denied`)
- `reason` — ENUM(`granted`, `unknown_face`, `membership_expired`, `no_active_membership`, `member_suspended`, `no_consent`, `liveness_failed`, `low_quality_image`)
- `similarity_score` — REAL, nullable
- `liveness_score` — REAL, nullable
- `accessed_at` — TIMESTAMPTZ

### `admin_users`

- `admin_id` — UUID, PK
- `username` — VARCHAR(50), UNIQUE
- `password_hash` — VARCHAR(255)
- `role` — ENUM(`owner`, `receptionist`, `viewer`)
- `is_active` — BOOLEAN
- `last_login_at` — TIMESTAMPTZ, nullable
- `created_at` — TIMESTAMPTZ

### `audit_logs`

- `audit_id` — UUID, PK
- `admin_id` — UUID, nullable FK to `admin_users.admin_id`
- `action` — VARCHAR(100)
- `entity_type` — VARCHAR(50)
- `entity_id` — UUID, nullable
- `details` — JSONB, nullable
- `created_at` — TIMESTAMPTZ

### UNCONFIRMED NULLABILITY

The following 36 columns have no explicit NULL/NOT NULL status established by the approved written design or ERD. Their nullability remains UNCONFIRMED; this document does not choose a value:

- `members`: `member_status`, `enrollment_status`, `created_at`, `updated_at`.
- `membership_plans`: `name`, `duration_days`, `price`, `is_active`, `created_at`.
- `memberships`: `start_date`, `end_date`, `membership_status`, `created_at`.
- `face_embeddings`: `embedding`, `model_version`, `is_active`, `created_at`.
- `biometric_consents`: `consent_status`, `consent_version`, `consent_date`.
- `devices`: `device_name`, `device_location`, `device_status`, `api_key_hash`, `created_at`.
- `access_logs`: `access_decision`, `reason`, `accessed_at`.
- `admin_users`: `username`, `password_hash`, `role`, `is_active`, `created_at`.
- `audit_logs`: `action`, `entity_type`, `created_at`.

This status is distinct from explicitly nullable fields, explicitly NOT NULL fields, primary keys (which are non-null by definition), and required relationship FKs identified in the ERD.

## 5. Relationships and Cardinality

### APPROVED DESIGN

- `members` 1:N `memberships`; `memberships.member_id` is a required (NOT NULL) FK.
- `membership_plans` 1:N `memberships`; `memberships.plan_id` is a required (NOT NULL) FK.
- `members` 1:N `face_embeddings`; `face_embeddings.member_id` is a required (NOT NULL) FK.
- `members` 1:N `biometric_consents`; `biometric_consents.member_id` is a required (NOT NULL) FK.
- `members` 1:N `access_logs`; `access_logs.member_id` is optional (nullable) for an unknown face.
- `devices` 1:N `access_logs`; `access_logs.device_id` is a required (NOT NULL) FK.
- `admin_users` 1:N `access_logs` through optional (nullable) `reviewed_by`.
- `admin_users` 1:N `audit_logs` through optional (nullable) `admin_id`.

These cardinalities and optional relationships must remain consistent with the approved ERD image.

## 6. ENUM Types

The following approved ENUM value sets are used by the columns specified above:

- `member_status`: `active`, `suspended`, `banned`.
- `enrollment_status`: `pending`, `enrolled`, `failed`.
- `membership_status`: `active`, `expired`, `cancelled`, `frozen`.
- `consent_status`: `granted`, `revoked`.
- `device_status`: `online`, `offline`, `disabled`.
- `access_decision`: `allowed`, `denied`.
- `reason`: `granted`, `unknown_face`, `membership_expired`, `no_active_membership`, `member_suspended`, `no_consent`, `liveness_failed`, `low_quality_image`.
- `role`: `owner`, `receptionist`, `viewer`.

Creating, changing, or removing an ENUM value is a schema change and requires owner approval.

## 7. Vector Storage and Similarity Search

### APPROVED DESIGN

- Store face embeddings as pgvector `VECTOR(512)` values in `face_embeddings.embedding`.
- Use cosine similarity for face matching.
- Use an HNSW index for vector similarity search.
- Compare embeddings only when their `model_version` values match.
- Store embeddings only; do not store face images in this database design.

### IMPLEMENTATION NOTES — REVIEW REQUIRED

The exact HNSW index definition, operator class, tuning parameters, query threshold, and performance validation plan are not specified. They require technical review before implementation and must not be invented as approved decisions.

## 8. Data Integrity and Constraints

### APPROVED DESIGN

- UUID primary keys identify rows; foreign keys connect the relationships listed above.
- Phone numbers must use E.164 format, as indicated in the ERD.
- Phone and email uniqueness applies among non-deleted members, as indicated in the ERD.
- A member must not have overlapping memberships.
- A consent revocation requires deletion of that member's embeddings; new embeddings require valid consent.
- Access decision and reason must agree; `allowed` is used only when access is granted.
- Unknown faces are logged with `access_logs.member_id = NULL`.
- `audit_logs` is append-only; audit rows must not be updated or deleted.

### IMPLEMENTATION NOTES — REVIEW REQUIRED

The database mechanisms that enforce these rules (including E.164 validation, partial uniqueness, date-range exclusion, decision/reason validation, consent-revocation deletion, and append-only protection) are not defined by this document. Review and approve enforcement details before schema implementation. Do not assume that documentation alone enforces a rule.

## 9. Security and Biometric Privacy

### APPROVED REQUIREMENTS

- Never store plaintext passwords; `admin_users` stores `password_hash`.
- Store device API keys as hashes in `devices.api_key_hash`.
- Keep biometric data and processing local; do not upload sensitive member data to external services without approval.
- Require explicit, valid biometric consent before enrollment and recognition, and enforce consent revocation.
- Restrict access to biometric embeddings and protect them against unauthorized access.
- Protect secrets using environment variables; never commit credentials or secrets.
- Protect audit logs from unauthorized modification and preserve their append-only rule.

### IMPLEMENTATION NOTES — REVIEW REQUIRED

Specific authorization roles for application access, encryption and key management, secret rotation, backup handling, and operational retention procedures require review. This section does not approve a particular implementation for those items.

## 10. Indexing Strategy

### APPROVED DESIGN

- Use HNSW indexing for vector similarity search with cosine similarity.
- Preserve the uniqueness requirements shown in the approved ERD, including member contact uniqueness among non-deleted members.

### IMPLEMENTATION NOTES — REVIEW REQUIRED

No additional indexes are approved here. Indexes for foreign keys, filtering, timestamps, or other query patterns should be proposed with workload evidence and reviewed before adoption. Exact HNSW configuration remains open as described in Section 7.

## 11. Open Technical Decisions

The following items remain unresolved and require review before the relevant implementation decisions are made. They are implementation questions, not changes to the approved ERD:

1. NULL/NOT NULL status for the 36 columns listed in Section 4; do not infer these statuses.
2. pgvector extension setup and activation for the local PostgreSQL 17 environment.
3. Exact HNSW index configuration, operator class, and query behavior; HNSW with cosine similarity remains the approved requirement.
4. Enforcement of the no-overlapping-memberships rule.
5. Soft-delete uniqueness behavior for phone and email, including E.164 validation.
6. Foreign-key deletion and update policies.
7. Biometric deletion behavior and lifecycle details after consent revocation. The approved requirement to delete embeddings after revocation remains in force; implementation and retention details are open.
8. Database enforcement for access-decision and reason consistency, including the unknown-face relationship.
9. Mechanism for enforcing append-only audit logs.
10. Additional indexes, if justified by measured query workloads.

These are **IMPLEMENTATION NOTES — REVIEW REQUIRED**, not approved schema changes. No answer is assumed by this document.

## 12. Approval Checklist

### Approved baseline recorded

- [x] PostgreSQL 17, pgvector, and local-first architecture.
- [x] Nine approved tables and their listed columns and types.
- [x] Approved relationships and cardinalities, including the nullable unknown-face member reference.
- [x] `VECTOR(512)`, cosine similarity, and HNSW as the approved vector-search direction.
- [x] Security, consent, and database business rules stated in the approved requirements and ERD.
- [x] Original ERD image referenced without modification.

### Future review required before implementation details are finalized

- [ ] pgvector setup and exact HNSW parameters.
- [ ] Nullability decisions for the 36 UNCONFIRMED columns.
- [ ] Model-version compatibility behavior.
- [ ] Enforcement of membership date overlap and soft-delete uniqueness.
- [ ] Foreign-key deletion/update policies.
- [ ] Consent-revocation deletion lifecycle details.
- [ ] Enforcement mechanisms for decision/reason consistency and append-only audit logs.
- [ ] Any additional indexes or implementation improvements.

Checking an item above records its status in this specification; it does not authorize an unapproved schema change or database operation.
