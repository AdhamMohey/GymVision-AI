"""PostgreSQL-backed local MVP store plus legacy memory test double."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


@dataclass
class Member:
    member_id: UUID
    full_name: str
    phone: str | None
    email: str | None
    member_status: str
    consent_granted: bool
    membership_start: date
    membership_end: date
    embedding: list[float]
    model_version: str = "synthetic-demo-v1"
    enrollment_status: str = "enrolled"


class DuplicateMemberError(ValueError):
    """Duplicate member contact value with a field-specific API message."""

    def __init__(self, field: str) -> None:
        self.field = field
        message = {
            "phone": "Phone number already exists",
            "email": "Email address already exists",
        }[field]
        super().__init__(message)


def duplicate_member_error(constraint_name: str | None) -> DuplicateMemberError | None:
    """Map the approved partial unique indexes to their corresponding form fields."""
    constraints = {
        "members_phone_non_deleted_uidx": "phone",
        "members_email_non_deleted_uidx": "email",
    }
    field = constraints.get(constraint_name or "")
    return DuplicateMemberError(field) if field else None


@dataclass
class MemoryStore:
    """In-memory demo/test store; an explicit demo mode keeps records process-local."""
    members: dict[UUID, Member] = field(default_factory=dict)
    logs: list[dict[str, object]] = field(default_factory=list)
    revoked_members: set[UUID] = field(default_factory=set)
    max_members: int | None = None

    def _check_capacity(self) -> None:
        if self.max_members is not None and len(self.members) >= self.max_members:
            raise ValueError(f"The in-memory competition demo supports at most {self.max_members} members")

    def enroll(self, **values: object) -> Member:
        if not values["consent_granted"]:
            raise ValueError("valid biometric consent is required")
        self._check_capacity()
        phone, email = values["phone"], values["email"]
        if phone and any(m.phone == phone for m in self.members.values()):
            raise DuplicateMemberError("phone")
        if email and any(m.email == email for m in self.members.values()):
            raise DuplicateMemberError("email")
        model_version = values.get("model_version", "synthetic-demo-v1")
        member = Member(uuid4(), values["full_name"], phone, email, "active", True,
                        values["membership_start"], values["membership_end"],
                        list(values["embedding"]), model_version)
        self.members[member.member_id] = member
        return member

    def prepare_enrollment(self, *, full_name: str, phone: str | None, email: str | None,
                           membership_start: date, membership_end: date) -> dict[str, object]:
        """Create a consented in-memory member without persisting a face image."""
        self._check_capacity()
        if membership_end < membership_start:
            raise ValueError("membership end precedes start")
        if phone and any(m.phone == phone for m in self.members.values()):
            raise DuplicateMemberError("phone")
        if email and any(m.email == email for m in self.members.values()):
            raise DuplicateMemberError("email")
        member = Member(uuid4(), full_name, phone, email, "active", True,
                        membership_start, membership_end, [], "buffalo_s-512-v1", "pending")
        self.members[member.member_id] = member
        return {"member_id": member.member_id, "full_name": full_name, "phone": phone,
                "email": email, "member_status": member.member_status,
                "consent_granted": True, "membership_start": membership_start,
                "membership_end": membership_end, "enrollment_status": "pending"}

    def save_enrollment_embedding(self, member_id: UUID, embedding: list[float],
                                  model_version: str) -> None:
        member = self.members.get(member_id)
        if member is None:
            raise KeyError(member_id)
        if not member.consent_granted or member_id in self.revoked_members:
            raise ValueError("valid biometric consent is required before storing an embedding")
        if len(embedding) != 512 or not all(math.isfinite(value) for value in embedding):
            raise ValueError("embedding must contain 512 finite values")
        member.embedding = list(embedding)
        member.model_version = model_version
        member.enrollment_status = "enrolled"

    def mark_enrollment_failed(self, member_id: UUID) -> None:
        self.members[member_id].enrollment_status = "failed"

    def get_enrollment_status(self, member_id: UUID) -> dict[str, object] | None:
        member = self.members.get(member_id)
        if member is None:
            return None
        return {"member_id": member.member_id, "full_name": member.full_name,
                "phone": member.phone, "email": member.email,
                "member_status": member.member_status,
                "enrollment_status": member.enrollment_status,
                "deleted_at": None, "consent_granted": member.consent_granted}

    @staticmethod
    def cosine_similarity(left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right, strict=True))
        norm = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
        return dot / norm if norm else 0.0

    def check_in(self, embedding: list[float] | None, liveness_score: float | None,
                 *, member_id: UUID | None = None, threshold: float = 0.82,
                 liveness_threshold: float = 0.60,
                 model_version: str = "synthetic-demo-v1") -> dict[str, object]:
        now = datetime.now(timezone.utc)
        member = self.members.get(member_id) if member_id else None
        score: float | None = None
        if member_id is None and embedding:
            matches = [(self.cosine_similarity(embedding, candidate.embedding), candidate)
                       for candidate in self.members.values()
                       if candidate.consent_granted and candidate.member_id not in self.revoked_members
                       and candidate.model_version == model_version and candidate.embedding]
            score, member = max(matches, key=lambda item: item[0], default=(None, None))
        elif (member is not None and member.consent_granted
              and member.model_version == model_version and embedding and member.embedding):
            score = self.cosine_similarity(embedding, member.embedding)

        decision, reason = "denied", "unknown_face"
        matched = member.member_id if member_id and member else None
        if member is not None and (not member.consent_granted or member.member_id in self.revoked_members):
            reason, matched = "consent_revoked", member.member_id
        elif liveness_score is not None and liveness_score < liveness_threshold:
            reason = "liveness_failed"
        elif member is not None and score is not None and score >= threshold:
            matched = member.member_id
            if member.member_status != "active": reason = "member_suspended"
            elif not member.membership_start <= now.date() <= member.membership_end: reason = "membership_expired"
            else: decision, reason = "allowed", "granted"
        row = {"log_id": str(uuid4()), "member_id": str(matched) if matched else None,
               "member_name": member.full_name if matched else None, "access_decision": decision,
               "reason": reason, "similarity_score": round(score, 5) if score is not None else None,
               "liveness_score": round(liveness_score, 5) if liveness_score is not None else None,
               "member_status": member.member_status if matched and member else None,
               "membership_status": (
                   "expired" if member and now.date() > member.membership_end else
                   "not_started" if member and now.date() < member.membership_start else
                   "active" if member else None
               ),
               "membership_end": member.membership_end.isoformat() if matched and member else None,
               "accessed_at": now.isoformat(),
               "device_id": "local-demo-device"}
        self.logs.insert(0, row)
        return row

    def revoke_consent(self, member_id: UUID) -> None:
        member = self.members[member_id]
        member.consent_granted = False
        member.embedding.clear()
        self.revoked_members.add(member_id)

    def restore_consent(self, member_id: UUID, *, consent_version: str) -> dict[str, object]:
        if consent_version != "buffalo-s-v1":
            raise ValueError("unsupported consent version")
        member = self.members[member_id]
        member.consent_granted = True
        member.embedding.clear()
        member.enrollment_status = "pending"
        self.revoked_members.discard(member_id)
        return {"member_status": member.member_status,
                "consent_date": datetime.now(timezone.utc).isoformat()}

    def list_members(self) -> list[dict[str, object]]:
        return [{"member_id": str(m.member_id), "full_name": m.full_name, "phone": m.phone,
                 "email": m.email, "member_status": m.member_status, "consent_granted": m.consent_granted,
                 "enrollment_status": m.enrollment_status,
                 "consent_status": "granted" if m.consent_granted else "revoked",
                 "membership_start": m.membership_start.isoformat(), "membership_end": m.membership_end.isoformat()}
                for m in self.members.values()]

    def update_status(self, member_id: UUID, status: str) -> dict[str, object]:
        if status not in {"active", "suspended", "banned"}: raise ValueError("invalid member status")
        member = self.members[member_id]
        member.member_status = status
        return {"member_id": member.member_id, "member_status": member.member_status}

    def dashboard(self) -> dict[str, object]:
        today = date.today().isoformat()
        allowed = sum(log["access_decision"] == "allowed" and log["accessed_at"][:10] == today
                      for log in self.logs)
        denied = sum(log["access_decision"] == "denied" and log["accessed_at"][:10] == today
                     for log in self.logs)
        return {"member_count": len(self.members), "allowed_today": allowed,
                "denied_today": denied, "recent_logs": self.logs[:20],
                "members": self.list_members()}

    def health(self) -> dict[str, object]:
        return {"status": "ok", "mode": "memory-demo", "member_limit": self.max_members}


class PostgresStore:
    """Thin SQL store. Refuses all database names except the approved scratch DB."""

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or os.getenv("GYMVISION_DATABASE_URL", "")
        if not self.dsn:
            raise RuntimeError("GYMVISION_DATABASE_URL is required")
        from psycopg.conninfo import conninfo_to_dict
        dbname = conninfo_to_dict(self.dsn).get("dbname")
        if dbname != "gymvision_ai_test":
            raise RuntimeError("Refusing database connection: GYMVISION_DATABASE_URL must target gymvision_ai_test")

    def connect(self):
        return psycopg.connect(self.dsn, row_factory=dict_row, connect_timeout=3)

    def health(self) -> dict[str, object]:
        with self.connect() as conn:
            row = conn.execute("SELECT current_database() AS database, version() AS version, (SELECT extversion FROM pg_extension WHERE extname='vector') AS vector_version").fetchone()
        if row["database"] != "gymvision_ai_test":
            raise RuntimeError("Connected database name did not match the approved test database")
        return {"status": "ok", "mode": "postgres-demo", "database": row["database"], "pgvector": row["vector_version"]}

    @staticmethod
    def _vector(values: list[float]) -> str:
        if len(values) != 512 or not all(math.isfinite(v) for v in values):
            raise ValueError("embedding must contain 512 finite values")
        return "[" + ",".join(format(v, ".9g") for v in values) + "]"

    def _demo_refs(self, conn) -> tuple[UUID, UUID]:
        plan = conn.execute("SELECT plan_id FROM membership_plans WHERE name='Local MVP demo' LIMIT 1").fetchone()
        if not plan:
            plan_id = uuid4()
            conn.execute("INSERT INTO membership_plans(plan_id,name,duration_days,price,is_active,created_at) VALUES (%s,'Local MVP demo',30,0,true,now())", (plan_id,))
        else: plan_id = plan["plan_id"]
        device = conn.execute("SELECT device_id FROM devices WHERE device_name='Local MVP device' LIMIT 1").fetchone()
        if not device:
            device_id = uuid4()
            conn.execute("INSERT INTO devices(device_id,device_name,device_location,device_status,api_key_hash,created_at) VALUES (%s,'Local MVP device','Local demo','online','local-demo-not-an-api-key',now())", (device_id,))
        else: device_id = device["device_id"]
        return plan_id, device_id

    def enroll(self, *, full_name: str, phone: str, email: str | None, consent_granted: bool,
               embedding: list[float], membership_start: date, membership_end: date,
               model_version: str = "synthetic-demo-v1") -> Member:
        if not consent_granted: raise ValueError("valid biometric consent is required")
        if membership_end < membership_start: raise ValueError("membership end precedes start")
        vector = self._vector(embedding)
        member_id, now = uuid4(), datetime.now(timezone.utc)
        try:
            with self.connect() as conn:
                plan_id, _ = self._demo_refs(conn)
                conn.execute("INSERT INTO members(member_id,full_name,phone,email,member_status,enrollment_status,created_at,updated_at) VALUES (%s,%s,%s,%s,'active','enrolled',%s,%s)", (member_id, full_name, phone, email, now, now))
                conn.execute("INSERT INTO memberships(membership_id,member_id,plan_id,start_date,end_date,membership_status,created_at) VALUES (%s,%s,%s,%s,%s,'active',%s)", (uuid4(), member_id, plan_id, membership_start, membership_end, now))
                conn.execute("INSERT INTO biometric_consents(consent_id,member_id,consent_status,consent_version,consent_date) VALUES (%s,%s,'granted','demo-v1',%s)", (uuid4(), member_id, now))
                conn.execute("INSERT INTO face_embeddings(embedding_id,member_id,embedding,model_version,is_active,created_at) VALUES (%s,%s,%s::vector,%s,true,%s)", (uuid4(), member_id, vector, model_version, now))
        except psycopg.errors.UniqueViolation as exc:
            duplicate_error = duplicate_member_error(exc.diag.constraint_name)
            if duplicate_error is None:
                raise
            raise duplicate_error from exc
        except psycopg.Error as exc:
            if getattr(exc, "sqlstate", None) in {"23P01", "22007"}:
                raise ValueError("membership dates overlap or are invalid") from exc
            raise
        return Member(member_id, full_name, phone, email, "active", True, membership_start,
                      membership_end, list(embedding), model_version)

    def prepare_enrollment(self, *, full_name: str, phone: str, email: str | None,
                           membership_start: date, membership_end: date) -> dict[str, object]:
        """Persist the member, membership and consent before camera capture begins."""
        if membership_end < membership_start:
            raise ValueError("membership end precedes start")
        member_id, now = uuid4(), datetime.now(timezone.utc)
        try:
            with self.connect() as conn:
                plan_id, _ = self._demo_refs(conn)
                conn.execute("""INSERT INTO members(member_id,full_name,phone,email,member_status,
                    enrollment_status,created_at,updated_at)
                    VALUES (%s,%s,%s,%s,'active','pending',%s,%s)""",
                    (member_id, full_name, phone, email, now, now))
                conn.execute("""INSERT INTO memberships(membership_id,member_id,plan_id,start_date,
                    end_date,membership_status,created_at)
                    VALUES (%s,%s,%s,%s,%s,'active',%s)""",
                    (uuid4(), member_id, plan_id, membership_start, membership_end, now))
                conn.execute("""INSERT INTO biometric_consents(consent_id,member_id,consent_status,
                    consent_version,consent_date) VALUES (%s,%s,'granted','buffalo-s-v1',%s)""",
                    (uuid4(), member_id, now))
        except psycopg.errors.UniqueViolation as exc:
            duplicate_error = duplicate_member_error(exc.diag.constraint_name)
            if duplicate_error is None:
                raise
            raise duplicate_error from exc
        return {"member_id": member_id, "full_name": full_name, "phone": phone, "email": email,
                "member_status": "active", "consent_granted": True,
                "membership_start": membership_start, "membership_end": membership_end,
                "enrollment_status": "pending"}

    def save_enrollment_embedding(self, member_id: UUID, embedding: list[float],
                                  model_version: str) -> None:
        with self.connect() as conn:
            consent = conn.execute("""SELECT consent_status,revoked_at FROM biometric_consents
                WHERE member_id=%s ORDER BY consent_date DESC LIMIT 1 FOR UPDATE""",
                (member_id,)).fetchone()
            if (not consent or consent["consent_status"] != "granted"
                    or consent["revoked_at"] is not None):
                raise ValueError("valid biometric consent is required before storing an embedding")
            vector = self._vector(embedding)
            conn.execute("UPDATE face_embeddings SET is_active=false WHERE member_id=%s AND model_version=%s",
                         (member_id, model_version))
            conn.execute("""INSERT INTO face_embeddings(embedding_id,member_id,embedding,
                model_version,is_active,created_at) VALUES (%s,%s,%s::vector,%s,true,now())""",
                (uuid4(), member_id, vector, model_version))
            conn.execute("UPDATE members SET enrollment_status='enrolled',updated_at=now() WHERE member_id=%s", (member_id,))

    def mark_enrollment_failed(self, member_id: UUID) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE members SET enrollment_status='failed',updated_at=now() WHERE member_id=%s", (member_id,))

    def get_enrollment_status(self, member_id: UUID) -> dict[str, object] | None:
        with self.connect() as conn:
            return conn.execute("""SELECT m.member_id,m.full_name,m.phone,m.email,m.member_status,
                m.enrollment_status,m.deleted_at,
                EXISTS(SELECT 1 FROM biometric_consents c WHERE c.member_id=m.member_id
                    AND c.consent_status='granted' AND c.revoked_at IS NULL) AS consent_granted
                FROM members m WHERE m.member_id=%s""", (member_id,)).fetchone()

    def check_in(self, embedding: list[float] | None, liveness_score: float | None, *,
                 member_id: UUID | None = None, threshold: float = .82,
                 liveness_threshold: float = .60,
                 model_version: str = "synthetic-demo-v1") -> dict[str, object]:
        with self.connect() as conn:
            _, device_id = self._demo_refs(conn)
            match = None
            score: float | None = None
            resolved_member_id: UUID | None = None
            decision, reason = "denied", "unknown_face"

            # Resolve identity and consent before validating or querying any vector.
            if member_id is not None:
                consent = conn.execute("""SELECT consent_status,revoked_at FROM biometric_consents
                    WHERE member_id=%s ORDER BY consent_date DESC LIMIT 1 FOR UPDATE""",
                    (member_id,)).fetchone()
                identity = conn.execute("""SELECT member_id,full_name,member_status
                    FROM members WHERE member_id=%s AND deleted_at IS NULL""",
                    (member_id,)).fetchone()
                if identity:
                    identity["consent_status"] = consent["consent_status"] if consent else None
                    identity["revoked_at"] = consent["revoked_at"] if consent else None
                    resolved_member_id = identity["member_id"]
                    match = identity
                    if (identity["consent_status"] != "granted"
                            or identity["revoked_at"] is not None):
                        reason = "consent_revoked" if identity["consent_status"] == "revoked" else "no_consent"
                    elif embedding is None:
                        reason = "unknown_face"
                    else:
                        vector = self._vector(embedding)
                        match = conn.execute("""SELECT m.member_id,m.full_name,m.member_status,
                            ms.start_date,ms.end_date,ms.membership_status,
                            (fe.embedding <=> %s::vector) AS distance
                            FROM members m JOIN face_embeddings fe ON fe.member_id=m.member_id
                            JOIN LATERAL (SELECT start_date,end_date,membership_status FROM memberships x
                                WHERE x.member_id=m.member_id ORDER BY x.created_at DESC LIMIT 1) ms ON true
                            WHERE m.member_id=%s AND m.deleted_at IS NULL AND fe.is_active=true
                                AND fe.model_version=%s
                            ORDER BY fe.created_at DESC LIMIT 1""", (vector, member_id, model_version)).fetchone()
                        if match:
                            score = 1.0 - float(match["distance"])
                            if score >= threshold:
                                if match["member_status"] != "active": reason = "member_suspended"
                                elif match["membership_status"] != "active": reason = "no_active_membership"
                                elif (match["start_date"] is None or match["end_date"] is None
                                      or not match["start_date"] <= date.today() <= match["end_date"]): reason = "membership_expired"
                                elif (liveness_score is not None
                                      and liveness_score < liveness_threshold): reason = "liveness_failed"
                                else: decision, reason = "allowed", "granted"
                            else:
                                resolved_member_id = None
                                reason = "unknown_face"
                        else:
                            resolved_member_id = None
                            reason, score = "unknown_face", None
            elif embedding is not None:
                vector = self._vector(embedding)
                match = conn.execute("""SELECT m.member_id,m.full_name,m.member_status,
                    ms.start_date,ms.end_date,ms.membership_status,
                    (fe.embedding <=> %s::vector) AS distance
                    FROM face_embeddings fe JOIN members m ON m.member_id=fe.member_id
                    JOIN biometric_consents bs ON bs.member_id=m.member_id
                        AND bs.consent_status='granted' AND bs.revoked_at IS NULL
                    JOIN LATERAL (SELECT start_date,end_date,membership_status FROM memberships x
                        WHERE x.member_id=m.member_id ORDER BY x.created_at DESC LIMIT 1) ms ON true
                    WHERE fe.is_active=true AND fe.model_version=%s AND m.deleted_at IS NULL
                    ORDER BY fe.embedding <=> %s::vector LIMIT 1""", (vector, model_version, vector)).fetchone()
                score = 1.0 - float(match["distance"]) if match else None
                if match and score is not None and score >= threshold:
                    resolved_member_id = match["member_id"]
                    if match["member_status"] != "active": reason = "member_suspended"
                    elif match["membership_status"] != "active": reason = "no_active_membership"
                    elif (match["start_date"] is None or match["end_date"] is None
                          or not match["start_date"] <= date.today() <= match["end_date"]): reason = "membership_expired"
                    elif (liveness_score is not None
                          and liveness_score < liveness_threshold): reason = "liveness_failed"
                    else: decision, reason = "allowed", "granted"
                else:
                    reason = ("liveness_failed" if liveness_score is not None
                              and liveness_score < liveness_threshold else "unknown_face")

            now, log_id = datetime.now(timezone.utc), uuid4()
            # The approved access_reason enum calls revoked consent "no_consent".
            stored_reason = "no_consent" if reason == "consent_revoked" else reason
            conn.execute("INSERT INTO access_logs(log_id,member_id,device_id,access_decision,reason,similarity_score,liveness_score,accessed_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)", (log_id, resolved_member_id, device_id, decision, stored_reason, score, liveness_score, now))
        resolved_membership_status = None
        membership_end = None
        if match and resolved_member_id:
            membership_end = match.get("end_date")
            resolved_membership_status = match.get("membership_status")
            if membership_end and date.today() > membership_end:
                resolved_membership_status = "expired"
            elif match.get("start_date") and date.today() < match["start_date"]:
                resolved_membership_status = "not_started"
        return {"log_id": str(log_id), "member_id": str(resolved_member_id) if resolved_member_id else None,
                "member_name": match["full_name"] if match and resolved_member_id else None,
                "member_status": match.get("member_status") if match and resolved_member_id else None,
                "membership_status": resolved_membership_status,
                "membership_end": membership_end.isoformat() if membership_end else None,
                "access_decision": decision, "reason": reason,
                "similarity_score": round(score, 5) if score is not None else None,
                "liveness_score": round(liveness_score, 5) if liveness_score is not None else None,
                "accessed_at": now.isoformat(), "device_id": str(device_id)}

    def list_members(self) -> list[dict[str, object]]:
        with self.connect() as conn:
            rows = conn.execute("""SELECT m.member_id,m.full_name,m.phone,m.email,m.member_status,
             m.enrollment_status,
             EXISTS(SELECT 1 FROM biometric_consents c WHERE c.member_id=m.member_id AND c.consent_status='granted' AND c.revoked_at IS NULL) AS consent_granted,
             (SELECT c.consent_status FROM biometric_consents c WHERE c.member_id=m.member_id
                ORDER BY c.consent_date DESC,c.consent_id DESC LIMIT 1) AS consent_status,
             ms.start_date AS membership_start,ms.end_date AS membership_end FROM members m
             LEFT JOIN LATERAL (SELECT start_date,end_date FROM memberships x WHERE x.member_id=m.member_id ORDER BY created_at DESC LIMIT 1) ms ON true
             WHERE m.deleted_at IS NULL ORDER BY m.created_at DESC""").fetchall()
        return [{**r, "member_id": str(r["member_id"]), "membership_start": r["membership_start"].isoformat() if r["membership_start"] else None,
                 "membership_end": r["membership_end"].isoformat() if r["membership_end"] else None} for r in rows]

    def dashboard(self) -> dict[str, object]:
        with self.connect() as conn:
            counts = conn.execute("SELECT count(*) FILTER(WHERE access_decision='allowed' AND accessed_at::date=current_date) allowed, count(*) FILTER(WHERE access_decision='denied' AND accessed_at::date=current_date) denied FROM access_logs").fetchone()
            rows = conn.execute("SELECT l.log_id,l.member_id,m.full_name AS member_name,l.access_decision,l.reason,l.similarity_score,l.liveness_score,l.accessed_at,l.device_id FROM access_logs l LEFT JOIN members m USING(member_id) ORDER BY accessed_at DESC LIMIT 20").fetchall()
        logs = [{**r, "log_id": str(r["log_id"]), "member_id": str(r["member_id"]) if r["member_id"] else None,
                 "device_id": str(r["device_id"]), "accessed_at": r["accessed_at"].isoformat()} for r in rows]
        members = self.list_members()
        return {"member_count": len(members), "allowed_today": counts["allowed"], "denied_today": counts["denied"], "recent_logs": logs, "members": members}

    def update_status(self, member_id: UUID, status: str) -> dict[str, object]:
        if status not in {"active", "suspended", "banned"}: raise ValueError("invalid member status")
        with self.connect() as conn:
            row = conn.execute("UPDATE members SET member_status=%s,updated_at=now() WHERE member_id=%s AND deleted_at IS NULL RETURNING member_id,member_status", (status, member_id)).fetchone()
        if not row: raise KeyError(member_id)
        return row

    def revoke_consent(self, member_id: UUID) -> None:
        with self.connect() as conn:
            row = conn.execute("UPDATE biometric_consents SET consent_status='revoked',revoked_at=now() WHERE member_id=%s AND consent_status='granted' RETURNING consent_id", (member_id,)).fetchone()
            if not row:
                exists = conn.execute("SELECT 1 FROM members WHERE member_id=%s", (member_id,)).fetchone()
                if not exists: raise KeyError(member_id)
            conn.execute("UPDATE face_embeddings SET is_active=false WHERE member_id=%s", (member_id,))

    def restore_consent(self, member_id: UUID, *, consent_version: str = "buffalo-s-v1") -> dict[str, object]:
        """Append a newly confirmed consent event without changing member status."""
        if len(consent_version) > 20:
            raise ValueError("consent version must be at most 20 characters")
        consent_id, audit_id = uuid4(), uuid4()
        with self.connect() as conn:
            previous = conn.execute("""SELECT consent_status,revoked_at FROM biometric_consents
                WHERE member_id=%s ORDER BY consent_date DESC LIMIT 1 FOR UPDATE""",
                (member_id,)).fetchone()
            member = conn.execute("""SELECT member_id,member_status FROM members
                WHERE member_id=%s AND deleted_at IS NULL FOR UPDATE""", (member_id,)).fetchone()
            if not member:
                raise KeyError(member_id)
            if not previous or previous["consent_status"] != "revoked":
                raise ValueError("Member must have revoked consent before it can be restored")

            consent = conn.execute("""INSERT INTO biometric_consents
                (consent_id,member_id,consent_status,consent_version,consent_date,revoked_at)
                VALUES (%s,%s,'granted',%s,now(),NULL)
                RETURNING consent_id,consent_date""",
                (consent_id, member_id, consent_version)).fetchone()
            # A new consent authorizes a new capture; historical vectors stay inactive.
            conn.execute("UPDATE face_embeddings SET is_active=false WHERE member_id=%s", (member_id,))
            conn.execute("UPDATE members SET enrollment_status='pending',updated_at=now() WHERE member_id=%s",
                         (member_id,))
            conn.execute("""INSERT INTO audit_logs
                (audit_id,action,entity_type,entity_id,details,created_at)
                VALUES (%s,'biometric_consent_restored','member',%s,%s,now())""",
                (audit_id, member_id, Jsonb({"consent_id": str(consent_id),
                                             "consent_version": consent_version,
                                             "confirmation": "operator_confirmed"})))
        return {"member_id": member_id, "member_status": member["member_status"],
                "consent_id": consent["consent_id"], "consent_date": consent["consent_date"],
                "consent_status": "granted", "enrollment_status": "pending"}
