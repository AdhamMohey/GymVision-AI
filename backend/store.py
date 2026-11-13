"""In-memory data store for explicitly non-persistent competition demos."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from uuid import UUID, uuid4


@dataclass
class Member:
    member_id: UUID
    full_name: str
    phone: str
    email: str | None
    member_status: str
    consent_granted: bool
    membership_start: date
    membership_end: date
    embedding: list[float]
    model_version: str = "demo-vector-v1"


@dataclass
class MemoryStore:
    members: dict[UUID, Member] = field(default_factory=dict)
    logs: list[dict[str, object]] = field(default_factory=list)
    revoked_members: set[UUID] = field(default_factory=set)

    def enroll(self, *, full_name: str, phone: str, email: str | None,
               consent_granted: bool, embedding: list[float],
               membership_start: date, membership_end: date) -> Member:
        if not consent_granted:
            raise ValueError("valid biometric consent is required")
        if any(m.phone == phone for m in self.members.values()):
            raise ValueError("phone already exists")
        if email and any(m.email == email for m in self.members.values()):
            raise ValueError("email already exists")
        if membership_end < membership_start:
            raise ValueError("membership end precedes start")
        member = Member(uuid4(), full_name, phone, email, "active", True,
                        membership_start, membership_end, list(embedding))
        self.members[member.member_id] = member
        return member

    @staticmethod
    def cosine_similarity(left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)

    def check_in(self, embedding: list[float], liveness_score: float,
                 *, threshold: float = 0.82, liveness_threshold: float = 0.60) -> dict[str, object]:
        now = datetime.now(timezone.utc)
        matches = [
            (self.cosine_similarity(embedding, member.embedding), member)
            for member in self.members.values()
            if member.consent_granted and member.member_id not in self.revoked_members
        ]
        best_score, member = max(matches, key=lambda item: item[0], default=(None, None))
        reason = "unknown_face"
        decision = "denied"
        matched_id: UUID | None = None

        if liveness_score < liveness_threshold:
            reason = "liveness_failed"
        elif member is None or best_score is None or best_score < threshold:
            reason = "unknown_face"
        else:
            matched_id = member.member_id
            today = now.date()
            if not member.consent_granted:
                reason = "no_consent"
            elif member.member_status != "active":
                reason = "member_suspended"
            elif today < member.membership_start or today > member.membership_end:
                reason = "membership_expired"
            else:
                decision, reason = "allowed", "granted"

        record: dict[str, object] = {
            "log_id": str(uuid4()), "member_id": str(matched_id) if matched_id else None,
            "member_name": member.full_name if matched_id and member else None,
            "access_decision": decision, "reason": reason,
            "similarity_score": round(best_score, 5) if best_score is not None else None,
            "liveness_score": round(liveness_score, 5), "accessed_at": now.isoformat(),
            "device_id": "local-demo-device",
        }
        self.logs.insert(0, record)
        return record

    def revoke_consent(self, member_id: UUID) -> None:
        member = self.members[member_id]
        member.consent_granted = False
        member.embedding.clear()
        self.revoked_members.add(member_id)

    def list_members(self) -> list[dict[str, object]]:
        return [
            {"member_id": str(m.member_id), "full_name": m.full_name,
             "phone": m.phone, "email": m.email, "member_status": m.member_status,
             "consent_granted": m.consent_granted,
             "membership_start": m.membership_start.isoformat(),
             "membership_end": m.membership_end.isoformat()}
            for m in self.members.values()
        ]

    def update_status(self, member_id: UUID, status: str) -> Member:
        if status not in {"active", "suspended", "banned"}:
            raise ValueError("invalid member status")
        member = self.members[member_id]
        member.member_status = status
        return member
