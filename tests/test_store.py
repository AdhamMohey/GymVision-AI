"""Business-rule tests for the in-memory demo decision workflow."""

from datetime import date, timedelta

import pytest

from backend.store import MemoryStore


def valid_membership() -> tuple[date, date]:
    today = date.today()
    return today - timedelta(days=1), today + timedelta(days=30)


def enroll(store: MemoryStore, embedding: list[float] | None = None):
    start, end = valid_membership()
    return store.enroll(
        full_name="Demo Member", phone="+201012345678", email=None,
        consent_granted=True, embedding=embedding or [1.0] + [0.0] * 511,
        membership_start=start, membership_end=end,
    )


def test_matching_active_consented_member_is_allowed() -> None:
    store = MemoryStore()
    member = enroll(store)

    result = store.check_in([1.0] + [0.0] * 511, 0.95)

    assert result["access_decision"] == "allowed"
    assert result["reason"] == "granted"
    assert result["member_id"] == str(member.member_id)


def test_unknown_vector_is_denied_and_logged_without_member() -> None:
    store = MemoryStore()
    enroll(store)

    result = store.check_in([0.0, 1.0] + [0.0] * 510, 0.95)

    assert result["access_decision"] == "denied"
    assert result["reason"] == "unknown_face"
    assert result["member_id"] is None
    assert len(store.logs) == 1


def test_low_liveness_denies_access() -> None:
    store = MemoryStore()
    enroll(store)

    result = store.check_in([1.0] + [0.0] * 511, 0.2)

    assert result["access_decision"] == "denied"
    assert result["reason"] == "liveness_failed"


def test_suspended_member_is_denied() -> None:
    store = MemoryStore()
    member = enroll(store)
    store.update_status(member.member_id, "suspended")

    result = store.check_in([1.0] + [0.0] * 511, 0.95)

    assert result["access_decision"] == "denied"
    assert result["reason"] == "member_suspended"


def test_revoking_consent_deletes_embedding_and_prevents_matching() -> None:
    store = MemoryStore()
    member = enroll(store)

    store.revoke_consent(member.member_id)
    result = store.check_in([1.0] + [0.0] * 511, 0.95)

    assert member.embedding == []
    assert member.consent_granted is False
    assert result["reason"] == "unknown_face"
    assert result["member_id"] is None


def test_enrollment_requires_consent() -> None:
    store = MemoryStore()
    start, end = valid_membership()

    with pytest.raises(ValueError, match="consent"):
        store.enroll(full_name="Demo Member", phone="+201012345678", email=None,
                     consent_granted=False, embedding=[1.0] + [0.0] * 511,
                     membership_start=start, membership_end=end)
