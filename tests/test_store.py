"""Business-rule tests for the in-memory demo decision workflow."""

from datetime import date, timedelta
from uuid import uuid4

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from backend import app as app_module
from backend.store import DuplicateMemberError, MemoryStore
from backend.models import CheckInRequest, CheckInResponse, ConsentRevocation, EnrollmentRequest, MemberStatusUpdate


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


def test_check_in_response_contract_uses_access_decision_field() -> None:
    store = MemoryStore()
    enroll(store)

    response = CheckInResponse.model_validate(
        store.check_in([1.0] + [0.0] * 511, 0.95)
    )

    assert response.access_decision == "allowed"


def test_check_in_api_returns_store_field_name(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeStore:
        def check_in(self, embedding: list[float] | None, liveness_score: float,
                     *, member_id=None, model_version="synthetic-demo-v1") -> dict[str, object]:
            assert model_version == "synthetic-demo-v1"
            return {
                "access_decision": "denied",
                "reason": "unknown_face",
                "member_id": None,
                "member_name": None,
                "similarity_score": None,
                "liveness_score": liveness_score,
                "accessed_at": "2026-11-13T12:00:00+00:00",
            }

    monkeypatch.setattr(app_module, "get_store", lambda: FakeStore())
    payload = CheckInRequest(embedding=[1.0] + [0.0] * 511, liveness_score=0.95)

    response = app_module.check_in(payload, "test-session")

    assert response["access_decision"] == "denied"
    assert "decision" not in response


def test_real_enrollment_records_consent_before_opening_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    member_id = uuid4()
    events = []

    class FakeStore:
        def prepare_enrollment(self, **values):
            events.append("consent_recorded")
            return {**values, "member_id": member_id, "member_status": "active",
                    "consent_granted": True, "enrollment_status": "pending"}

        def save_enrollment_embedding(self, saved_member_id, embedding, model_version):
            events.append("embedding_stored")
            assert saved_member_id == member_id
            assert len(embedding) == 512
            assert model_version == "buffalo_s-512-v1"

    monkeypatch.setattr(app_module, "get_recognition_mode", lambda: "buffalo_s")
    monkeypatch.setattr(app_module, "get_store", lambda: FakeStore())
    monkeypatch.setattr(app_module, "get_recognizer", lambda: pytest.fail("camera must wait for preview confirmation"))
    payload = EnrollmentRequest(
        full_name="Research Member", phone="+201012345678", consent_granted=True,
        embedding=None, membership_start=date.today(), membership_end=date.today() + timedelta(days=30),
    )

    response = app_module.enroll_member(payload, "test-session")

    assert events == ["consent_recorded"]
    assert response["member_id"] == str(member_id)
    assert response["enrollment_status"] == "pending"


def test_real_enrollment_capture_uses_preview_session_after_consent(monkeypatch: pytest.MonkeyPatch) -> None:
    member_id = uuid4()
    events = []

    class FakeStore:
        def get_enrollment_status(self, requested_id):
            assert requested_id == member_id
            return {"deleted_at": None, "member_status": "active", "enrollment_status": "pending",
                    "consent_granted": True}

        def save_enrollment_embedding(self, saved_id, embedding, model_version):
            events.append("embedding_stored")
            assert saved_id == member_id
            assert len(embedding) == 512
            assert model_version == "buffalo_s-512-v1"

    class FakeRecognizer:
        def capture_preview_embedding(self):
            events.append("preview_captured")
            return [1.0] + [0.0] * 511

    monkeypatch.setattr(app_module, "get_recognition_mode", lambda: "buffalo_s")
    monkeypatch.setattr(app_module, "get_store", lambda: FakeStore())
    monkeypatch.setattr(app_module, "get_recognizer", lambda: FakeRecognizer())

    response = app_module.capture_pending_member(member_id, "test-session")

    assert events == ["preview_captured", "embedding_stored"]
    assert response == {"member_id": str(member_id), "enrollment_status": "enrolled"}


def test_restore_consent_requires_operator_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "get_recognition_mode", lambda: "buffalo_s")
    monkeypatch.setattr(app_module, "get_store", lambda: pytest.fail("do not restore without confirmation"))

    with pytest.raises(HTTPException) as error:
        app_module.restore_member_consent(uuid4(), ConsentRevocation(confirmed=False), "test-session")

    assert error.value.status_code == 422


def test_restore_consent_records_new_event_and_preserves_suspended_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    member_id = uuid4()
    consent_date = "2026-11-14T12:00:00+00:00"
    calls = []

    class FakeStore:
        def restore_consent(self, restored_member_id, *, consent_version):
            calls.append((restored_member_id, consent_version))
            return {"member_status": "suspended", "consent_date": consent_date}

    monkeypatch.setattr(app_module, "get_recognition_mode", lambda: "buffalo_s")
    monkeypatch.setattr(app_module, "get_store", lambda: FakeStore())

    result = app_module.restore_member_consent(
        member_id, ConsentRevocation(confirmed=True), "test-session"
    )

    assert calls == [(member_id, "buffalo-s-v1")]
    assert result == {"member_id": str(member_id), "consent_status": "granted",
                      "consent_date": consent_date, "member_status": "suspended",
                      "enrollment_status": "pending"}


def test_real_enrollment_rejects_missing_consent_before_store_or_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module, "get_recognition_mode", lambda: "buffalo_s")
    monkeypatch.setattr(app_module, "get_store", lambda: pytest.fail("must not create member without consent"))
    monkeypatch.setattr(app_module, "get_recognizer", lambda: pytest.fail("must not open camera without consent"))
    payload = EnrollmentRequest(
        full_name="Research Member", phone="+201012345678", consent_granted=False,
        membership_start=date.today(), membership_end=date.today() + timedelta(days=30),
    )

    with pytest.raises(HTTPException) as error:
        app_module.enroll_member(payload, "test-session")

    assert error.value.status_code == 422


def test_real_check_in_captures_embedding_without_fake_liveness(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = []

    class FakeStore:
        def check_in(self, embedding, liveness_score, *, threshold, model_version):
            captured.append((embedding, liveness_score, threshold, model_version))
            return {"access_decision": "denied", "reason": "unknown_face"}

    class FakeRecognizer:
        def capture_preview_embedding(self):
            return [1.0] + [0.0] * 511

    monkeypatch.setattr(app_module, "get_recognition_mode", lambda: "buffalo_s")
    monkeypatch.setattr(app_module, "get_store", lambda: FakeStore())
    monkeypatch.setattr(app_module, "get_recognizer", lambda: FakeRecognizer())
    payload = CheckInRequest(consent_confirmed=True)

    response = app_module.check_in(payload, "test-session")

    assert response["access_decision"] == "denied"
    assert captured[0][1] is None
    assert captured[0][2] == 0.82
    assert captured[0][3] == "buffalo_s-512-v1"


def test_real_check_in_requires_explicit_consent_before_opening_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module, "get_recognition_mode", lambda: "buffalo_s")
    monkeypatch.setattr(
        app_module,
        "get_recognizer",
        lambda: pytest.fail("camera must not open without explicit consent"),
    )

    with pytest.raises(HTTPException) as error:
        app_module.check_in(CheckInRequest(), "test-session")

    assert error.value.status_code == 403


def test_similarity_threshold_is_configurable_separately(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GYMVISION_SIMILARITY_THRESHOLD", "0.73")

    assert app_module.get_similarity_threshold() == 0.73


def test_invalid_similarity_threshold_fails_with_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GYMVISION_SIMILARITY_THRESHOLD", "1.4")

    with pytest.raises(HTTPException, match="GYMVISION_SIMILARITY_THRESHOLD"):
        app_module.get_similarity_threshold()


def test_memory_demo_enrollment_and_matching_are_ephemeral_and_consent_checked() -> None:
    demo = MemoryStore(max_members=3)
    start, end = valid_membership()
    pending = demo.prepare_enrollment(
        full_name="Demo Person", phone="+201011112222", email=None,
        membership_start=start, membership_end=end,
    )
    member_id = pending["member_id"]
    assert pending["enrollment_status"] == "pending"
    assert demo.members[member_id].embedding == []

    demo.save_enrollment_embedding(member_id, [1.0] + [0.0] * 511, "buffalo_s-512-v1")
    result = demo.check_in(
        [1.0] + [0.0] * 511, None, threshold=0.82,
        model_version="buffalo_s-512-v1",
    )

    assert result["access_decision"] == "allowed"
    assert result["member_id"] == str(member_id)
    assert result["membership_status"] == "active"
    assert demo.dashboard()["recent_logs"][0]["member_name"] == "Demo Person"


def test_memory_demo_denies_expired_membership_and_caps_members() -> None:
    demo = MemoryStore(max_members=3)
    expired_start = date.today() - timedelta(days=30)
    expired_end = date.today() - timedelta(days=1)
    pending = demo.prepare_enrollment(
        full_name="Expired Demo", phone="+201022223333", email=None,
        membership_start=expired_start, membership_end=expired_end,
    )
    member_id = pending["member_id"]
    demo.save_enrollment_embedding(member_id, [1.0] + [0.0] * 511, "buffalo_s-512-v1")
    result = demo.check_in(
        [1.0] + [0.0] * 511, None, threshold=0.82,
        model_version="buffalo_s-512-v1",
    )
    assert result["access_decision"] == "denied"
    assert result["reason"] == "membership_expired"
    assert result["membership_status"] == "expired"

    for index in (1, 2):
        demo.prepare_enrollment(
            full_name=f"Demo {index}", phone=f"+20103333444{index}", email=None,
            membership_start=date.today(), membership_end=date.today() + timedelta(days=30),
        )
    with pytest.raises(ValueError, match="at most 3 members"):
        demo.prepare_enrollment(
            full_name="Fourth Demo", phone="+201099998888", email=None,
            membership_start=date.today(), membership_end=date.today() + timedelta(days=30),
        )


def test_real_mode_config_exposes_distinct_detection_and_similarity_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module, "get_recognition_mode", lambda: "buffalo_s")
    monkeypatch.setenv("GYMVISION_SIMILARITY_THRESHOLD", "0.76")

    config = app_module.app_config("test-session")

    assert config["camera_source"] == "DroidCam Video via PyAV/DirectShow"
    assert config["detector_size"] == "320x320"
    assert config["detector_threshold"] == 0.50
    assert config["similarity_threshold"] == 0.76


def test_memory_demo_storage_is_capped_and_never_constructs_postgres(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module, "store", None)
    monkeypatch.setenv("GYMVISION_DEMO_STORAGE", "memory")
    monkeypatch.setattr(
        app_module,
        "PostgresStore",
        lambda: pytest.fail("memory demo must not construct PostgresStore"),
    )

    demo = app_module.get_store()

    assert isinstance(demo, MemoryStore)
    assert demo.max_members == 3
    assert demo.health()["mode"] == "memory-demo"


def test_get_recognizer_reuses_one_instance_with_named_camera_and_demo_detector_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class FakeRecognizer:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(app_module, "recognizer", None)
    monkeypatch.setattr(app_module, "BuffaloSRecognizer", FakeRecognizer)

    first = app_module.get_recognizer()
    second = app_module.get_recognizer()

    assert first is second
    assert len(calls) == 1
    assert calls[0]["camera_factory"] is app_module.create_droidcam_capture
    assert calls[0]["det_size"] == (320, 320)
    assert calls[0]["det_threshold"] == 0.50


def test_recapture_success_uses_buffalo_version_without_reactivating_suspended_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    member_id = uuid4()
    state = {"consent_granted": True, "member_status": "suspended", "enrollment_status": "pending"}

    class FakeStore:
        def get_enrollment_status(self, requested_id):
            assert requested_id == member_id
            return {"deleted_at": None, **state}

        def save_enrollment_embedding(self, requested_id, embedding, model_version):
            assert requested_id == member_id
            assert len(embedding) == 512
            assert model_version == "buffalo_s-512-v1"
            state["enrollment_status"] = "enrolled"

    class FakeRecognizer:
        def capture_preview_embedding(self):
            return [1.0] + [0.0] * 511

    monkeypatch.setattr(app_module, "get_recognition_mode", lambda: "buffalo_s")
    monkeypatch.setattr(app_module, "get_store", lambda: FakeStore())
    monkeypatch.setattr(app_module, "get_recognizer", lambda: FakeRecognizer())

    result = app_module.capture_pending_member(member_id, "test-session")

    assert result == {"member_id": str(member_id), "enrollment_status": "enrolled"}
    assert state["member_status"] == "suspended"
    assert state["enrollment_status"] == "enrolled"


def test_failed_recapture_keeps_restored_consent_and_marks_enrollment_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    member_id = uuid4()
    state = {"consent_granted": True, "member_status": "active", "enrollment_status": "pending"}

    class FakeStore:
        def get_enrollment_status(self, _member_id):
            return {"deleted_at": None, **state}

        def mark_enrollment_failed(self, _member_id):
            state["enrollment_status"] = "failed"

    class FakeRecognizer:
        def capture_preview_embedding(self):
            raise app_module.NoFaceDetected("No face detected")

    monkeypatch.setattr(app_module, "get_recognition_mode", lambda: "buffalo_s")
    monkeypatch.setattr(app_module, "get_store", lambda: FakeStore())
    monkeypatch.setattr(app_module, "get_recognizer", lambda: FakeRecognizer())

    with pytest.raises(HTTPException) as error:
        app_module.capture_pending_member(member_id, "test-session")

    assert error.value.status_code == 422
    assert state["consent_granted"] is True
    assert state["enrollment_status"] == "failed"


@pytest.mark.parametrize("host", ["127.0.0.1", "::1"])
def test_camera_preview_allows_loopback_clients(host: str) -> None:
    request = Request({"type": "http", "method": "POST", "headers": [],
                       "client": (host, 1234), "server": ("localhost", 8000),
                       "scheme": "http", "path": "/api/camera/session", "query_string": b""})

    app_module._require_local_camera_request(request)


def test_camera_preview_rejects_non_local_clients() -> None:
    request = Request({"type": "http", "method": "POST", "headers": [],
                       "client": ("192.0.2.20", 1234), "server": ("localhost", 8000),
                       "scheme": "http", "path": "/api/camera/session", "query_string": b""})

    with pytest.raises(HTTPException) as error:
        app_module._require_local_camera_request(request)

    assert error.value.status_code == 403


def test_embeddings_from_different_model_versions_never_match() -> None:
    store = MemoryStore()
    start, end = valid_membership()
    store.enroll(
        full_name="Synthetic Member", phone="+201012345678", email=None,
        consent_granted=True, embedding=[1.0] + [0.0] * 511,
        membership_start=start, membership_end=end,
        model_version="synthetic-demo-v1",
    )

    result = store.check_in(
        [1.0] + [0.0] * 511, 0.95, model_version="buffalo_s-512-v1"
    )

    assert result["access_decision"] == "denied"
    assert result["reason"] == "unknown_face"
    assert result["member_id"] is None


@pytest.mark.parametrize("status", ["suspended", "active"])
def test_member_status_api_accepts_store_dictionary(
    monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    member_id = uuid4()

    class FakeStore:
        def update_status(self, requested_member_id, requested_status):
            assert requested_member_id == member_id
            assert requested_status == status
            return {"member_id": member_id, "member_status": status}

    monkeypatch.setattr(app_module, "get_store", lambda: FakeStore())

    response = app_module.update_member_status(
        member_id, MemberStatusUpdate(member_status=status), "test-session"
    )

    assert response == {"member_id": str(member_id), "member_status": status}


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


def test_expired_membership_is_denied() -> None:
    store = MemoryStore()
    member = enroll(store)
    member.membership_end = date.today() - timedelta(days=1)

    result = store.check_in([1.0] + [0.0] * 511, 0.95)

    assert result["access_decision"] == "denied"
    assert result["reason"] == "membership_expired"


def test_revoking_consent_deletes_embedding_and_prevents_matching() -> None:
    store = MemoryStore()
    member = enroll(store)

    store.revoke_consent(member.member_id)
    result = store.check_in(None, 0.95, member_id=member.member_id)

    assert member.embedding == []
    assert member.consent_granted is False
    assert result["access_decision"] == "denied"
    assert result["reason"] == "consent_revoked"
    assert result["member_id"] == str(member.member_id)
    assert store.logs[0]["reason"] == "consent_revoked"


def test_missing_vector_is_denied_and_member_is_still_identified() -> None:
    store = MemoryStore()
    member = enroll(store)
    member.embedding.clear()

    result = store.check_in(None, 0.95, member_id=member.member_id)

    assert result["access_decision"] == "denied"
    assert result["reason"] == "unknown_face"
    assert result["member_id"] == str(member.member_id)
    assert store.logs[0]["member_id"] == str(member.member_id)


def test_enrollment_requires_consent() -> None:
    store = MemoryStore()
    start, end = valid_membership()

    with pytest.raises(ValueError, match="consent"):
        store.enroll(full_name="Demo Member", phone="+201012345678", email=None,
                     consent_granted=False, embedding=[1.0] + [0.0] * 511,
                     membership_start=start, membership_end=end)


@pytest.mark.parametrize(
    ("duplicate_field", "duplicate_value", "expected_message"),
    [
        ("phone", "+201012345678", "Phone number already exists"),
        ("email", "member@example.com", "Email address already exists"),
    ],
)
def test_duplicate_contacts_are_validated_independently(
    duplicate_field: str, duplicate_value: str, expected_message: str
) -> None:
    store = MemoryStore()
    start, end = valid_membership()
    store.enroll(
        full_name="Existing Member", phone="+201012345678", email="member@example.com",
        consent_granted=True, embedding=[1.0] + [0.0] * 511,
        membership_start=start, membership_end=end,
    )
    values = {
        "full_name": "Second Member",
        "phone": "+201099999999",
        "email": "other@example.com",
        "consent_granted": True,
        "embedding": [1.0] + [0.0] * 511,
        "membership_start": start,
        "membership_end": end,
    }
    values[duplicate_field] = duplicate_value

    with pytest.raises(DuplicateMemberError) as error:
        store.enroll(**values)

    assert error.value.field == duplicate_field
    assert str(error.value) == expected_message


@pytest.mark.parametrize(
    ("field", "message"),
    [("phone", "Phone number already exists"), ("email", "Email address already exists")],
)
def test_duplicate_contact_api_error_identifies_own_field(
    monkeypatch: pytest.MonkeyPatch, field: str, message: str
) -> None:
    class FakeStore:
        def enroll(self, **_values):
            raise DuplicateMemberError(field)

    monkeypatch.setattr(app_module, "get_recognition_mode", lambda: "simulated")
    monkeypatch.setattr(app_module, "get_store", lambda: FakeStore())
    payload = EnrollmentRequest(
        full_name="New Member", phone="+201012345678", email="new@example.com",
        consent_granted=True, embedding=[1.0] + [0.0] * 511,
        membership_start=date.today(), membership_end=date.today() + timedelta(days=30),
    )

    with pytest.raises(HTTPException) as error:
        app_module.enroll_member(payload, "test-session")

    assert error.value.status_code == 409
    assert error.value.detail == {"field": field, "message": message}
