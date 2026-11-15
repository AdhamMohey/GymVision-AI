"""Safe unit checks for the PostgreSQL adapter configuration and vectors."""

import pytest
from uuid import uuid4
from datetime import date, timedelta
from datetime import datetime, timezone

from backend.store import PostgresStore, duplicate_member_error


def test_postgres_store_refuses_any_database_except_scratch() -> None:
    with pytest.raises(RuntimeError, match="must target gymvision_ai_test"):
        PostgresStore("postgresql://local:local@localhost:5432/gymvision_ai")


def test_postgres_store_accepts_scratch_database_url() -> None:
    store = PostgresStore("postgresql://local:local@localhost:5432/gymvision_ai_test")
    assert store.dsn.endswith("/gymvision_ai_test")


def test_vector_encoder_requires_512_finite_values() -> None:
    encoded = PostgresStore._vector([0.25] + [0.0] * 511)
    assert encoded.startswith("[0.25,0,0")
    with pytest.raises(ValueError, match="512 finite"):
        PostgresStore._vector([1.0])
    with pytest.raises(ValueError, match="512 finite"):
        PostgresStore._vector([float("nan")] + [0.0] * 511)


@pytest.mark.parametrize(
    ("constraint", "field", "message"),
    [
        ("members_phone_non_deleted_uidx", "phone", "Phone number already exists"),
        ("members_email_non_deleted_uidx", "email", "Email address already exists"),
    ],
)
def test_duplicate_index_maps_to_the_matching_contact_field(constraint, field, message):
    error = duplicate_member_error(constraint)

    assert error is not None
    assert error.field == field
    assert str(error) == message


def test_revoked_consent_access_attempt_is_logged_without_reading_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    member_id = uuid4()

    class Result:
        def __init__(self, row=None):
            self.row = row

        def fetchone(self):
            return self.row

    class Connection:
        def __init__(self):
            self.queries = []
            self.access_log_values = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, query, params=None):
            self.queries.append(query)
            if "SELECT plan_id" in query or "SELECT device_id" in query:
                return Result()
            if "FROM biometric_consents" in query:
                return Result({"consent_status": "revoked", "revoked_at": None})
            if "FROM members WHERE" in query:
                return Result({"member_id": member_id, "full_name": "Adham Mohey",
                               "member_status": "active"})
            if "INSERT INTO access_logs" in query:
                self.access_log_values = params
            return Result()

    fake_connection = Connection()
    store = PostgresStore("postgresql://local:local@localhost:5432/gymvision_ai_test")
    monkeypatch.setattr(store, "connect", lambda: fake_connection)

    result = store.check_in(None, 0.95, member_id=member_id)

    assert result["access_decision"] == "denied"
    assert result["reason"] == "consent_revoked"
    assert result["member_id"] == str(member_id)
    assert fake_connection.access_log_values[1] == member_id
    assert fake_connection.access_log_values[3:5] == ("denied", "no_consent")
    assert not any("face_embeddings" in query for query in fake_connection.queries)


def test_postgres_matching_query_is_scoped_to_requested_model_version(monkeypatch: pytest.MonkeyPatch) -> None:
    member_id = uuid4()
    version = "buffalo_s-512-v1"

    class Result:
        def __init__(self, row=None):
            self.row = row

        def fetchone(self):
            return self.row

    class Connection:
        def __init__(self):
            self.match_query = None
            self.match_params = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, query, params=None):
            if "SELECT plan_id" in query or "SELECT device_id" in query:
                return Result({"plan_id": uuid4()} if "plan_id" in query else {"device_id": uuid4()})
            if "ORDER BY fe.embedding <=>" in query:
                self.match_query, self.match_params = query, params
                today = date.today()
                return Result({"member_id": member_id, "full_name": "Consented member",
                               "member_status": "active", "start_date": today - timedelta(days=1),
                               "end_date": today + timedelta(days=1), "membership_status": "active",
                               "distance": 0.0})
            return Result()

    connection = Connection()
    store = PostgresStore("postgresql://local:local@localhost:5432/gymvision_ai_test")
    monkeypatch.setattr(store, "connect", lambda: connection)

    result = store.check_in([1.0] + [0.0] * 511, None, model_version=version)

    assert result["access_decision"] == "allowed"
    assert result["liveness_score"] is None
    assert "fe.model_version=%s" in connection.match_query
    assert "bs.consent_status='granted'" in connection.match_query
    assert "bs.revoked_at IS NULL" in connection.match_query
    assert connection.match_params[1] == version


def test_restore_consent_appends_event_disables_old_vectors_and_preserves_status(monkeypatch: pytest.MonkeyPatch) -> None:
    member_id = uuid4()
    restored_at = datetime.now(timezone.utc)

    class Result:
        def __init__(self, row=None):
            self.row = row

        def fetchone(self):
            return self.row

    class Connection:
        def __init__(self):
            self.queries = []
            self.consent_params = None
            self.audit_params = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, query, params=None):
            self.queries.append(query)
            if "SELECT member_id,member_status FROM members" in query:
                return Result({"member_id": member_id, "member_status": "suspended"})
            if "SELECT consent_status,revoked_at FROM biometric_consents" in query:
                return Result({"consent_status": "revoked", "revoked_at": restored_at})
            if "INSERT INTO biometric_consents" in query:
                self.consent_params = params
                return Result({"consent_id": params[0], "consent_date": restored_at})
            if "INSERT INTO audit_logs" in query:
                self.audit_params = params
            return Result()

    connection = Connection()
    store = PostgresStore("postgresql://local:local@localhost:5432/gymvision_ai_test")
    monkeypatch.setattr(store, "connect", lambda: connection)

    result = store.restore_consent(member_id)

    assert result["consent_status"] == "granted"
    assert result["consent_date"] == restored_at
    assert result["member_status"] == "suspended"
    assert result["enrollment_status"] == "pending"
    assert connection.consent_params == (result["consent_id"], member_id, "buffalo-s-v1")
    assert "VALUES (%s,%s,'granted',%s,now(),NULL)" in next(
        query for query in connection.queries if "INSERT INTO biometric_consents" in query
    )
    assert any("UPDATE face_embeddings SET is_active=false" in query for query in connection.queries)
    assert any("UPDATE members SET enrollment_status='pending'" in query for query in connection.queries)
    assert not any("UPDATE members SET member_status" in query for query in connection.queries)
    assert connection.audit_params[1] == member_id
    assert "biometric_consent_restored" in next(
        query for query in connection.queries if "INSERT INTO audit_logs" in query
    )
    assert connection.audit_params[2].obj["confirmation"] == "operator_confirmed"


def test_restore_consent_refuses_to_duplicate_an_active_consent(monkeypatch: pytest.MonkeyPatch) -> None:
    member_id = uuid4()

    class Result:
        def __init__(self, row=None):
            self.row = row

        def fetchone(self):
            return self.row

    class Connection:
        def __init__(self):
            self.inserted = False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, query, _params=None):
            if "SELECT consent_status,revoked_at FROM biometric_consents" in query:
                return Result({"consent_status": "granted", "revoked_at": None})
            if "SELECT member_id,member_status FROM members" in query:
                return Result({"member_id": member_id, "member_status": "active"})
            if "INSERT INTO biometric_consents" in query:
                self.inserted = True
            return Result()

    connection = Connection()
    store = PostgresStore("postgresql://local:local@localhost:5432/gymvision_ai_test")
    monkeypatch.setattr(store, "connect", lambda: connection)

    with pytest.raises(ValueError, match="revoked consent"):
        store.restore_consent(member_id)

    assert connection.inserted is False
