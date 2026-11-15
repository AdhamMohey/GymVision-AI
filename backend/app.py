"""FastAPI application for the local, in-memory competition demo."""

from __future__ import annotations

import os
import secrets
import time
import ipaddress
import threading
from pathlib import Path
from typing import Annotated
from uuid import UUID

import bcrypt
import psycopg
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from ai.recognition import (
    BUFFALO_S_MODEL_VERSION,
    BuffaloSRecognizer,
    CameraSessionActive,
    CameraSessionNotActive,
    CaptureCancelled,
    MultipleFacesDetected,
    NoFaceDetected,
    RecognitionError,
)
from ai.droidcam_camera import create_droidcam_capture
from backend.models import CheckInRequest, ConsentRevocation, EnrollmentRequest, MemberStatusUpdate
from backend.store import DuplicateMemberError, MemoryStore, PostgresStore

app = FastAPI(title="GymVision AI MVP", version="0.1.0")
store: PostgresStore | MemoryStore | None = None
recognizer: BuffaloSRecognizer | None = None
_recognizer_lock = threading.Lock()
sessions: dict[str, float] = {}

def load_local_env() -> None:
    """Read simple KEY=VALUE entries without adding a dotenv dependency."""
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if not env_file.is_file():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("\"'")
        if key and key.replace("_", "").isalnum():
            os.environ.setdefault(key, value)


load_local_env()
dashboard_file = Path(__file__).resolve().parent.parent / "dashboard" / "index.html"
SYNTHETIC_MODEL_VERSION = "synthetic-demo-v1"
RECOGNITION_MODES = {"simulated", "buffalo_s"}

ADMIN_USERNAME = os.getenv("GYMVISION_ADMIN_USERNAME", "")
ADMIN_PASSWORD_HASH = os.getenv("GYMVISION_ADMIN_PASSWORD_HASH", "")
ADMIN_PASSWORD = os.getenv("GYMVISION_ADMIN_PASSWORD", "")
if not ADMIN_PASSWORD_HASH and ADMIN_PASSWORD:
    ADMIN_PASSWORD_HASH = bcrypt.hashpw(ADMIN_PASSWORD.encode(), bcrypt.gensalt()).decode()


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: Annotated[str, Field(min_length=1, max_length=50)]
    password: Annotated[str, Field(min_length=1, max_length=256)]


def require_admin(authorization: Annotated[str | None, Header()] = None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Admin login required")
    token = authorization.removeprefix("Bearer ")
    expires_at = sessions.get(token)
    if expires_at is None or expires_at < time.time():
        sessions.pop(token, None)
        raise HTTPException(status_code=401, detail="Session expired; log in again")
    return token


AdminSession = Annotated[str, Depends(require_admin)]


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(dashboard_file)


@app.get("/health")
def health() -> dict[str, object]:
    try:
        return get_store().health()
    except (psycopg.Error, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}") from exc


def get_storage_mode() -> str:
    mode = os.getenv("GYMVISION_DEMO_STORAGE", "postgres").strip().lower()
    if mode not in {"postgres", "memory"}:
        raise HTTPException(status_code=503, detail="GYMVISION_DEMO_STORAGE must be postgres or memory")
    return mode


def get_store() -> PostgresStore | MemoryStore:
    global store
    if store is None:
        try:
            store = MemoryStore(max_members=3) if get_storage_mode() == "memory" else PostgresStore()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return store


def get_recognition_mode() -> str:
    mode = os.getenv("GYMVISION_RECOGNITION_MODE", "simulated").strip().lower()
    if mode not in RECOGNITION_MODES:
        raise HTTPException(status_code=503, detail="GYMVISION_RECOGNITION_MODE must be simulated or buffalo_s")
    return mode


def get_recognizer() -> BuffaloSRecognizer:
    global recognizer
    if recognizer is None:
        with _recognizer_lock:
            if recognizer is None:
                recognizer = BuffaloSRecognizer(
                    camera_factory=create_droidcam_capture,
                    det_size=(320, 320),
                    det_threshold=0.50,
                )
    return recognizer


def get_similarity_threshold() -> float:
    """Read the cosine identity threshold; detector confidence is separate."""
    raw_value = os.getenv("GYMVISION_SIMILARITY_THRESHOLD", "0.82").strip()
    try:
        threshold = float(raw_value)
    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail="GYMVISION_SIMILARITY_THRESHOLD must be a number from 0 to 1",
        ) from exc
    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(
            status_code=503,
            detail="GYMVISION_SIMILARITY_THRESHOLD must be a number from 0 to 1",
        )
    return threshold


@app.post("/api/session")
def create_session(payload: LoginRequest) -> dict[str, str]:
    if not ADMIN_USERNAME or not ADMIN_PASSWORD_HASH:
        raise HTTPException(status_code=503, detail="Set GYMVISION_ADMIN_USERNAME and GYMVISION_ADMIN_PASSWORD")
    username_ok = secrets.compare_digest(payload.username, ADMIN_USERNAME)
    try:
        password_ok = bcrypt.checkpw(payload.password.encode(), ADMIN_PASSWORD_HASH.encode())
    except ValueError:
        password_ok = False
    if not username_ok or not password_ok:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = secrets.token_urlsafe(32)
    sessions[token] = time.time() + 8 * 60 * 60
    return {"access_token": token, "token_type": "bearer"}


@app.delete("/api/session", status_code=204)
def delete_session(token: AdminSession) -> None:
    sessions.pop(token, None)


@app.get("/api/dashboard")
def dashboard_summary(_: AdminSession) -> dict[str, object]:
    try:
        return get_store().dashboard()
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}") from exc


@app.get("/api/config")
def app_config(_: AdminSession) -> dict[str, object]:
    mode = get_recognition_mode()
    return {
        "recognition_mode": mode,
        "model_version": BUFFALO_S_MODEL_VERSION if mode == "buffalo_s" else SYNTHETIC_MODEL_VERSION,
        "liveness_detection": "unavailable" if mode == "buffalo_s" else "simulated_operator_score",
        "camera_source": "DroidCam Video via PyAV/DirectShow" if mode == "buffalo_s" else "synthetic",
        "detector_size": "320x320" if mode == "buffalo_s" else None,
        "detector_threshold": 0.50 if mode == "buffalo_s" else None,
        "similarity_threshold": get_similarity_threshold() if mode == "buffalo_s" else None,
        "storage_mode": get_storage_mode(),
    }


@app.post("/api/demo/vector")
def create_demo_vector(_: AdminSession) -> dict[str, list[float]]:
    """Create random synthetic data for UI demonstrations; never a face embedding."""
    if get_recognition_mode() != "simulated":
        raise HTTPException(status_code=409, detail="Synthetic vectors are disabled in buffalo_s mode")
    import random

    vector = [random.SystemRandom().gauss(0, 1) for _ in range(512)]
    return {"embedding": vector}


@app.post("/api/members", status_code=201)
def enroll_member(payload: EnrollmentRequest, _: AdminSession) -> dict[str, object]:
    if not payload.consent_granted:
        raise HTTPException(status_code=422, detail="Valid biometric consent is required")
    mode = get_recognition_mode()
    if mode == "buffalo_s":
        if get_storage_mode() != "memory" and payload.phone is None:
            raise HTTPException(status_code=422, detail="Phone is required when using database storage")
        if payload.embedding is not None:
            raise HTTPException(status_code=422, detail="Real mode captures embeddings locally; do not submit an embedding")
        try:
            pending = get_store().prepare_enrollment(
                full_name=payload.full_name, phone=payload.phone, email=payload.email,
                membership_start=payload.membership_start, membership_end=payload.membership_end,
            )
        except DuplicateMemberError as exc:
            raise HTTPException(status_code=409, detail={"field": exc.field, "message": str(exc)}) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (psycopg.Error, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=f"Database error: {exc}") from exc
        return {**pending, "member_id": str(pending["member_id"]), "enrollment_status": "pending"}

    if payload.phone is None:
        raise HTTPException(status_code=422, detail="Phone is required in simulated mode")
    if payload.embedding is None:
        raise HTTPException(status_code=422, detail="embedding is required in simulated mode")
    try:
        member = get_store().enroll(**payload.model_dump(), model_version=SYNTHETIC_MODEL_VERSION)
    except DuplicateMemberError as exc:
        raise HTTPException(status_code=409, detail={"field": exc.field, "message": str(exc)}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (psycopg.Error, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {exc}") from exc
    return {"member_id": str(member.member_id), "full_name": member.full_name,
            "phone": member.phone, "member_status": member.member_status,
            "consent_granted": member.consent_granted,
            "membership_start": member.membership_start.isoformat(),
            "membership_end": member.membership_end.isoformat()}


def _mark_enrollment_failed(member_id: UUID) -> None:
    try:
        get_store().mark_enrollment_failed(member_id)
    except (psycopg.Error, RuntimeError):
        # The original enrollment/capture failure is more useful to the caller.
        pass


def _require_local_camera_request(request: Request) -> None:
    """Restrict raw camera preview frames to requests from this computer."""
    host = request.client.host if request.client else ""
    try:
        local = ipaddress.ip_address(host).is_loopback
    except ValueError:
        local = False
    if not local:
        raise HTTPException(status_code=403, detail="Camera preview is available only on this computer")


@app.post("/api/camera/session")
def start_camera_session(request: Request, _: AdminSession) -> dict[str, str]:
    _require_local_camera_request(request)
    if get_recognition_mode() != "buffalo_s":
        raise HTTPException(status_code=409, detail="Live camera preview is available only in buffalo_s mode")
    try:
        get_recognizer().start_preview()
    except CameraSessionActive as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RecognitionError as exc:
        raise HTTPException(status_code=503, detail=f"Camera unavailable: {exc}") from exc
    return {"status": "camera_ready"}


@app.get("/api/camera/session/frame")
def camera_session_frame(request: Request, _: AdminSession) -> Response:
    _require_local_camera_request(request)
    try:
        frame = get_recognizer().preview_jpeg()
    except CameraSessionNotActive as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RecognitionError as exc:
        get_recognizer().cancel_preview()
        raise HTTPException(status_code=503, detail=f"Camera preview failed: {exc}") from exc
    return Response(frame, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.delete("/api/camera/session")
def cancel_camera_session(request: Request, _: AdminSession) -> dict[str, str]:
    _require_local_camera_request(request)
    get_recognizer().cancel_preview()
    return {"status": "camera_closed"}


@app.post("/api/members/{member_id}/capture")
def capture_pending_member(member_id: UUID, _: AdminSession) -> dict[str, object]:
    if get_recognition_mode() != "buffalo_s":
        raise HTTPException(status_code=409, detail="Capture retries are available only in buffalo_s mode")
    try:
        member = get_store().get_enrollment_status(member_id)
    except (psycopg.Error, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {exc}") from exc
    if member is None or member["deleted_at"] is not None:
        raise HTTPException(status_code=404, detail="Member not found")
    if member["member_status"] == "banned":
        raise HTTPException(status_code=409, detail="A banned member cannot be enrolled for face recognition")
    if member["enrollment_status"] not in {"pending", "failed"}:
        raise HTTPException(status_code=409, detail="Member does not have a pending face enrollment")
    if not member["consent_granted"]:
        raise HTTPException(status_code=403, detail="Valid biometric consent is required before camera capture")
    try:
        embedding = get_recognizer().capture_preview_embedding()
        get_store().save_enrollment_embedding(member_id, embedding, BUFFALO_S_MODEL_VERSION)
    except (CameraSessionNotActive, CaptureCancelled) as exc:
        _mark_enrollment_failed(member_id)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (NoFaceDetected, MultipleFacesDetected) as exc:
        _mark_enrollment_failed(member_id)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RecognitionError as exc:
        _mark_enrollment_failed(member_id)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        _mark_enrollment_failed(member_id)
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except psycopg.Error as exc:
        _mark_enrollment_failed(member_id)
        raise HTTPException(status_code=503, detail=f"Database error: {exc}") from exc
    return {"member_id": str(member_id), "enrollment_status": "enrolled"}


@app.post("/api/check-in")
def check_in(payload: CheckInRequest, _: AdminSession) -> dict[str, object]:
    mode = get_recognition_mode()
    if mode == "buffalo_s":
        if payload.embedding is not None or payload.member_id is not None:
            raise HTTPException(status_code=422, detail="Real mode captures a live face locally; do not submit embedding or member_id")
        if not payload.consent_confirmed:
            raise HTTPException(status_code=403, detail="Explicit participant consent is required before recognition")
        try:
            embedding = get_recognizer().capture_preview_embedding()
            return get_store().check_in(
                embedding,
                None,
                threshold=get_similarity_threshold(),
                model_version=BUFFALO_S_MODEL_VERSION,
            )
        except CameraSessionNotActive as exc:
            raise HTTPException(status_code=409, detail="Start the camera preview before evaluating access") from exc
        except CaptureCancelled as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except NoFaceDetected as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except MultipleFacesDetected as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (RecognitionError, psycopg.Error, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=f"Recognition unavailable: {exc}") from exc

    if payload.liveness_score is None:
        raise HTTPException(status_code=422, detail="liveness_score is required in simulated mode")
    if payload.embedding is None and payload.member_id is None:
        raise HTTPException(status_code=422, detail="embedding or member_id is required in simulated mode")
    try:
        return get_store().check_in(payload.embedding, payload.liveness_score,
                                    member_id=payload.member_id,
                                    model_version=SYNTHETIC_MODEL_VERSION)
    except (psycopg.Error, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {exc}") from exc


@app.patch("/api/members/{member_id}/status")
def update_member_status(member_id: UUID, payload: MemberStatusUpdate,
                         _: AdminSession) -> dict[str, object]:
    try:
        member = get_store().update_status(member_id, payload.member_status)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Member not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (psycopg.Error, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {exc}") from exc
    return {"member_id": str(member["member_id"]), "member_status": member["member_status"]}


@app.post("/api/members/{member_id}/revoke-consent")
def revoke_consent(member_id: UUID, payload: ConsentRevocation,
                   _: AdminSession) -> dict[str, str]:
    if not payload.confirmed:
        raise HTTPException(status_code=422, detail="Explicit confirmation is required")
    try:
        get_store().revoke_consent(member_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Member not found") from exc
    except (psycopg.Error, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {exc}") from exc
    return {"member_id": str(member_id), "consent_status": "revoked",
            "embedding_status": "deactivated; retained in database for audit-safe consent withdrawal"}


@app.post("/api/members/{member_id}/restore-consent")
def restore_member_consent(member_id: UUID, payload: ConsentRevocation,
                           _: AdminSession) -> dict[str, object]:
    if get_recognition_mode() != "buffalo_s":
        raise HTTPException(status_code=409, detail="Face ID setup is available only in buffalo_s mode")
    if not payload.confirmed:
        raise HTTPException(status_code=422, detail="Confirm that the member has provided consent again")
    try:
        restored = get_store().restore_consent(member_id, consent_version="buffalo-s-v1")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Member not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (psycopg.Error, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {exc}") from exc
    return {"member_id": str(member_id), "consent_status": "granted",
            "consent_date": restored["consent_date"],
            "member_status": restored["member_status"],
            "enrollment_status": "pending"}
