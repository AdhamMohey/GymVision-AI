"""FastAPI application for the local, in-memory competition demo."""

from __future__ import annotations

import os
import secrets
import time
from pathlib import Path
from typing import Annotated
from uuid import UUID

import bcrypt
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from backend.models import CheckInRequest, ConsentRevocation, EnrollmentRequest, MemberStatusUpdate
from backend.store import MemoryStore

app = FastAPI(title="GymVision AI MVP", version="0.1.0")
store = MemoryStore()
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
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "in-memory-demo"}


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
    allowed = sum(row["access_decision"] == "allowed" for row in store.logs)
    denied = len(store.logs) - allowed
    return {"member_count": len(store.members), "allowed_today": allowed,
            "denied_today": denied, "recent_logs": store.logs[:20],
            "members": store.list_members()}


@app.post("/api/demo/vector")
def create_demo_vector(_: AdminSession) -> dict[str, list[float]]:
    """Create random synthetic data for UI demonstrations; never a face embedding."""
    import random

    vector = [random.SystemRandom().gauss(0, 1) for _ in range(512)]
    return {"embedding": vector}


@app.post("/api/members", status_code=201)
def enroll_member(payload: EnrollmentRequest, _: AdminSession) -> dict[str, object]:
    if not payload.consent_granted:
        raise HTTPException(status_code=422, detail="Valid biometric consent is required")
    try:
        member = store.enroll(**payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"member_id": str(member.member_id), "full_name": member.full_name,
            "phone": member.phone, "member_status": member.member_status,
            "consent_granted": member.consent_granted,
            "membership_start": member.membership_start.isoformat(),
            "membership_end": member.membership_end.isoformat()}


@app.post("/api/check-in")
def check_in(payload: CheckInRequest, _: AdminSession) -> dict[str, object]:
    return store.check_in(payload.embedding, payload.liveness_score)


@app.patch("/api/members/{member_id}/status")
def update_member_status(member_id: UUID, payload: MemberStatusUpdate,
                         _: AdminSession) -> dict[str, object]:
    try:
        member = store.update_status(member_id, payload.member_status)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Member not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"member_id": str(member.member_id), "member_status": member.member_status}


@app.post("/api/members/{member_id}/revoke-consent")
def revoke_consent(member_id: UUID, payload: ConsentRevocation,
                   _: AdminSession) -> dict[str, str]:
    if not payload.confirmed:
        raise HTTPException(status_code=422, detail="Explicit confirmation is required")
    try:
        store.revoke_consent(member_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Member not found") from exc
    return {"member_id": str(member_id), "consent_status": "revoked",
            "embedding_status": "deleted from demo memory"}
