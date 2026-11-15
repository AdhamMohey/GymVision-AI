"""Validated API models for the local demo."""

from datetime import date
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

Embedding = Annotated[list[float], Field(min_length=512, max_length=512)]
Phone = Annotated[str, StringConstraints(pattern=r"^\+[1-9]\d{7,14}$")]


class EnrollmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    full_name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=150)]
    phone: Phone | None = None
    email: Annotated[str | None, StringConstraints(strip_whitespace=True, max_length=255)] = None
    consent_granted: bool
    embedding: Embedding | None = None
    membership_start: date
    membership_end: date

    @field_validator("embedding")
    @classmethod
    def embedding_must_be_finite(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return None
        import math

        if not all(math.isfinite(item) for item in value):
            raise ValueError("embedding values must be finite")
        if not any(value):
            raise ValueError("embedding cannot be a zero vector")
        return value

    @field_validator("membership_end")
    @classmethod
    def end_not_before_start(cls, value: date, info: object) -> date:
        start = getattr(info, "data", {}).get("membership_start")
        if start is not None and value < start:
            raise ValueError("membership_end must be on or after membership_start")
        return value

    @field_validator("email")
    @classmethod
    def email_has_basic_shape(cls, value: str | None) -> str | None:
        if value is not None and ("@" not in value or "." not in value.rsplit("@", 1)[-1]):
            raise ValueError("email must have a valid basic format")
        return value


class CheckInRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    member_id: UUID | None = None
    embedding: Embedding | None = None
    liveness_score: Annotated[float, Field(ge=0, le=1)] | None = None
    consent_confirmed: bool = False

    @field_validator("embedding")
    @classmethod
    def embedding_must_be_finite(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return None
        import math

        if not all(math.isfinite(item) for item in value):
            raise ValueError("embedding values must be finite")
        if not any(value):
            raise ValueError("embedding cannot be a zero vector")
        return value


class MemberStatusUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    member_status: str


class ConsentRevocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirmed: bool


class CheckInResponse(BaseModel):
    access_decision: str
    reason: str
    member_id: UUID | None
    member_name: str | None
    similarity_score: float | None
    liveness_score: float | None
    accessed_at: str
