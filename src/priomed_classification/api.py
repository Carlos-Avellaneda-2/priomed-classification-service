"""API REST del servicio de clasificación (contrato interno, Sección 6.3 del artículo).

Ejecutar:  uvicorn priomed_classification.api:app --reload
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .bootstrap import build_default_pipeline

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["pipeline"] = build_default_pipeline()
    yield
    _state.clear()


def allowed_origins() -> list[str]:
    """Orígenes autorizados a llamar la API desde un navegador (CORS).

    Se leen de PRIOMED_CORS_ORIGINS (lista separada por comas). Por defecto solo el
    frontend en desarrollo local; nunca se usa el comodín "*".
    """
    raw = os.getenv("PRIOMED_CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://localhost:4173"

app = FastAPI(title="PrioMed Classification Service", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_methods=["GET", "POST"],
    allow_headers=["Accept", "Content-Type"],
)


class ClassifyRequest(BaseModel):
    referral_id: str
    text: str = Field(min_length=1, description="Texto libre de la remisión")
    structured_urgency: int = Field(ge=0, le=2, description="0=baja, 1=media, 2=alta (registrada por el médico)")


class ClassifyResponse(BaseModel):
    referral_id: str
    priority: str
    source: str
    high_score: float
    alarm_signs: list[str]
    requires_human_review: bool


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    result = _state["pipeline"].classify(req.text, req.structured_urgency)
    return ClassifyResponse(
        referral_id=req.referral_id,
        priority=result.priority.name,
        source=result.source,
        high_score=round(result.high_score, 4),
        alarm_signs=result.alarm_signs,
        requires_human_review=result.requires_human_review,
    )
