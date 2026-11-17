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
    referral_id: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=5000, description="Texto libre de la remisión")
    structured_urgency: int = Field(ge=0, le=2, description="0=baja, 1=media, 2=alta (registrada por el médico)")


class AlarmEvidence(BaseModel):
    sign: str = Field(description="Señal de alarma canónica, p. ej. dolor_toracico")
    phrase: str = Field(description="Frase del diccionario que coincidió (normalizada, sin tildes)")
    negated: bool = Field(description="True si la frase está dentro del alcance de una negación")


class ClassifyResponse(BaseModel):
    referral_id: str
    priority: str
    source: str
    high_score: float
    high_threshold: float | None = Field(description="Umbral Neyman-Pearson sobre P(HIGH) usado por el modelo")
    alarm_signs: list[str]
    evidence: list[AlarmEvidence] = Field(description="Frases de alarma encontradas, en orden de aparición")
    requires_human_review: bool


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    pipeline = _state["pipeline"]
    result = pipeline.classify(req.text, req.structured_urgency)
    threshold = pipeline.high_threshold
    return ClassifyResponse(
        referral_id=req.referral_id,
        priority=result.priority.name,
        source=result.source,
        high_score=round(result.high_score, 4),
        high_threshold=None if threshold is None else round(threshold, 4),
        alarm_signs=result.alarm_signs,
        evidence=[AlarmEvidence(sign=h.sign, phrase=h.phrase, negated=h.negated) for h in result.evidence],
        requires_human_review=result.requires_human_review,
    )
