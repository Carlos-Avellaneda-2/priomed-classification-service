"""Pipeline completo por capas: guardrail (A) + ML con umbral NP (B).

La capa 5 (explicación LLM, Alternativa C) y la 6 (validación humana) viven fuera de este
servicio: aquí el resultado SIEMPRE sale marcado `requires_human_review=True`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .alarm_extractor import AlarmHit, extract_alarms
from .model import PriorityModel
from .taxonomy import Priority


@dataclass
class Classification:
    priority: Priority
    source: str  # "guardrail" | "ml"
    high_score: float
    alarm_signs: list[str] = field(default_factory=list)
    requires_human_review: bool = True
    # Todas las frases de alarma encontradas, negadas o no (explicación para el revisor).
    evidence: list[AlarmHit] = field(default_factory=list)


class PrioMedPipeline:
    def __init__(self, model: PriorityModel, high_threshold: float | None = None):
        self.model = model
        self.high_threshold = high_threshold

    def classify(self, text: str, structured_urgency: int) -> Classification:
        proba = self.model.predict_proba([text], [structured_urgency])
        score = float(proba[0, int(Priority.HIGH)])

        hits = extract_alarms(text)
        signs = sorted({h.sign for h in hits if not h.negated})
        if signs:  # capa 3: guardrail, independiente de lo que diga el modelo
            return Classification(Priority.HIGH, "guardrail", score, signs, evidence=hits)

        pred = int(self.model.decide(proba, self.high_threshold)[0])
        return Classification(Priority(pred), "ml", score, [], evidence=hits)
