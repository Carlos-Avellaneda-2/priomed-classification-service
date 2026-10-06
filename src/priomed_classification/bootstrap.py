"""Construye un pipeline listo para usar entrenando sobre datos SINTÉTICOS (prototipo)."""
from __future__ import annotations

from .model import PriorityModel
from .np_threshold import np_threshold
from .pipeline import PrioMedPipeline
from .synthetic_data import generate
from .taxonomy import Priority


def build_default_pipeline(alpha: float = 0.05, delta: float = 0.05, seed: int = 7) -> PrioMedPipeline:
    train = generate(6000, seed=seed)
    calib = [r for r in generate(3000, seed=seed + 1) if r.true_priority == Priority.HIGH]
    model = PriorityModel(seed=seed).fit(train)
    t = np_threshold(model.high_score([r.text for r in calib], [r.structured_urgency for r in calib]), alpha, delta)
    return PrioMedPipeline(model, high_threshold=t)
