"""Capa 4 del pipeline: scoring con ML clásico (Alternativa B)."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .synthetic_data import Referral
from .taxonomy import Priority
from .text_utils import normalize


def _doc(r_text: str, structured_urgency: int) -> str:
    # La urgencia estructurada que registró el médico entra como un token más.
    return f"{normalize(r_text)} urg{structured_urgency}"


class PriorityModel:
    """TF-IDF (1-2 gramas) + regresión logística multinomial. P(HIGH) es el puntaje de riesgo."""

    def __init__(self, C: float = 2.0, seed: int = 7):
        self.pipe = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                ("clf", LogisticRegression(C=C, max_iter=1000, random_state=seed)),
            ]
        )

    def fit(self, data: list[Referral]) -> "PriorityModel":
        X = [_doc(r.text, r.structured_urgency) for r in data]
        y = [int(r.true_priority) for r in data]
        self.pipe.fit(X, y)
        return self

    def predict_proba(self, texts: list[str], urgencies: list[int]) -> np.ndarray:
        """Matriz (n, 3) con columnas ordenadas LOW, MEDIUM, HIGH."""
        X = [_doc(t, u) for t, u in zip(texts, urgencies)]
        return self.pipe.predict_proba(X)

    def high_score(self, texts: list[str], urgencies: list[int]) -> np.ndarray:
        return self.predict_proba(texts, urgencies)[:, int(Priority.HIGH)]

    @staticmethod
    def decide(proba: np.ndarray, high_threshold: float | None = None) -> np.ndarray:
        """argmax por defecto; con `high_threshold`, HIGH si P(HIGH) >= umbral (umbral NP)."""
        pred = proba.argmax(axis=1)
        if high_threshold is not None:
            is_high = proba[:, int(Priority.HIGH)] >= high_threshold
            non_high_argmax = proba[:, : int(Priority.HIGH)].argmax(axis=1)
            pred = np.where(is_high, int(Priority.HIGH), non_high_argmax)
        return pred
