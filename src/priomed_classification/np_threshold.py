"""Calibración de umbral tipo Neyman-Pearson para controlar la tasa de subtriaje.

Problema: queremos un clasificador binario "¿es de alta prioridad?" tal que la tasa de
omisión (HIGH clasificado como no-HIGH, el error costoso) sea <= alpha con probabilidad
>= 1 - delta, y entre los umbrales que lo garantizan, el que genere menos falsas alarmas.

Método (inspirado en el algoritmo "NP umbrella", Tong, Feng & Li, 2018; y en la
conexión NP / aprendizaje sensible al costo de Tian & Feng): se toman los puntajes de un
conjunto de CALIBRACIÓN solo de casos HIGH, ordenados ascendentemente s_(1) <= ... <= s_(n).
Si se clasifica como HIGH cuando puntaje >= s_(k), la tasa de omisión verdadera de ese
umbral es F(s_(k)) ~ Beta(k, n-k+1) (para puntajes continuos), por lo que

    P(omisión > alpha) = P(Binomial(n, alpha) <= k-1).

Se elige el MAYOR k con P(Binomial(n, alpha) <= k-1) <= delta (umbral más alto = menos
falsas alarmas). Si ni k=1 cumple, no hay garantía posible: faltan casos HIGH de
calibración (se necesita n >= ln(delta) / ln(1-alpha)).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import binom


class InfeasibleError(ValueError):
    """No hay suficientes casos de la clase crítica para garantizar alpha con confianza 1-delta."""


def min_calibration_size(alpha: float, delta: float) -> int:
    """Mínimo n de casos HIGH de calibración para que la garantía sea factible."""
    return math.ceil(math.log(delta) / math.log(1 - alpha))


def np_threshold(high_scores: np.ndarray, alpha: float = 0.05, delta: float = 0.05) -> float:
    """Umbral t tal que P(tasa de omisión de 'score >= t' > alpha) <= delta."""
    s = np.sort(np.asarray(high_scores, dtype=float))
    n = len(s)
    if n < min_calibration_size(alpha, delta):
        raise InfeasibleError(
            f"Se necesitan >= {min_calibration_size(alpha, delta)} casos HIGH de calibración "
            f"para alpha={alpha}, delta={delta}; hay {n}."
        )
    # Mayor k (1-indexado) con P(Bin(n, alpha) <= k-1) <= delta
    k = 1
    for cand in range(1, n + 1):
        if binom.cdf(cand - 1, n, alpha) <= delta:
            k = cand
        else:
            break
    return float(s[k - 1])
