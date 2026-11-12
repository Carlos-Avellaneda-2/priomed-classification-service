"""Extracción de señales de alarma por diccionario + reglas (capa 2 del pipeline, Sección 6.2)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .taxonomy import ALARM_SIGNS
from .text_utils import is_negated, normalize_with_clauses

# Cada frase se busca como palabras completas: "ahogo" no debe coincidir en "desahogo".
_PHRASE_PATTERNS: dict[str, list[tuple[str, re.Pattern[str]]]] = {
    sign: [(p, re.compile(r"\b" + re.escape(p) + r"\b")) for p in phrases] for sign, phrases in ALARM_SIGNS.items()
}


@dataclass(frozen=True)
class AlarmHit:
    sign: str  # nombre canónico, p. ej. "dolor_toracico"
    phrase: str  # frase que coincidió
    negated: bool  # True si se detectó negación previa a ESA ocurrencia


def extract_alarms(text: str) -> list[AlarmHit]:
    """Devuelve todas las ocurrencias de señales de alarma en `text`.

    Se evalúan TODAS las ocurrencias de cada frase (no solo la primera), de modo que
    "niega dolor toracico ... luego dolor toracico intenso" produce una ocurrencia
    negada y una activa. Se prioriza la sensibilidad: mejor una falsa alarma
    (revisión humana extra) que una señal real omitida.
    """
    norm = normalize_with_clauses(text)
    hits: list[AlarmHit] = []
    for sign, patterns in _PHRASE_PATTERNS.items():
        for phrase, pattern in patterns:
            for m in pattern.finditer(norm):
                hits.append(AlarmHit(sign=sign, phrase=phrase, negated=is_negated(norm, m.start())))
    return hits


def active_alarm_signs(text: str) -> set[str]:
    """Señales de alarma con al menos una ocurrencia NO negada."""
    return {h.sign for h in extract_alarms(text) if not h.negated}


def has_active_alarm(text: str) -> bool:
    return bool(active_alarm_signs(text))
