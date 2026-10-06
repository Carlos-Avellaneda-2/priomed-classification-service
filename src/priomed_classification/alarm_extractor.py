"""Extracción de señales de alarma por diccionario + reglas (capa 2 del pipeline, Sección 6.2)."""
from __future__ import annotations

from dataclasses import dataclass

from .taxonomy import ALARM_SIGNS
from .text_utils import is_negated, normalize


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
    norm = normalize(text)
    hits: list[AlarmHit] = []
    for sign, phrases in ALARM_SIGNS.items():
        for phrase in phrases:
            start = 0
            while (idx := norm.find(phrase, start)) != -1:
                hits.append(AlarmHit(sign=sign, phrase=phrase, negated=is_negated(norm, idx)))
                start = idx + len(phrase)
    return hits


def active_alarm_signs(text: str) -> set[str]:
    """Señales de alarma con al menos una ocurrencia NO negada."""
    return {h.sign for h in extract_alarms(text) if not h.negated}


def has_active_alarm(text: str) -> bool:
    return bool(active_alarm_signs(text))
