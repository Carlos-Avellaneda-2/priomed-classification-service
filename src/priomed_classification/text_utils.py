"""Normalización de texto en español para extracción de señales."""
from __future__ import annotations

import re
import unicodedata


def normalize(text: str) -> str:
    """Minúsculas, sin tildes, espacios colapsados."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# Marcadores de negación que anulan una señal si aparecen justo antes de ella.
NEGATION_CUES = ("niega", "sin", "no presenta", "no refiere", "descarta", "ausencia de", "no hay", "no tiene")


def is_negated(normalized_text: str, start: int, window: int = 30) -> bool:
    """True si hay un marcador de negación en la ventana previa a la posición `start`.

    Heurística deliberadamente simple (prototipo). Una negación mal detectada que
    OCULTA una alarma real es el error costoso; por eso la ventana es corta y la
    extracción es conservadora: ante duda, NO se descarta la señal (ver
    `alarm_extractor.extract_alarms`, parámetro `conservative`).
    """
    prefix = normalized_text[max(0, start - window) : start]
    return any(cue in prefix for cue in NEGATION_CUES)
