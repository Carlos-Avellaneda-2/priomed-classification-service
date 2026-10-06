"""Taxonomía de motivos de consulta y niveles de prioridad.

IMPORTANTE: la asignación motivo -> severidad es una HIPÓTESIS DE TRABAJO derivada de la
entrevista exploratoria con un médico general (ver artículo, Sección 2.2). Debe ser
validada por un profesional clínico antes de usarse con datos reales.
"""
from __future__ import annotations

from enum import IntEnum


class Priority(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2


# Motivo de consulta -> (severidad base, expresiones de texto libre que lo describen)
REASONS: dict[str, tuple[Priority, list[str]]] = {
    # Señales de alarma (HIGH)
    "dolor_toracico": (Priority.HIGH, ["dolor toracico", "dolor en el pecho", "opresion en el pecho"]),
    "ideacion_suicida": (
        Priority.HIGH,
        ["ideacion suicida", "pensamientos de muerte", "quiere quitarse la vida", "pensamientos suicidas"],
    ),
    "dificultad_respiratoria": (
        Priority.HIGH,
        ["dificultad para respirar", "falta de aire", "disnea", "ahogo"],
    ),
    "sincope": (Priority.HIGH, ["desmayo", "sincope", "perdida de conocimiento"]),
    "herida_arma_fuego": (Priority.HIGH, ["herida por arma de fuego", "disparo"]),
    # Severidad intermedia (MEDIUM)
    "dolor_abdominal": (Priority.MEDIUM, ["dolor abdominal", "dolor de estomago", "colicos abdominales"]),
    "vomito": (Priority.MEDIUM, ["vomito", "vomitos persistentes", "nauseas con vomito"]),
    "fiebre": (Priority.MEDIUM, ["fiebre", "fiebre alta", "temperatura elevada"]),
    "mareo": (Priority.MEDIUM, ["mareo", "vertigo", "sensacion de mareo"]),
    "herida_cortante": (Priority.MEDIUM, ["herida cortante", "corte profundo", "laceracion"]),
    "golpe_caida": (Priority.MEDIUM, ["golpe", "caida", "trauma por caida"]),
    "debilidad": (Priority.MEDIUM, ["debilidad", "astenia", "adinamia"]),
    "diarrea": (Priority.MEDIUM, ["diarrea", "deposiciones liquidas"]),
    "agitacion": (Priority.MEDIUM, ["agitacion", "inquietud psicomotora"]),
    # Baja severidad (LOW)
    "cefalea": (Priority.LOW, ["dolor de cabeza", "cefalea"]),
    "rinorrea": (Priority.LOW, ["rinorrea", "secrecion nasal", "moqueo"]),
    "tos": (Priority.LOW, ["tos seca", "tos con flema", "tos"]),
    "erupcion_cutanea": (Priority.LOW, ["erupcion en la piel", "brote cutaneo", "rash"]),
    "prurito": (Priority.LOW, ["prurito", "picazon"]),
    "hormigueo": (Priority.LOW, ["hormigueo", "parestesias"]),
}

# Señales de alarma que activan el guardrail de reglas (Escenario 1, Sección 5.4)
ALARM_SIGNS: dict[str, list[str]] = {
    name: phrases for name, (prio, phrases) in REASONS.items() if prio == Priority.HIGH
}
