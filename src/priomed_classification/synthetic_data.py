"""Generador de remisiones SINTÉTICAS en español para evaluar el pipeline.

No contiene datos reales de pacientes. La etiqueta de severidad proviene de la taxonomía
de `taxonomy.py` (hipótesis de trabajo, pendiente de validación clínica), con ruido de
etiqueta y casos difíciles inyectados a propósito para que el modelo NO sea perfecto:

  * ruido de etiqueta: un porcentaje de casos con severidad ajustada +/-1 nivel;
  * casos "desajuste" (Escenario 1): el médico registra urgencia baja en el campo
    estructurado pero el texto libre contiene una señal de alarma;
  * negaciones: "niega dolor toracico" no debe disparar el guardrail;
  * texto ruidoso: sin tildes, abreviaturas y muletillas.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .taxonomy import REASONS, Priority

SPECIALTIES = ["cardiologia", "neurologia", "gastroenterologia", "psiquiatria", "neumologia", "dermatologia"]

OPENERS = [
    "Paciente remitido por",
    "Remision por cuadro de",
    "Consulta por",
    "Refiere",
    "Acude por",
    "Paciente con",
]
FILLERS = [
    "de varios dias de evolucion",
    "de inicio reciente",
    "con empeoramiento progresivo",
    "sin mejoria con manejo ambulatorio",
    "de 2 semanas",
    "intermitente",
    "",
]
DISTRACTORS = [
    "antecedente de hipertension arterial",
    "toma acetaminofen ocasional",
    "refiere buen estado general",
    "sin alergias conocidas",
    "ultimo control hace 6 meses",
]

# Paráfrasis de motivos HIGH que NO están en el diccionario del guardrail. Sirven para
# probar qué pasa cuando el diccionario falla (cobertura léxica incompleta).
#   "train"  -> aparecen en entrenamiento/calibración/prueba
#   "unseen" -> SOLO en el conjunto de prueba (estrés fuera de distribución)
PARAPHRASES: dict[str, dict[str, list[str]]] = {
    "dolor_toracico": {
        "train": ["me aprieta el pecho", "ardor y presion en el torax"],
        "unseen": ["siento el corazon oprimido", "punzada fuerte en el esternon"],
    },
    "ideacion_suicida": {
        "train": ["ya no quiere seguir viviendo", "habla de hacerse dano"],
        "unseen": ["dice que seria mejor no despertar", "expresa deseos de no estar vivo"],
    },
    "dificultad_respiratoria": {
        "train": ["le cuesta tomar aire", "respira con dificultad"],
        "unseen": ["no le entra el aire", "se agita al hablar y se asfixia"],
    },
    "sincope": {
        "train": ["se cayo sin conocimiento", "episodio de desvanecimiento"],
        "unseen": ["quedo inconsciente unos segundos", "se apago de repente"],
    },
    "herida_arma_fuego": {
        "train": ["lesion por proyectil", "herida de bala"],
        "unseen": ["impacto de bala en extremidad", "lesion penetrante por arma de fuego"],
    },
}


@dataclass
class Referral:
    text: str
    structured_urgency: int  # 0=baja, 1=media, 2=alta (registrado por el médico)
    reason: str  # motivo principal verdadero (para análisis por motivo)
    true_priority: Priority
    specialty: str
    is_mismatch: bool = False  # alarma en texto libre + urgencia estructurada baja


def _phrase(rng: random.Random, reason: str, paraphrase_rate: float = 0.0, paraphrase_set: str = "train") -> str:
    if reason in PARAPHRASES and rng.random() < paraphrase_rate:
        return rng.choice(PARAPHRASES[reason][paraphrase_set])
    return rng.choice(REASONS[reason][1])


def generate(
    n: int = 4000,
    seed: int = 7,
    label_noise: float = 0.05,
    mismatch_rate: float = 0.30,
    paraphrase_rate: float = 0.15,
    paraphrase_set: str = "train",
) -> list[Referral]:
    """Genera `n` remisiones sintéticas.

    mismatch_rate: fracción de los casos HIGH en los que el médico registró urgencia
    estructurada BAJA (mientras el texto libre sí menciona la alarma).
    paraphrase_rate: fracción de los motivos HIGH redactados con una paráfrasis que NO
    está en el diccionario del guardrail (set "train" o "unseen").
    """
    rng = random.Random(seed)
    reasons = list(REASONS.keys())
    # Prevalencia realista: HIGH minoritario
    weights = [{Priority.HIGH: 0.5, Priority.MEDIUM: 1.0, Priority.LOW: 1.5}[REASONS[r][0]] for r in reasons]

    out: list[Referral] = []
    for _ in range(n):
        reason = rng.choices(reasons, weights=weights, k=1)[0]
        base = REASONS[reason][0]
        true_priority = base
        if rng.random() < label_noise:
            true_priority = Priority(min(2, max(0, int(base) + rng.choice([-1, 1]))))

        parts = [rng.choice(OPENERS), _phrase(rng, reason, paraphrase_rate, paraphrase_set), rng.choice(FILLERS)]
        if rng.random() < 0.6:
            parts.append(rng.choice(DISTRACTORS))
        # Negación de una señal de alarma ajena (distractor que NO debe disparar el guardrail)
        if rng.random() < 0.15:
            neg_reason = rng.choice([r for r, (p, _) in REASONS.items() if p == Priority.HIGH and r != reason])
            parts.append(f"niega {_phrase(rng, neg_reason)}")
        rest = parts[1:]
        rng.shuffle(rest)  # deja el opener primero
        text = " ".join(p for p in [parts[0], *rest] if p).strip()

        # Urgencia estructurada que registra el médico (imprecisa)
        if true_priority == Priority.HIGH:
            structured = 2
            mismatch = False
            if rng.random() < mismatch_rate:
                structured = rng.choice([0, 1])
                mismatch = base == Priority.HIGH  # la alarma sí está en el texto
        else:
            structured = int(true_priority)
            if rng.random() < 0.12:
                structured = rng.choice([0, 1, 2])
            mismatch = False

        out.append(
            Referral(
                text=text,
                structured_urgency=structured,
                reason=reason,
                true_priority=true_priority,
                specialty=rng.choice(SPECIALTIES),
                is_mismatch=mismatch,
            )
        )
    return out
