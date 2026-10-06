"""Evaluación comparativa de estrategias de priorización (Sección 4.3 del artículo).

Estrategias comparadas, todas sobre el MISMO conjunto de prueba:
  S0  urgencia estructurada registrada por el médico (línea base)
  S1  solo reglas: guardrail; si no hay alarma, urgencia estructurada   (Alternativa A)
  S2  ML con argmax                                                    (Alternativa B)
  S3  ML con umbral Neyman-Pearson sobre P(HIGH)
  S4  guardrail + ML argmax
  S5  guardrail + ML con umbral NP  (pipeline PrioMed completo)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .alarm_extractor import has_active_alarm
from .model import PriorityModel
from .np_threshold import np_threshold
from .synthetic_data import Referral, generate
from .taxonomy import Priority

HIGH = int(Priority.HIGH)


@dataclass
class Metrics:
    name: str
    high_sensitivity: float  # P(pred=HIGH | true=HIGH)  ← métrica foco
    high_miss_rate: float  # 1 - sensibilidad
    mismatch_sensitivity: float  # sensibilidad en casos "desajuste" (Escenario 1)
    high_false_alarm_rate: float  # P(pred=HIGH | true != HIGH)
    undertriage_rate: float  # P(pred < true) sobre todos los casos
    overtriage_rate: float  # P(pred > true)
    accuracy: float
    n_high: int


def _safe_div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def compute_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray, mismatch: np.ndarray) -> Metrics:
    is_high = y_true == HIGH
    pred_high = y_pred == HIGH
    sens = _safe_div((is_high & pred_high).sum(), is_high.sum())
    mm = is_high & mismatch
    return Metrics(
        name=name,
        high_sensitivity=sens,
        high_miss_rate=1 - sens,
        mismatch_sensitivity=_safe_div((mm & pred_high).sum(), mm.sum()),
        high_false_alarm_rate=_safe_div((~is_high & pred_high).sum(), (~is_high).sum()),
        undertriage_rate=float((y_pred < y_true).mean()),
        overtriage_rate=float((y_pred > y_true).mean()),
        accuracy=float((y_pred == y_true).mean()),
        n_high=int(is_high.sum()),
    )


def run_experiment(
    n_train: int = 6000,
    n_calib: int = 3000,
    n_test: int = 3000,
    alpha: float = 0.05,
    delta: float = 0.05,
    seed: int = 7,
    unseen_paraphrases: bool = False,
) -> tuple[list[Metrics], dict]:
    """Entrena, calibra el umbral NP y evalúa las seis estrategias.

    unseen_paraphrases=True -> el test usa paráfrasis HIGH nunca vistas (ni en el diccionario
    del guardrail ni en entrenamiento): prueba de estrés de cobertura léxica.
    """
    train = generate(n_train, seed=seed)
    calib = generate(n_calib, seed=seed + 1)
    test = generate(n_test, seed=seed + 2, paraphrase_set="unseen" if unseen_paraphrases else "train")

    model = PriorityModel(seed=seed).fit(train)

    calib_high = [r for r in calib if r.true_priority == Priority.HIGH]
    t_np = np_threshold(
        model.high_score([r.text for r in calib_high], [r.structured_urgency for r in calib_high]),
        alpha=alpha,
        delta=delta,
    )

    texts = [r.text for r in test]
    urg = np.array([r.structured_urgency for r in test])
    y_true = np.array([int(r.true_priority) for r in test])
    mismatch = np.array([r.is_mismatch for r in test])
    proba = model.predict_proba(texts, list(urg))
    guard = np.array([has_active_alarm(t) for t in texts])

    s0 = urg
    s1 = np.where(guard, HIGH, urg)
    s2 = PriorityModel.decide(proba)
    s3 = PriorityModel.decide(proba, t_np)
    s4 = np.where(guard, HIGH, s2)
    s5 = np.where(guard, HIGH, s3)

    strategies = {
        "S0 urgencia estructurada": s0,
        "S1 solo reglas (A)": s1,
        "S2 ML argmax (B)": s2,
        "S3 ML + umbral NP": s3,
        "S4 guardrail + ML": s4,
        "S5 guardrail + ML + NP (PrioMed)": s5,
    }
    metrics = [compute_metrics(name, y_true, pred, mismatch) for name, pred in strategies.items()]
    info = {
        "np_threshold": t_np,
        "alpha": alpha,
        "delta": delta,
        "n_calibration_high": len(calib_high),
        "unseen_paraphrases": unseen_paraphrases,
        "test_size": n_test,
    }
    return metrics, info


def to_markdown(metrics: list[Metrics], info: dict) -> str:
    head = (
        f"Umbral NP = {info['np_threshold']:.3f} (alpha={info['alpha']}, delta={info['delta']}, "
        f"n_calib_HIGH={info['n_calibration_high']}) · test n={info['test_size']} · "
        f"paráfrasis no vistas={'sí' if info['unseen_paraphrases'] else 'no'}\n\n"
    )
    rows = [
        "| Estrategia | Sens. HIGH | Sens. desajuste | Falsa alarma HIGH | Subtriaje | Sobretriaje | Exactitud |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in metrics:
        rows.append(
            f"| {m.name} | {m.high_sensitivity:.3f} | {m.mismatch_sensitivity:.3f} | "
            f"{m.high_false_alarm_rate:.3f} | {m.undertriage_rate:.3f} | {m.overtriage_rate:.3f} | {m.accuracy:.3f} |"
        )
    return head + "\n".join(rows) + "\n"
