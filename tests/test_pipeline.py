import pytest

from priomed_classification.bootstrap import build_default_pipeline
from priomed_classification.taxonomy import Priority


@pytest.fixture(scope="module")
def pipe():
    return build_default_pipeline()


def test_guardrail_forces_high_even_if_physician_recorded_low(pipe):
    # Escenario 1 (Sección 5.4): urgencia estructurada baja, alarma en texto libre
    r = pipe.classify("Paciente con dolor toracico de inicio reciente", structured_urgency=0)
    assert r.priority == Priority.HIGH and r.source == "guardrail"
    assert "dolor_toracico" in r.alarm_signs


def test_benign_case_is_not_escalated(pipe):
    r = pipe.classify("Consulta por rinorrea y tos seca de 2 semanas", structured_urgency=0)
    assert r.priority == Priority.LOW


def test_result_always_requires_human_review(pipe):
    assert pipe.classify("Consulta por tos", 0).requires_human_review is True
