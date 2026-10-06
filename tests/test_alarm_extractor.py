from priomed_classification.alarm_extractor import active_alarm_signs, extract_alarms, has_active_alarm


def test_detects_basic_alarm_with_accents_and_case():
    assert "dolor_toracico" in active_alarm_signs("Paciente con DOLOR TORÁCICO desde ayer")


def test_negated_alarm_is_not_active():
    assert not has_active_alarm("Refiere cefalea. Niega dolor toracico.")


def test_negation_does_not_hide_a_second_real_occurrence():
    text = "Niega dolor toracico. Hoy consulta por dolor toracico intenso."
    hits = [h for h in extract_alarms(text) if h.sign == "dolor_toracico"]
    assert {h.negated for h in hits} == {True, False}
    assert has_active_alarm(text)


def test_no_alarm_in_benign_referral():
    assert not has_active_alarm("Consulta por rinorrea y tos seca de 2 semanas")


def test_suicidal_ideation_detected():
    assert "ideacion_suicida" in active_alarm_signs("Paciente con pensamientos de muerte")
