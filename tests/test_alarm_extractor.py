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


def test_negation_cue_inside_another_word_does_not_hide_alarm():
    # "sin" dentro de "sintomas" / "sintetico" / "casino" no es una negación.
    assert "dolor_toracico" in active_alarm_signs("Consulta por síntomas de dolor torácico")
    assert "dolor_toracico" in active_alarm_signs("Caso sintético: dolor torácico opresivo")
    assert "sincope" in active_alarm_signs("Estaba en el casino y tuvo un desmayo")


def test_negation_does_not_cross_clause_boundaries():
    # Regresión: "sin"/"niega" de una cláusula anterior ocultaban la alarma siguiente.
    assert "dolor_toracico" in active_alarm_signs("Sin alergias conocidas. Dolor torácico intenso")
    assert "dificultad_respiratoria" in active_alarm_signs("Sin fiebre. Disnea severa")
    assert "sincope" in active_alarm_signs("Niega fiebre; presentó desmayo en casa")
    assert "dificultad_respiratoria" in active_alarm_signs("Niega dolor torácico, refiere falta de aire")


def test_scope_terminator_ends_negation():
    assert active_alarm_signs("Paciente sin dolor toracico pero con disnea") == {"dificultad_respiratoria"}
    assert "sincope" in active_alarm_signs("No hay mejoria y presenta desmayo")


def test_sin_embargo_is_not_a_negation():
    assert "dolor_toracico" in active_alarm_signs("Sin embargo dolor toracico persistente")


def test_alarm_phrase_must_be_whole_words():
    assert not has_active_alarm("Refiere desahogo emocional tras la consulta")


def test_synthetic_alarm_cases_are_not_hidden_by_distractors():
    from priomed_classification.synthetic_data import generate
    from priomed_classification.taxonomy import ALARM_SIGNS

    cases = [r for r in generate(3000, seed=9, paraphrase_rate=0.0) if r.reason in ALARM_SIGNS]
    missed = [r.text for r in cases if r.reason not in active_alarm_signs(r.text)]
    assert not missed, missed[:5]


def test_whole_word_negation_cues_still_negate():
    assert not has_active_alarm("Paciente sin dolor toracico")
    assert not has_active_alarm("No refiere disnea ni desmayo")
    assert not has_active_alarm("Se descarta sincope")
