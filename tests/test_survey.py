from __future__ import annotations

import pandas as pd

from audiencekit import PersonaTemplate, Study, SyntheticPanel, build_survey_prompt, parse_json_response


class FakeBackend:
    def get_completion(self, prompt, image=None, **kwargs):
        assert "Return a single JSON object" in prompt
        assert kwargs["temperature"] == 0.9
        return '{"fit": 4, "verbatim": "Looks useful."}'


class CapturingBackend:
    def __init__(self):
        self.prompts = []

    def get_completion(self, prompt, image=None, **kwargs):
        self.prompts.append(prompt)
        return '{"fit": 5}'


def test_parse_json_response_tolerates_fenced_json() -> None:
    assert parse_json_response('```json\n{"score": 5}\n```') == {"score": 5}


def test_study_from_dict_validates_question_shape() -> None:
    study = Study.from_dict(
        {
            "title": "Concept test",
            "stimulus": {"description": "A compact EV for city commuters."},
            "questions": [
                {"id": "fit", "type": "likert", "text": "How well does it fit your life?"},
                {"id": "verbatim", "type": "text", "text": "What is your first reaction?"},
            ],
        }
    )

    assert study.question_ids == ["fit", "verbatim"]
    assert study.to_dict()["title"] == "Concept test"


def test_build_survey_prompt_accepts_study_objects() -> None:
    study = Study.from_dict(
        {
            "title": "Concept test",
            "questions": [{"id": "fit", "type": "likert", "text": "Fit?"}],
        }
    )

    prompt = build_survey_prompt({"age": 35, "sex": "Male"}, study)

    assert '"fit" (integer 1-5): Fit?' in prompt


def test_synthetic_panel_accepts_injected_backend() -> None:
    respondents = pd.DataFrame(
        [{"id": "r1", "age": "35", "sex": "Male", "region": "Pacific", "income16": "$50,000 to $59,999"}]
    )
    study = Study.from_dict(
        {
            "title": "Concept test",
            "questions": [
                {"id": "fit", "type": "likert", "text": "Fit?"},
                {"id": "verbatim", "type": "text", "text": "Reaction?"},
            ],
        }
    )

    panel = SyntheticPanel(respondents, backend=FakeBackend(), temperature=0.9)
    results = panel.run_survey(study, verbose=False)

    assert results.loc[0, "valid"] is True
    assert results.loc[0, "fit"] == 4
    assert results.loc[0, "verbatim"] == "Looks useful."


def test_synthetic_panel_defaults_to_gemini_backend(monkeypatch) -> None:
    calls = []

    def fake_make_backend(backend_type, model):
        calls.append((backend_type, model))
        return FakeBackend()

    monkeypatch.setattr("audiencekit.survey.make_backend", fake_make_backend)

    SyntheticPanel(pd.DataFrame([{"id": "r1"}]))

    assert calls == [("gemini", None)]


def test_synthetic_panel_accepts_custom_persona_template_for_non_gss_data() -> None:
    respondents = pd.DataFrame(
        [{"person_id": "u1", "age": 41, "region": "Midwest", "category": "running shoes"}]
    )
    study = Study.from_dict(
        {
            "title": "Concept test",
            "questions": [{"id": "fit", "type": "likert", "text": "Fit?"}],
        }
    )
    backend = CapturingBackend()

    panel = SyntheticPanel(
        respondents,
        backend=backend,
        persona_template=PersonaTemplate("You are {age}, live in {region}, and buy {category}."),
    )
    results = panel.run_survey(study, verbose=False)

    assert results.loc[0, "fit"] == 5
    assert "You are 41, live in Midwest, and buy running shoes." in backend.prompts[0]
