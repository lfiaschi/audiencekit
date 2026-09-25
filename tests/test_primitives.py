from __future__ import annotations

import pandas as pd
import pytest

from audiencekit import AudienceFrame, PersonaTemplate


def test_audience_frame_samples_any_weighted_dataset() -> None:
    data = pd.DataFrame(
        {
            "person_id": ["a", "b", "c"],
            "survey_weight": [0, 0, 5],
            "age": [25, 35, 45],
            "cohort": ["student", "parent", "parent"],
        }
    )

    frame = AudienceFrame(data, id_column="person_id", weight_column="survey_weight")
    sampled = frame.sample(
        n=2, segment=lambda row: row["cohort"] == "parent", segment_name="parents", seed=3, replace=True
    )

    assert sampled["person_id"].tolist() == ["c", "c"]
    assert sampled["segment"].tolist() == ["parents", "parents"]


def test_sample_raises_when_n_exceeds_pool() -> None:
    frame = AudienceFrame(pd.DataFrame({"id": ["a", "b"], "weight": [1.0, 1.0]}))
    with pytest.raises(ValueError, match="exceeds the 2 matching respondents"):
        frame.sample(n=3)


def test_sample_with_replace_true_allows_oversampling() -> None:
    frame = AudienceFrame(pd.DataFrame({"id": ["a", "b"], "weight": [1.0, 1.0]}))
    assert len(frame.sample(n=3, replace=True)) == 3


def test_persona_template_renders_missing_fields_as_unknown() -> None:
    template = PersonaTemplate("You are {age}, live in {region}, and buy {category}.")

    assert template.render({"age": 35, "category": "coffee"}) == (
        "You are 35, live in Unknown, and buy coffee."
    )
