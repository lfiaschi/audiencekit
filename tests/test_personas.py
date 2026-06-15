from __future__ import annotations

import zipfile

import pandas as pd
import pytest

import audiencekit as ak
from audiencekit import build_persona, load_panel, sample_panel
from audiencekit.gss import load_gss, prepare_gss_persona_frame


def test_sample_panel_accepts_callable_segment_and_uses_weights() -> None:
    df = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "weight": [0.0, 0.0, 10.0],
            "income16": ["Under $1,000", "$170,000 or over", "$170,000 or over"],
            "class": ["Working Class", "Middle Class", "Upper Class"],
            "degree": ["High school", "Bachelor's", "Graduate"],
            "prestg10": ["Low", "High", "High"],
            "finrela": ["Average", "Above average", "Far above average"],
        }
    )

    sampled = sample_panel(
        df,
        n=3,
        segment=lambda row: row["income16"] == "$170,000 or over",
        seed=7,
        weighted=True,
        segment_name="affluent",
    )

    assert sampled["id"].tolist() == ["c", "c", "c"]
    assert sampled["segment"].tolist() == ["affluent", "affluent", "affluent"]


def test_load_panel_defaults_to_packaged_2024_panel() -> None:
    panel = load_panel()

    assert {"id", "weight", "age", "sex"}.issubset(panel.columns)
    assert len(panel) > 1000
    assert panel["id"].str.startswith("2024-").all()


def test_load_gss_reads_zipped_stata_file(tmp_path) -> None:
    raw = pd.DataFrame(
        {
            "id": [301],
            "year": [2024],
            "wtssnrps": [1.5],
            "age": [52],
            "sex": [1],
            "race": [1],
            "region": [4],
            "res16": [3],
            "marital": [1],
            "educ": [16],
            "degree": [3],
            "income16": [24],
            "occ10": [10],
            "prestg10": [65],
            "finrela": [4],
            "satfin": [1],
            "partyid": [3],
            "polviews": [4],
            "relig": [4],
            "attend": [0],
            "childs": [2],
            "happy": [2],
            "health": [2],
            "tvhours": [1],
            "usewww": [1],
            "getahead": [1],
        }
    )
    dta_path = tmp_path / "GSS2024.dta"
    zip_path = tmp_path / "2024_stata.zip"
    raw.to_stata(dta_path, write_index=False)
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(dta_path, arcname="2024/GSS2024.dta")

    panel = load_gss(zip_path, years=[2024])

    assert panel["id"].tolist() == ["2024-301"]
    assert panel.loc[0, "sex"] == "Male"


def test_weighted_sampling_requires_positive_weight_column() -> None:
    df = pd.DataFrame({"id": ["a"], "weight": [0]})

    with pytest.raises(ValueError, match="positive"):
        sample_panel(df, n=1, weighted=True)


def test_prepare_gss_persona_frame_filters_years_and_maps_values() -> None:
    raw = pd.DataFrame(
        {
            "id": [101, 102],
            "year": [2022, 2024],
            "wtssnrps": [1.2, 2.3],
            "age": [35, 44],
            "sex": [1, 2],
            "race": [1, 2],
            "region": [1, 4],
            "res16": [3, 5],
            "marital": [1, 5],
            "educ": [16, 14],
            "degree": [3, 1],
            "income16": [26, -99],
            "class": [3, 2],
            "occ10": [10, 4700],
            "prestg10": [70, 40],
            "finrela": [4, 3],
            "satfin": [1, 2],
            "partyid": [3, 0],
            "polviews": [4, 2],
            "relig": [4, 1],
            "attend": [0, 8],
            "childs": [0, 2],
            "happy": [2, 1],
            "health": [2, 1],
            "tvhours": [1, 3],
            "usewww": [1, 2],
            "getahead": [1, 2],
        }
    )

    panel = prepare_gss_persona_frame(raw, years=[2024])

    assert panel["id"].tolist() == ["2024-102"]
    assert panel["weight"].tolist() == [2.3]
    assert panel.loc[0, "sex"] == "Female"
    assert panel.loc[0, "race"] == "Black"
    assert panel.loc[0, "income16"] is None


def test_prepare_gss_persona_frame_keeps_2024_region_and_graduate_education() -> None:
    raw = pd.DataFrame(
        {
            "id": [301],
            "year": [2024],
            "wtssnrps": [1.5],
            "age": [44],
            "sex": [2],
            "race": [1],
            "region": [4],
            "res16": [6],
            "marital": [5],
            "educ": [20],
            "degree": [4],
            "income16": [26],
        }
    )

    panel = prepare_gss_persona_frame(raw, years=[2024])

    assert panel["id"].tolist() == ["2024-301"]
    assert panel.loc[0, "region"] == "West"
    assert panel.loc[0, "educ"] == "8 or more years of college"


def test_prepare_gss_persona_frame_maps_rich_high_coverage_persona_fields() -> None:
    raw = pd.DataFrame(
        {
            "id": [302],
            "year": [2024],
            "wtssnrps": [2.0],
            "age": [39],
            "sex": [1],
            "race": [3],
            "racecen1": [16],
            "region": [3],
            "res16": [5],
            "marital": [1],
            "educ": [16],
            "degree": [3],
            "income16": [24],
            "class": [3],
            "wrkstat": [1],
            "weekswrk": [52],
            "wrkslf": [2],
            "earnrs": [3],
            "adults": [2],
            "born": [1],
            "sibs": [6],
            "madeg": [3],
            "occ10": [1020],
            "prestg10": [61],
            "finrela": [4],
            "satfin": [1],
            "partyid": [3],
            "polviews": [4],
            "reltrad": [7],
            "relpersn": [4],
            "attend": [0],
            "childs": [0],
            "happy": [2],
            "health": [2],
            "natsoc": [1],
        }
    )

    panel = prepare_gss_persona_frame(raw, years=[2024])

    assert panel.loc[0, "race_detail"] == "Hispanic"
    assert panel.loc[0, "wrkstat"] == "working full time"
    assert panel.loc[0, "weekswrk"] == "52 weeks"
    assert panel.loc[0, "wrkslf"] == "working for someone else"
    assert panel.loc[0, "earnrs"] == "3 or more earners"
    assert panel.loc[0, "adults"] == "2 adults"
    assert panel.loc[0, "born"] == "born in the United States"
    assert panel.loc[0, "sibs"] == "6 or more siblings"
    assert panel.loc[0, "madeg"] == "Bachelor's"
    assert panel.loc[0, "occ10"] == "management, business, science, and arts occupations"
    assert panel.loc[0, "reltrad"] == "nonaffiliated"
    assert panel.loc[0, "relpersn"] == "not religious at all"
    assert panel.loc[0, "natsoc"] == "too little"


def test_gss_persona_fields_use_high_coverage_defaults() -> None:
    expected = {
        "race_detail",
        "wrkstat",
        "weekswrk",
        "wrkslf",
        "earnrs",
        "adults",
        "born",
        "sibs",
        "madeg",
        "reltrad",
        "relpersn",
        "natsoc",
    }
    low_coverage = {"tvhours", "usewww", "getahead"}

    assert expected.issubset(set(ak.GSS_PERSONA_FIELDS))
    assert low_coverage.isdisjoint(set(ak.GSS_PERSONA_FIELDS))


def test_build_persona_renders_unknown_for_missing_values() -> None:
    persona = build_persona({"age": 44, "sex": "Female", "race": None})

    assert "44 year old Female adult" in persona
    assert "race or ethnicity as Unknown" in persona
    assert "reported family income last year before taxes was Unknown" in persona


def test_build_persona_frames_income16_as_reported_family_income_not_salary() -> None:
    persona = build_persona(
        {
            "age": 68,
            "sex": "Female",
            "race": "White",
            "region": "South",
            "income16": "$10,000 to $12,499",
            "class": "Working Class",
            "finrela": "Below average",
            "satfin": "Not Satisfied",
        }
    )

    assert "reported family income last year before taxes was $10,000 to $12,499" in persona
    assert "from all family sources, not just salary" in persona


def test_public_gss_persona_template_renders_rows_like_build_persona() -> None:
    row = {
        "age": "68",
        "sex": "Female",
        "race_detail": "White",
        "region": "South",
        "res16": "in a large city (over 250,000)",
        "born": "born in the United States",
        "marital": "Never Married",
        "educ": "4 years of college",
        "degree": "Bachelor's",
        "madeg": "High school",
        "income16": "$10,000 to $12,499",
        "earnrs": "1 earner",
        "class": "Working Class",
        "wrkstat": "retired",
        "weekswrk": "0 weeks",
        "wrkslf": "working for someone else",
        "occ10": "service occupations",
        "prestg10": "Low",
        "finrela": "Below average",
        "satfin": "Not Satisfied",
        "partyid": "Independent",
        "polviews": "Moderate",
        "reltrad": "nonaffiliated",
        "relpersn": "not religious at all",
        "attend": "Never",
        "childs": "0",
        "adults": "1 adult",
        "sibs": "2 siblings",
        "happy": "Pretty Happy",
        "health": "Good",
        "natsoc": "too little",
    }

    assert isinstance(ak.GSS_PERSONA_TEMPLATE, ak.PersonaTemplate)
    assert "income16" in ak.GSS_PERSONA_FIELDS
    assert ak.GSS_PERSONA_TEMPLATE.render(row) == ak.build_persona(row)
