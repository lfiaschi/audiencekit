"""Run structured studies against a panel of synthetic audience members.

A survey is a plain dict (usually authored as JSON by the /survey skill):

    {
      "title": "Concept test",
      "stimulus": {
        "description": "A compact EV designed for city commuters.",
        "image": "examples/my_study/stimulus.jpg"   # optional
      },
      "questions": [
        {"id": "fit",            "type": "likert", "text": "How well does this fit your life?"},
        {"id": "consideration",  "type": "likert", "text": "How likely would you be to consider it?"},
        {"id": "verdict",        "type": "choice", "text": "What is your bottom-line reaction?", "options": ["positive", "mixed", "negative"]},
        {"id": "first_reaction", "type": "text",   "text": "What is your honest first reaction?"}
      ]
    }

Each respondent answers the whole questionnaire in one LLM call and returns
JSON keyed by question id. Likert answers are integers 1-5.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from .backends import make_backend
from .personas import build_persona
from .primitives import PersonaTemplate
from .ssr import SemanticSimilarityRater

LIKERT_ANCHORS = (
    "1 = very low / strongly disagree\n"
    "2 = low / disagree\n"
    "3 = neutral or mixed\n"
    "4 = high / agree\n"
    "5 = very high / strongly agree"
)


@dataclass(frozen=True)
class Question:
    """One survey question in an AudienceKit study."""

    id: str
    type: str
    text: str
    options: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Question":
        missing = [key for key in ["id", "type", "text"] if not data.get(key)]
        if missing:
            raise ValueError(f"Question is missing required fields: {missing}")
        qtype = data["type"]
        if qtype not in {"likert", "choice", "text"}:
            raise ValueError(f"Unsupported question type {qtype!r}")
        options = tuple(data.get("options") or ())
        if qtype == "choice" and not options:
            raise ValueError(f"Choice question {data['id']!r} needs options")
        return cls(id=str(data["id"]), type=qtype, text=str(data["text"]), options=options)

    def to_dict(self) -> dict[str, Any]:
        data = {"id": self.id, "type": self.type, "text": self.text}
        if self.options:
            data["options"] = list(self.options)
        return data


@dataclass(frozen=True)
class Study:
    """Validated survey specification for synthetic audience studies."""

    title: str
    questions: tuple[Question, ...]
    stimulus: dict[str, Any] = field(default_factory=dict)
    treatments: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Study":
        questions = tuple(Question.from_dict(item) for item in data.get("questions", []))
        if not questions:
            raise ValueError("Study needs at least one question")
        ids = [question.id for question in questions]
        duplicates = sorted({qid for qid in ids if ids.count(qid) > 1})
        if duplicates:
            raise ValueError(f"Duplicate question ids: {duplicates}")
        return cls(
            title=str(data.get("title") or "Untitled study"),
            stimulus=dict(data.get("stimulus") or {}),
            treatments=dict(data.get("treatments") or {}),
            questions=questions,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "Study":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @property
    def question_ids(self) -> list[str]:
        return [question.id for question in self.questions]

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "title": self.title,
            "questions": [question.to_dict() for question in self.questions],
        }
        if self.stimulus:
            data["stimulus"] = dict(self.stimulus)
        if self.treatments:
            data["treatments"] = dict(self.treatments)
        return data


def _as_study_dict(study: Study | dict[str, Any]) -> dict[str, Any]:
    return study.to_dict() if isinstance(study, Study) else study


def render_persona(
    attributes: dict[str, Any],
    persona_template: PersonaTemplate | str | Any | None = None,
) -> str:
    """Render a persona from row attributes.

    If no template is provided, use the built-in GSS-oriented renderer. A
    template can be a ``PersonaTemplate``, a format string, or any callable
    that accepts the attribute dict and returns text.
    """
    if persona_template is None:
        return build_persona(attributes)
    if isinstance(persona_template, PersonaTemplate):
        return persona_template.render(attributes)
    if isinstance(persona_template, str):
        return PersonaTemplate(persona_template).render(attributes)
    if callable(persona_template):
        return str(persona_template(attributes))
    raise TypeError("persona_template must be a PersonaTemplate, format string, callable, or None")


def build_survey_prompt(
    attributes: dict,
    survey: Study | dict[str, Any],
    *,
    persona_template: PersonaTemplate | str | Any | None = None,
) -> str:
    survey_dict = _as_study_dict(survey)
    persona = render_persona(attributes, persona_template)
    stimulus = survey_dict.get("stimulus") or {}
    stimulus_block = ""
    if stimulus.get("description"):
        stimulus_block = f"\n# What you are shown\n{stimulus['description']}\n"
        if stimulus.get("image"):
            stimulus_block += "You are also looking at an image of it.\n"

    lines = []
    likert_ids = []
    for q in survey_dict["questions"]:
        if q["type"] == "likert":
            lines.append(f'- "{q["id"]}" (integer 1-5): {q["text"]}')
            likert_ids.append(q["id"])
        elif q["type"] == "choice":
            lines.append(f'- "{q["id"]}" (one of {q["options"]}): {q["text"]}')
        else:
            lines.append(f'- "{q["id"]}" (short natural sentence or two): {q["text"]}')
    question_block = "\n".join(lines)

    likert_block = f"\nFor 1-5 ratings use this scale:\n{LIKERT_ANCHORS}\n" if likert_ids else ""

    return f"""# Role
You are a consumer taking part in a market research interview.
Answer naturally from your own point of view and stay consistent with this profile.
There are no right answers. Be direct and honest — you can like some aspects and dislike others.

# Who you are
{persona}
{stimulus_block}
# Questionnaire
Answer ALL of the following fields:
{question_block}
{likert_block}
Return a single JSON object with exactly those field names and nothing else.
No markdown, no commentary."""


def build_ssr_survey_prompt(
    attributes: dict,
    survey: Study | dict[str, Any],
    *,
    persona_template: PersonaTemplate | str | Any | None = None,
) -> str:
    """Build a prompt that elicits text for Likert items before SSR scoring."""
    survey_dict = _as_study_dict(survey)
    persona = render_persona(attributes, persona_template)
    stimulus = survey_dict.get("stimulus") or {}
    stimulus_block = ""
    if stimulus.get("description"):
        stimulus_block = f"\n# What you are shown\n{stimulus['description']}\n"
        if stimulus.get("image"):
            stimulus_block += "You are also looking at an image of it.\n"

    lines = []
    has_likert = False
    for q in survey_dict["questions"]:
        if q["type"] == "likert":
            lines.append(f'- "{q["id"]}" (short natural-language answer, not a number): {q["text"]}')
            has_likert = True
        elif q["type"] == "choice":
            lines.append(f'- "{q["id"]}" (one of {q["options"]}): {q["text"]}')
        else:
            lines.append(f'- "{q["id"]}" (short natural sentence or two): {q["text"]}')
    question_block = "\n".join(lines)

    ssr_instruction = (
        "\nDo not answer Likert questions with numbers or scale labels. "
        "Write the thought you would naturally express before choosing a rating.\n"
        if has_likert
        else ""
    )

    return f"""# Role
You are a consumer taking part in a market research interview.
Answer naturally from your own point of view and stay consistent with this profile.
There are no right answers. Be direct and honest — you can like some aspects and dislike others.

# Who you are
{persona}
{stimulus_block}
# Questionnaire
Answer ALL of the following fields:
{question_block}
{ssr_instruction}
Return a single JSON object with exactly those field names and nothing else.
No markdown, no commentary."""


def parse_json_response(raw: str) -> Optional[dict]:
    """Parse a model response into a dict, tolerating code fences and prose."""
    if not raw:
        return None
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


class SyntheticPanel:
    """A panel runner for respondent rows rendered as synthetic audience members."""

    def __init__(
        self,
        respondents: pd.DataFrame,
        backend_type: str = "gemini",
        model: Optional[str] = None,
        backend: Any | None = None,
        persona_template: PersonaTemplate | str | Any | None = None,
        prompt_builder: Any | None = None,
        temperature: float = 0.7,
        max_workers: int = 8,
    ):
        self.respondents = respondents.reset_index(drop=True)
        self.backend = backend or make_backend(backend_type, model)
        self.persona_template = persona_template
        self.prompt_builder = prompt_builder
        self.temperature = temperature
        self.max_workers = max_workers

    def __len__(self) -> int:
        return len(self.respondents)

    def run_survey(
        self,
        survey: Study | dict[str, Any],
        image: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Run the survey across the panel; one row per respondent.

        image overrides survey['stimulus']['image'] when given.
        Adds respondent id plus a few demographics for cuts, and a
        'valid' flag (False when the response could not be parsed).
        """
        survey_dict = _as_study_dict(survey)
        stimulus = survey_dict.get("stimulus") or {}
        image_path = image or stimulus.get("image")

        def _one(idx: int) -> dict:
            row = self.respondents.iloc[idx].to_dict()
            if self.prompt_builder:
                prompt = self.prompt_builder(row, survey_dict)
            else:
                prompt = build_survey_prompt(row, survey_dict, persona_template=self.persona_template)
            try:
                raw = self.backend.get_completion(
                    prompt, image=image_path, temperature=self.temperature
                )
                parsed = parse_json_response(raw)
            except RuntimeError:
                parsed = None
            record = {
                "respondent_id": row.get("id"),
                "age": row.get("age"),
                "sex": row.get("sex"),
                "region": row.get("region"),
                "income16": row.get("income16"),
                "segment": row.get("segment", "broad"),
                "valid": parsed is not None,
            }
            if parsed:
                for q in survey_dict["questions"]:
                    record[q["id"]] = parsed.get(q["id"])
            if verbose:
                print(".", end="", flush=True)
            return record

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            records = list(pool.map(_one, range(len(self.respondents))))
        if verbose:
            valid = sum(r["valid"] for r in records)
            print(f" done ({valid}/{len(records)} valid)")

        df = pd.DataFrame(records)
        if "valid" in df.columns:
            df["valid"] = df["valid"].astype(object)
        for q in survey_dict["questions"]:
            if q["type"] == "likert" and q["id"] in df.columns:
                df[q["id"]] = pd.to_numeric(df[q["id"]], errors="coerce")
        return df

    def run_ssr_survey(
        self,
        survey: Study | dict[str, Any],
        *,
        rater: SemanticSimilarityRater | Mapping[str, SemanticSimilarityRater],
        reference_set_id: str | Mapping[str, str] = "mean",
        image: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Run a survey with textual Likert elicitation and SSR scoring.

        Likert question values are expected scores. Raw elicited text is stored
        in ``{question_id}_text`` and per-score PMFs in
        ``{question_id}_pmf_{score}`` columns.
        """
        survey_dict = _as_study_dict(survey)
        stimulus = survey_dict.get("stimulus") or {}
        image_path = image or stimulus.get("image")

        def _one(idx: int) -> dict:
            row = self.respondents.iloc[idx].to_dict()
            if self.prompt_builder:
                prompt = self.prompt_builder(row, survey_dict)
            else:
                prompt = build_ssr_survey_prompt(row, survey_dict, persona_template=self.persona_template)
            try:
                raw = self.backend.get_completion(
                    prompt, image=image_path, temperature=self.temperature
                )
                parsed = parse_json_response(raw)
            except RuntimeError:
                parsed = None
            record = {
                "respondent_id": row.get("id"),
                "age": row.get("age"),
                "sex": row.get("sex"),
                "region": row.get("region"),
                "income16": row.get("income16"),
                "segment": row.get("segment", "broad"),
                "valid": parsed is not None,
            }
            if parsed:
                for q in survey_dict["questions"]:
                    value = parsed.get(q["id"])
                    if q["type"] == "likert":
                        record[f"{q['id']}_text"] = None if value is None else str(value)
                    else:
                        record[q["id"]] = value
            if verbose:
                print(".", end="", flush=True)
            return record

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            records = list(pool.map(_one, range(len(self.respondents))))
        if verbose:
            valid = sum(r["valid"] for r in records)
            print(f" done ({valid}/{len(records)} valid)")

        df = pd.DataFrame(records)
        if "valid" in df.columns:
            df["valid"] = df["valid"].astype(object)

        for q in survey_dict["questions"]:
            if q["type"] != "likert":
                continue
            qid = q["id"]
            text_col = f"{qid}_text"
            if text_col not in df.columns:
                continue
            scorer = rater[qid] if isinstance(rater, Mapping) else rater
            ref_id = reference_set_id[qid] if isinstance(reference_set_id, Mapping) else reference_set_id
            mask = df["valid"].astype(bool) & df[text_col].notna()
            if not mask.any():
                continue

            scored = scorer.score_texts(df.loc[mask, text_col].astype(str).tolist(), reference_set_id=ref_id)
            scored_frame = scored.to_frame(prefix=qid)
            for column in scored_frame.columns:
                if column not in df.columns:
                    df[column] = pd.NA
                df.loc[mask, column] = scored_frame[column].to_numpy()

        return df
