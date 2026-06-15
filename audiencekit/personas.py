"""Build synthetic-consumer personas from real GSS respondent attributes.

Each persona is a first-person identity card rendered from one row of the
General Social Survey (NORC, University of Chicago). The card conditions an
LLM to answer survey questions as that respondent would.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .primitives import AudienceFrame, PersonaTemplate

DEFAULT_PANEL_RESOURCE = "2024_stata.zip"

# Income values used as the affluence anchor for the luxury segment.
HIGH_INCOME_VALUES = {
    "$130,000 to $149,999",
    "$150,000 to $169,999",
    "$170,000 or over",
}

PERSONA_TEMPLATE = """\
You are a {age} year old {sex} {race} adult living in the {region} region of the United States, raised {res16}.
You are {marital}. Your highest education is {educ} (degree: {degree}).
Your reported family income last year before taxes was {income16}, from all family sources, not just salary.
You describe your social class as {class_}.
You work in: {occ10} (occupational prestige: {prestg10}). Compared with other households, your financial situation is {finrela}, and you feel {satfin} about it.
Politically you identify as {partyid} and consider yourself {polviews}.
Your religious preference is {relig}; you attend services {attend}.
You have {childs} children. You describe yourself as {happy} overall and your health as {health}.
You watch about {tvhours} hours of TV per day; internet user: {usewww}.
You believe getting ahead in life comes from {getahead}."""
GSS_PERSONA_TEMPLATE = PersonaTemplate(PERSONA_TEMPLATE)


def normalize(value: object) -> str:
    """Map missing/invalid values to the literal string 'Unknown'."""
    if value is None:
        return "Unknown"
    try:
        if pd.isna(value):
            return "Unknown"
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "invalid", "refused", "don't know"}:
        return "Unknown"
    return text


def build_persona(attributes: Mapping[str, Any]) -> str:
    """Render the persona card for one respondent row (dict or Series)."""
    fields: dict[str, Any] = {
        key: attributes.get(key)
        for key in [
            "age", "sex", "race", "region", "res16", "marital", "educ",
            "degree", "income16", "occ10", "prestg10", "finrela", "satfin",
            "partyid", "polviews", "relig", "attend", "childs", "happy",
            "health", "tvhours", "usewww", "getahead",
        ]
    }
    fields["class_"] = attributes.get("class")
    return GSS_PERSONA_TEMPLATE.render(fields)


def load_panel(path: str | Path | None = None) -> pd.DataFrame:
    """Load a prepared AudienceKit persona panel.

    With no path, loads the bundled GSS 2024 public-use Stata file and prepares
    it as an AudienceKit frame. Explicit paths can point to either a prepared
    panel CSV or a raw GSS source file supported by ``audiencekit.gss.load_gss``.
    """
    if path is None:
        with resources.as_file(resources.files("audiencekit").joinpath("data", DEFAULT_PANEL_RESOURCE)) as bundled:
            from .gss import load_gss

            df = load_gss(bundled, years=[2024])
    elif Path(path).suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype=str)
    else:
        from .gss import load_gss

        df = load_gss(path)
    if "id" not in df.columns:
        raise ValueError(f"Panel file {path} has no 'id' column")
    return df


def is_luxury_household(row: Mapping[str, Any]) -> bool:
    """Affluent-household proxy: an affluence anchor plus a second signal."""
    score = 0
    anchored = False
    if normalize(row.get("income16")) in HIGH_INCOME_VALUES:
        score += 1
        anchored = True
    if normalize(row.get("class")) == "Upper Class":
        score += 1
        anchored = True
    if normalize(row.get("degree")) in {"Bachelor's", "Graduate"}:
        score += 1
    if normalize(row.get("prestg10")) == "High":
        score += 1
    if normalize(row.get("finrela")) in {"Above average", "Far above average"}:
        score += 1
    return anchored and score >= 2


def sample_panel(
    df: pd.DataFrame,
    n: int = 30,
    segment: str | Callable[[Mapping[str, Any]], bool] | None = "broad",
    seed: int = 42,
    weighted: bool = True,
    weight_column: str = "weight",
    segment_name: str | None = None,
) -> pd.DataFrame:
    """Sample respondents for a survey run.

    segment: ``"broad"``/``None`` for the whole frame, ``"luxury"`` for the
    built-in affluent-household proxy, or a callable that receives each row and
    returns True for respondents to keep.

    weighted (default True): draw with each respondent's GSS survey weight
    (``weight_column``, usually ``wtssnrps`` renamed to ``weight``), so a broad
    panel is representative of the weighted sampling frame rather than of raw
    respondent rows. Segment samples are weighted within the segment. Set
    weighted=False for a plain uniform draw over rows.
    """
    pool = df.copy()
    label = segment_name or ("broad" if segment is None else segment)
    if segment in {None, "broad"}:
        label = segment_name or "broad"
    elif segment == "luxury":
        mask = df.apply(is_luxury_household, axis=1)
        pool = df[mask]
        label = segment_name or "luxury"
    elif callable(segment):
        mask = df.apply(segment, axis=1)
        pool = df[mask]
        label = segment_name or "custom"
    else:
        raise ValueError("segment must be 'broad', 'luxury', None, or a callable row filter")
    if pool.empty:
        raise ValueError(f"No respondents available for segment '{label}'")

    try:
        return AudienceFrame(pool, weight_column=weight_column).sample(
            n=n,
            seed=seed,
            weighted=weighted,
            segment_name=str(label),
        )
    except ValueError as exc:
        if weight_column in str(exc):
            raise ValueError(
                f"weighted=True needs a {weight_column!r} column with positive values; "
                "prepare GSS data with audiencekit.gss.load_gss or pass weighted=False"
            ) from exc
        raise
