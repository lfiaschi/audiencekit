"""Prepare General Social Survey rows for AudienceKit personas.

AudienceKit does not require a bundled GSS extract. Users can download the
full cumulative GSS file from NORC, read it through this module, and sample
weighted synthetic audience panels from the prepared frame.
"""

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path
from typing import Iterable

import pandas as pd

PERSONA_COLUMNS = [
    "id",
    "year",
    "wtssnrps",
    "age",
    "sex",
    "race",
    "region",
    "res16",
    "marital",
    "educ",
    "degree",
    "income16",
    "class",
    "occ10",
    "prestg10",
    "finrela",
    "satfin",
    "partyid",
    "polviews",
    "relig",
    "attend",
    "childs",
    "happy",
    "health",
    "tvhours",
    "usewww",
    "getahead",
]

# Rows must be complete only on post-stratification fields plus a valid weight.
# Income and other attributes may be missing and render as "Unknown".
CORE_FIELDS = ["age", "sex", "race", "region", "educ"]

GSS_MISSING_VALUES = {-100, -99, -98, -97, -90, -80, -70}

MAPPINGS = {
    "sex": {1: "Male", 2: "Female"},
    "race": {1: "White", 2: "Black", 3: "Other"},
    "marital": {1: "Married", 2: "Widowed", 3: "Divorced", 4: "Separated", 5: "Never Married"},
    "happy": {1: "Very Happy", 2: "Pretty Happy", 3: "Not Too Happy"},
    "health": {1: "Excellent", 2: "Good", 3: "Fair", 4: "Poor"},
    "polviews": {
        1: "Extremely Liberal",
        2: "Liberal",
        3: "Slightly Liberal",
        4: "Moderate",
        5: "Slightly Conservative",
        6: "Conservative",
        7: "Extremely Conservative",
    },
    "attend": {
        0: "Never",
        1: "Less Than Once A Year",
        2: "Once A Year",
        3: "Several Times A Year",
        4: "Once A Month",
        5: "2-3 Times A Month",
        6: "Nearly Every Week",
        7: "Every Week",
        8: "More Than Once A Week",
    },
    "region": {
        1: "New England",
        2: "Middle Atlantic",
        3: "East North Central",
        4: "West North Central",
        5: "South Atlantic",
        6: "East South Central",
        7: "West South Central",
        8: "Mountain",
        9: "Pacific",
    },
    "educ": {
        **{i: f"{i}th Grade" for i in range(4, 13)},
        0: "No Formal Schooling",
        1: "1st Grade",
        2: "2nd Grade",
        3: "3rd Grade",
        13: "1 Year College",
        14: "2 Years College",
        15: "3 Years College",
        16: "4 Years College",
        17: "5+ Years College",
    },
    # INCOME16 is the expanded current-family-income card. It is the right
    # persona-facing GSS income field; INCOM16 is a different childhood-origin
    # variable and should not be used as current income.
    "income16": {
        1: "Under $1,000",
        2: "$1,000 to $2,999",
        3: "$3,000 to $3,999",
        4: "$4,000 to $4,999",
        5: "$5,000 to $5,999",
        6: "$6,000 to $6,999",
        7: "$7,000 to $7,999",
        8: "$8,000 to $9,999",
        9: "$10,000 to $12,499",
        10: "$12,500 to $14,999",
        11: "$15,000 to $17,499",
        12: "$17,500 to $19,999",
        13: "$20,000 to $22,499",
        14: "$22,500 to $24,999",
        15: "$25,000 to $29,999",
        16: "$30,000 to $34,999",
        17: "$35,000 to $39,999",
        18: "$40,000 to $49,999",
        19: "$50,000 to $59,999",
        20: "$60,000 to $74,999",
        21: "$75,000 to $89,999",
        22: "$90,000 to $109,999",
        23: "$110,000 to $129,999",
        24: "$130,000 to $149,999",
        25: "$150,000 to $169,999",
        26: "$170,000 or over",
    },
    "partyid": {
        0: "Strong Democrat",
        1: "Democrat",
        2: "Independent, Near Democrat",
        3: "Independent",
        4: "Independent, Near Republican",
        5: "Republican",
        6: "Strong Republican",
        7: "Other",
    },
    "relig": {1: "Protestant", 2: "Catholic", 3: "Jewish", 4: "None", 5: "Other"},
    "class": {1: "Lower Class", 2: "Working Class", 3: "Middle Class", 4: "Upper Class"},
    "satfin": {1: "Satisfied", 2: "More or less satisfied", 3: "Not Satisfied"},
    "getahead": {1: "Hard Work", 2: "Connections", 3: "Luck"},
    "res16": {
        1: "in open country but not on a farm",
        2: "on a farm",
        3: "in a small city or town (under 50,000)",
        4: "in a medium-size city (50,000-250,000)",
        5: "in a suburb near a large city",
        6: "in a large city (over 250,000)",
    },
    "degree": {
        0: "Less than high school",
        1: "High school",
        2: "Associate/Junior college",
        3: "Bachelor's",
        4: "Graduate",
    },
    "finrela": {
        1: "Far below average",
        2: "Below average",
        3: "Average",
        4: "Above average",
        5: "Far above average",
    },
    "occ10": {
        10: "Management, professional, and related occupations",
        3600: "Service occupations",
        4700: "Sales and office occupations",
        6005: "Natural resources, construction, and maintenance occupations",
        7700: "Production, transportation, and material moving occupations",
        9830: "Military specific occupations",
    },
    "usewww": {1: "Yes", 2: "No"},
    "age": lambda x: str(int(x)) if 0 <= x <= 100 else None,
    "childs": lambda x: str(int(x)) if 0 <= x <= 20 else None,
    "tvhours": lambda x: str(int(x)) if 0 <= x <= 24 else None,
    "prestg10": lambda x: "Low" if 0 <= x < 50 else "High" if 50 <= x <= 100 else None,
}


def load_gss(
    path: str | Path,
    *,
    years: Iterable[int] | int | None = None,
    columns: list[str] | None = None,
    weight_column: str = "wtssnrps",
) -> pd.DataFrame:
    """Read a full GSS file and return an AudienceKit persona frame.

    Supported inputs are Stata ``.dta``, zipped Stata files, CSV, and Parquet
    files. For Stata, ``convert_categoricals=False`` is used so the mapping
    layer handles codes consistently across releases.
    """
    source = Path(path)
    read_columns = columns or PERSONA_COLUMNS
    if weight_column not in read_columns:
        read_columns = [*read_columns, weight_column]

    if source.suffix.lower() == ".dta":
        raw = _read_stata_selected(str(source), read_columns)
    elif source.suffix.lower() == ".zip":
        raw = _read_stata_zip(source, read_columns)
    elif source.suffix.lower() == ".csv":
        raw = pd.read_csv(source, usecols=lambda col: col in set(read_columns))
    elif source.suffix.lower() in {".parquet", ".pq"}:
        raw = pd.read_parquet(source, columns=read_columns)
    else:
        raise ValueError(f"Unsupported GSS file type: {source.suffix}")

    return prepare_gss_persona_frame(raw, years=years, weight_column=weight_column)


def _read_stata_zip(source: Path, columns: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(source) as archive:
        stata_members = [
            name
            for name in archive.namelist()
            if not name.endswith("/") and name.lower().endswith(".dta")
        ]
        if not stata_members:
            raise ValueError(f"No .dta file found in {source}")
        if len(stata_members) > 1:
            raise ValueError(f"Expected one .dta file in {source}, found {stata_members}")
        payload = BytesIO(archive.read(stata_members[0]))
    return _read_stata_selected(payload, columns)


def _read_stata_selected(source, columns: list[str]) -> pd.DataFrame:
    try:
        return pd.read_stata(source, convert_categoricals=False, columns=columns)
    except ValueError as exc:
        if "not found in the Stata data set" not in str(exc):
            raise
        if hasattr(source, "seek"):
            source.seek(0)
        raw = pd.read_stata(source, convert_categoricals=False)
        available = [column for column in columns if column in raw.columns]
        return raw[available]


def prepare_gss_persona_frame(
    df: pd.DataFrame,
    *,
    years: Iterable[int] | int | None = None,
    weight_column: str = "wtssnrps",
) -> pd.DataFrame:
    """Map raw GSS rows to the persona frame consumed by AudienceKit.

    ``years`` can be a single year, a list of years, or ``None`` for all years
    in the provided data. Sampling weights are preserved as ``weight``.
    """
    prepared = df.copy()
    _require_columns(prepared, ["id", "year", weight_column, *CORE_FIELDS])

    if years is not None:
        selected_years = {int(years)} if isinstance(years, int) else {int(year) for year in years}
        prepared = prepared[prepared["year"].astype(int).isin(selected_years)].copy()

    for column in PERSONA_COLUMNS:
        if column not in prepared.columns and column != "wtssnrps":
            prepared[column] = None

    for column in prepared.columns:
        if column in MAPPINGS:
            prepared[column] = _map_values(prepared[column], MAPPINGS[column])

    prepared = prepared.dropna(subset=CORE_FIELDS + [weight_column]).copy()
    prepared[weight_column] = pd.to_numeric(prepared[weight_column], errors="coerce")
    prepared = prepared[prepared[weight_column] > 0].reset_index(drop=True)

    year_text = prepared["year"].astype(int).astype(str)
    id_text = prepared["id"].astype(int).astype(str)
    prepared["id"] = year_text + "-" + id_text
    prepared = prepared.rename(columns={weight_column: "weight"})

    output_columns = ["id", "weight", *[col for col in PERSONA_COLUMNS if col not in {"id", "year", "wtssnrps"}]]
    return prepared[output_columns]


def write_gss_panel(
    source_path: str | Path,
    output_path: str | Path,
    *,
    years: Iterable[int] | int | None = None,
    columns: list[str] | None = None,
    weight_column: str = "wtssnrps",
) -> Path:
    """Prepare a GSS persona panel and write it to CSV."""
    panel = load_gss(source_path, years=years, columns=columns, weight_column=weight_column)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out, index=False)
    return out


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"GSS data is missing required columns: {missing}")


def _map_values(series: pd.Series, mapping) -> pd.Series:
    def convert(value):
        if pd.isna(value):
            return None
        if isinstance(value, (int, float)) and int(value) in GSS_MISSING_VALUES:
            return None
        if callable(mapping):
            return mapping(value)
        return mapping.get(value)

    return pd.Series([convert(value) for value in series], index=series.index, dtype=object)
