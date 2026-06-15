---
name: survey
description: Create and run a synthetic audience research study with AudienceKit. Use when a user wants to test a product concept, message, price, ad, brand idea, or treatment comparison against a weighted audience frame.
---

# Synthetic Audience Survey

Use AudienceKit to turn a research brief into a structured study, run it on a sampled audience frame, and summarize directional findings.

## Inputs

Extract these from the brief:

- Research objective: the decision or hypothesis being pressure-tested.
- Stimulus: product, message, ad, concept, price, or image path.
- Audience frame: prepared CSV, GSS file, or custom DataFrame.
- Segment: broad population, named segment, or explicit row-filter rule.
- Treatments: alternate framings or concepts to compare.
- Panel size: default 25 for live exploration; use 100+ only when requested.
- Backend: Gemini is the default; use OpenAI, Anthropic, or a custom backend only when requested.

Ask one clarifying question only when the core research objective or audience is ambiguous.

## Study Spec

Create a JSON/YAML-compatible `audiencekit.Study` with:

- 3-8 questions total.
- At least one `likert` item and one `text` item.
- `choice` items for forced verdicts or treatment comparisons.
- Neutral wording. Avoid leading language.
- Short snake_case question ids.

Use benchmarks/reference cells when interpreting a score would otherwise be ambiguous.

## Run

Use the Python API first:

```python
import audiencekit as ak

pool = ak.load_panel()
respondents = ak.sample_panel(pool, n=50, segment="broad", seed=42)
study = ak.Study.from_json("examples/my_study/study.json")
results = ak.SyntheticPanel(respondents).run_survey(study)
```

For a user-prepared panel:

```python
pool = ak.load_panel("data/gss_panel.csv")
```

For full GSS files:

```python
pool = ak.load_gss("path/to/gss7224_r3.dta", years=[2024])
```

For non-GSS datasets, create an `ak.AudienceFrame` and pass a custom persona template:

```python
frame = ak.AudienceFrame(df, id_column="person_id", weight_column="survey_weight")
respondents = frame.sample(n=50, segment_name="target")
template = ak.PersonaTemplate("You are {age}, live in {region}, and buy {category}.")
results = ak.SyntheticPanel(respondents, persona_template=template).run_survey(study)
```

Use `prompt_builder(row, study_dict)` only when the default survey prompt is too generic for the research design. Preserve JSON-only output instructions and exact question ids.

## Report

Summarize:

- Valid and invalid response counts.
- Mean scores and distributions for Likert items.
- Treatment effects against baseline or benchmark cells.
- Demographic or segment cuts only when they are material.
- 3-5 verbatims that explain the pattern.

Frame every result as a directional pressure test, not as a fieldwork substitute. Include limitations around model choice, prompt sensitivity, panel fit, and response-distribution compression.
