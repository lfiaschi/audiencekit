# AudienceKit

AudienceKit is a Python library for synthetic audience research grounded in
real respondent rows.

The core idea is simple: start from a real sampling frame, render each row as a
persona, run a structured study through an LLM backend, and analyze the output
like a directional research instrument. GSS is the first included data adapter,
but the primitives are dataset-agnostic.

## Install

```bash
uv venv
uv pip install -e ".[dev]"
```

Set a model API key:

```bash
export OPENAI_API_KEY=...
```

## Quick Start

```python
import audiencekit as ak

pool = ak.load_panel()
respondents = ak.sample_panel(pool, n=50, segment="broad", seed=42)

study = ak.Study.from_dict({
    "title": "Concept test",
    "stimulus": {"description": "A compact EV designed for city commuters."},
    "questions": [
        {"id": "fit", "type": "likert", "text": "How well does this fit your life?"},
        {"id": "first_reaction", "type": "text", "text": "What is your first reaction?"},
    ],
})

results = ak.SyntheticPanel(respondents).run_survey(study)
```

## Generic Primitives

AudienceKit is not tied to GSS. New datasets should be added as adapters that
produce ordinary DataFrames, then use the same primitives:

```python
frame = ak.AudienceFrame(my_dataframe, id_column="person_id", weight_column="survey_weight")
sample = frame.sample(n=100, segment=lambda row: row["country"] == "US")

template = ak.PersonaTemplate("You are {age}, live in {region}, and buy {category}.")
persona = template.render(sample.iloc[0].to_dict())

panel = ak.SyntheticPanel(sample, persona_template=template, backend=my_backend)
results = panel.run_survey(study)
```

## GSS Adapter

`ak.load_panel()` loads a small bundled GSS 2022 sample panel for examples and
smoke tests. For production studies, download the full General Social Survey
cumulative file from NORC, then prepare a weighted persona frame:

```python
pool = ak.load_gss("path/to/gss7224_r3.dta", years=[2024])
respondents = ak.sample_panel(pool, n=600, weighted=True)
```

`audiencekit.gss` maps selected GSS codes to readable labels, preserves the GSS
survey weight as `weight`, and keeps missing non-core persona attributes as
`Unknown` rather than dropping those respondents.

The Apache License 2.0 covers AudienceKit code. Bundled sample data and
example assets are documented separately in `NOTICE.md` and should be treated
according to their source terms.

## Examples

- `examples/ferrari_luce/` contains the Ferrari Luce concept-test study spec,
  stimulus assets, and a notebook-style walkthrough.
- `skills/` contains optional agent workflows for survey generation and
  persona website browsing.
- `scripts/` contains small utility scripts; the Python API is the primary
  interface.

## Methodological Grounding

AudienceKit should be used as a structured hypothesis generator, not as a
replacement for fieldwork.

The strongest current validation signal is the SSR paper:
[LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation of Likert Ratings](https://arxiv.org/abs/2510.08338).
It finds that directly asking LLMs for numeric Likert ratings can distort
response distributions, while text-first semantic similarity rating performs
substantially better against human purchase-intent studies.

AudienceKit v0.1 keeps direct structured Likert questions because they are
simple, inspectable, and useful for within-run pressure tests. Treat SSR-style
text-first scoring as the stronger validation direction for future adapters or
custom backends, not as a feature this release already implements.

The Ferrari Luce example shows the discipline this library is meant to
support: benchmark cells, fixed stimuli, treatment arms, item diagnostics,
bootstrap intervals, and an explicit skeptical-review section. Add the
production case-study URL here once the post is live.

When reporting results, be precise:

- Claims are conditional on the model, prompt, stimuli, and audience frame.
- Synthetic confidence intervals are not human survey sampling intervals.
- Benchmark/reference cells are safer than interpreting raw scores in isolation.
- Open-ended text is often more useful than compressed Likert numbers.
- A good synthetic run sharpens a human study; it should not replace one.

## Development

```bash
uv run --extra dev python -m pytest tests
```
