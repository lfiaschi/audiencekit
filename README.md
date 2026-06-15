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
export GEMINI_API_KEY=...
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

By default, `SyntheticPanel` uses Gemini with `gemini-2.5-flash`.

## Model Backends

AudienceKit supports Gemini, OpenAI, Anthropic, and custom backend objects.
Gemini is the default:

```python
panel = ak.SyntheticPanel(respondents)  # Gemini, gemini-2.5-flash
```

Select another managed backend:

```python
panel = ak.SyntheticPanel(respondents, backend_type="openai", model="gpt-4o-mini")
panel = ak.SyntheticPanel(respondents, backend_type="anthropic")
```

Required API keys:

- Gemini: `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`

For tests, local models, or another provider, pass any object with a
`get_completion(prompt, image=None, **kwargs)` method:

```python
class MyBackend:
    def get_completion(self, prompt, image=None, **kwargs):
        return call_my_model(prompt, image=image, **kwargs)

panel = ak.SyntheticPanel(respondents, backend=MyBackend())
```

## Extending Datasets

AudienceKit is not tied to GSS. A dataset adapter only needs to produce a
DataFrame with one row per audience member, an id column, and optionally a
weight column. Then use the generic primitives:

```python
frame = ak.AudienceFrame(my_dataframe, id_column="person_id", weight_column="survey_weight")
sample = frame.sample(n=100, segment=lambda row: row["country"] == "US")

template = ak.PersonaTemplate("You are {age}, live in {region}, and buy {category}.")
persona = template.render(sample.iloc[0].to_dict())

panel = ak.SyntheticPanel(sample, persona_template=template)
results = panel.run_survey(study)
```

Recommended adapter shape:

```python
def load_my_panel(path):
    raw = read_my_source(path)
    return raw.rename(columns={"respondent_id": "id", "survey_weight": "weight"})
```

Keep dataset-specific cleaning, labels, and missing-value rules in the adapter.
Keep `AudienceFrame`, `PersonaTemplate`, `Study`, and `SyntheticPanel`
dataset-neutral.

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

## Skills

AudienceKit ships two optional agent skills:

- `skills/survey/`: turn a research brief into a study spec, sample an
  audience frame, run a panel, and summarize directional findings.
- `skills/persona-browse/`: sample one persona and run a short qualitative
  website walkthrough in that persona's voice.

To use them, copy or symlink the skill folders into your agent's skill
directory, or point your agent at this repository's `skills/` directory if your
runtime supports repo-local skills. The skills are workflow guides; the Python
API remains the source of truth.

## Customizing Prompts

AudienceKit has two prompt customization layers.

Customize persona rendering with `PersonaTemplate`:

```python
template = ak.PersonaTemplate(
    "You are {age}, live in {region}, shop for {category}, "
    "and describe price sensitivity as {price_sensitivity}."
)

panel = ak.SyntheticPanel(respondents, persona_template=template)
```

For full control, pass a `prompt_builder(row, study_dict)` callable. This
replaces AudienceKit's default survey prompt while keeping sampling, backend
calls, parsing, and report utilities:

```python
def prompt_builder(row, study):
    return f"""
You are responding as this audience member:
age={row["age"]}, segment={row["segment"]}

Answer this study as JSON with these fields:
{[q["id"] for q in study["questions"]]}
"""

panel = ak.SyntheticPanel(respondents, prompt_builder=prompt_builder)
```

You can also subclass or implement a backend when a provider needs a different
message format.

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
