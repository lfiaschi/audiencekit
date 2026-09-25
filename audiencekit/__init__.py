"""AudienceKit — synthetic audience studies grounded in real survey rows.

Quick start:

    import audiencekit as ak

    pool = ak.load_panel()
    respondents = ak.sample_panel(pool, n=30)
    study = ak.Study.from_dict({...})
    results = ak.SyntheticPanel(respondents).run_survey(study)
"""

from .backends import AnthropicBackend, GeminiBackend, LLMBackend, Media, OpenAIBackend, make_backend, media_from_path
from .gss import load_gss, prepare_gss_persona_frame, write_gss_panel
from .personas import (
    GSS_PERSONA_FIELDS,
    GSS_PERSONA_TEMPLATE,
    build_persona,
    is_luxury_household,
    load_panel,
    sample_panel,
)
from .primitives import AudienceFrame, PersonaTemplate
from .survey import Question, Study, SyntheticPanel, build_survey_prompt, parse_json_response, render_persona

__all__ = [
    "AudienceFrame",
    "AnthropicBackend",
    "GeminiBackend",
    "GSS_PERSONA_FIELDS",
    "GSS_PERSONA_TEMPLATE",
    "LLMBackend",
    "Media",
    "OpenAIBackend",
    "PersonaTemplate",
    "Question",
    "Study",
    "SyntheticPanel",
    "build_persona",
    "build_survey_prompt",
    "is_luxury_household",
    "load_gss",
    "load_panel",
    "make_backend",
    "media_from_path",
    "parse_json_response",
    "prepare_gss_persona_frame",
    "render_persona",
    "sample_panel",
    "write_gss_panel",
]
