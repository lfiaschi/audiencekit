"""AudienceKit — synthetic audience studies grounded in real survey rows.

Quick start:

    import audiencekit as ak

    pool = ak.load_panel()
    respondents = ak.sample_panel(pool, n=30)
    study = ak.Study.from_dict({...})
    results = ak.SyntheticPanel(respondents).run_survey(study)

Persona-driven browsing (the persona reads each page and picks its own next move):

    row = ak.sample_panel(pool, n=1, seed=13).iloc[0].to_dict()
    nav = ak.PersonaNavigator(row, task="find a dinner plan for next week")
    steps = ak.run_browse_session(nav, browser, "https://example.com/")
"""

from .backends import AnthropicBackend, GeminiBackend, LLMBackend, OpenAIBackend, make_backend
from .browse import (
    DEFAULT_REACTION_FIELDS,
    LEAVE,
    BrowseAction,
    Browser,
    PageView,
    PersonaNavigator,
    PersonaStep,
    build_browse_prompt,
    run_browse_session,
)
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
    "BrowseAction",
    "Browser",
    "DEFAULT_REACTION_FIELDS",
    "GeminiBackend",
    "GSS_PERSONA_FIELDS",
    "GSS_PERSONA_TEMPLATE",
    "LEAVE",
    "LLMBackend",
    "OpenAIBackend",
    "PageView",
    "PersonaNavigator",
    "PersonaStep",
    "PersonaTemplate",
    "Question",
    "Study",
    "SyntheticPanel",
    "build_browse_prompt",
    "build_persona",
    "build_survey_prompt",
    "is_luxury_household",
    "load_gss",
    "load_panel",
    "make_backend",
    "parse_json_response",
    "prepare_gss_persona_frame",
    "render_persona",
    "run_browse_session",
    "sample_panel",
    "write_gss_panel",
]
