"""AudienceKit — synthetic audience studies grounded in real survey rows.

Quick start:

    import audiencekit as ak

    pool = ak.load_panel()
    respondents = ak.sample_panel(pool, n=30)
    study = ak.Study.from_dict({...})
    results = ak.SyntheticPanel(respondents).run_survey(study)
"""

from .backends import AnthropicBackend, GeminiBackend, LLMBackend, OpenAIBackend, make_backend
from .gss import load_gss, prepare_gss_persona_frame, write_gss_panel
from .personas import build_persona, is_luxury_household, load_panel, sample_panel
from .primitives import AudienceFrame, PersonaTemplate
from .ssr import (
    SSRAnchorSet,
    SSRResult,
    SemanticSimilarityRater,
    SentenceTransformerEmbeddings,
    purchase_intent_anchor_sets,
)
from .survey import (
    Question,
    Study,
    SyntheticPanel,
    build_ssr_survey_prompt,
    build_survey_prompt,
    parse_json_response,
    render_persona,
)

__all__ = [
    "AudienceFrame",
    "AnthropicBackend",
    "GeminiBackend",
    "LLMBackend",
    "OpenAIBackend",
    "PersonaTemplate",
    "Question",
    "SSRAnchorSet",
    "SSRResult",
    "SemanticSimilarityRater",
    "SentenceTransformerEmbeddings",
    "Study",
    "SyntheticPanel",
    "build_persona",
    "build_ssr_survey_prompt",
    "build_survey_prompt",
    "is_luxury_household",
    "load_gss",
    "load_panel",
    "make_backend",
    "parse_json_response",
    "prepare_gss_persona_frame",
    "purchase_intent_anchor_sets",
    "render_persona",
    "sample_panel",
    "write_gss_panel",
]
