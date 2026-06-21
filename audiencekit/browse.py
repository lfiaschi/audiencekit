"""Persona-driven website navigation for synthetic audience members.

Where :mod:`audiencekit.survey` asks a persona to react to a *static* stimulus,
this module lets a persona *browse*: on each page it reads what is actually on
screen, reacts in first person, and chooses one real on-page action — click this
link, type into that field, scroll, or leave. The persona is the decision maker;
AudienceKit never guesses the next move with heuristics.

AudienceKit stays browser-agnostic on purpose (the same reason it does not bundle
a single LLM vendor). It owns the *decision layer*; the actual automation is
injected through the small :class:`Browser` protocol, so callers can plug in
``agent-browser``, Playwright, Selenium, or a recorded fixture.

Quick start::

    import audiencekit as ak

    pool = ak.load_panel()
    row = ak.sample_panel(pool, n=1, seed=13).iloc[0].to_dict()
    nav = ak.PersonaNavigator(row, backend_type="gemini",
                              task="find a dinner plan for next week")

    # `browser` is any object implementing the Browser protocol below.
    steps = ak.run_browse_session(nav, browser, "https://example.com/", max_steps=8)
    for step in steps:
        print(step.milestone, "->", step.quote)

Driving a single page manually::

    page = browser.read()
    step = nav.step(page, browser.actions())
    if not step.left:
        browser.execute(step.action)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence, Union, runtime_checkable

from .backends import make_backend
from .primitives import PersonaTemplate
from .survey import parse_json_response, render_persona

# Reserved id for the always-available "I would leave this site" choice.
LEAVE = "leave"

# The in-character fields every browse reaction returns. Callers can request more
# via PersonaNavigator(extra_fields=...), e.g. ("price_feel", "how the price feels").
DEFAULT_REACTION_FIELDS: tuple[tuple[str, str], ...] = (
    ("notices", "the one thing you notice first on this screen, in your words"),
    ("reaction", "your overall reaction: one of positive | mixed | negative"),
    ("believable", "how the main claim reads: one of believable | generic | exaggerated | unsure"),
    ("trust_shift", "how your trust moved: one of up | down | none"),
    ("intent", "your willingness to continue: one of continue | hesitant | leaning_out"),
)


@dataclass(frozen=True)
class PageView:
    """A snapshot of the page the persona is reacting to."""

    text: str = ""
    url: str = ""
    title: str = ""
    milestone: str = ""
    screenshot: Optional[Union[str, Path]] = None


@dataclass(frozen=True)
class BrowseAction:
    """One thing the persona can do on the current page.

    ``id`` is what the persona returns to pick this action. ``payload`` is opaque
    to AudienceKit and carried back to the caller's :class:`Browser` for execution
    (e.g. an element ref, a value to type, an action kind).
    """

    id: str
    label: str
    payload: Any = None


@dataclass(frozen=True)
class PersonaStep:
    """The outcome of one in-character browsing decision."""

    page: PageView
    reaction: dict[str, Any]
    action: Optional[BrowseAction]
    left: bool
    raw: str
    valid: bool

    @property
    def quote(self) -> str:
        return str(self.reaction.get("quote", "")).strip()

    @property
    def milestone(self) -> str:
        return self.page.milestone or self.page.url

    @property
    def leave_reason(self) -> str:
        return str(self.reaction.get("leave_reason", "")).strip()

    def summary(self) -> str:
        """A short line for the running history fed back into later steps."""
        where = self.page.milestone or self.page.title or self.page.url or "page"
        did = "left the site" if self.left else (f"clicked '{self.action.label}'" if self.action else "looked around")
        return f"{where}: {self.quote or 'reacted'} ({did})"


def _trim(text: str, limit: int = 6000) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "\n[...truncated...]"


def _render_actions(actions: Sequence[BrowseAction]) -> str:
    return "\n".join(f"  {action.id}: {action.label}" for action in actions) or "  (no actions)"


def build_browse_prompt(
    attributes: dict[str, Any],
    page: PageView,
    actions: Sequence[BrowseAction],
    *,
    task: str = "",
    history: Sequence[str] | None = None,
    reaction_fields: Sequence[tuple[str, str]] = DEFAULT_REACTION_FIELDS,
    persona_template: PersonaTemplate | str | Any | None = None,
) -> str:
    """Build the prompt that asks a persona to react and choose one action."""
    persona = render_persona(attributes, persona_template)
    task_block = f"\nYour goal: {task}\n" if task else ""
    trail = ""
    if history:
        trail = "\n# What you have already seen and done\n" + "\n".join(f"- {item}" for item in list(history)[-5:]) + "\n"
    where = page.milestone or page.title or "this screen"
    field_lines = "\n".join(f'- "{fid}": {desc}' for fid, desc in reaction_fields)
    return f"""# Role
You are a real person browsing a website, not a usability expert.
React naturally and stay consistent with who you are. You will not pay or place an order.

# Who you are
{persona}
{task_block}{trail}
# Where you are now
You are on "{where}" ({page.url}). Look at the attached screenshot if there is one.
Page content for reference:
\"\"\"
{_trim(page.text)}
\"\"\"

# What you can actually do here (choose exactly ONE by its id)
{_render_actions(actions)}

# Respond
React in character to THIS screen, then decide what YOU do next by picking one action id.
A real person rarely leaves on the very first screen — look around first; pick the '{LEAVE}'
action only once you have genuinely seen enough to walk away.

Return ONE JSON object with exactly these keys and nothing else (no markdown):
{field_lines}
- "quote": one vivid first-person sentence reacting to this exact screen
- "action_choice": the id of the single action you take next (use '{LEAVE}' to leave)
- "leave_reason": if you chose '{LEAVE}', why in your words; otherwise ''
"""


@runtime_checkable
class Browser(Protocol):
    """Minimal automation surface AudienceKit drives. Inject your own.

    Implementations wrap agent-browser, Playwright, Selenium, or a fixture.
    AudienceKit only reads pages, lists candidate actions, and asks to execute
    the one the persona chose — it never decides the next move itself.
    """

    def open(self, url: str) -> None: ...

    def read(self) -> PageView: ...

    def actions(self) -> Sequence[BrowseAction]: ...

    def execute(self, action: BrowseAction) -> None: ...

    def close(self) -> None: ...


class PersonaNavigator:
    """A sampled persona that reacts to pages and chooses its own next action."""

    def __init__(
        self,
        attributes: dict[str, Any],
        backend_type: str = "gemini",
        model: Optional[str] = None,
        backend: Any | None = None,
        *,
        task: str = "",
        persona_template: PersonaTemplate | str | Any | None = None,
        extra_fields: Sequence[tuple[str, str]] = (),
        temperature: float = 0.7,
        max_tokens: int = 1300,
    ):
        self.attributes = dict(attributes)
        self.backend = backend or make_backend(backend_type, model)
        self.task = task
        self.persona_template = persona_template
        self.reaction_fields = tuple(DEFAULT_REACTION_FIELDS) + tuple(extra_fields)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def step(
        self,
        page: PageView,
        actions: Sequence[BrowseAction],
        history: Sequence[str] | None = None,
        **backend_kwargs: Any,
    ) -> PersonaStep:
        """Elicit one in-character reaction and the persona's chosen action.

        A '{LEAVE}' action is always offered, so the persona can disengage on any
        page. ``backend_kwargs`` are forwarded to ``backend.get_completion`` (for
        example a Gemini ``thinking_config``).
        """
        menu = list(actions)
        if not any(action.id == LEAVE for action in menu):
            menu.append(BrowseAction(id=LEAVE, label="leave — I have seen enough and would not continue here"))
        by_id = {action.id: action for action in menu}

        prompt = build_browse_prompt(
            self.attributes,
            page,
            menu,
            task=self.task,
            history=history,
            reaction_fields=self.reaction_fields,
            persona_template=self.persona_template,
        )
        try:
            raw = self.backend.get_completion(
                prompt,
                image=str(page.screenshot) if page.screenshot else None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **backend_kwargs,
            )
        except RuntimeError:
            raw = ""
        parsed = parse_json_response(raw)
        valid = parsed is not None
        reaction = parsed or {}

        choice = str(reaction.get("action_choice", "")).strip()
        chosen = by_id.get(choice)
        left = chosen is None or chosen.id == LEAVE
        return PersonaStep(
            page=page,
            reaction=reaction,
            action=None if left else chosen,
            left=left,
            raw=raw or "",
            valid=valid,
        )


def run_browse_session(
    navigator: PersonaNavigator,
    browser: Browser,
    start_url: str,
    *,
    max_steps: int = 8,
    boundary: Callable[[PageView], bool] | None = None,
    verbose: bool = False,
) -> list[PersonaStep]:
    """Drive a full persona-led walkthrough and return the ordered steps.

    The persona decides every move. The session ends when the persona leaves, a
    page has no actionable next step, ``max_steps`` is reached, or ``boundary``
    returns True for the current page (use it to stop at out-of-scope walls such
    as payment, password, phone, or captcha).
    """
    steps: list[PersonaStep] = []
    history: list[str] = []
    try:
        browser.open(start_url)
        for _ in range(max_steps):
            page = browser.read()
            actions = list(browser.actions())
            step = navigator.step(page, actions, history=history)
            steps.append(step)
            history.append(step.summary())
            if verbose:
                print(f"{step.milestone}: {step.quote}")
            if boundary is not None and boundary(page):
                break
            if step.left or step.action is None:
                break
            browser.execute(step.action)
    finally:
        browser.close()
    return steps
