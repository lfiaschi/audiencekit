from __future__ import annotations

import json

from audiencekit import (
    BrowseAction,
    PageView,
    PersonaNavigator,
    build_browse_prompt,
    run_browse_session,
)


class ScriptedBackend:
    """Returns a queued JSON reaction per call and records prompts."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def get_completion(self, prompt, image=None, **kwargs):
        self.prompts.append((prompt, image, kwargs))
        return self.responses.pop(0)


def _row():
    return {"age": 71, "sex": "Female", "region": "South", "income16": "$20,000 to $22,499"}


def test_build_browse_prompt_lists_actions_and_persona():
    prompt = build_browse_prompt(
        _row(),
        PageView(text="Menu. $11.49/serving.", url="https://x.test/menu", milestone="menu"),
        [BrowseAction(id="a1", label="click the 'See plans' button")],
        task="find a dinner plan",
    )
    assert "find a dinner plan" in prompt
    assert "a1: click the 'See plans' button" in prompt
    assert "action_choice" in prompt
    assert "71 year old" in prompt or "71" in prompt


def test_step_resolves_chosen_action():
    backend = ScriptedBackend([json.dumps({"action_choice": "a2", "quote": "Let me see the menu."})])
    nav = PersonaNavigator(_row(), backend=backend)
    actions = [BrowseAction("a1", "click 'Pricing'"), BrowseAction("a2", "click 'Menu'", payload={"ref": "e9"})]
    step = nav.step(PageView(text="home", url="u"), actions)
    assert step.valid is True
    assert step.left is False
    assert step.action is not None and step.action.id == "a2"
    assert step.action.payload == {"ref": "e9"}
    assert step.quote == "Let me see the menu."


def test_step_leave_choice_marks_session_done():
    backend = ScriptedBackend([json.dumps({"action_choice": "leave", "leave_reason": "Too pricey.", "quote": "Not for me."})])
    nav = PersonaNavigator(_row(), backend=backend)
    step = nav.step(PageView(text="$$$"), [BrowseAction("a1", "click 'Plans'")])
    assert step.left is True
    assert step.action is None
    assert step.leave_reason == "Too pricey."


def test_step_unparseable_or_unknown_choice_defaults_to_leave():
    backend = ScriptedBackend(["not json at all", json.dumps({"action_choice": "a99", "quote": "?"})])
    nav = PersonaNavigator(_row(), backend=backend)
    bad = nav.step(PageView(text="x"), [BrowseAction("a1", "click")])
    assert bad.valid is False and bad.left is True
    unknown = nav.step(PageView(text="x"), [BrowseAction("a1", "click")])
    assert unknown.left is True  # unknown id falls back to leaving


def test_extra_fields_are_requested_in_prompt():
    backend = ScriptedBackend([json.dumps({"action_choice": "leave"})])
    nav = PersonaNavigator(_row(), backend=backend, extra_fields=(("price_feel", "how the price feels to you"),))
    nav.step(PageView(text="x"), [BrowseAction("a1", "click")])
    assert "price_feel" in backend.prompts[0][0]


class FakeBrowser:
    """A two-page fixture site driven through the Browser protocol."""

    def __init__(self):
        self.opened = None
        self.executed = []
        self.closed = False
        self._page = 0

    def open(self, url):
        self.opened = url

    def read(self):
        if self._page == 0:
            return PageView(text="Homepage", url="https://x.test/", milestone="homepage")
        return PageView(text="Menu with prices", url="https://x.test/menu", milestone="menu")

    def actions(self):
        return [BrowseAction("a1", "click 'Menu'", payload="menu-ref")]

    def execute(self, action):
        self.executed.append(action)
        self._page += 1

    def close(self):
        self.closed = True


def test_run_browse_session_lets_persona_drive_then_leave():
    backend = ScriptedBackend(
        [
            json.dumps({"action_choice": "a1", "quote": "I'll open the menu."}),
            json.dumps({"action_choice": "leave", "leave_reason": "Too expensive.", "quote": "Out of my budget."}),
        ]
    )
    nav = PersonaNavigator(_row(), backend=backend)
    browser = FakeBrowser()
    steps = run_browse_session(nav, browser, "https://x.test/", max_steps=8)

    assert browser.opened == "https://x.test/"
    assert browser.closed is True
    assert len(steps) == 2
    assert [s.page.milestone for s in steps] == ["homepage", "menu"]
    assert steps[0].action.payload == "menu-ref"
    assert steps[-1].left is True
    assert len(browser.executed) == 1  # second step left, so no further navigation
    assert "homepage" in steps[1].summary() or "menu" in steps[1].summary()


def test_run_browse_session_stops_at_boundary():
    backend = ScriptedBackend([json.dumps({"action_choice": "a1", "quote": "Looks like a payment wall."})])
    nav = PersonaNavigator(_row(), backend=backend)
    browser = FakeBrowser()
    steps = run_browse_session(
        nav, browser, "https://x.test/", max_steps=8, boundary=lambda page: "Homepage" in page.text
    )
    assert len(steps) == 1  # boundary on the first page halts the walkthrough
    assert browser.executed == []
    assert browser.closed is True
