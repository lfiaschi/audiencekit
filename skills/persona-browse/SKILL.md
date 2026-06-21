---
name: persona-browse
description: Have one synthetic audience member browse a website and narrate a first-person stream of consciousness. Use for synthetic UX walkthroughs, landing-page reactions, checkout friction, product discovery, and qualitative website testing.
---

# Persona Browse

Run a short qualitative website walkthrough from one sampled persona's point of view.

## Inputs

- URL to browse.
- Audience frame or persona constraints.
- Optional task, e.g. "find pricing", "evaluate the product", or "decide whether to sign up".

If no persona is specified, sample one respondent from the available frame with a fresh seed.

## Cast The Persona

Use AudienceKit to sample and render the persona:

```python
import audiencekit as ak

pool = ak.load_panel()
row = ak.sample_panel(pool, n=1, seed=13).iloc[0].to_dict()
print(ak.build_persona(row))
```

For non-GSS data, use `ak.AudienceFrame` plus `ak.PersonaTemplate`.

Show the persona card before browsing. Keep the attributes in working memory and let them shape attention, skepticism, budget sensitivity, and vocabulary.

## Programmatic API

For reproducible, scripted walkthroughs (many personas, many sites, structured
output), use the native `PersonaNavigator` instead of free-form narration. The
persona reads each page and **chooses its own next action** from the real on-page
options — AudienceKit never picks the move with heuristics. The browser is
injected through the `ak.Browser` protocol, so plug in `agent-browser`,
Playwright, Selenium, or a fixture:

```python
import audiencekit as ak

row = ak.sample_panel(ak.load_panel(), n=1, seed=13).iloc[0].to_dict()
nav = ak.PersonaNavigator(row, backend_type="gemini",
                          task="find a dinner plan for next week",
                          extra_fields=(("price_feel", "how the price feels to you"),))

steps = ak.run_browse_session(nav, browser, "https://example.com/", max_steps=8)
for step in steps:
    print(step.milestone, "->", step.quote, "| left" if step.left else f"| {step.action.label}")
```

A minimal `Browser` adapter over the `agent-browser` CLI looks like:

```python
class AgentBrowser:
    def __init__(self, session="persona"): self.session = session
    def open(self, url): run("open", url)
    def read(self):
        return ak.PageView(text=run("snapshot", "--compact"), url=run("get", "url"),
                           title=run("get", "title"))
    def actions(self):
        # parse clickable refs from the accessibility snapshot into BrowseAction(id, label, payload=ref)
        return [...]
    def execute(self, action): run("click", action.payload)
    def close(self): run("close")
```

Pass `boundary=lambda page: "payment" in page.text.lower()` to `run_browse_session`
to stop before out-of-scope payment, password, phone, or captcha walls.

## Browse

Use the configured browser automation tool for navigation. Make 4-6 moves maximum:

- On each page, inspect the visible content before acting.
- Narrate what the persona notices first.
- Choose one plausible next action.
- Let the persona leave if the page would lose them.

When using `agent-browser`, start headed mode explicitly on the first call:

```bash
agent-browser --headed open <url>
```

Put `--headed` before the subcommand. Do not rely on a local `agent-browser.json` file being read from the current directory.

## Narration Format

After every page interaction, print:

```text
Page/action: <where the persona is>
Inner voice: "<2-4 first-person sentences, grounded in the persona and concrete page details>"
Next action: <one action or leave>
```

Voice rules:

- First person, direct, and colloquial.
- Match reading level and category familiarity to the persona.
- React to concrete page elements, not generic UX theory.
- Preserve boredom, confusion, sticker shock, and distrust when they appear.

## Debrief

End with:

- Overall impression in the persona's voice.
- One thing that helped.
- One thing that lost them.
- Whether they would return or convert.
- One researcher sentence with the most actionable UX insight.
