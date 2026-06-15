"""Run a survey spec against an AudienceKit panel.

Usage:
    uv run python scripts/run_survey.py --spec examples/ferrari_luce/study.json \
        [--n 30] [--segment broad|luxury] [--treatment NAME] [--seed 42] \
        [--backend gemini|openai|anthropic] [--out results/my_survey]

The spec is the JSON survey format accepted by audiencekit.Study.
If the spec has a "treatments" map ({name: stimulus description}), pass
--treatment to select one; the chosen description replaces
spec["stimulus"]["description"] and the arm name is recorded in the output.

Writes <out>/<segment>_<treatment>_responses.csv and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audiencekit import Study, SyntheticPanel, load_panel, sample_panel
from audiencekit.report import likert_summary, text_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="Path to survey spec JSON")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--segment", default="broad", choices=["broad", "luxury"])
    parser.add_argument("--treatment", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", default="gemini", choices=["gemini", "openai", "anthropic"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", default=None, help="Output directory (default: results/<spec stem>)")
    args = parser.parse_args()

    spec_path = Path(args.spec)
    survey = Study.from_dict(json.loads(spec_path.read_text()))

    treatment = args.treatment or "default"
    if args.treatment:
        treatments = survey.treatments
        if args.treatment not in treatments:
            sys.exit(f"Treatment {args.treatment!r} not in spec (have: {list(treatments)})")
        survey_dict = survey.to_dict()
        survey_dict.setdefault("stimulus", {})["description"] = treatments[args.treatment]
        survey = Study.from_dict(survey_dict)

    pool = load_panel()
    respondents = sample_panel(pool, n=args.n, segment=args.segment, seed=args.seed)
    panel = SyntheticPanel(respondents, backend_type=args.backend, model=args.model)

    print(f"Running '{survey.title or spec_path.stem}' — "
          f"{args.n} {args.segment} respondents, treatment={treatment}")
    df = panel.run_survey(survey)
    df["treatment"] = treatment

    out_dir = Path(args.out) if args.out else Path("results") / spec_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{args.segment}_{treatment}_responses.csv"
    df.to_csv(out_csv, index=False)

    print(f"\nSaved {out_csv}")
    print("\n=== Likert summary ===")
    print(likert_summary(df, survey).round(2).to_string(index=False))
    for qid in text_ids(survey):
        sample_answers = df[qid].dropna().head(3)
        if not sample_answers.empty:
            print(f"\n=== {qid} (first 3) ===")
            for answer in sample_answers:
                print(f'  - "{answer}"')


if __name__ == "__main__":
    main()
