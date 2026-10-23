"""Measure how stable requirement extraction is across repeated runs and models.

Every screening starts by turning the job description into must-haves (with
years), nice-to-haves, total years and a degree. Must-haves are knockouts, so
when they change between runs or models, different candidates are rejected.
This extracts each job description ``--runs`` times per model and reports, per
field, how many distinct answers came back and how often two runs agree, and in
how many runs chat would ask the recruiter to confirm open requirements.

    python -m scripts.requirements_stability --runs 5 --models gpt-5.6-terra,gpt-6-luna
    python -m scripts.requirements_stability "React and 3+ years experience" jd.txt --runs 3
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

from dotenv import load_dotenv

from profile_matching import llm, providers
from profile_matching.config import get_settings
from profile_matching.rag import skills_vocab
from profile_matching.requirements import extract_requirements, load_job_description, open_questions
from profile_matching.schemas import Requirements

JD_DIR = Path("tests/fixtures/job_descriptions")
REQUESTS = (
    "Find me candidates with React and 3+ years experience",
    "Senior Python backend engineer with Django and PostgreSQL, 5+ years",
    "Machine learning engineer with PyTorch; LLM experience is a plus",
)
# What a run's requirements are compared on.
FIELDS = {
    "must_have": lambda r: tuple(sorted(r.must_have_skills)),
    "must_have_years": lambda r: tuple(sorted((m.skill, m.min_years) for m in r.must_have)),
    "nice_skills": lambda r: tuple(sorted(s for s in r.nice_to_have if s in skills_vocab.SKILLS)),
    # Preferences that are not skills ("financial services"): the model's wording.
    "nice_other": lambda r: tuple(sorted(s.lower() for s in r.nice_to_have if s not in skills_vocab.SKILLS)),
    "total_years": lambda r: r.min_total_years,
    "degree": lambda r: r.min_degree,
}


def agreement(values: list) -> float:
    """Share of run pairs that gave the same value."""
    pairs = list(combinations(values, 2))
    return sum(a == b for a, b in pairs) / len(pairs) if pairs else 1.0


def measure(runs: dict[str, list[Requirements]]) -> dict[str, dict]:
    """Per field: agreement within each model, across all runs, and the distinct answers."""
    everything = [r for model_runs in runs.values() for r in model_runs]
    report = {}
    for field, key in FIELDS.items():
        report[field] = {
            "within": {model: agreement([key(r) for r in model_runs]) for model, model_runs in runs.items()},
            "overall": agreement([key(r) for r in everything]),
            "answers": Counter(key(r) for r in everything),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("jds", nargs="*", help="Job descriptions (text or files); default: the fixtures and 3 requests.")
    parser.add_argument("--runs", type=int, default=5, help="Extractions per model (default: 5).")
    parser.add_argument("--models", help="Comma-separated models, as names or provider:model (default: configured).")
    parser.add_argument("--verbose", action="store_true", help="Print every distinct answer.")
    args = parser.parse_args(argv)
    inputs = args.jds or [*map(str, sorted(JD_DIR.glob("*.txt"))), *REQUESTS]
    model_names = [m.strip() for m in (args.models or get_settings().llm_model).split(",") if m.strip()]
    try:
        models = {name: llm.make_chat_model(name) for name in model_names}
    except providers.cli_errors() as exc:
        print(f"Error: {providers.error_message(exc)}", file=sys.stderr)
        return 1

    jobs = [(jd, name) for jd in inputs for name in models for _ in range(args.runs)]
    texts = {jd: load_job_description(jd) for jd in inputs}
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda job: extract_requirements(texts[job[0]], model=models[job[1]]), jobs))

    totals: dict[str, list[float]] = {field: [] for field in FIELDS}
    for jd in inputs:
        runs = {name: [r for (j, n), r in zip(jobs, results) if j == jd and n == name] for name in models}
        report = measure(runs)
        everything = [r for model_runs in runs.values() for r in model_runs]
        asked = sum(bool(open_questions(texts[jd], r)) for r in everything)
        print(f"\n== {Path(jd).name if Path(jd).exists() else jd}  (chat would ask the recruiter in "
              f"{asked}/{len(everything)} runs)")
        for field, data in report.items():
            totals[field].append(data["overall"])
            within = "  ".join(f"{m}: {v:.2f}" for m, v in data["within"].items())
            print(f"   {field:<16} agreement {data['overall']:.2f} ({within})  distinct answers: {len(data['answers'])}")
            if args.verbose or (field.startswith("must") and len(data["answers"]) > 1):
                for answer, count in data["answers"].most_common():
                    print(f"      {count}x {answer}")
    print("\n== mean agreement over all job descriptions (1.00 = every run identical)")
    for field, values in totals.items():
        print(f"   {field:<16} {sum(values) / len(values):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
