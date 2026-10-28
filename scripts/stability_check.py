"""Measure how stable the agent's screening is across repeated runs and across models.

Each run screens the index from scratch (no cached assessments). Reported per
model, and across all models together: top-k overlap and Kendall tau between
runs, agreement on interview recommendations, the spread of each candidate's deep score, and
whether the requirements came out the same.

Run from the project root, e.g. on the generated sample resumes:
    python -m scripts.generate_sample_resumes /tmp/sample_resumes
    python -m scripts.stability_check "React and 3+ years experience" --resumes /tmp/sample_resumes --runs 3
    python -m scripts.stability_check jd.txt --resumes /tmp/sample_resumes --models gpt-5.6-terra,gpt-6-luna
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from itertools import combinations
from statistics import mean

from dotenv import load_dotenv

import matching_agent
from resume_shortlister import llm, providers
from resume_shortlister.services import Services


def kendall_tau(first: list[str], second: list[str]) -> float:
    """Rank agreement (-1..1) over the candidates both lists contain; 1.0 if fewer than two."""
    common = [c for c in first if c in second]
    if len(common) < 2:
        return 1.0
    position = {c: second.index(c) for c in common}
    pairs = list(combinations(common, 2))  # each pair is in ``first`` order
    concordant = sum(position[a] < position[b] for a, b in pairs)
    return (2 * concordant - len(pairs)) / len(pairs)


def summarise(runs: list[dict]) -> dict:
    """Stability metrics over the final states of several runs."""
    rankings = [[m.id for m in state["shortlist"]] for state in runs]
    k = max(len(r) for r in rankings) or 1
    pairs = list(combinations(range(len(runs)), 2))
    in_all = set.intersection(*(set(r) for r in rankings)) if rankings else set()

    decisions = {
        cid: {next(m.recommendation.decision for m in s["shortlist"] if m.id == cid) for s in runs}
        for cid in in_all
        if all(m.recommendation for s in runs for m in s["shortlist"] if m.id == cid)
    }
    deep = {
        cid: [next(m.assessment.deep_score for m in s["shortlist"] if m.id == cid) for s in runs]
        for cid in in_all
        if all(m.assessment for s in runs for m in s["shortlist"] if m.id == cid)
    }
    spreads = [max(scores) - min(scores) for scores in deep.values()]
    requirement_sets = {
        (tuple(sorted(r.label for r in s["requirements"].must_have)), tuple(sorted(s["requirements"].nice_to_have)))
        for s in runs
    }
    return {
        "runs": len(runs),
        "top_k_overlap": mean(len(set(rankings[a]) & set(rankings[b])) / k for a, b in pairs) if pairs else 1.0,
        "kendall_tau": mean(kendall_tau(rankings[a], rankings[b]) for a, b in pairs) if pairs else 1.0,
        "decision_agreement": mean(len(d) == 1 for d in decisions.values()) if decisions else 1.0,
        "deep_spread_mean": mean(spreads) if spreads else 0.0,
        "deep_spread_max": max(spreads, default=0),
        "identical_requirements": len(requirement_sets) == 1,
        "rankings": rankings,
        "interviews": [sorted(m.id for m in s["shortlist"] if m.recommendation and m.recommendation.decision == "interview")
                  for s in runs],
    }


def print_metrics(title: str, metrics: dict, labels: list[str]) -> None:
    print(f"\n== {title}")
    for name, value in metrics.items():
        if name not in {"rankings", "interviews"}:
            print(f"{name:>22}: {value:.2f}" if isinstance(value, float) else f"{name:>22}: {value}")
    for label, ranking, interviews in zip(labels, metrics["rankings"], metrics["interviews"], strict=True):
        print(f"{label:>22}: {', '.join(ranking)}")
        print(f"{'interview':>22}: {', '.join(interviews) or 'none'}")


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Repeat a screening and measure how stable it is.")
    parser.add_argument("jd", help="Job description text, a short request or a JD file path.")
    parser.add_argument("--resumes", required=True, help="Folder of resumes to screen.")
    parser.add_argument("--runs", type=int, default=3, help="Identical runs per model (default: 3).")
    parser.add_argument("--models", help="Comma-separated models to compare, as names or provider:model (default: the configured model).")
    args = parser.parse_args(argv)
    models = [m.strip() for m in (args.models or "").split(",") if m.strip()]
    if args.runs * max(1, len(models)) < 2:
        parser.error("need at least two runs in total")
    try:
        services = Services.from_settings()
        services.llm_cache = None  # this measures the model's own variance: always ask it
        print(services.use_folder(args.resumes).summary(), file=sys.stderr)
        services.check_ready()
    except providers.cli_errors() as exc:
        print(f"Error: {providers.error_message(exc)}", file=sys.stderr)
        return 1
    services.settings = dataclasses.replace(services.settings, save_reports=False)

    models = models or [services.settings.llm_model]
    by_model: dict[str, list[dict]] = {}
    for name in models:
        model_services = dataclasses.replace(services, model=llm.make_chat_model(name))
        for number in range(1, args.runs + 1):
            print(f"{name}: run {number}/{args.runs}…", file=sys.stderr)
            by_model.setdefault(name, []).append(matching_agent.screen(model_services, args.jd))

    for name, runs in by_model.items():
        if len(runs) > 1:
            print_metrics(name, summarise(runs), [f"run {i}" for i in range(1, len(runs) + 1)])
    if len(by_model) > 1:
        everything = [state for runs in by_model.values() for state in runs]
        labels = [f"{name} #{i}" for name, runs in by_model.items() for i in range(1, len(runs) + 1)]
        print_metrics("across all models", summarise(everything), labels)
    return 0


if __name__ == "__main__":
    sys.exit(main())
