"""Measure round-1 ranking quality for an embedder, and compare two runs.

Builds the sample resumes (seeded, so every run sees the same 100 files),
indexes them with the chosen embedder and runs the real round-1 code
(``screening.evaluate``) for fixed requirements: one query per role family
with must-haves, plus open queries without any, where the embedding does most
of the work. Requirements are written here rather than extracted by an LLM,
and the indexer's LLM fallback is off, so only the retrieval stack varies.

Every resume's role family is known, which gives relevance labels. Per query:

- P@10: share of the top 10 (passing every must-have) from a relevant family;
- nDCG@10 over the same list;
- AUC: the chance that a relevant candidate's match score beats an irrelevant
  one's, over every candidate. Below 1.0 means some weak fit outscores a
  strong one somewhere in the pool.

``compare`` checks two runs candidate by candidate: identical knockouts, top-10
overlap, rank correlation and the largest match-score change.

    python -m scripts.ranking_eval run --embedder openai:text-embedding-3-small \\
        --out base.json --cache embeddings.json
    python -m scripts.ranking_eval compare base.json new.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean

from profile_matching import llm, screening
from profile_matching.config import Settings
from profile_matching.rag import resume_rag
from profile_matching.schemas import Requirements
from profile_matching.services import Services
from scripts import generate_sample_resumes

TOP_K = 10
# Role families of the hand-written resumes (the synthetic ones carry theirs).
HANDWRITTEN_FAMILIES = {
    "resume_john_doe": "python_backend", "resume_jane_smith": "data_science",
    "resume_alex_lee": "java_backend", "resume_maria_garcia": "frontend",
    "resume_priya_patel": "ml_engineer", "resume_daniel_kim": "python_backend",
    "resume_fatima_ahmed": "devops", "resume_liam_oconnor": "data_engineer",
    "resume_sofia_rossi": "ios", "resume_wei_chen": "fullstack",
    "resume_olivia_brown": "ml_engineer", "resume_carlos_mendes": "ml_engineer",
}


@dataclass(frozen=True)
class Query:
    name: str
    relevant: frozenset[str]  # role families that fit
    must: tuple[tuple[str, float | None], ...] = ()  # (skill, min years)
    nice: tuple[str, ...] = ()

    def requirements(self) -> Requirements:
        return Requirements.model_validate({
            "role_title": self.name, "min_total_years": None, "min_degree": None,
            "must_have": [{"skill": skill, "min_years": years} for skill, years in self.must],
            "nice_to_have": list(self.nice),
        })


def _q(name: str, relevant: str, must=(), nice=()) -> Query:
    return Query(name, frozenset(relevant.split()), tuple(must), tuple(nice))


QUERIES = (
    _q("frontend_react", "frontend fullstack", [("React", 3), ("TypeScript", None)], ["Next.js", "Jest", "Tailwind CSS"]),
    _q("fullstack_node", "fullstack", [("React", None), ("Node.js", None)], ["PostgreSQL", "Docker", "TypeScript"]),
    _q("python_backend", "python_backend", [("Python", 3), ("Django", None)], ["PostgreSQL", "Celery", "Docker"]),
    _q("java_backend", "java_backend", [("Java", 5), ("Spring Boot", None)], ["Kafka", "Microservices", "Kubernetes"]),
    _q("data_science", "data_science", [("Python", None), ("Statistics", None)], ["scikit-learn", "A/B Testing", "SQL"]),
    _q("ml_engineer", "ml_engineer data_science", [("Machine Learning", None), ("PyTorch", None)], ["MLflow", "Kubernetes", "LLMs"]),
    _q("data_engineer", "data_engineer", [("Spark", None), ("Airflow", None)], ["Snowflake", "Kafka"]),
    _q("devops", "devops", [("Kubernetes", None), ("Terraform", None)], ["AWS", "Prometheus", "Helm"]),
    _q("ios", "ios", [("Swift", None)], ["SwiftUI", "XCTest"]),
    _q("android", "android", [("Kotlin", None)], ["Jetpack Compose", "Firebase"]),
    _q("qa", "qa", [("Selenium", None)], ["pytest", "Cypress", "Jenkins"]),
    # No must-haves: nothing is knocked out and the semantic score weighs most.
    _q("open_mobile", "ios android", nice=["Swift", "Kotlin", "Flutter"]),
    _q("open_cloud", "devops", nice=["AWS", "Kubernetes", "Terraform", "Linux"]),
    _q("open_pipelines", "data_engineer", nice=["ETL", "Spark", "Airflow", "dbt"]),
    _q("open_web_ui", "frontend fullstack", nice=["JavaScript", "CSS", "HTML"]),
    _q("open_ai", "ml_engineer", nice=["LLMs", "NLP", "Hugging Face"]),
)


class CachedEmbedder:
    """Wraps an embedder with a JSON cache, so repeated runs embed byte-identical vectors."""

    def __init__(self, inner: resume_rag.Embedder, path: Path):
        self.inner, self.path, self.name = inner, path, inner.name
        self.cache = json.loads(path.read_text()) if path.exists() else {}

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.name}\n{text}".encode()).hexdigest()

    def embed(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in dict.fromkeys(texts) if self._key(t) not in self.cache]
        if missing:
            for text, vector in zip(missing, self.inner.embed(missing), strict=True):
                self.cache[self._key(text)] = vector
            self.path.write_text(json.dumps(self.cache))
        return [self.cache[self._key(t)] for t in texts]


# --- metrics ---------------------------------------------------------------------------------------


def ndcg(relevance: list[int], k: int = TOP_K) -> float:
    gains = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance[:k]))
    ideal = sum(rel / math.log2(i + 2) for i, rel in enumerate(sorted(relevance, reverse=True)[:k]))
    return gains / ideal if ideal else 1.0


def auc(scores: list[tuple[float, bool]]) -> float:
    """P(relevant score > irrelevant score), ties counting half."""
    pos = [s for s, rel in scores if rel]
    neg = [s for s, rel in scores if not rel]
    if not pos or not neg:
        return 1.0
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def kendall_tau(first: list[str], second: list[str]) -> float:
    position = {item: i for i, item in enumerate(second)}
    pairs = list(combinations(first, 2))
    concordant = sum(1 if position[a] < position[b] else -1 for a, b in pairs)
    return concordant / len(pairs) if pairs else 1.0


# --- run -------------------------------------------------------------------------------------------


def run(embedder: resume_rag.Embedder, workdir: Path) -> dict:
    resumes, db = workdir / "resumes", workdir / "index"
    dataset = generate_sample_resumes.build_dataset()
    generate_sample_resumes.generate(resumes)
    families = {stem: data.get("family") or HANDWRITTEN_FAMILIES[stem] for stem, data in dataset.items()}

    llm.metadata_fallback = lambda: None  # rule-based metadata only: runs stay comparable
    services = Services(settings=Settings(db_dir=str(db), embedder="hashing"))
    services.use_folder(resumes, embedder=embedder)
    family_of = {p.id: families[Path(p.resume_path).stem] for p in services.candidates.all()}

    results = {"embedder": embedder.name, "queries": {}}
    for query in QUERIES:
        ranked = screening.evaluate(services, query.requirements())
        top = screening.shortlist(ranked, TOP_K)
        relevance = [int(family_of[m.id] in query.relevant) for m in top]
        results["queries"][query.name] = {
            "p_at_10": mean(relevance) if relevance else 0.0,
            "ndcg_at_10": ndcg(relevance),
            "auc": auc([(m.match_score, family_of[m.id] in query.relevant) for m in ranked]),
            "ranking": [
                {"id": m.id, "score": m.match_score, "passes": m.passes_must_haves, "family": family_of[m.id],
                 **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.score_breakdown.model_dump().items()}}
                for m in ranked
            ],
        }
    return results


def summarise(results: dict) -> dict:
    queries = results["queries"].values()
    return {metric: mean(q[metric] for q in queries) for metric in ("p_at_10", "ndcg_at_10", "auc")}


def print_run(results: dict) -> None:
    print(f"embedder: {results['embedder']}")
    print(f"{'query':<16} {'P@10':>6} {'nDCG':>6} {'AUC':>6}  top 3")
    for name, q in results["queries"].items():
        top3 = ", ".join(f"{r['id']}({r['family']},{r['score']})" for r in q["ranking"][:3])
        print(f"{name:<16} {q['p_at_10']:>6.2f} {q['ndcg_at_10']:>6.2f} {q['auc']:>6.3f}  {top3}")
    overall = summarise(results)
    print(f"{'mean':<16} {overall['p_at_10']:>6.3f} {overall['ndcg_at_10']:>6.3f} {overall['auc']:>6.3f}")


def compare(first: dict, second: dict) -> bool:
    """Print the differences; True when the two runs rank every query identically."""
    print(f"A: {first['embedder']}\nB: {second['embedder']}")
    print(f"{'query':<16} {'knockouts':>9} {'top10':>6} {'tau':>6} {'max|Δ|':>7} {'meanΔ':>6}  ΔP@10  ΔAUC")
    identical = True
    for name, a in first["queries"].items():
        b = second["queries"][name]
        ids_a, ids_b = [r["id"] for r in a["ranking"]], [r["id"] for r in b["ranking"]]
        passes_a = {r["id"] for r in a["ranking"] if r["passes"]}
        passes_b = {r["id"] for r in b["ranking"] if r["passes"]}
        top_a = [r["id"] for r in a["ranking"] if r["passes"]][:TOP_K]
        top_b = [r["id"] for r in b["ranking"] if r["passes"]][:TOP_K]
        score_b = {r["id"]: r["score"] for r in b["ranking"]}
        deltas = [score_b[r["id"]] - r["score"] for r in a["ranking"]]
        identical &= ids_a == ids_b and not any(deltas)
        print(f"{name:<16} {'same' if passes_a == passes_b else 'DIFFER':>9} {len(set(top_a) & set(top_b)):>4}/{len(top_a):<1}"
              f" {kendall_tau(ids_a, ids_b):>6.3f} {max(map(abs, deltas)):>7} {mean(deltas):>+6.1f}"
              f"  {b['p_at_10'] - a['p_at_10']:>+5.2f} {b['auc'] - a['auc']:>+5.3f}")
    sa, sb = summarise(first), summarise(second)
    print("mean  " + "  ".join(f"{k}: {sa[k]:.3f} -> {sb[k]:.3f}" for k in sa))
    print("IDENTICAL rankings and scores" if identical else "Rankings or scores differ")
    return identical


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="Index the sample resumes and measure ranking quality.")
    run_parser.add_argument("--embedder", default="hashing:512", help="Embedder name, e.g. openai:text-embedding-3-small.")
    run_parser.add_argument("--out", type=Path, required=True, help="Where to write the results (JSON).")
    run_parser.add_argument("--cache", type=Path, help="Embedding cache (JSON), reused across runs.")
    compare_parser = commands.add_parser("compare", help="Compare two runs.")
    compare_parser.add_argument("first", type=Path)
    compare_parser.add_argument("second", type=Path)
    args = parser.parse_args(argv)

    if args.command == "compare":
        compare(json.loads(args.first.read_text()), json.loads(args.second.read_text()))
        return 0
    embedder = resume_rag.embedder_from_name(args.embedder)
    if args.cache:
        embedder = CachedEmbedder(embedder, args.cache)
    with tempfile.TemporaryDirectory() as workdir:
        results = run(embedder, Path(workdir))
    args.out.write_text(json.dumps(results, indent=1))
    print_run(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
