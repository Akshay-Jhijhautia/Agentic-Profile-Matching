"""The documentation stays true: its graph diagram is the compiled graph, and its links resolve."""

import re

import matching_agent
from tests.fakes import FakeChatModel
from tests.paths import ROOT

# "a --> b" and "a -.->|label| b", with node shapes such as "__end__([END])" allowed.
EDGE_RE = re.compile(r"^\s*(\w+)(?:\(\[\w+\]\))?\s*-\.?->\s*(?:\|[^|]*\|\s*)?(\w+)", re.MULTILINE)
LINK_RE = re.compile(r"\]\(([^)\s]+)\)")
CODE_RE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)
DOCS = ["README.md", "CHANGELOG.md", "SECURITY.md", "frontend/README.md", *sorted(
    p.relative_to(ROOT).as_posix() for p in (ROOT / "docs").glob("*.md"))]


def test_the_documented_diagram_matches_the_compiled_graph(services):
    guide = (ROOT / "docs" / "how-it-works.md").read_text()
    diagram = guide.split("```mermaid", 1)[1].split("```", 1)[0]
    documented = set(EDGE_RE.findall(diagram))

    services.model = FakeChatModel({})
    graph = matching_agent.build_graph(services, conversational=True).get_graph()
    actual = {(edge.source, edge.target) for edge in graph.edges}
    assert documented == actual


def _anchors(markdown: str) -> set[str]:
    """Heading anchors as GitHub makes them: lower case, punctuation dropped, spaces as hyphens."""
    headings = re.findall(r"^#+ (.+)$", CODE_RE.sub("", markdown), re.MULTILINE)
    return {re.sub(r"[^\w\- ]", "", h.strip().lower()).replace(" ", "-") for h in headings}


def test_every_link_between_the_docs_resolves():
    broken = []
    for doc in DOCS:
        source = ROOT / doc
        for link in LINK_RE.findall(CODE_RE.sub("", source.read_text())):
            if re.match(r"[a-z]+:", link):  # https:, mailto:
                continue
            path, _, anchor = link.partition("#")
            target = (source.parent / path).resolve() if path else source
            if not target.exists():
                broken.append(f"{doc}: {link} (no such file)")
            elif anchor and target.suffix == ".md" and anchor not in _anchors(target.read_text()):
                broken.append(f"{doc}: {link} (no such heading)")
    assert broken == []
