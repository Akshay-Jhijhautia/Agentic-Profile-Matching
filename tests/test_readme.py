import re

import matching_agent
from tests.fakes import FakeChatModel
from tests.paths import ROOT

# "a --> b" and "a -.->|label| b", with node shapes such as "__end__([END])" allowed.
EDGE_RE = re.compile(r"^\s*(\w+)(?:\(\[\w+\]\))?\s*-\.?->\s*(?:\|[^|]*\|\s*)?(\w+)", re.MULTILINE)


def test_readme_diagram_matches_the_compiled_graph(services):
    readme = (ROOT / "README.md").read_text()
    diagram = readme.split("```mermaid", 1)[1].split("```", 1)[0]
    documented = set(EDGE_RE.findall(diagram))

    services.model = FakeChatModel({})
    graph = matching_agent.build_graph(services, conversational=True).get_graph()
    actual = {(edge.source, edge.target) for edge in graph.edges}
    assert documented == actual
