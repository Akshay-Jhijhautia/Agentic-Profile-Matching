"""The browser app: its build is current, its API types are current, and it is served."""

import hashlib
import json
import re

from fastapi.testclient import TestClient

from profile_matching.web import app as web_app
from profile_matching.web.app import TOKEN_HEADER, create_app
from scripts import export_openapi
from tests.paths import ROOT

FRONTEND = ROOT / "frontend"
STATIC = ROOT / "profile_matching" / "web" / "static"
# Must match frontend/scripts/build-info.mjs.
SOURCES = ["index.html", "package.json", "package-lock.json", "tsconfig.json", "vite.config.ts", "src"]


def source_hash() -> str:
    files = []
    for source in SOURCES:
        path = FRONTEND / source
        found = path.rglob("*") if path.is_dir() else [path]
        files += [f for f in found if f.is_file() and not re.search(r"\.test\.tsx?$", f.name)]
    digest = hashlib.sha256()
    for path in sorted(f.relative_to(FRONTEND).as_posix() for f in files):
        digest.update(path.encode() + b"\0")
        digest.update((FRONTEND / path).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def test_the_committed_build_matches_the_frontend_source():
    recorded = json.loads((STATIC / "build-info.json").read_text())["source_hash"]
    assert recorded == source_hash(), "frontend/ changed since the last build: run `npm run build` in frontend/"


def test_the_api_types_match_the_api():
    assert (FRONTEND / "openapi.json").read_text() == export_openapi.schema_text(), (
        "the API changed: run `python -m scripts.export_openapi`, then `npm run api-types` and `npm run build` in frontend/"
    )


def _client(make_services=None):
    app = create_app(make_services or (lambda: None), token="t0ken")
    return TestClient(app, base_url="http://127.0.0.1", headers={TOKEN_HEADER: "t0ken"})


def test_the_page_carries_the_token_and_its_assets_are_served():
    page = _client().get("/")
    assert page.status_code == 200 and page.headers["cache-control"] == "no-store"
    assert '<meta name="profile-match-token" content="t0ken"' in page.text
    assert "__PROFILE_MATCH_TOKEN__" not in page.text
    for asset in re.findall(r'(?:src|href)="(/assets/[^"]+)"', page.text):
        response = _client().get(asset)
        assert response.status_code == 200 and response.content, asset


def test_a_missing_build_is_explained(monkeypatch, tmp_path):
    monkeypatch.setattr(web_app, "STATIC_DIR", tmp_path)
    page = _client().get("/")
    assert page.status_code == 503 and "profile-match chat" in page.text
