"""Write the web API's OpenAPI schema to frontend/openapi.json.

The browser app's TypeScript types are generated from it (``npm run api-types``
in frontend/), and a test fails when it is older than the API.

    python -m scripts.export_openapi
"""

from __future__ import annotations

import json
from pathlib import Path

from profile_matching.web.app import create_app

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "frontend" / "openapi.json"


def schema_text() -> str:
    return json.dumps(create_app(token="schema-export").openapi(), indent=2, sort_keys=True) + "\n"


def main() -> None:
    SCHEMA_PATH.write_text(schema_text())
    print(f"Wrote {SCHEMA_PATH}")


if __name__ == "__main__":
    main()
