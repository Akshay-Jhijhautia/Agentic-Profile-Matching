# Profile Match UI

The browser interface of Profile Match: React 19, TypeScript and Vite. It is
built into the Python package (`profile_matching/web/static/`), so people who
install Profile Match never need Node; `profile-match web` serves it.

## Develop

Run the API and the dev server side by side, with the same token:

```bash
# terminal 1, from the repository root
PROFILE_MATCH_TOKEN=dev profile-match web --no-browser

# terminal 2, in frontend/
npm install
PROFILE_MATCH_TOKEN=dev npm run dev     # http://localhost:5173, reloads as you edit
```

The dev server forwards `/api` to `http://127.0.0.1:8765` (`PROFILE_MATCH_API`
changes it) and adds the token, which the built app reads from its page instead.

## Before committing a UI change

```bash
npm run typecheck && npm test
npm run build      # writes profile_matching/web/static/ and its source fingerprint
```

Commit the build with the source: `pytest` fails when `static/build-info.json`
does not match the source (`tests/test_web_ui.py`).

## When the API changes

```bash
python -m scripts.export_openapi   # from the repository root: writes frontend/openapi.json
npm run api-types                  # regenerates src/api/schema.ts
npm run build
```

`pytest` fails when `openapi.json` is older than the API.

## Layout

| Path | Purpose |
| --- | --- |
| `src/api/client.ts` | Typed API calls; turns stream their events (`sse.ts`) |
| `src/api/schema.ts` | Generated from `openapi.json`; do not edit |
| `src/useSession.ts` | The session the page works on (kept in the URL) |
| `src/components/` | Resumes, role, progress, confirmation, results |
| `scripts/build-info.mjs` | The source fingerprint written with each build |
