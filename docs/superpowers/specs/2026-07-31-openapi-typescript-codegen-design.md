# OpenAPI → TypeScript codegen

Date: 2026-07-31  
Status: Approved for planning

## Goal

Close dual-contract drift between `backend/app/schemas/*.py` and hand-written types in `frontend/src/lib/api.ts` by generating TypeScript types from FastAPI’s OpenAPI schema.

## Decisions

| Topic | Choice |
| --- | --- |
| Codegen scope | Types only (keep thin `fetch*` wrappers) |
| Schema source of truth | Committed `frontend/openapi.json` |
| Generated types | Committed `frontend/src/lib/api.schema.d.ts` |
| Call-site migration | Keep `Api*` names via aliases in `api.ts` |
| Tooling | `openapi-typescript` |
| CI | Fail if OpenAPI dump or regenerated types differ from committed files |
| Runtime validation | Out of scope (no response parsing / zod) |
| Typed HTTP client | Out of scope for v1 (`openapi-fetch` / Orval deferred) |

## Architecture

```text
backend/app/schemas/*.py  (Pydantic, response_model)
        │
        ▼  export script (import FastAPI app → app.openapi())
frontend/openapi.json     (committed)
        │
        ▼  openapi-typescript
frontend/src/lib/api.schema.d.ts  (committed, do not edit by hand)
        │
        ▼  type aliases
frontend/src/lib/api.ts   (Api* re-exports + thin fetch* wrappers)
        │
        ▼
hooks / components        (unchanged import paths)
```

**Developer workflow when a response shape changes**

1. Edit Pydantic models / route `response_model`.
2. Run OpenAPI export → updates `frontend/openapi.json`.
3. Run `npm run generate:api` → updates `api.schema.d.ts`.
4. Commit schema + OpenAPI + generated types together.
5. Add or adjust an `Api*` alias in `api.ts` only when introducing a new frontend-facing model name.

## Components

| Piece | Role |
| --- | --- |
| Export script (e.g. `scripts/export_openapi.py`) | `from app.main import app`; write stable, sorted `frontend/openapi.json` |
| `frontend/openapi.json` | Committed contract snapshot |
| `frontend/src/lib/api.schema.d.ts` | `openapi-typescript` output; never hand-edit |
| `frontend/src/lib/api.ts` | `API_BASE`, `fetch*` helpers, `Api*` aliases into `components["schemas"][…]` |
| `frontend/package.json` | `generate:api`; optional `check:api` that fails on dirty tree |
| `.github/workflows/pages.yml` | After backend install: export + diff OpenAPI; after frontend install: generate + diff types (before vitest/build) |

### Alias map (v1 frontend surface)

| Frontend export | OpenAPI / Pydantic schema |
| --- | --- |
| `ApiGameStatus` | Status enum from OpenAPI (`GameStatus` if named; else field pick from `WnbaGame`) |
| `ApiWnbaTeam` | `WnbaTeam` |
| `ApiWnbaGame` | `WnbaGame` |
| `WnbaScoreboardResponse` | `WnbaScoreboardResponse` |
| `ApiWnbaGameDetail` and nested `ApiGameDetail*` | `WnbaGameDetail` / `GameDetail*` |
| `ApiWnbaLeader*` / `ApiWnbaLeadersResponse` | matching `WnbaLeader*` / `WnbaLeadersResponse` |
| `ApiWnbaStandings*` / `ApiWnbaStandingsResponse` | matching `WnbaStandings*` |
| `ApiWnbaOdds*` / `ApiWnbaOddsResponse` | matching `WnbaOdds*` |
| `ApiWnbaProp*` / `ApiWnbaPropsResponse` | matching `WnbaProp*` / `WnbaPropsResponse` |

NBA DB-backed route schemas still appear in OpenAPI and generated types, but v1 does not add `Api*` aliases or fetchers for them.

## CI contract check

1. Install backend deps with `PYTHONPATH=backend`.
2. Run export script; `git diff --exit-code frontend/openapi.json` (or equivalent file compare).
3. Install frontend deps; run `generate:api`; `git diff --exit-code frontend/src/lib/api.schema.d.ts`.
4. Proceed with existing pytest / vitest / build / Pages deploy jobs.

Export must work without a live database: importing `app.main` must not require `SUPABASE_DB_URL` or network.

## Error handling

| Failure | Behavior |
| --- | --- |
| Stale `openapi.json` or `api.schema.d.ts` | CI fails; message tells developer to re-export and regenerate |
| Export cannot import app | Fix import-time side effects; do not require DB for OpenAPI dump |
| Schema rename breaks an `Api*` alias | TypeScript build fails; update alias map in `api.ts` |
| Hand-edited generated `.d.ts` | CI types diff fails |

Runtime `fetch*` behavior stays as today: `VITE_API_BASE_URL`, `cache: "no-store"`, throw on non-OK. Codegen does not validate JSON at runtime.

## Testing

- Existing vitest suite must pass after replacing hand-written interfaces with aliases (compile-time contract).
- Optional guard: assert committed OpenAPI includes the six WNBA paths used by the frontend.
- No new runtime response-schema tests in v1.

## Out of scope

- Replacing `fetch*` with `openapi-fetch` / Orval
- Renaming call sites away from `Api*`
- Runtime zod (or similar) validation
- Generating types only for a subset of routes (full OpenAPI is fine; aliases are selective)

## Success criteria

1. Hand-written response interfaces removed from `api.ts` (aliases + fetchers only).
2. `npm test` and `npm run build` pass.
3. CI blocks merge when Pydantic ↔ `openapi.json` ↔ `api.schema.d.ts` drift.
