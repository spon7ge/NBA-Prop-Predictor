# OpenAPI → TypeScript Codegen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate TypeScript API types from FastAPI OpenAPI so `frontend/src/lib/api.ts` stops hand-mirroring `backend/app/schemas/*.py`.

**Architecture:** Export `app.openapi()` to committed `frontend/openapi.json`; run `openapi-typescript` into committed `frontend/src/lib/api.schema.d.ts`; keep thin `fetch*` wrappers and `Api*` aliases in `api.ts`. CI fails if either committed artifact drifts.

**Tech Stack:** FastAPI OpenAPI, Python export script, pytest, `openapi-typescript`, npm scripts, GitHub Actions (`pages.yml`)

## Global Constraints

- Types only — do not replace `fetch*` with `openapi-fetch` / Orval.
- Commit both `frontend/openapi.json` and `frontend/src/lib/api.schema.d.ts`.
- Keep existing `Api*` export names via aliases; do not rename call sites.
- Export must work with `PYTHONPATH=backend` and no live DB / network.
- Runtime fetch behavior unchanged: `VITE_API_BASE_URL`, `cache: "no-store"`, throw on non-OK.
- Spec: `docs/superpowers/specs/2026-07-31-openapi-typescript-codegen-design.md`

---

## File Structure

| File | Responsibility |
| --- | --- |
| `scripts/export_openapi.py` | Import FastAPI app; write sorted `frontend/openapi.json` |
| `backend/tests/test_export_openapi.py` | Assert export includes the six WNBA frontend paths |
| `frontend/openapi.json` | Committed OpenAPI snapshot |
| `frontend/src/lib/api.schema.d.ts` | Generated types (`openapi-typescript`); never hand-edit |
| `frontend/src/lib/api.ts` | `Api*` aliases + `fetch*` helpers only |
| `frontend/package.json` | `generate:api` / `check:api` scripts; `openapi-typescript` devDependency |
| `.github/workflows/pages.yml` | OpenAPI drift check in backend job; types drift check in frontend job |

---

### Task 1: OpenAPI export script + path guard test

**Files:**
- Create: `scripts/export_openapi.py`
- Create: `backend/tests/test_export_openapi.py`
- Create (via script): `frontend/openapi.json`

**Interfaces:**
- Consumes: `app.main.app` (`FastAPI`)
- Produces: `export_openapi(path: Path | None = None) -> Path` writing JSON; default path = repo `frontend/openapi.json`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_export_openapi.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from export_openapi import REQUIRED_WNBA_PATHS, export_openapi


def test_export_openapi_includes_wnba_frontend_paths(tmp_path: Path) -> None:
    out = tmp_path / "openapi.json"
    written = export_openapi(out)
    assert written == out
    spec = json.loads(out.read_text(encoding="utf-8"))
    paths = spec["paths"]
    for path in REQUIRED_WNBA_PATHS:
        assert path in paths, f"missing OpenAPI path: {path}"
```

Note: the test imports `export_openapi` from the repo-root script. Either:
- put the module on `sys.path` in the test, or
- prefer importing via a thin package path.

**Preferred layout for importability:** implement helpers in `backend/app/openapi_export.py` and keep `scripts/export_openapi.py` as a CLI wrapper. Then the test imports from `app.openapi_export`.

Adjust Step 1 to:

```python
from __future__ import annotations

import json
from pathlib import Path

from app.openapi_export import REQUIRED_WNBA_PATHS, export_openapi


def test_export_openapi_includes_wnba_frontend_paths(tmp_path: Path) -> None:
    out = tmp_path / "openapi.json"
    written = export_openapi(out)
    assert written == out
    spec = json.loads(out.read_text(encoding="utf-8"))
    paths = spec["paths"]
    for path in REQUIRED_WNBA_PATHS:
        assert path in paths, f"missing OpenAPI path: {path}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor
PYTHONPATH=backend python -m pytest backend/tests/test_export_openapi.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'app.openapi_export'` (or import error).

- [ ] **Step 3: Implement export module + CLI**

Create `backend/app/openapi_export.py`:

```python
"""Export FastAPI OpenAPI schema to a stable JSON file for frontend codegen."""

from __future__ import annotations

import json
from pathlib import Path

from app.main import app

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "frontend" / "openapi.json"

# Paths used by frontend/src/lib/api.ts fetch helpers (with /api prefix as mounted).
REQUIRED_WNBA_PATHS = (
    "/api/wnba/scoreboard/today",
    "/api/wnba/games/{espn_event_id}",
    "/api/wnba/leaders",
    "/api/wnba/standings",
    "/api/wnba/odds/today",
    "/api/wnba/props/today",
)


def export_openapi(path: Path | None = None) -> Path:
    """Write sorted OpenAPI JSON to ``path`` (default: frontend/openapi.json)."""
    out = path or DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    schema = app.openapi()
    text = json.dumps(schema, indent=2, sort_keys=True) + "\n"
    out.write_text(text, encoding="utf-8")
    return out
```

Create `scripts/export_openapi.py`:

```python
#!/usr/bin/env python3
"""CLI: dump FastAPI OpenAPI to frontend/openapi.json."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.openapi_export import DEFAULT_OUT, export_openapi  # noqa: E402


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    written = export_openapi(out)
    print(f"Wrote {written}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=backend python -m pytest backend/tests/test_export_openapi.py -v
```

Expected: PASS.

If FastAPI path keys omit the `/api` prefix, inspect `list(app.openapi()["paths"])` and update `REQUIRED_WNBA_PATHS` to the actual keys (routers are mounted with `prefix="/api"` in `main.py`, so keys should include `/api/...`).

- [ ] **Step 5: Write committed OpenAPI snapshot**

```bash
PYTHONPATH=backend python scripts/export_openapi.py
```

Expected: creates/updates `frontend/openapi.json`.

- [ ] **Step 6: Commit**

```bash
git add backend/app/openapi_export.py scripts/export_openapi.py \
  backend/tests/test_export_openapi.py frontend/openapi.json
git commit -m "$(cat <<'EOF'
feat: export FastAPI OpenAPI for frontend codegen

Add a stable JSON dump of the API schema and guard the six WNBA paths the UI calls.
EOF
)"
```

---

### Task 2: `openapi-typescript` + generate scripts + committed types

**Files:**
- Modify: `frontend/package.json`
- Modify: `frontend/package-lock.json` (via npm)
- Create: `frontend/src/lib/api.schema.d.ts`

**Interfaces:**
- Consumes: `frontend/openapi.json`
- Produces: `npm run generate:api` → `frontend/src/lib/api.schema.d.ts` with `components["schemas"][...]`

- [ ] **Step 1: Install openapi-typescript**

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor/frontend
npm install -D openapi-typescript
```

- [ ] **Step 2: Add npm scripts**

In `frontend/package.json` `"scripts"`, add:

```json
"generate:api": "openapi-typescript openapi.json -o src/lib/api.schema.d.ts",
"check:api": "npm run generate:api && git diff --exit-code -- src/lib/api.schema.d.ts"
```

Keep existing `dev`, `build`, `preview`, `test` scripts.

- [ ] **Step 3: Generate committed types**

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor/frontend
npm run generate:api
```

Expected: creates `frontend/src/lib/api.schema.d.ts` containing `components` / `schemas` entries for `WnbaGame`, `WnbaScoreboardResponse`, `WnbaGameDetail`, `WnbaLeadersResponse`, `WnbaStandingsResponse`, `WnbaOddsResponse`, `WnbaPropsResponse`, etc.

Confirm schema keys exist:

```bash
grep -E "WnbaGame:|WnbaScoreboardResponse:|WnbaGameDetail:|WnbaPropsResponse:" src/lib/api.schema.d.ts
```

Expected: matches for each.

- [ ] **Step 4: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/src/lib/api.schema.d.ts
git commit -m "$(cat <<'EOF'
feat: generate TypeScript types from OpenAPI

Add openapi-typescript and commit api.schema.d.ts as the frontend contract source of truth.
EOF
)"
```

---

### Task 3: Replace hand-written types in `api.ts` with aliases

**Files:**
- Modify: `frontend/src/lib/api.ts`
- Test: existing frontend vitest suite (no new unit test file required — compile-time contract)

**Interfaces:**
- Consumes: `components["schemas"]` from `./api.schema`
- Produces: same public exports as today (`ApiWnbaGame`, `fetchWnbaScoreboard`, …)

- [ ] **Step 1: Rewrite type section of `api.ts`**

Replace all hand-written `export type … = { … }` response interfaces with aliases. Keep `API_BASE` and every `fetch*` function body unchanged.

Full target shape for the type exports (fetchers unchanged from current file):

```typescript
import type { components } from "./api.schema";

type Schemas = components["schemas"];

export type ApiGameStatus = Schemas["WnbaGame"]["status"];
export type ApiWnbaTeam = Schemas["WnbaTeam"];
export type ApiWnbaGame = Schemas["WnbaGame"];
export type ApiGameDetailTeam = Schemas["GameDetailTeam"];
export type ApiGameDetailShot = Schemas["GameDetailShot"];
export type ApiGameDetailPlay = Schemas["GameDetailPlay"];
export type ApiGameDetailLatestPlay = Schemas["GameDetailLatestPlay"];
export type ApiGameDetailWinProbabilityPoint =
  Schemas["GameDetailWinProbabilityPoint"];
export type ApiGameDetailTeamStat = Schemas["GameDetailTeamStat"];
export type ApiGameDetailWinProbability = Schemas["GameDetailWinProbability"];
export type ApiGameDetailMatchupPrediction =
  Schemas["GameDetailMatchupPrediction"];
export type ApiGameDetailStarter = Schemas["GameDetailStarter"];
export type ApiGameDetailProjectedStarters =
  Schemas["GameDetailProjectedStarters"];
export type ApiGameDetailSeasonLeader = Schemas["GameDetailSeasonLeader"];
export type ApiGameDetailSeasonLeaders = Schemas["GameDetailSeasonLeaders"];
export type ApiGameDetailInjury = Schemas["GameDetailInjury"];
export type ApiGameDetailInjuries = Schemas["GameDetailInjuries"];
export type ApiGameDetailBoxScorePlayer = Schemas["GameDetailBoxScorePlayer"];
export type ApiGameDetailBoxScore = Schemas["GameDetailBoxScore"];
export type ApiWnbaGameDetail = Schemas["WnbaGameDetail"];
export type WnbaScoreboardResponse = Schemas["WnbaScoreboardResponse"];

export type ApiWnbaLeaderRow = Schemas["WnbaLeaderRow"];
export type ApiWnbaLeaderCategory = Schemas["WnbaLeaderCategory"];
export type ApiWnbaLeadersResponse = Schemas["WnbaLeadersResponse"];

export type ApiWnbaStandingsRow = Schemas["WnbaStandingsRow"];
export type ApiWnbaStandingsConference = Schemas["WnbaStandingsConference"];
export type ApiWnbaStandingsResponse = Schemas["WnbaStandingsResponse"];

export type ApiWnbaOddsGame = Schemas["WnbaOddsGame"];
export type ApiWnbaOddsResponse = Schemas["WnbaOddsResponse"];

export type ApiWnbaPropBookQuote = Schemas["WnbaPropBookQuote"];
export type ApiWnbaPropLine = Schemas["WnbaPropLine"];
export type ApiWnbaPropsResponse = Schemas["WnbaPropsResponse"];

/**
 * Origin of the HoopVista API, without a trailing slash.
 *
 * Empty in local dev, where Vite's `/api` proxy forwards to the backend. Static
 * hosts (GitHub Pages and friends) have no proxy, so their builds must set
 * `VITE_API_BASE_URL` to the live API origin or every request 404s.
 */
const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

// … keep existing fetchWnbaScoreboard, fetchGameDetail, fetchWnbaLeaders,
// fetchWnbaStandings, fetchWnbaOdds, fetchWnbaProps implementations exactly …
```

If TypeScript reports a missing schema key (e.g. OpenAPI inlined a nested model), open `api.schema.d.ts`, find the actual key name, and update that one alias. Prefer fixing Pydantic/`response_model` naming only if a model is truly absent.

If `import type { components } from "./api.schema"` fails because the file is `.d.ts`, try `./api.schema.d.ts` or rename output to `api.schema.ts` **only if required** — prefer keeping the spec path `api.schema.d.ts` and adjusting tsconfig/`paths` only if the compiler cannot resolve it.

- [ ] **Step 2: Run frontend tests + build**

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor/frontend
npm test
npm run build
```

Expected: both PASS. If build fails on optional/nullable mismatches vs old hand types, fix the **backend schema** (source of truth) or tighten/loosen the alias — do not reintroduce hand-written interfaces.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/lib/api.ts
git commit -m "$(cat <<'EOF'
refactor: alias Api* types from generated OpenAPI schema

Remove hand-written response interfaces from api.ts; keep fetch helpers and export names.
EOF
)"
```

---

### Task 4: CI drift checks

**Files:**
- Modify: `.github/workflows/pages.yml`

**Interfaces:**
- Consumes: `scripts/export_openapi.py`, `npm run generate:api`
- Produces: CI failure when committed OpenAPI or `api.schema.d.ts` is stale

- [ ] **Step 1: Add OpenAPI drift check to `backend-tests`**

After “Install backend deps” and **before** pytest, add:

```yaml
      - name: Check OpenAPI snapshot is up to date
        env:
          PYTHONPATH: backend
        run: |
          python scripts/export_openapi.py
          git diff --exit-code -- frontend/openapi.json
```

If the diff fails, the job message from `git diff` is enough; optionally prepend:

```bash
echo "OpenAPI drift detected. Run: PYTHONPATH=backend python scripts/export_openapi.py"
```

- [ ] **Step 2: Add types drift check to `frontend-tests`**

After “Install frontend deps” and **before** vitest, add:

```yaml
      - name: Check generated API types are up to date
        working-directory: frontend
        run: |
          npm run generate:api
          git diff --exit-code -- src/lib/api.schema.d.ts
```

- [ ] **Step 3: Verify workflow YAML locally (syntax sanity)**

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/pages.yml'))"
```

Expected: no exception. (If PyYAML missing: `pip install pyyaml` or skip and rely on GitHub.)

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/pages.yml
git commit -m "$(cat <<'EOF'
ci: fail when OpenAPI or generated API types drift

Gate Pages CI on a fresh FastAPI OpenAPI dump and openapi-typescript output matching git.
EOF
)"
```

---

### Task 5: End-to-end verification

**Files:** none new (verification only)

- [ ] **Step 1: Re-export + regenerate and confirm clean tree**

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor
PYTHONPATH=backend python scripts/export_openapi.py
cd frontend && npm run generate:api
cd .. && git status --short
```

Expected: no modified `frontend/openapi.json` or `frontend/src/lib/api.schema.d.ts` (clean for those paths).

- [ ] **Step 2: Simulate drift detection**

```bash
# Break OpenAPI snapshot intentionally
echo "{}" > frontend/openapi.json
PYTHONPATH=backend python scripts/export_openapi.py
git diff --exit-code -- frontend/openapi.json; echo "exit=$?"
# Restore
git checkout -- frontend/openapi.json
```

After restore + real export, `git diff --exit-code` should exit 0.

- [ ] **Step 3: Full local gate**

```bash
PYTHONPATH=backend python -m pytest backend/tests/ -q
cd frontend && npm test && npm run build
```

Expected: all green.

- [ ] **Step 4: Final commit only if Task 5 produced fixes**

If verification required fixes, commit them with a message describing the fix. Otherwise no commit.

---

## Spec coverage checklist

| Spec requirement | Task |
| --- | --- |
| Types-only codegen | Tasks 2–3 |
| Committed `openapi.json` | Task 1 |
| Committed `api.schema.d.ts` | Task 2 |
| `Api*` aliases in `api.ts` | Task 3 |
| `openapi-typescript` | Task 2 |
| CI drift on OpenAPI + types | Task 4 |
| Export without live DB | Task 1 (import `app.main` only) |
| Optional six-path guard | Task 1 test |
| Keep `fetch*` / runtime behavior | Task 3 |
| Vitest + build pass | Tasks 3, 5 |
