# WNBA Player Bio Showcase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enrich the existing WNBA player header with ESPN-style bio facts (jersey, full position, height, birthdate+age, college, draft) from `commonplayerinfo`, keeping averages on the right.

**Architecture:** Extend `WnbaPlayerResponse` with nullable bio fields; parse/format them in `wnba_player.py` from the info payload already fetched; restyle `PlayerHeader` left column. No new routes or upstream hosts.

**Tech Stack:** FastAPI · Pydantic · pytest · React 19 · TypeScript · Vitest · openapi-typescript

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-01-wnba-player-bio-showcase-design.md`
- Same endpoint: `GET /api/wnba/player/{player_id}`
- Fields: jersey, position (full name when possible), height, birthdate `(age)`, college, draft_info
- Omit missing rows; UI prefixes `#` on jersey
- Draft format: `"{year}: Rd {round}, Pk {number} ({team})"` when parts exist
- Charcoal league hub visual language; larger headshot; averages unchanged
- Verify backend: `cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor && PYTHONPATH=backend python3 -m pytest backend/tests/test_wnba_player.py -v`
- Verify frontend: `cd frontend && npm run test -- --run src/components/league/PlayerHeader.test.tsx src/pages/LeaguePlayerPage.test.tsx && npm run build`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `backend/app/schemas/wnba_player.py` | Add nullable bio fields on response |
| `backend/app/services/wnba_player.py` | Format helpers + wire into normalize |
| `backend/tests/fixtures/stats_wnba_player_info.json` | Expand with bio headers/values |
| `backend/tests/test_wnba_player.py` | Unit tests for formatters + normalize |
| `frontend/openapi.json` + `src/lib/api.schema.d.ts` | Regenerate after schema change |
| `frontend/src/components/league/PlayerHeader.tsx` | ESPN-style left column |
| `frontend/src/components/league/PlayerHeader.test.tsx` | Bio rows + omit-empty |

---

### Task 1: Schema + bio formatters + normalize

**Files:**
- Modify: `backend/app/schemas/wnba_player.py`
- Modify: `backend/app/services/wnba_player.py`
- Modify: `backend/tests/fixtures/stats_wnba_player_info.json`
- Modify: `backend/tests/test_wnba_player.py`

**Interfaces:**
- Extends `WnbaPlayerResponse` with:
  - `jersey: str | None = None`
  - `height: str | None = None`
  - `birthdate: str | None = None`
  - `college: str | None = None`
  - `draft_info: str | None = None`
  - `position` remains `str | None` but normalize now returns full display name when possible
- Produces helpers:
  - `format_jersey(raw) -> str | None`
  - `format_height(raw) -> str | None`
  - `format_birthdate(raw, *, today: date | None = None) -> str | None`
  - `format_position(raw) -> str | None`
  - `format_draft(year, round_, number, team_abbrev) -> str | None`
  - `format_college(raw) -> str | None` (strip; treat empty / `"N/A"` as None)

Upstream `CommonPlayerInfo` headers to read (defensive — try aliases):
- Jersey: `JERSEY`
- Height: `HEIGHT` (often `"5-10"` → display `5' 10"`)
- Birthdate: `BIRTHDATE` (ISO `YYYY-MM-DDTHH:MM:SS` or date-only)
- College: `SCHOOL` then `LAST_AFFILIATION` (if affiliation looks like `"School/country"`, take part before `/` when useful; prefer `SCHOOL`)
- Draft: `DRAFT_YEAR`, `DRAFT_ROUND`, `DRAFT_NUMBER`; team from info `TEAM_ABBREVIATION` or dash abbrev passed in
- Position: `POSITION` via `format_position`

- [ ] **Step 1: Expand info fixture**

Update `backend/tests/fixtures/stats_wnba_player_info.json` CommonPlayerInfo headers/row to include bio fields for A'ja Wilson–style sample, e.g.:

```json
"headers": [
  "PERSON_ID", "DISPLAY_FIRST_LAST", "BIRTHDATE", "SCHOOL",
  "HEIGHT", "JERSEY", "POSITION", "TEAM_NAME", "TEAM_ABBREVIATION",
  "DRAFT_YEAR", "DRAFT_ROUND", "DRAFT_NUMBER"
],
"rowSet": [[
  1628932, "A'ja Wilson", "1996-08-08T00:00:00", "South Carolina",
  "6-4", "22", "C", "Las Vegas Aces", "LVA",
  "2018", "1", "1"
]]
```

- [ ] **Step 2: Write failing formatter + normalize tests**

```python
from datetime import date

def test_format_height():
    assert svc.format_height("5-10") == "5' 10\""
    assert svc.format_height("6-4") == "6' 4\""
    assert svc.format_height(None) is None


def test_format_birthdate_with_age():
    assert svc.format_birthdate(
        "2003-01-29T00:00:00", today=date(2026, 8, 1)
    ) == "1/29/2003 (23)"


def test_format_position_expands_abbrev():
    assert svc.format_position("G") == "Guard"
    assert svc.format_position("F") == "Forward"
    assert svc.format_position("C") == "Center"
    assert svc.format_position("Guard") == "Guard"


def test_format_draft():
    assert (
        svc.format_draft("2026", "1", "2", "MIN")
        == "2026: Rd 1, Pk 2 (MIN)"
    )
    assert svc.format_draft(None, None, None, "MIN") is None


def test_normalize_includes_bio_fields():
    result = svc.normalize_wnba_player(
        player_id="1628932",
        season=2026,
        dash=_load("stats_wnba_player_dash.json"),
        info=_load("stats_wnba_player_info.json"),
        gamelog=_load("stats_wnba_player_gamelog.json"),
    )
    assert result is not None
    assert result.jersey == "22"
    assert result.position == "Center"
    assert result.height == "6' 4\""
    assert result.college == "South Carolina"
    assert result.draft_info == "2018: Rd 1, Pk 1 (LVA)"
    assert result.birthdate  # contains year and age parens
```

Also add a sparse-info test: info row without HEIGHT/SCHOOL → those fields `None`, player still returns.

- [ ] **Step 3: Run — expect FAIL**

`PYTHONPATH=backend python3 -m pytest backend/tests/test_wnba_player.py -v`

- [ ] **Step 4: Implement schema + helpers + wire normalize**

Add fields to `WnbaPlayerResponse`. Replace `_position_from_info` usage with `format_position(...)`. Extract bio from `info_rows[0]` when present. Pass `team_abbrev` into draft formatter as fallback team.

Age calculation: completed years from birthdate to `today` (default `date.today()`), careful with birthday not yet reached this year.

Height: if already contains `'`, return stripped; if `feet-inches` pattern, convert; else return stripped raw or None if empty.

- [ ] **Step 5: Run — expect PASS**

- [ ] **Step 6: Commit**

```bash
git add backend/app/schemas/wnba_player.py backend/app/services/wnba_player.py \
  backend/tests/fixtures/stats_wnba_player_info.json backend/tests/test_wnba_player.py
git commit -m "$(cat <<'EOF'
feat: parse WNBA player bio fields from commonplayerinfo

EOF
)"
```

---

### Task 2: OpenAPI sync

**Files:**
- Regenerate: `frontend/openapi.json`, `frontend/src/lib/api.schema.d.ts`

**Interfaces:**
- Schema types gain optional `jersey`, `height`, `birthdate`, `college`, `draft_info`

- [ ] **Step 1: Export + generate**

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor
PYTHONPATH=backend python3 scripts/export_openapi.py
cd frontend && npm run generate:api
```

Confirm new properties exist on `WnbaPlayerResponse` in `api.schema.d.ts`.

- [ ] **Step 2: Commit**

```bash
git add frontend/openapi.json frontend/src/lib/api.schema.d.ts
git commit -m "$(cat <<'EOF'
chore: sync OpenAPI for WNBA player bio fields

EOF
)"
```

---

### Task 3: PlayerHeader ESPN-style layout

**Files:**
- Modify: `frontend/src/components/league/PlayerHeader.tsx`
- Modify: `frontend/src/components/league/PlayerHeader.test.tsx`
- Modify if needed: `frontend/src/pages/LeaguePlayerPage.test.tsx` (fixtures must include new optional fields or remain valid)

**Interfaces:**
- Consumes new nullable fields on `ApiWnbaPlayerResponse`
- Subtitle: join non-empty `[jersey && \`#${jersey}\`, position, team_name]` with ` · `
- Fact rows (label → value), skip nulls:
  - Height → `height`
  - Birthdate → `birthdate`
  - College → `college`
  - Draft Info → `draft_info`

- [ ] **Step 1: Update failing/expanded tests**

```tsx
const player: ApiWnbaPlayerResponse = {
  // ...existing
  position: "Center",
  jersey: "22",
  height: "6' 4\"",
  birthdate: "8/8/1996 (29)",
  college: "South Carolina",
  draft_info: "2018: Rd 1, Pk 1 (LVA)",
};

it("renders ESPN-style bio facts", () => {
  render(<PlayerHeader player={player} />);
  expect(screen.getByText(/#22/)).toBeInTheDocument();
  expect(screen.getByText(/Center/)).toBeInTheDocument();
  expect(screen.getByText("Height")).toBeInTheDocument();
  expect(screen.getByText("6' 4\"")).toBeInTheDocument();
  expect(screen.getByText("Birthdate")).toBeInTheDocument();
  expect(screen.getByText("College")).toBeInTheDocument();
  expect(screen.getByText("South Carolina")).toBeInTheDocument();
  expect(screen.getByText("Draft Info")).toBeInTheDocument();
  expect(screen.getByText("2018: Rd 1, Pk 1 (LVA)")).toBeInTheDocument();
});

it("omits missing bio rows", () => {
  render(
    <PlayerHeader
      player={{ ...player, height: null, college: null, draft_info: null, birthdate: null }}
    />,
  );
  expect(screen.queryByText("Height")).not.toBeInTheDocument();
  expect(screen.queryByText("College")).not.toBeInTheDocument();
  expect(screen.queryByText("Draft Info")).not.toBeInTheDocument();
});
```

Update the old “renders bio and season averages” test: position text is now full name; subtitle may be one combined string — assert with flexible matchers (`getByText` content function or regex).

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement layout**

```tsx
// Sketch
<section className="rounded-xl border border-white/10 bg-white/[0.03] p-4">
  <div className="flex flex-col gap-6 md:flex-row md:items-start md:justify-between">
    <div className="flex min-w-0 flex-1 gap-4">
      {/* larger headshot ~ size-24 */}
      <div className="min-w-0 flex-1">
        <h2>...</h2>
        <p className="text-sm text-white/45">{subtitle}</p>
        <dl className="mt-4 space-y-2 text-sm">
          {rows.map(({ label, value }) => (
            <div key={label} className="grid grid-cols-[7rem_1fr] gap-2">
              <dt className="text-white/35">{label}</dt>
              <dd className="text-white">{value}</dd>
            </div>
          ))}
        </dl>
      </div>
    </div>
    {/* existing avg tiles */}
  </div>
</section>
```

- [ ] **Step 4: Run tests + build**

`npm run test -- --run src/components/league/PlayerHeader.test.tsx src/pages/LeaguePlayerPage.test.tsx`  
`npm run build`

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/league/PlayerHeader.tsx \
  frontend/src/components/league/PlayerHeader.test.tsx \
  frontend/src/pages/LeaguePlayerPage.test.tsx
git commit -m "$(cat <<'EOF'
feat: showcase jersey, bio facts in WNBA player header

EOF
)"
```

---

## Spec coverage checklist

| Spec requirement | Task |
| --- | --- |
| Bio fields on API | 1 |
| Format height / birthdate+age / position / draft | 1 |
| Omit missing | 1, 3 |
| OpenAPI types | 2 |
| ESPN-style left column + larger headshot | 3 |
| Averages unchanged on right | 3 |

## Plan self-review

- No TBD placeholders; draft/position/height rules explicit.
- `position` field reused (display-expanded) — matches spec “keep single position string”.
- Fixture expansion keeps A'ja Wilson id consistent with existing dash/gamelog fixtures.
