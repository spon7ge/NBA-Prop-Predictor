# PrizePicks DataDome Playwright Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add DataDome detection, clearer errors, and a headed Playwright last-resort fetch to `prizepicks_scraper.py`.

**Architecture:** Keep curl_cffi → requests; after both fail, open Chromium, clear/wait for DataDome on app.prizepicks.com, then `page.evaluate(fetch(API_URL))`. Unit-test detection helpers and error text without live network.

**Tech Stack:** Python 3, requests, curl_cffi (existing), playwright (optional), pytest

## Global Constraints

- Headed by default; `PRIZEPICKS_HEADLESS` truthy (`1`/`true`/`yes`) → headless
- Playwright optional; missing import → skip with log, do not crash before final error
- Captcha wait ≤ ~120s
- Do not change league_id / extract / save / CLI beyond env docs
- No live PrizePicks calls in CI tests
- Do not commit unless user asks

---

## File Structure

- Modify: `src/scrapers/prizepicks_scraper.py` — detection helpers, Playwright fetch, wire into `fetch_projections_payload`, update error text
- Create: `tests/scrapers/test_prizepicks_scraper.py` — unit tests for detection + error builder (+ headless env parsing)

---

### Task 1: DataDome detection helpers + error text

**Files:**
- Modify: `src/scrapers/prizepicks_scraper.py`
- Create: `tests/scrapers/test_prizepicks_scraper.py`

**Interfaces:**
- Produces: `is_datadome_challenge(body: str, headers: dict[str, str] | None = None) -> bool`
- Produces: `is_bot_challenge(body: str, headers: dict[str, str] | None = None) -> bool` (DataDome or legacy PerimeterX)
- Produces: `build_fetch_failure_message() -> str`
- Produces: `headless_from_env() -> bool`

- [x] **Step 1: Write failing tests** for detection fixtures and failure message contents
- [x] **Step 2: Implement helpers**; keep `is_perimeterx_challenge` as thin wrapper or fold into `is_bot_challenge`
- [x] **Step 3: Use `build_fetch_failure_message()` in `fetch_projections_payload`**
- [x] **Step 4: Run pytest on the new test file**

### Task 2: Playwright fetch fallback

**Files:**
- Modify: `src/scrapers/prizepicks_scraper.py`

**Interfaces:**
- Produces: `try_fetch_with_playwright() -> dict[str, Any] | None`
- Consumes: `API_URL`, `headless_from_env()`, `is_datadome_challenge`
- Wire: `fetch_projections_payload` calls Playwright after requests fails

- [x] **Step 1: Implement `try_fetch_with_playwright`** (sync Playwright; headed default; poll fetch ≤120s)
- [x] **Step 2: Wire into `fetch_projections_payload`**
- [x] **Step 3: Update module docstring / argparse epilog for new env vars**
- [x] **Step 4: Log DataDome on HTTP 403 bodies in existing fetch helpers**
- [x] **Step 5: Re-run unit tests**

### Task 3: Local smoke (manual)

- [x] Install Chromium; smoke headless Playwright path (may still need headed captcha)
