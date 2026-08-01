# PrizePicks scraper — DataDome + Playwright fallback

Date: 2026-07-31  
Status: Approved for planning

## Goal

Unblock `prizepicks_scraper.py` when PrizePicks returns DataDome 403 challenges. Keep the existing fast HTTP path; add a headed Playwright fallback that clears (or lets the user clear) the challenge, then fetches projections in-page. Improve failure messages so they name DataDome, not PerimeterX.

## Decisions

| Topic | Choice |
| --- | --- |
| Primary fetch | Unchanged: `curl_cffi` impersonation, then `requests` |
| Browser fallback | Playwright Chromium after both HTTP paths fail |
| Browser mode | Headed by default; `PRIZEPICKS_HEADLESS=1` forces headless |
| Challenge handling | Wait up to ~120s; if captcha still present, log that user should solve it in the open window |
| API fetch in browser | `page.evaluate` → `fetch(API_URL, { credentials: 'include' })` (same pattern as Rotowire scraper) |
| Cookie env vars | Keep `PRIZEPICKS_COOKIE` / `PRIZEPICKS_COOKIE_FILE` for HTTP paths |
| Bot detection naming | Detect DataDome; retire PerimeterX-centric troubleshooting copy |
| Scope | `src/scrapers/prizepicks_scraper.py` + unit tests for detection / error text |
| Dependencies | Optional import of Playwright (already in `requirements-etl.txt`); clear install hint if missing |

## Architecture

```
fetch_projections_payload()
  ├── try_fetch_with_curl_cffi()
  ├── try_fetch_with_requests()
  └── try_fetch_with_playwright()   # new last resort
        ├── launch Chromium (headed unless PRIZEPICKS_HEADLESS=1)
        ├── goto https://app.prizepicks.com/
        ├── wait for DataDome clearance (or user captcha, ≤ ~120s)
        └── page.evaluate(fetch API_URL) → JSON with "data"
```

On total failure, raise with troubleshooting that mentions DataDome, Playwright install (`pip install playwright && playwright install chromium`), headed captcha solve, and optional cookie env vars.

## Behavior details

### DataDome detection

A response is treated as a DataDome block when any of:

- Body / URL contains `captcha-delivery.com` or `geo.captcha-delivery.com`
- Headers include `X-DataDome` (or body JSON has a captcha `url` pointing at captcha-delivery)
- Legacy PerimeterX markers remain recognized for logging only if still seen

HTTP helpers should log status + “DataDome challenge” at DEBUG when detected, instead of a silent skip.

### Playwright flow

1. If Playwright is not installed → log skip reason and return `None` (do not crash before the final error).
2. Launch browser headed unless `PRIZEPICKS_HEADLESS` is truthy (`1`, `true`, `yes`).
3. Navigate to `https://app.prizepicks.com/` with a generous timeout.
4. Poll until either:
   - In-page `fetch(API_URL)` returns JSON containing `data`, or
   - Timeout (~120s) elapses.
5. While waiting, if a captcha interstitial is detected, log once: solve captcha in the browser window.
6. Close browser in a `finally` block.
7. Return parsed payload or `None`.

### Error message updates

Replace PerimeterX-first troubleshooting with:

1. DataDome is blocking automated clients
2. Install Playwright + Chromium if using browser fallback
3. Re-run headed and solve captcha if prompted
4. Optional: `PRIZEPICKS_COOKIE` / cookie file for HTTP path
5. `LOG_LEVEL=DEBUG` for diagnostics
6. `PRIZEPICKS_HEADLESS=1` only when no interactive captcha is expected

## Out of scope

- Changing `league_id` / NBA vs WNBA targeting
- Airflow / ETL pipeline wiring
- Persisting Playwright cookies to disk across runs
- Captcha-solving services or third-party anti-bot APIs
- Frontend or backend API changes

## Testing

- Unit tests for DataDome / challenge detection helpers (fixture bodies and header-like dicts)
- Unit tests that failure message text mentions DataDome and Playwright (not PerimeterX as primary)
- Do not require live PrizePicks network in CI; Playwright fetch path may be lightly smoke-tested locally only

## Success criteria

- When HTTP paths get DataDome 403, scraper attempts Playwright instead of failing immediately
- Headed run can succeed after manual captcha if needed
- Failure output correctly identifies DataDome and documents Playwright + cookie options
- Existing extract/save/CLI behavior unchanged when fetch succeeds
