# HoopVista

## What the website does

The public site (under `docs/`) is a **read-only dashboard** for your NBA player prop model. It does not run picks on a server in the browser. Instead it loads **precomputed JSON** you (or CI) export from the modeling pipeline.

- **Top Pairs** — Shows ranked **2-leg parlays** by **expected value (EV)**. You pick a **book** (PrizePicks or Underdog); the page loads that book’s slate file and renders each parlay as a card with EV, hit probability, Kelly, both legs (lines, sides, model context, form, matchup notes), and basic game info (total, spread).
- **All Players** — A searchable table of **every modeled line** (not only pairs): odds, EV by side, “best” side, and rolling hit rates, driven by a JSON Lines export.

The UI is built for **clarity on phone and desktop**: two main views, book selection, search, and stat filters on the player table.

---

## Tech stack

| Area | Stack |
|------|--------|
| **Model & data pipeline** | Python: **pandas**, **NumPy**, **SciPy**; tree models (**XGBoost**, **NGBoost**), **scikit-learn**; **nba-api**; notebooks and scripts under `notebooks/`, `src/` |
| **Optional app UI** | **Streamlit** (for internal/exploratory views, not required for the static site) |
| **Public website** | **Static HTML/CSS/JavaScript** — `docs/index.html`, `docs/slate.css`, `docs/slate.js`; no React/build step for the slate page |
| **Hosting** | **GitHub Pages** (see `.github/workflows/pages.yml`): copies EV JSON into `docs/data/` and deploys the `docs/` folder |
| **Data APIs** | **NBA API** (and related sources as wired in your pipeline); environment config via **python-dotenv** where needed |

---

## How it works

1. **Offline pipeline** — Your code trains/scores props, computes EV and parlay math, and **writes JSON** under `data/props/ev_analysis/` (for example `prizepicks.json`, `underdog.json`, and `all_line_probs.json` for the player table).

2. **Static site** — The slate page is plain HTML/CSS/JS. On load, the browser **fetches those JSON files** (paths resolve for both local dev and GitHub Pages). There is **no backend API** on the site itself.

3. **Deploy** — On push to `main`, the Pages workflow can **copy** the latest EV JSON from `data/props/ev_analysis/` into `docs/data/props/ev_analysis/`, bump asset cache keys, and publish `docs/` so visitors always see files that match your repo’s exported data.

4. **“Last updated”** — The Top Pairs header can show a last-updated time from `docs/build-meta.json`. Regenerate it with `python scripts/update_build_meta.py` before you commit (see script docstring); CI may also stamp build metadata depending on your workflow.

In short: **Python produces the numbers → JSON ships with the site → the browser renders them.**

---

## Disclaimer

This project is for **educational purposes only**. Sports betting involves risk. Gamble responsibly.
