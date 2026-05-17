# HoopVista

## What the website does

HoopVista is a personal NBA player-prop research stack: a probabilistic/modeling pipeline that produces exported top legs for the day. The site is a decision-support dashboard: you pick props and edges using EV, hit rates, lineup/context-style fields, and book-specific outputs, mainly via Top Legs (EV-ranked parlays) and All Players (full modeled slate in a searchable table).

- **Top Legs** — Shows ranked **parlays** by **expected value (EV)**. You pick a **book** (PrizePicks, Underdog, DraftKings Pick 6, Betr) and **leg count** (2, 3, 5, or 6 — 4-leg is skipped). The page loads that book’s slate JSON and renders each parlay as a card with EV, hit probability, Kelly, every leg (lines, sides, model context, form, matchup notes), and basic game info (total, spread).
- **All Players** — A searchable table of **every modeled line** (not only pairs): odds, EV by side, “best” side, and rolling hit rates, driven by a JSON Lines export.

---

## Disclaimer

This project was made for **educational purposes only**. Sports betting involves risk and my model is no where near as good as the actual bookmakers. Gamble responsibly.
