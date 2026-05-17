# HoopVista — How It Works (Plain English)

This page explains what this project does without any coding or math jargon. Think of it like a homework helper for NBA pick’em apps—not a crystal ball, and not a guarantee you’ll win.

---

## What problem are we solving?

On apps like **PrizePicks**, **Underdog**, **DraftKings Pick 6**, and **Betr**, you pick whether a player will do **more** or **less** than a number—for example, “LeBron over 24.5 points.”

That sounds simple, but there are hundreds of players and lines every night. This project tries to answer one question for each line:

> **“Based on history and today’s situation, how likely is over vs. under?”**

Then it highlights picks that look **better than the payout**—not just “probably right,” but “worth playing if you played this many times.”

---

## The big picture (three steps on game day)

Every day with NBA games, the project runs three chores in order—like making lunch: get ingredients, read the menu, then cook.

```
1. Update player stats     →  “What did everyone do lately?”
2. Fetch today’s lines     →  “What are the apps offering tonight?”
3. Run the daily pipeline  →  “Guess, adjust, rank, and save picks”
```

After step 3, results show up on the **HoopVista** website: **Top Legs** (suggested parlays) and **All Players** (every line the model looked at).

---

## Step 1 — Remember what players have been doing

**What it does:** Collects recent game stats from the NBA (points, assists, rebounds, minutes, etc.) and adds helpful context like team spreads and game totals.

**Non-technical version:** Before guessing tonight, we look at each player’s report card from past games—how many points they usually get, how many minutes they play, and whether their team is expected to score a lot or win big.

**Why it matters:** You can’t guess tonight well if you don’t know what “normal” looks like for each player.

---

## Step 2 — See what the apps are offering tonight

**What it does:** Downloads today’s prop lines from betting data sources—both the pick’em apps and sharper “traditional” sportsbooks.

**Non-technical version:** We read today’s menu: “App says Jayson Tatum 27.5 points—over or under?”

**Why it matters:** Our guesses only matter when compared to a **real line** someone can actually play.

---

## Step 3 — The “brain” runs (the daily pipeline)

This is the main event. Here’s what happens inside, in simple terms.

### A. Check who’s actually playing

The project looks at **starters** and **injuries** for today’s games.

**Non-technical version:** If a star is out, their teammate might shoot more. We don’t want to predict like the star is still there.

---

### B. Four mini-brains learned from the past

Over time (not every day), the project **trains models** on thousands of old games. There are four related predictors:

| Mini-brain | Guesses… |
|------------|----------|
| Minutes | How long will they be on the court? |
| Points per minute | When they play, how fast do they score? |
| Assists per minute | Same for assists |
| Rebounds per minute | Same for rebounds |

**Minutes × rate ≈ stat** (e.g., 32 minutes × 0.75 points/minute → about 24 points).

Each prediction isn’t one number—it’s a **range**: a low guess, middle guess, and high guess (like “probably between 20 and 28 points”).

**Non-technical version:** We taught four calculators by showing them old games. One calculator knows playing time; another knows scoring speed. Multiply them to get points, assists, or rebounds.

---

### C. Tweak for today’s story

Raw guesses get nudged based on things like:

- A star teammate is **out**
- The game might be **fast** or **slow**
- The spread suggests a **blowout** (bench players play more)
- The **defense** they’re facing

**Non-technical version:** The calculators give a first answer, then we say, “Wait—it’s a tough matchup tonight” or “His buddy is injured so he’ll shoot more,” and we adjust a little—not wildly.

---

### D. Roll dice thousands of times

For each player and each line (e.g., 24.5 points), the computer **simulates** the game over and over—like rolling dice 10,000 times—and counts how often the player goes **over** vs **under**.

That gives **P_OVER** and **P_UNDER** (chances out of 100%).

**Non-technical version:** Instead of one guess, we pretend the game happens 10,000 times in a video game and see how often he beats the line.

---

### E. Compare our guess to “sharp” books

Pick’em apps don’t always show the same odds as serious sportsbooks. The project checks whether **smarter market lines** agree with our lean.

| Label | Plain meaning |
|-------|----------------|
| **Sharp verified** | Our side matches what sharper books imply—feels more trustworthy |
| **DFS only** | Only the pick’em app has the line; no sharp book to double-check |
| **Conflict** | We lean one way; sharper books lean the other—be careful |
| **No model** | We couldn’t model this line reliably (gap too big, etc.) |

**Non-technical version:** We ask a second opinion from stricter “grading teachers” (sharp books) when we can.

---

### F. Build suggested parlays (Top Legs)

Apps pay more if you string picks together (2, 3, 5, or 6 legs). The project:

1. Finds combinations where the model likes **all** legs
2. Estimates **hit chance** and **expected value (EV)**—whether the payout is fair for the risk
3. Ranks the best stacks per app (PrizePicks, Underdog, etc.)

**Expected value (EV)** in one sentence: *If you made this bet thousands of times, would you come out ahead on average?* Positive EV doesn’t mean you win tonight—it means the math favors the bet long-term.

**Non-technical version:** We bundle a few “I like over” picks into a parlay ticket and sort them so the best math deals are on top.

---

### G. Save everything and show it on the site

Outputs land in data files; the **HoopVista** site reads them and shows:

- **Top Legs** — ranked parlay cards with EV, hit rate, each leg’s line and context
- **All Players** — searchable table of every modeled line, both sides, and recent form

**Non-technical version:** We write down today’s homework answers and put them on a website so you can browse instead of reading spreadsheets.

---

## What about “training” the brain?

Training doesn’t run every day. Occasionally (e.g., start of playoffs), the project retrains on historical games with rules like:

- Only use information that existed **before** each game (no cheating with future stats)
- Treat **stars** and **role players** differently so averages don’t blur together
- Respect **time order**—old games teach the model; recent games matter most

**Non-technical version:** The calculators go to school on old seasons, then get a report card. On game nights we only **use** what they learned—we don’t re-teach them every morning.

---

## What this project is *not*

- **Not a sportsbook** — it doesn’t take bets or move money
- **Not financial advice** — it’s a research tool for learning how modeling and EV thinking work
- **Not better than the books** — sportsbooks employ huge teams; this is one person’s educational project
- **Not a guarantee** — even good math loses often in the short run; variance is real

**Please gamble responsibly.** Only bet what you can afford to lose.

---

## One-sentence summary

**HoopVista collects NBA history and tonight’s prop lines, uses trained models plus game-day context to estimate over/under chances, ranks picks and parlays by expected value, and displays them on a website so you can research—not blindly follow—player prop decisions.**

---

## Where to learn more (still friendly)

- **README.md** — what the website pages show
- **workflow.md** — technical step-by-step for developers
- **Disclaimer in README** — educational use only

If something on the site looks confusing, start with **All Players** for single picks, then **Top Legs** when you’re ready to think about parlays and EV together.
