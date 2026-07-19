import { useState, type ReactNode } from "react";
import type {
  ApiGradedPick,
  ApiHitRateBucket,
  ApiLeagueFilter,
  ApiParlaySummary,
} from "@/types/api";
import type { Book } from "@/types/slate";
import { BOOK_LABELS, BOOKS, LEG_LABELS, SLATE_LEG_COUNTS } from "@/lib/constants";
import {
  buildBankrollSeries,
  computeLegsRoi,
  formatRoiPct,
  formatUsd,
} from "@/lib/legsRoi";
import { usePerformance, type ResultsLegsFilter } from "@/lib/queries";
import { BankrollCurve } from "@/components/BankrollCurve";
import { Dropdown } from "@/components/Dropdown";
import { LoadingMessage } from "@/components/LoadingSkeleton";
import { TicketStripList } from "@/components/TicketStrip";

const LEAGUE_OPTIONS: { value: ApiLeagueFilter; label: string }[] = [
  { value: "wnba", label: "WNBA" },
  { value: "nba", label: "NBA" },
];

const LEGS_OPTIONS: { value: ResultsLegsFilter; label: string }[] = [
  { value: "all", label: "All" },
  { value: "singles", label: "Singles" },
  ...SLATE_LEG_COUNTS.map((n) => ({
    value: String(n) as ResultsLegsFilter,
    label: LEG_LABELS[n],
  })),
];

/** Map API bookmaker display names → Book slug. */
const BOOK_DISPLAY_TO_SLUG: Record<string, Book> = {
  prizepicks: "prizepicks",
  underdog: "underdog",
  draftkings: "draftkings",
  "draftkings pick6": "draftkings",
  "draftkings pick 6": "draftkings",
  betr: "betr",
  "betr dfs": "betr",
};

function bookSlugFromDisplay(raw: string | null | undefined): Book | null {
  if (!raw) return null;
  const key = raw.trim().toLowerCase();
  if (key in BOOK_DISPLAY_TO_SLUG) return BOOK_DISPLAY_TO_SLUG[key];
  for (const book of BOOKS) {
    if (key.includes(book)) return book;
  }
  return null;
}

function bookLabel(raw: string): string {
  const slug = bookSlugFromDisplay(raw);
  return slug ? BOOK_LABELS[slug] : raw;
}

function pct(rate: number | null | undefined): string {
  if (rate == null || Number.isNaN(rate)) return "—";
  return `${Math.round(rate * 100)}%`;
}

function recordLine(bucket: ApiHitRateBucket | undefined): string {
  if (!bucket || bucket.n <= 0) return "no graded props yet";
  const misses = bucket.n - bucket.hits;
  return `${bucket.hits}–${misses} (${pct(bucket.hit_rate)})`;
}

function formatSide(side: string): string {
  return side.toLowerCase() === "under" ? "Under" : "Over";
}

function formatActual(stat: number | null | undefined): string {
  if (stat == null) return "—";
  const n = Number(stat);
  return Number.isInteger(n) ? String(n) : n.toFixed(1);
}

function formatLine(line: number | null | undefined): string {
  if (line == null) return "—";
  return Number.isInteger(line) ? String(line) : line.toFixed(1);
}

function parlayStatus(parlay: {
  cashed: boolean | null;
  legs_hit: number;
  n_legs: number;
}): { label: string; className: string } {
  // Hit only when every leg cashed; anything else is a miss (or pending).
  if (parlay.cashed === true) {
    return { label: "CASH", className: "results-pick--hit" };
  }
  if (parlay.cashed === false) {
    return { label: "MISS", className: "results-pick--miss" };
  }
  return { label: "OPEN", className: "results-pick--pending" };
}

function parlayHeadline(summary: ApiParlaySummary | undefined): string | null {
  if (!summary) return null;
  const parts: string[] = [];
  if (summary.decided > 0) {
    parts.push(
      `Parlays ${summary.cashed}/${summary.decided} cashed (${pct(summary.cash_rate)})`,
    );
  }
  if (summary.legs_scored > 0) {
    parts.push(`Legs ${pct(summary.leg_hit_rate)}`);
  }
  return parts.length ? parts.join(" · ") : null;
}

function legsFilterLabel(legs: ResultsLegsFilter): string {
  if (legs === "all") return "All";
  if (legs === "singles") return "Singles";
  return LEG_LABELS[Number(legs) as 2 | 3 | 5 | 6] ?? legs;
}

interface ResultsViewProps {
  league: ApiLeagueFilter;
  onLeagueChange: (league: ApiLeagueFilter) => void;
}

export function ResultsView({ league, onLeagueChange }: ResultsViewProps) {
  const [activeBook, setActiveBook] = useState<Book | "all">("all");
  const [activeLegs, setActiveLegs] = useState<ResultsLegsFilter>("all");
  const [bankroll, setBankroll] = useState(1000);
  const [stakePerTicket, setStakePerTicket] = useState(10);
  const { data, isLoading, isError, error } = usePerformance(
    league,
    7,
    activeBook,
    activeLegs,
  );

  const bookOptions: { value: Book | "all"; label: string }[] = [
    { value: "all", label: "All Books" },
    ...BOOKS.map((book) => ({ value: book, label: BOOK_LABELS[book] })),
  ];

  const showParlays = activeLegs !== "singles";
  const showSingles = activeLegs === "all" || activeLegs === "singles";
  const isParlayMode = ["2", "3", "5", "6"].includes(activeLegs);
  const showRoi = showParlays;

  let content: ReactNode;
  if (isError) {
    content = (
      <p className="load-msg load-err">
        {error instanceof Error
          ? error.message
          : "Couldn't load results. Grade props after box scores land."}
      </p>
    );
  } else if (isLoading || !data) {
    content = <LoadingMessage>Loading results…</LoadingMessage>;
  } else if (
    data.last_n_days.n <= 0 &&
    data.last_night.n <= 0 &&
    (data.graded_parlays?.length ?? 0) <= 0
  ) {
    content = (
      <p className="load-msg">
        No graded props yet. Results appear after the midnight pipeline grades
        last night&apos;s live props against box scores.
      </p>
    );
  } else {
    const night = data.last_night;
    const week = data.last_n_days;
    const markets = data.by_market.filter((m) =>
      ["PTS", "REB", "AST"].includes(m.key),
    );
    const picks = data.recent_picks.slice(0, 24);
    const allParlays = data.graded_parlays ?? [];
    const parlays = allParlays.slice(0, 16);
    const bookTitle =
      activeBook === "all" ? "All Books" : BOOK_LABELS[activeBook];
    const legsTitle = legsFilterLabel(activeLegs);
    const legsLine = showParlays ? parlayHeadline(data.parlay_summary) : null;
    const roi = showRoi
      ? computeLegsRoi(allParlays, bankroll, stakePerTicket)
      : null;
    const bankrollPoints = showRoi
      ? buildBankrollSeries(allParlays, bankroll, stakePerTicket)
      : [];

    content = (
      <>
        <p className="results-headline" aria-live="polite">
          {isParlayMode ? (
            <>
              Last night {night.n > 0 ? `${night.hits}–${night.n - night.hits} cashed (${pct(night.hit_rate)})` : "no decided parlays"}
              <span className="results-headline-sep" aria-hidden="true">
                ·
              </span>
              Last {data.days} days {pct(week.hit_rate)}
              {week.n > 0 && (
                <span className="results-headline-n">
                  {" "}
                  ({week.hits}/{week.n} cashed)
                </span>
              )}
            </>
          ) : (
            <>
              Last night {recordLine(night)}
              <span className="results-headline-sep" aria-hidden="true">
                ·
              </span>
              Last {data.days} days {pct(week.hit_rate)}
              {week.n > 0 && (
                <span className="results-headline-n"> ({week.hits}/{week.n})</span>
              )}
            </>
          )}
        </p>
        {legsLine && <p className="results-subhead">{legsLine}</p>}

        {showRoi && roi && (
          <div className="results-roi" aria-label="Legs ROI calculator">
            <div className="results-roi-inputs">
              <label className="results-roi-field">
                <span className="results-roi-label">Bankroll</span>
                <span className="results-roi-input-wrap">
                  <span className="results-roi-prefix" aria-hidden="true">
                    $
                  </span>
                  <input
                    type="number"
                    min={0}
                    step={10}
                    inputMode="decimal"
                    className="results-roi-input"
                    value={Number.isFinite(bankroll) ? bankroll : 0}
                    onChange={(e) => setBankroll(Number(e.target.value) || 0)}
                  />
                </span>
              </label>
              <label className="results-roi-field">
                <span className="results-roi-label">Stake / ticket</span>
                <span className="results-roi-input-wrap">
                  <span className="results-roi-prefix" aria-hidden="true">
                    $
                  </span>
                  <input
                    type="number"
                    min={0}
                    step={1}
                    inputMode="decimal"
                    className="results-roi-input"
                    value={Number.isFinite(stakePerTicket) ? stakePerTicket : 0}
                    onChange={(e) =>
                      setStakePerTicket(Number(e.target.value) || 0)
                    }
                  />
                </span>
              </label>
            </div>
            <div className="results-roi-stats">
              <div className="results-roi-stat">
                <span className="results-roi-stat-label">ROI</span>
                <span
                  className={`results-roi-stat-value${
                    roi.roi != null && roi.roi > 0
                      ? " results-roi-stat-value--pos"
                      : roi.roi != null && roi.roi < 0
                        ? " results-roi-stat-value--neg"
                        : ""
                  }`}
                >
                  {formatRoiPct(roi.roi)}
                </span>
              </div>
              <div className="results-roi-stat">
                <span className="results-roi-stat-label">P/L</span>
                <span
                  className={`results-roi-stat-value${
                    roi.profit > 0
                      ? " results-roi-stat-value--pos"
                      : roi.profit < 0
                        ? " results-roi-stat-value--neg"
                        : ""
                  }`}
                >
                  {formatUsd(roi.profit)}
                </span>
              </div>
              <div className="results-roi-stat">
                <span className="results-roi-stat-label">Ending</span>
                <span className="results-roi-stat-value">
                  {formatUsd(roi.endingBankroll)}
                </span>
              </div>
              <div className="results-roi-stat">
                <span className="results-roi-stat-label">Tickets</span>
                <span className="results-roi-stat-value">
                  {roi.cashed}/{roi.decided}
                </span>
              </div>
            </div>
            <BankrollCurve points={bankrollPoints} startBankroll={bankroll} />
            <p className="results-roi-note">
              Flat ${stakePerTicket} per decided ticket · platform net payouts
              (2-leg ~3× return, etc.) · OPEN tickets excluded
            </p>
          </div>
        )}

        {markets.length > 0 && (
          <div className="results-markets" role="list" aria-label="Hit rate by market">
            {markets.map((m) => (
              <div key={m.key} className="results-market" role="listitem">
                <span className="results-market-label">{m.key}</span>
                <span className="results-market-rate">{pct(m.hit_rate)}</span>
                <span className="results-market-n">
                  {m.hits}/{m.n}
                </span>
              </div>
            ))}
          </div>
        )}

        {showParlays && (
          <>
            <h3 className="results-list-title">
              Ticket strips
              <span className="results-list-title-book">
                {" "}
                · {bookTitle} · {legsTitle}
              </span>
            </h3>
            <TicketStripList
              parlays={allParlays}
              stakePerTicket={stakePerTicket}
              bookLabel={bookLabel}
            />

            <h3 className="results-list-title">
              Top Legs results
              <span className="results-list-title-book">
                {" "}
                · {bookTitle} · {legsTitle}
              </span>
            </h3>
            {parlays.length === 0 ? (
              <p className="load-msg">
                No graded parlays for {bookTitle} / {legsTitle} in this window.
              </p>
            ) : (
              <ul className="results-parlay-list">
                {parlays.map((parlay, i) => {
                  const status = parlayStatus(parlay);
                  return (
                    <li
                      key={`${parlay.game_date}-${parlay.bookmaker}-${parlay.n_legs}-${i}`}
                      className={`results-parlay ${status.className}`}
                    >
                      <div className="results-parlay-head">
                        <span className="results-pick-status">{status.label}</span>
                        <span className="results-parlay-book">
                          {bookLabel(parlay.bookmaker)}
                        </span>
                        <span className="results-parlay-meta">
                          {parlay.n_legs}-leg
                          <span className="results-parlay-legs-count">
                            {" "}
                            · {parlay.legs_hit}/{parlay.legs_scored || parlay.n_legs}{" "}
                            legs
                            {parlay.legs_pending > 0
                              ? ` (${parlay.legs_pending} pending)`
                              : ""}
                          </span>
                        </span>
                      </div>
                      <ul className="results-parlay-legs">
                        {parlay.legs.map((leg, j) => {
                          const legClass =
                            leg.hit === true
                              ? "results-leg--hit"
                              : leg.hit === false
                                ? "results-leg--miss"
                                : "results-leg--pending";
                          const mark =
                            leg.hit === true ? "✓" : leg.hit === false ? "✗" : "·";
                          return (
                            <li
                              key={`${leg.player_name}-${leg.market}-${j}`}
                              className={`results-leg ${legClass}`}
                            >
                              <span className="results-leg-mark" aria-hidden="true">
                                {mark}
                              </span>
                              <span className="results-leg-body">
                                <span className="results-leg-player">
                                  {leg.player_name}
                                  {leg.team_abbr ? ` ${leg.team_abbr}` : ""}
                                </span>
                                <span className="results-leg-lean">
                                  {formatSide(leg.side)} {leg.market}{" "}
                                  {formatLine(leg.line)}
                                  {leg.actual_stat != null && (
                                    <span className="results-leg-actual">
                                      {" "}
                                      → {formatActual(leg.actual_stat)}
                                    </span>
                                  )}
                                  {leg.hit === null && leg.miss_reason ? (
                                    <span className="results-pick-reason">
                                      {" "}
                                      · {leg.miss_reason}
                                    </span>
                                  ) : null}
                                </span>
                              </span>
                            </li>
                          );
                        })}
                      </ul>
                    </li>
                  );
                })}
              </ul>
            )}
          </>
        )}

        {showSingles && (
          <>
            <h3 className="results-list-title">
              Graded picks
              {activeLegs === "singles" && (
                <span className="results-list-title-book"> · Singles</span>
              )}
            </h3>
            {picks.length === 0 ? (
              <p className="load-msg">No scored picks in this window.</p>
            ) : (
              <ul className="results-pick-list">
                {picks.map((pick: ApiGradedPick) => (
                  <li
                    key={`${pick.game_date}-${pick.player_name}-${pick.market}-${pick.bookmaker}-${pick.side}`}
                    className={`results-pick${pick.hit ? " results-pick--hit" : " results-pick--miss"}`}
                  >
                    <span
                      className="results-pick-status"
                      aria-label={pick.hit ? "Hit" : "Miss"}
                    >
                      {pick.hit ? "HIT" : "MISS"}
                    </span>
                    <span className="results-pick-body">
                      <span className="results-pick-player">
                        {pick.player_name}
                        {pick.team_abbr ? (
                          <span className="results-pick-team"> {pick.team_abbr}</span>
                        ) : null}
                      </span>
                      <span className="results-pick-lean">
                        {formatSide(pick.side)} {pick.market} {formatLine(pick.line)}
                        {pick.bookmaker ? (
                          <span className="results-pick-book">
                            {" "}
                            · {bookLabel(pick.bookmaker)}
                          </span>
                        ) : null}
                        {pick.stat_q50 != null && (
                          <span className="results-pick-q50">
                            {" "}
                            · model {pick.stat_q50.toFixed(1)}
                          </span>
                        )}
                      </span>
                      <span className="results-pick-actual">
                        Actual {formatActual(pick.actual_stat)}
                        {!pick.hit && pick.miss_reason ? (
                          <span className="results-pick-reason">
                            {" "}
                            · {pick.miss_reason}
                          </span>
                        ) : null}
                      </span>
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </>
        )}
      </>
    );
  }

  return (
    <section className="view-section" aria-labelledby="headingResults">
      <header className="view-section-head view-section-head--results">
        <h2 className="view-section-title" id="headingResults">
          Results
        </h2>
      </header>
      <div className="toolbar book-toolbar">
        <Dropdown
          id="resultsBookDropdown"
          label="Book"
          value={activeBook}
          options={bookOptions}
          onChange={setActiveBook}
          classPrefix="book"
        />
        <Dropdown
          id="resultsLegsDropdown"
          label="Legs"
          value={activeLegs}
          options={LEGS_OPTIONS}
          onChange={setActiveLegs}
          classPrefix="legs"
        />
        <Dropdown
          id="resultsLeagueDropdown"
          label="League"
          value={league === "nba" ? "nba" : "wnba"}
          options={LEAGUE_OPTIONS}
          onChange={onLeagueChange}
          classPrefix="league"
        />
      </div>
      {content}
    </section>
  );
}
