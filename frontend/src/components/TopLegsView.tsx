import { useEffect, useState, type ReactNode } from "react";
import type { ApiLeagueFilter } from "@/types/api";
import type { Book, LegCount } from "@/types/slate";
import { BOOK_LABELS, BOOKS, LEG_LABELS, SLATE_LEG_COUNTS } from "@/lib/constants";
import { hasAnySlates, mapRowN } from "@/lib/slate";
import { useLiveSlates } from "@/lib/queries";
import { Dropdown } from "@/components/Dropdown";
import { LoadingMessage } from "@/components/LoadingSkeleton";
import { ParlayCard } from "@/components/ParlayCard";

const LEAGUE_OPTIONS: { value: ApiLeagueFilter; label: string }[] = [
  { value: "wnba", label: "WNBA" },
  { value: "nba", label: "NBA" },
];

function getInitialLeague(): ApiLeagueFilter {
  try {
    const q = new URLSearchParams(window.location.search);
    const league = q.get("league");
    if (league === "nba" || league === "wnba") return league;
  } catch {
    /* ignore */
  }
  return "wnba";
}

export function TopLegsView() {
  const [league, setLeague] = useState<ApiLeagueFilter>(getInitialLeague);
  const { data, isLoading, isError, error } = useLiveSlates(league);
  const [activeBook, setActiveBook] = useState<Book>("prizepicks");
  const [activeLegs, setActiveLegs] = useState<LegCount>(2);

  useEffect(() => {
    const url = new URL(window.location.href);
    url.searchParams.set("league", league);
    window.history.replaceState(null, "", url);
  }, [league]);

  const bookOptions = BOOKS.map((book) => ({ value: book, label: BOOK_LABELS[book] }));
  const legOptions = SLATE_LEG_COUNTS.map((legs) => ({ value: legs, label: LEG_LABELS[legs] }));

  const slates = data?.slates ?? null;

  const emptyHint = (
    <p className="load-msg">
      No parlays for this slate yet. Try another book, league, or leg count.
    </p>
  );

  let content: ReactNode;
  if (isError) {
    content = (
      <p className="load-msg load-err">
        {error instanceof Error ? error.message : "Couldn't load parlays. Try again later."}
      </p>
    );
  } else if (isLoading || !slates) {
    content = (
      <>
        <LoadingMessage>Loading slates…</LoadingMessage>
        <div className="parlay-skeleton-list" aria-hidden="true">
          {Array.from({ length: 3 }, (_, i) => (
            <div key={i} className="card card--skeleton">
              <span className="skeleton-block skeleton-block--card" />
            </div>
          ))}
        </div>
      </>
    );
  } else if (!hasAnySlates(slates)) {
    content = emptyHint;
  } else if (SLATE_LEG_COUNTS.indexOf(activeLegs) === -1) {
    content = emptyHint;
  } else {
    const sorted = slates[activeLegs][activeBook] ?? [];
    if (!sorted.length) {
      content = emptyHint;
    } else {
      content = sorted.map((row, i) => (
        <ParlayCard key={i} parlay={mapRowN(row, activeLegs)} rank={i + 1} nLegs={activeLegs} />
      ));
    }
  }

  return (
    <section className="view-section" aria-labelledby="headingPairs">
      <header className="view-section-head view-section-head--pairs">
        <h2 className="view-section-title" id="headingPairs">
          Top Legs
        </h2>
      </header>
      <div className="toolbar book-toolbar">
        <Dropdown
          id="bookDropdown"
          label="Book"
          value={activeBook}
          options={bookOptions}
          onChange={setActiveBook}
          classPrefix="book"
        />
        <Dropdown
          id="leagueDropdown"
          label="League"
          value={league}
          options={LEAGUE_OPTIONS}
          onChange={setLeague}
          classPrefix="league"
        />
        <Dropdown
          id="legsDropdown"
          label="Legs"
          value={activeLegs}
          options={legOptions}
          onChange={setActiveLegs}
          classPrefix="legs"
        />
      </div>
      <div id="cards">{content}</div>
    </section>
  );
}
