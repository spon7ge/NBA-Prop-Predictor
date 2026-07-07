import { useEffect, useState, type ReactNode } from "react";
import type { Book, FlatParlayRow, LegCount } from "@/types/slate";
import { BOOK_LABELS, BOOKS, LEG_LABELS, SLATE_LEG_COUNTS } from "@/lib/constants";
import { hasAnySlates, loadAllSlates, mapRowN, slateJsonFilename } from "@/lib/slate";
import { Dropdown } from "@/components/Dropdown";
import { ParlayCard } from "@/components/ParlayCard";

type SlatesState = Record<LegCount, Record<Book, FlatParlayRow[]>>;

export function TopLegsView() {
  const [slates, setSlates] = useState<SlatesState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeBook, setActiveBook] = useState<Book>("prizepicks");
  const [activeLegs, setActiveLegs] = useState<LegCount>(2);

  useEffect(() => {
    let cancelled = false;
    loadAllSlates()
      .then((data) => {
        if (cancelled) return;
        if (!hasAnySlates(data)) {
          setError("Could not load slate JSON.");
        } else {
          setSlates(data);
        }
      })
      .catch(() => {
        if (!cancelled) setError("Failed to load slates.");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const bookOptions = BOOKS.map((book) => ({ value: book, label: BOOK_LABELS[book] }));
  const legOptions = SLATE_LEG_COUNTS.map((legs) => ({ value: legs, label: LEG_LABELS[legs] }));

  let content: ReactNode;
  if (error) {
    content = (
      <p className="load-msg load-err">
        {error} Serve the site over HTTP and place slate files under{" "}
        <code>data/props/ev_analysis/</code>.
      </p>
    );
  } else if (!slates) {
    content = <p className="load-msg">Loading slates…</p>;
  } else if (SLATE_LEG_COUNTS.indexOf(activeLegs) === -1) {
    content = (
      <p className="load-msg">
        No {LEG_LABELS[activeLegs]} slate is available yet. Export JSON for this leg count into{" "}
        <code>data/props/ev_analysis/</code>.
      </p>
    );
  } else {
    const sorted = slates[activeLegs][activeBook] ?? [];
    if (!sorted.length) {
      content = (
        <p className="load-msg">
          No parlays in the {BOOK_LABELS[activeBook]} slate. Export{" "}
          <code>{slateJsonFilename(activeBook, activeLegs)}</code> into{" "}
          <code>data/props/ev_analysis/</code>.
        </p>
      );
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
