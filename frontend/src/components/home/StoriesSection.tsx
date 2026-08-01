import type { Story } from "./types";
import { normalizeLiveGames } from "./format";
import { SectionHeading } from "./SectionHeading";

type StoriesSectionProps = {
  stories?: Story[];
};

/** Default marketing stories when no `stories` prop is provided. */
export const DEFAULT_STORIES: Story[] = [
  {
    id: "nba-summer-league",
    league: "nba",
    headline: "Summer League is over. Who won?",
    dateLabel: "JUL 27, 2026",
    summary:
      "The final buzzer sounded in Vegas. Here's who actually moved the needle — and who was just noise.",
    graphic: "bracket",
  },
  {
    id: "wnba-deadline",
    league: "wnba",
    headline: "Trade deadline pressure is building",
    dateLabel: "JUL 28, 2026",
    summary:
      "Contenders are shopping for one more piece. The window is short — and the rumors are getting louder.",
    graphic: "trade",
    daysLeft: 3,
  },
  {
    id: "nba-lebron",
    league: "nba",
    headline: "LeBron is a Sixer. Now what?",
    dateLabel: "JUL 26, 2026",
    summary:
      "The East just tilted. Lineups, props, and the new pecking order after a move nobody saw coming.",
    graphic: "crown",
  },
  {
    id: "wnba-all-star",
    league: "wnba",
    headline: "All-Star voting is almost closed",
    dateLabel: "JUL 25, 2026",
    summary:
      "A few names are locks. The real fight is for the last roster spots — and fan ballots still matter.",
    graphic: "checklist",
    daysLeft: 2,
  },
  {
    id: "nba-summer-signal",
    league: "nba",
    headline: "What Summer League actually tells you",
    dateLabel: "JUL 24, 2026",
    summary:
      "Not every highlight is a breakout. How to separate real signal from Vegas noise before tip-off.",
    graphic: "arc",
  },
  {
    id: "wnba-standings",
    league: "wnba",
    headline: "Playoff math is getting real",
    dateLabel: "JUL 23, 2026",
    summary:
      "Seeds are tightening. Here's which teams still control their path — and which need help.",
    graphic: "diamond",
  },
];

const DEFAULT_STORY_LIMIT = 3;

function resolveStories(stories: Story[] | undefined): Story[] {
  if (stories === undefined) {
    return DEFAULT_STORIES.slice(0, DEFAULT_STORY_LIMIT);
  }
  return normalizeLiveGames(stories);
}

function StoryCard({ story }: { story: Story }) {
  return (
    <article className="rounded-xl border border-white/10 px-5 py-6">
      <p className="text-[11px] font-medium tracking-wide text-white/35 uppercase">
        {story.league} · {story.dateLabel}
      </p>
      <h3 className="mt-3 text-base font-semibold leading-snug tracking-tight text-white sm:text-lg">
        {story.headline}
      </h3>
      <p className="mt-2 text-sm leading-relaxed text-white/45">{story.summary}</p>
    </article>
  );
}

export function StoriesSection({ stories }: StoriesSectionProps) {
  const list = resolveStories(stories);

  return (
    <section
      id="stories"
      className="mx-auto max-w-6xl border-t border-white/10 px-4 py-16 sm:px-6 sm:py-20"
    >
      <SectionHeading
        title="Stories"
        subtitle="A few things worth knowing before tip-off."
      />

      {list.length === 0 ? (
        <p className="text-sm text-white/40">No stories yet.</p>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {list.map((story) => (
            <StoryCard key={story.id} story={story} />
          ))}
        </div>
      )}
    </section>
  );
}
