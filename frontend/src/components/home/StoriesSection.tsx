import type { LucideIcon } from "lucide-react";
import {
  Clock,
  Crown,
  Goal,
  ListChecks,
  Network,
  Trophy,
  Waypoints,
} from "lucide-react";
import type { HomeLeague, Story, StoryGraphic } from "./types";
import { normalizeLiveGames } from "./format";
import { SectionHeading } from "./SectionHeading";

type StoriesSectionProps = {
  stories?: Story[];
};

const leaguePill: Record<HomeLeague, string> = {
  nba: "bg-sky-600/90 text-white",
  wnba: "bg-orange-500/90 text-white",
};

const graphicMeta: Record<
  StoryGraphic,
  { Icon: LucideIcon; color: string }
> = {
  bracket: { Icon: Trophy, color: "text-sky-400" },
  crown: { Icon: Crown, color: "text-sky-400" },
  arc: { Icon: Goal, color: "text-sky-400" },
  trade: { Icon: Network, color: "text-orange-400" },
  diamond: { Icon: Waypoints, color: "text-orange-400" },
  checklist: { Icon: ListChecks, color: "text-orange-400" },
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

function resolveStories(stories: Story[] | undefined): Story[] {
  if (stories === undefined) return DEFAULT_STORIES;
  return normalizeLiveGames(stories);
}

function StoryCard({ story }: { story: Story }) {
  const { Icon, color } = graphicMeta[story.graphic];

  return (
    <article className="flex gap-4 rounded-xl border border-white/10 bg-[#141414] p-5">
      <div className="min-w-0 flex-1 space-y-2.5">
        <div className="flex flex-wrap items-center gap-2">
          <span
            className={`rounded-full px-2 py-0.5 text-[10px] font-semibold tracking-wide uppercase ${leaguePill[story.league]}`}
          >
            {story.league}
          </span>
          {story.daysLeft != null ? (
            <span className="inline-flex items-center gap-1 rounded-full bg-red-600/90 px-2 py-0.5 text-[10px] font-semibold tracking-wide text-white uppercase">
              <Clock className="size-2.5" aria-hidden />
              {story.daysLeft} days left
            </span>
          ) : null}
        </div>
        <h3 className="text-base font-semibold leading-snug text-white sm:text-lg">
          {story.headline}
        </h3>
        <p className="text-[11px] font-medium tracking-wide text-white/40 uppercase">
          {story.dateLabel}
        </p>
        <p className="text-sm leading-relaxed text-white/50">{story.summary}</p>
      </div>
      <div
        className={`flex size-16 shrink-0 items-center justify-center self-center sm:size-20 ${color}`}
        aria-hidden
      >
        <Icon className="size-10 sm:size-12" strokeWidth={1.25} />
      </div>
    </article>
  );
}

export function StoriesSection({ stories }: StoriesSectionProps) {
  const list = resolveStories(stories);

  return (
    <section id="stories" className="mx-auto max-w-6xl px-4 py-10 sm:px-6">
      <SectionHeading title="Stories" />

      {list.length === 0 ? (
        <p className="text-sm text-white/40">No stories yet.</p>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2">
          {list.map((story) => (
            <StoryCard key={story.id} story={story} />
          ))}
        </div>
      )}
    </section>
  );
}
