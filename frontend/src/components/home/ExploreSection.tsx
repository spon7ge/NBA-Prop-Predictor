import type { LucideIcon } from "lucide-react";
import {
  Activity,
  BarChart3,
  ChartNoAxesCombined,
  ListOrdered,
  Radar,
  Sparkles,
} from "lucide-react";
import type {
  ExploreGraphic,
  ExploreItem,
  HomeLeague,
} from "./types";
import { normalizeLiveGames } from "./format";
import { SectionHeading } from "./SectionHeading";

type ExploreSectionProps = {
  items?: ExploreItem[];
};

const EXPLORE_SKELETON_COUNT = 4;

const leaguePill: Record<HomeLeague, string> = {
  nba: "bg-sky-600/90 text-white",
  wnba: "bg-orange-500/90 text-white",
};

const graphicMeta: Record<
  ExploreGraphic,
  { Icon: LucideIcon; color: string; glow: string }
> = {
  chart: {
    Icon: ChartNoAxesCombined,
    color: "text-sky-400",
    glow: "from-sky-500/20",
  },
  bars: {
    Icon: BarChart3,
    color: "text-orange-400",
    glow: "from-orange-500/20",
  },
  dots: {
    Icon: Sparkles,
    color: "text-sky-300",
    glow: "from-sky-400/15",
  },
  standings: {
    Icon: ListOrdered,
    color: "text-orange-300",
    glow: "from-orange-400/15",
  },
  pulse: {
    Icon: Activity,
    color: "text-sky-400",
    glow: "from-cyan-500/20",
  },
  radar: {
    Icon: Radar,
    color: "text-orange-400",
    glow: "from-rose-500/15",
  },
};

/** Default marketing explore cards when no `items` prop is provided. */
export const DEFAULT_EXPLORE_ITEMS: ExploreItem[] = [
  {
    id: "nba-props-explained",
    league: "nba",
    headline: "Player props, explained",
    summary:
      "Lines, over/unders, and why the number on the board isn’t always the story. A quick primer before you dig into tonight’s slate.",
    graphic: "pulse",
    featured: true,
  },
  {
    id: "nba-usage",
    league: "nba",
    headline: "Usage spikes to watch",
    summary:
      "When minutes and touches jump, props follow. Here’s how to spot the bump before tip-off.",
    graphic: "chart",
  },
  {
    id: "wnba-pace",
    league: "wnba",
    headline: "Pace and the over",
    summary:
      "Fast games stack counting stats. Read tempo signals that push totals and combos higher.",
    graphic: "bars",
  },
  {
    id: "nba-defense",
    league: "nba",
    headline: "Matchup defense, decoded",
    summary:
      "Which schemes choke threes, paint touches, or assists — and how that shows up in the line.",
    graphic: "dots",
  },
  {
    id: "wnba-clutch",
    league: "wnba",
    headline: "WNBA clutch minutes",
    summary:
      "Who actually closes games — and which late-game roles quietly move the needle on props.",
    graphic: "radar",
  },
  {
    id: "nba-standings",
    league: "nba",
    headline: "Playoff race pressure",
    summary:
      "Seeds tighten motivation. Teams fighting for positioning often play different rotations.",
    graphic: "standings",
  },
  {
    id: "wnba-standings",
    league: "wnba",
    headline: "WNBA standings watch",
    summary:
      "Who controls their path, who’s fading, and what that means for tonight’s minutes.",
    graphic: "standings",
  },
];

function resolveItems(items: ExploreItem[] | undefined): ExploreItem[] {
  if (items === undefined) return DEFAULT_EXPLORE_ITEMS;
  return normalizeLiveGames(items);
}

function ExploreCard({
  item,
  featured = false,
}: {
  item: ExploreItem;
  featured?: boolean;
}) {
  const { Icon, color, glow } = graphicMeta[item.graphic];

  return (
    <article
      className={`relative overflow-hidden rounded-xl border border-white/10 bg-[#141414] ${
        featured ? "p-6 sm:p-8" : "p-5"
      }`}
    >
      <div
        className={`pointer-events-none absolute inset-y-0 right-0 w-1/2 bg-gradient-to-l ${glow} to-transparent`}
        aria-hidden
      />
      <div
        className={`relative flex gap-4 ${featured ? "sm:gap-8" : ""}`}
      >
        <div className="min-w-0 flex-1 space-y-2.5">
          <div className="flex flex-wrap items-center gap-2">
            <span
              className={`rounded-full px-2 py-0.5 text-[10px] font-semibold tracking-wide uppercase ${leaguePill[item.league]}`}
            >
              {item.league}
            </span>
            {featured ? (
              <span className="rounded-full bg-white/10 px-2 py-0.5 text-[10px] font-semibold tracking-wide text-white/70 uppercase">
                Featured
              </span>
            ) : null}
          </div>
          <h3
            className={`font-heading font-semibold leading-snug text-white ${
              featured
                ? "text-2xl sm:text-3xl"
                : "text-base sm:text-lg"
            }`}
          >
            {item.headline}
          </h3>
          <p
            className={`leading-relaxed text-white/50 ${
              featured ? "max-w-xl text-sm sm:text-base" : "text-sm"
            }`}
          >
            {item.summary}
          </p>
        </div>
        <div
          className={`flex shrink-0 items-center justify-center self-center ${color} ${
            featured ? "size-20 sm:size-28" : "size-16 sm:size-20"
          }`}
          aria-hidden
        >
          <Icon
            className={featured ? "size-12 sm:size-16" : "size-10 sm:size-12"}
            strokeWidth={1.25}
          />
        </div>
      </div>
    </article>
  );
}

function SkeletonExploreCard({ featured = false }: { featured?: boolean }) {
  return (
    <article
      className={`rounded-xl border border-white/10 bg-[#141414] ${
        featured ? "p-6 sm:p-8" : "p-5"
      }`}
      aria-hidden
    >
      <div className="flex gap-4">
        <div className="min-w-0 flex-1 space-y-3">
          <span className="block h-4 w-14 animate-pulse rounded-full bg-white/10" />
          <span
            className={`block animate-pulse rounded bg-white/10 ${
              featured ? "h-7 w-3/4" : "h-5 w-2/3"
            }`}
          />
          <span className="block h-3 w-full animate-pulse rounded bg-white/10" />
          <span className="block h-3 w-5/6 animate-pulse rounded bg-white/10" />
        </div>
        <span
          className={`shrink-0 animate-pulse rounded-lg bg-white/10 ${
            featured ? "size-20 sm:size-28" : "size-16"
          }`}
        />
      </div>
    </article>
  );
}

export function ExploreSection({ items }: ExploreSectionProps) {
  const list = resolveItems(items);
  const isEmpty = list.length === 0;
  const featured = list.find((item) => item.featured);
  const gridItems = list.filter((item) => !item.featured);

  return (
    <section id="explore" className="mx-auto max-w-6xl px-4 py-10 sm:px-6">
      <SectionHeading title="Explore" />

      {isEmpty ? (
        <>
          <p className="mb-4 text-sm text-white/40">
            Browse leagues and topics — coming soon.
          </p>
          <div className="space-y-4">
            <SkeletonExploreCard featured />
            <div className="grid gap-4 sm:grid-cols-2">
              {Array.from({ length: EXPLORE_SKELETON_COUNT }, (_, i) => (
                <SkeletonExploreCard key={i} />
              ))}
            </div>
          </div>
        </>
      ) : (
        <div className="space-y-4">
          {featured ? (
            <ExploreCard item={featured} featured />
          ) : null}
          {gridItems.length > 0 ? (
            <div className="grid gap-4 sm:grid-cols-2">
              {gridItems.map((item) => (
                <ExploreCard key={item.id} item={item} />
              ))}
            </div>
          ) : null}
        </div>
      )}
    </section>
  );
}
