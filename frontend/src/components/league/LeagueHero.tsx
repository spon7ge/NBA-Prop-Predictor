import nbaLogo from "@/assets/nba_logo.png";
import wnbaLogo from "@/assets/wnba_logo.png";
import type { LeagueSlug } from "./types";

type LeagueHeroProps = {
  league: LeagueSlug;
  /** YYYY-MM-DD slate date from the scoreboard (ET). Falls back to ET "now". */
  dateEt?: string | null;
};

const leagueContent = {
  wnba: {
    label: "WNBA",
    title: "Women’s Basketball",
    blurb:
      "Tonight's matchups, league leaders, and standings, plus the playoff race and a clutch tab for who delivers late in tight games.",
    image: wnbaLogo,
  },
  nba: {
    label: "NBA",
    title: "Men’s Basketball",
    blurb:
      "Tonight's matchups, league leaders, standings, and the playoff race—all in one place.",
    image: nbaLogo,
  },
  mlb: {
    label: "MLB",
    title: "Major League Baseball",
    blurb:
      "Tonight's matchups and live scores—standings, leaders, and props coming soon.",
    image: "https://a.espncdn.com/i/teamlogos/leagues/500/mlb.png",
  },
} as const;

/** Format a YYYY-MM-DD (or Date) as `WED, JUL 29` in America/New_York. */
export function formatSlateDateLabel(dateEt?: string | null): string {
  const date =
    dateEt && /^\d{4}-\d{2}-\d{2}$/.test(dateEt)
      ? new Date(`${dateEt}T12:00:00-04:00`)
      : new Date();
  return new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York",
    weekday: "short",
    month: "short",
    day: "numeric",
  })
    .format(date)
    .toUpperCase();
}

export function LeagueHero({ league, dateEt }: LeagueHeroProps) {
  const content = leagueContent[league];
  const dateLabel = formatSlateDateLabel(dateEt);

  return (
    <section className="mx-auto max-w-6xl px-4 pt-12 sm:px-6 sm:pt-16">
      <div className="grid items-center gap-8 border-b border-white/10 pb-10 sm:grid-cols-[1.2fr_0.8fr] sm:gap-12 sm:pb-12">
        <div className="max-w-2xl">
          <div className="mb-4 flex flex-wrap items-center gap-3">
            <span className="text-[11px] font-medium tracking-[0.18em] text-white/40 uppercase">
              {content.label}
            </span>
            <time
              dateTime={dateEt ?? undefined}
              className="text-[11px] font-medium tracking-[0.12em] text-white/35 uppercase"
            >
              {dateLabel}
            </time>
          </div>
          <h1 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">
            {content.title}
          </h1>
          <p className="mt-3 max-w-xl text-sm leading-relaxed text-white/45 sm:text-base">
            {content.blurb}
          </p>
        </div>

        <div className="flex justify-center sm:justify-end">
          <img
            src={content.image}
            alt={`${content.label} logo`}
            className="size-36 object-contain sm:size-44 lg:size-52"
          />
        </div>
      </div>
    </section>
  );
}
