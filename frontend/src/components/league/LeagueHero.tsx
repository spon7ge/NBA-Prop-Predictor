import basketball from "@/assets/basketball.png";
import wnbaBasketball from "@/assets/wnba_basketball.png";
import type { LeagueSlug } from "./types";

type LeagueHeroProps = {
  league: LeagueSlug;
};

const leagueContent = {
  wnba: {
    label: "WNBA",
    title: "Women’s Basketball",
    blurb:
      "Tonight's matchups, league leaders, and standings, plus the playoff race and a clutch tab for who delivers late in tight games.",
    image: wnbaBasketball,
    pillClassName: "bg-violet-500/15 text-violet-300 ring-violet-400/30",
  },
  nba: {
    label: "NBA",
    title: "Men’s Basketball",
    blurb:
      "Tonight's matchups, league leaders, standings, and the playoff race—all in one place.",
    image: basketball,
    pillClassName: "bg-sky-500/15 text-sky-300 ring-sky-400/30",
  },
} as const;

function formatToday() {
  return new Intl.DateTimeFormat("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
  })
    .format(new Date())
    .toUpperCase();
}

export function LeagueHero({ league }: LeagueHeroProps) {
  const content = leagueContent[league];

  return (
    <section className="mx-auto max-w-6xl px-4 pt-8 sm:px-6 sm:pt-10">
      <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-[#121212] px-6 py-8 sm:px-10 sm:py-10">
        <div className="relative z-10 max-w-2xl">
          <div className="mb-5 flex flex-wrap items-center gap-3">
            <span
              className={`rounded-full px-3 py-1 text-xs font-semibold tracking-wider ring-1 ${content.pillClassName}`}
            >
              {content.label}
            </span>
            <time className="text-xs font-medium tracking-[0.16em] text-white/45">
              {formatToday()}
            </time>
          </div>
          <h1 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">
            {content.title}
          </h1>
          <p className="mt-3 max-w-xl text-sm leading-relaxed text-white/55 sm:text-base">
            {content.blurb}
          </p>
        </div>

        <img
          src={content.image}
          alt=""
          aria-hidden
          className="pointer-events-none absolute -right-8 -bottom-20 size-64 object-contain opacity-[0.08] sm:right-8 sm:size-72"
        />
      </div>
    </section>
  );
}
